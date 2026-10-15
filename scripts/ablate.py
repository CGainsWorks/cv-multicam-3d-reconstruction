"""The ablation: does the learned refiner earn its place?

    python scripts/ablate.py --data data/turntable --out results

Four experiments, on the held-out pose stations only.

**A. The deployment question.**  Classical alone, against classical followed by
the network.  This is the question the brief asks and it is the one that
decides whether the network ships.

**B. The controlled head-to-head.**  Both refiners started from the *same*
deliberately perturbed pose, across several magnitude bands.  A alone cannot
separate "the network is a better refiner" from "the classical estimate was
already at the noise floor and there was nothing left to correct".  B can.

**C. The iteration curve.**  Zero to four network steps.  Refiners are usually
quoted at whatever iteration count flatters them; the whole curve is cheaper to
report than an argument about it.

**D. Starved correspondences.**  The classical estimator is given a random
subset of its matches, simulating a poorly textured or partly occluded part,
and the network's contribution is measured as the classical stage degrades.
If the network is worth anything, this is where it shows.

Whatever the numbers say is what gets written to the README.  A refinement
network that does not beat a well implemented Levenberg-Marquardt on clean
synthetic data is an entirely ordinary outcome and saying so is worth more
than burying it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from multicam.evaluation.metrics import pose_error_table
from multicam.io.dataset import Dataset
from multicam.pose import ObjectModel, estimate_pose, match_to_model
from multicam.pose.classical import Correspondences, _reproject, refine_pose
from multicam.pose.dataset import PerturbationSpec
from multicam.pose.network import load_network
from multicam.pose.refine import refine
from multicam.pose.render_compare import ModelRenderer
from multicam.reconstruction import detect, make_detector
from multicam.types import RigCalibration


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--checkpoint", default=None, help="default <repo>/models/refiner.pt")
    p.add_argument("--rig", default=None)
    p.add_argument("--stations", type=int, default=0)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--max-iterations", type=int, default=4)
    p.add_argument("--contrast", type=float, default=0.025)
    p.add_argument("--ratio", type=float, default=0.80)
    p.add_argument("--threshold-px", type=float, default=3.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--only", default=None,
                   help="run one experiment: deployment, headtohead, iterations, starved")
    p.add_argument("--tag", default="ablation")
    p.add_argument("--quiet", action="store_true")
    return p


def summarise(estimates, truths, model_points, diameter, failures=0):
    pairs = [(e, t) for e, t in zip(estimates, truths) if e is not None]
    return pose_error_table(
        [e for e, _ in pairs], [t for _, t in pairs], model_points, diameter,
        failures=failures + (len(estimates) - len(pairs)),
    )


def line(name, table):
    return ("  %-28s ADD %7.3f mm | trans %7.3f mm | rot %7.4f deg | "
            "recall@2%% %5.1f%% | n %d"
            % (name, table["add_mm"]["median"], table["translation_mm"]["median"],
               table["rotation_deg"]["median"],
               100 * table["add_recall_2_pct_diameter"], table["n_solved"]))


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    data = Dataset(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rig_path = Path(args.rig) if args.rig else out / "rig_estimated.yaml"
    rig = RigCalibration.load(rig_path) if rig_path.exists() else data.truth_rig
    cad = data.object_mesh
    model_path = out / "object_model.npz"
    if not model_path.exists():
        raise SystemExit("%s not found; run scripts/pose_eval.py first" % model_path)
    model = ObjectModel.load(model_path)

    checkpoint = (Path(args.checkpoint) if args.checkpoint
                  else _bootstrap.ROOT / "models" / "refiner.pt")
    if not checkpoint.exists():
        raise SystemExit(
            "%s not found; run scripts/train_refiner.py first" % checkpoint
        )
    from multicam.pose.train import resolve_device

    device = resolve_device(args.device)
    net, blob = load_network(checkpoint, device=device)
    renderer = ModelRenderer(cad)
    say("refiner: %d parameters, trained %s, device %s"
        % (sum(p.numel() for p in net.parameters()),
           blob.get("config", {}).get("perturbation", {}), device))

    _, test_frames = data.pose_split()
    if args.stations:
        test_frames = test_frames[: args.stations]
    say("held-out stations: %d" % len(test_frames))

    T_rig_world = data.T_rig_world
    model_points, _ = cad.sample_surface(2048, np.random.default_rng(0))
    diameter = cad.diameter

    # -- shared setup: images, correspondences and the classical estimate ---
    detector = make_detector(contrast_threshold=args.contrast)
    images, correspondences, truths = {}, {}, {}
    interactive = sys.stdout.isatty() and not args.quiet
    t0 = time.time()
    for i, f in enumerate(test_frames):
        images[f] = {c: data.pose_image(f, c) for c in range(data.n_cameras)}
        correspondences[f] = [
            match_to_model(detect(images[f][c], detector, camera=c, frame=f), model,
                           ratio=args.ratio)
            for c in range(data.n_cameras)
        ]
        truths[f] = T_rig_world @ data.T_world_object_pose(f)
        if interactive:
            sys.stdout.write("\r  matching %d/%d " % (i + 1, len(test_frames)))
            sys.stdout.flush()
    if interactive:
        sys.stdout.write("\r%-32s\r" % "")
    say("matched %d stations in %.1f s (median %d correspondences)"
        % (len(test_frames), time.time() - t0,
           int(np.median([sum(len(c) for c in correspondences[f]) for f in test_frames]))))

    classical = {}
    for f in test_frames:
        r = estimate_pose(rig, correspondences[f], threshold_px=args.threshold_px,
                          rng=np.random.default_rng(f))
        classical[f] = r.T_rig_object if r.ok else None

    record = {
        "tag": args.tag,
        "dataset": str(Path(args.data).as_posix()),
        "commit": _bootstrap.git_commit(),
        "versions": _bootstrap.library_versions(),
        "checkpoint": str(checkpoint.as_posix()),
        "stations": len(test_frames),
        "device": device,
        "network": blob.get("stats", {}).get("network", {}),
        "training": blob.get("config", {}),
    }
    wanted = args.only

    # -- A. deployment -----------------------------------------------------
    if wanted in (None, "deployment"):
        say("\nA. Classical alone against classical plus the network")
        base = [classical[f] for f in test_frames]
        truth_list = [truths[f] for f in test_frames]
        t0 = time.time()
        refined = [
            None if classical[f] is None else
            refine(net, renderer, images[f], rig, classical[f],
                   iterations=args.iterations, device=device)
            for f in test_frames
        ]
        refine_seconds = time.time() - t0

        table_base = summarise(base, truth_list, model_points, diameter)
        table_net = summarise(refined, truth_list, model_points, diameter)
        say(line("classical", table_base))
        say(line("classical + network (%d it)" % args.iterations, table_net))

        delta = table_net["add_mm"]["median"] - table_base["add_mm"]["median"]
        verdict = ("the network improves ADD by %.3f mm" % -delta) if delta < 0 else \
                  ("the network makes ADD worse by %.3f mm" % delta)
        say("  -> %s (%.1f%% relative)" % (
            verdict, 100 * abs(delta) / max(table_base["add_mm"]["median"], 1e-9)))
        record["deployment"] = {
            "classical": table_base,
            "classical_plus_network": table_net,
            "add_median_delta_mm": delta,
            "network_helps": bool(delta < 0),
            "refine_seconds_total": round(refine_seconds, 2),
            "refine_ms_per_station": round(1000 * refine_seconds / max(len(test_frames), 1), 1),
            "iterations": args.iterations,
        }

    # -- B. head to head ---------------------------------------------------
    if wanted in (None, "headtohead"):
        say("\nB. Same perturbed start, classical LM against the network")
        bands = [
            ("0.5 mm / 0.3 deg", 0.0005, 0.3),
            ("2 mm / 1 deg", 0.002, 1.0),
            ("5 mm / 3 deg", 0.005, 3.0),
            ("12 mm / 6 deg", 0.012, 6.0),
        ]
        record["head_to_head"] = []
        for name, t_mag, r_mag in bands:
            spec = PerturbationSpec(t_mag, t_mag, r_mag, r_mag)
            starts, lm_out, net_out, truth_list = [], [], [], []
            for f in test_frames:
                rng = np.random.default_rng([7, int(f)])
                T_start = truths[f] @ spec.sample(rng)
                starts.append(T_start)
                truth_list.append(truths[f])
                # Classical refinement from the same start, on the same
                # correspondences.  `refine_pose` directly, not
                # `estimate_pose(initial=...)`: the latter reselects inliers at
                # the start pose and *refuses* if too few are within threshold,
                # which from a 12 mm start it almost always is.  That turned
                # this experiment into a comparison between a refiner that
                # always answers and one that answers two times in forty, and
                # the two medians were then over completely different subsets.
                #
                # Inliers are reselected at a threshold scaled to the
                # perturbation, so the classical refiner is not handed a
                # correspondence set chosen by the answer.
                errs = _reproject(rig, T_start, correspondences[f])
                gate = max(args.threshold_px, 3.0 * t_mag * 1000.0)
                inliers = {c.camera: errs[c.camera] < gate for c in correspondences[f]}
                T_lm, _ = refine_pose(rig, T_start, correspondences[f], inliers,
                                      huber_px=args.threshold_px)
                lm_out.append(T_lm)
                net_out.append(
                    refine(net, renderer, images[f], rig, T_start,
                           iterations=args.iterations, device=device)
                )
            t_start = summarise(starts, truth_list, model_points, diameter)
            t_lm = summarise(lm_out, truth_list, model_points, diameter)
            t_net = summarise(net_out, truth_list, model_points, diameter)
            say("  start %-18s" % name)
            say(line("    perturbed start", t_start))
            say(line("    classical LM", t_lm))
            say(line("    network", t_net))
            record["head_to_head"].append({
                "band": name,
                "translation_m": t_mag,
                "rotation_deg": r_mag,
                "start": t_start,
                "classical_lm": t_lm,
                "network": t_net,
                "network_better": bool(
                    t_net["add_mm"]["median"] < t_lm["add_mm"]["median"]
                ),
            })

    # -- C. iteration curve -------------------------------------------------
    if wanted in (None, "iterations"):
        say("\nC. How many network iterations")
        record["iterations"] = []
        for k in range(0, int(args.max_iterations) + 1):
            refined = [
                None if classical[f] is None else
                (classical[f] if k == 0 else
                 refine(net, renderer, images[f], rig, classical[f],
                        iterations=k, device=device))
                for f in test_frames
            ]
            table = summarise(refined, [truths[f] for f in test_frames],
                              model_points, diameter)
            say(line("%d iteration(s)" % k, table))
            record["iterations"].append({"iterations": k, "metrics": table})

    # -- D. starved correspondences ----------------------------------------
    if wanted in (None, "starved"):
        say("\nD. As the classical stage is starved of correspondences")
        record["starved"] = []
        for keep in (1.0, 0.35, 0.15, 0.07):
            base, refined, truth_list = [], [], []
            for f in test_frames:
                rng = np.random.default_rng([11, int(f), int(keep * 1000)])
                thinned = []
                for corr in correspondences[f]:
                    if not len(corr):
                        thinned.append(corr)
                        continue
                    n_keep = max(4, int(round(len(corr) * keep)))
                    idx = rng.choice(len(corr), size=min(n_keep, len(corr)), replace=False)
                    thinned.append(Correspondences(corr.camera, corr.xy[idx],
                                                   corr.points[idx]))
                r = estimate_pose(rig, thinned, threshold_px=args.threshold_px,
                                  rng=np.random.default_rng(f))
                T = r.T_rig_object if r.ok else None
                base.append(T)
                refined.append(
                    None if T is None else
                    refine(net, renderer, images[f], rig, T,
                           iterations=args.iterations, device=device)
                )
                truth_list.append(truths[f])
            t_base = summarise(base, truth_list, model_points, diameter)
            t_net = summarise(refined, truth_list, model_points, diameter)
            say("  keeping %.0f%% of correspondences" % (100 * keep))
            say(line("    classical", t_base))
            say(line("    classical + network", t_net))
            record["starved"].append({
                "keep_fraction": keep,
                "classical": t_base,
                "classical_plus_network": t_net,
                "add_median_delta_mm": (t_net["add_mm"]["median"]
                                        - t_base["add_mm"]["median"]),
            })

    path = out / ("%s.json" % args.tag)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=float),
                    encoding="utf-8")
    say("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
