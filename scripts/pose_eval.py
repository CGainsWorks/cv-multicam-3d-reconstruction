"""Classical 6-DoF pose estimation on the held-out pose stations.

    python scripts/pose_eval.py --data data/turntable --out results

Builds the appearance model from the sparse reconstruction, keeps the points
that lie on the CAD surface, then for every held-out station: detect, match to
the model, generalised-PnP under RANSAC across the whole rig, refine.

The per-station estimates are written out, because ``scripts/ablate.py`` starts
from them and re-running this to get them again would be both slow and a source
of disagreement between the two reports.
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
from multicam.geometry import se3
from multicam.io.dataset import Dataset
from multicam.pose import build_model, estimate_pose, match_to_model
from multicam.reconstruction import detect, make_detector
from multicam.types import RigCalibration


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--sparse", default=None)
    p.add_argument("--rig", default=None)
    p.add_argument("--split", choices=("test", "train", "all"), default="test")
    p.add_argument("--stations", type=int, default=0, help="limit, 0 for all")
    p.add_argument("--ratio", type=float, default=0.80)
    p.add_argument("--threshold-px", type=float, default=3.0)
    p.add_argument("--contrast", type=float, default=0.025)
    p.add_argument("--surface-mm", type=float, default=4.0)
    p.add_argument("--tag", default="pose_classical")
    p.add_argument("--quiet", action="store_true")
    return p


def load_model(data, out_dir, sparse_path, surface_mm, say):
    """The appearance model, from the sparse reconstruction, filtered to the CAD."""
    from multicam.evaluation.cad import SurfaceDistance

    sparse = np.load(sparse_path)
    cad = data.object_mesh
    distance = SurfaceDistance(cad)
    model = build_model(
        sparse["points_cad"], sparse["descriptors"], cad,
        surface_mm=surface_mm, distance=distance,
    )
    say("object model: %d points kept of %d reconstructed (median %.2f mm from the "
        "CAD surface)" % (model.stats["on_surface"], model.stats["candidates"],
                          model.stats["median_distance_mm"]))
    model.save(Path(out_dir) / "object_model.npz")
    return model, cad


def run_frames(data, rig, model, frames, args, images_loader, say):
    """Estimate a pose for each frame.  Returns estimates, truths and per-frame stats."""
    detector = make_detector(contrast_threshold=args.contrast)
    T_rig_world = data.T_rig_world

    estimates, truths, rows = [], [], []
    interactive = sys.stdout.isatty() and not args.quiet
    for i, frame in enumerate(frames):
        images = images_loader(frame)
        correspondences = []
        for c in range(data.n_cameras):
            features = detect(images[c], detector, camera=c, frame=frame)
            correspondences.append(match_to_model(features, model, ratio=args.ratio))

        t0 = time.time()
        result = estimate_pose(
            rig, correspondences, threshold_px=args.threshold_px,
            rng=np.random.default_rng(frame),
        )
        elapsed = time.time() - t0

        T_true = T_rig_world @ data.T_world_object_pose(frame)
        estimates.append(result.T_rig_object if result.ok else None)
        truths.append(T_true)
        row = {"frame": int(frame), "seconds": round(elapsed, 3), **result.summary()}
        if result.ok:
            t_err, r_err = se3.pose_error(result.T_rig_object, T_true)
            row["translation_error_mm"] = t_err * 1000.0
            row["rotation_error_deg"] = r_err
        rows.append(row)
        if interactive:
            sys.stdout.write("\r  station %d/%d  %s   " % (
                i + 1, len(frames),
                ("%.2f mm" % row.get("translation_error_mm", float("nan")))
                if result.ok else "FAILED"))
            sys.stdout.flush()
    if interactive:
        sys.stdout.write("\r%-42s\r" % "")
    return estimates, truths, rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    data = Dataset(args.data)

    sparse_path = Path(args.sparse) if args.sparse else Path(args.out) / "sparse.npz"
    if not sparse_path.exists():
        raise SystemExit("%s not found; run scripts/reconstruct.py first" % sparse_path)
    rig_path = Path(args.rig) if args.rig else Path(args.out) / "rig_estimated.yaml"
    rig = RigCalibration.load(rig_path) if rig_path.exists() else data.truth_rig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cad = load_model(data, out, sparse_path, args.surface_mm, say)

    train_frames, test_frames = data.pose_split()
    frames = {"test": test_frames, "train": train_frames,
              "all": sorted(set(train_frames) | set(test_frames))}[args.split]
    if args.stations:
        frames = frames[: args.stations]
    say("evaluating %d %s stations" % (len(frames), args.split))

    t0 = time.time()
    estimates, truths, rows = run_frames(
        data, rig, model, frames, args,
        lambda f: {c: data.pose_image(f, c) for c in range(data.n_cameras)}, say,
    )
    elapsed = time.time() - t0

    model_points, _ = cad.sample_surface(2048, np.random.default_rng(0))
    failures = sum(1 for e in estimates if e is None)
    table = pose_error_table(
        [e for e in estimates if e is not None],
        [t for e, t in zip(estimates, truths) if e is not None],
        model_points, cad.diameter, failures=failures,
    )

    say("\nClassical pose, %d stations (%d failed)" % (len(frames), failures))
    say("  translation: median %.3f mm, mean %.3f mm, 90th pct %.3f mm"
        % (table["translation_mm"]["median"], table["translation_mm"]["mean"],
           table["translation_mm"]["p90"]))
    say("  rotation:    median %.4f deg, mean %.4f deg, 90th pct %.4f deg"
        % (table["rotation_deg"]["median"], table["rotation_deg"]["mean"],
           table["rotation_deg"]["p90"]))
    say("  ADD:  median %.3f mm, mean %.3f mm" % (table["add_mm"]["median"],
                                                  table["add_mm"]["mean"]))
    say("  ADD-S median %.3f mm" % table["adds_mm"]["median"])
    say("  recall: %.1f%% at 2%% of diameter (%.2f mm), %.1f%% at 1 mm"
        % (100 * table["add_recall_2_pct_diameter"], 0.02 * cad.diameter * 1000,
           100 * table["add_recall_1mm"]))
    solved = [r for r in rows if r["solved"]]
    if solved:
        say("  correspondences: median %d, inlier ratio median %.2f"
            % (int(np.median([r["correspondences"] for r in solved])),
               float(np.median([r["inlier_ratio"] for r in solved]))))
        say("  %.0f ms per station" % (1000 * float(np.mean([r["seconds"] for r in solved]))))

    record = {
        "tag": args.tag,
        "dataset": str(Path(args.data).as_posix()),
        "split": args.split,
        "commit": _bootstrap.git_commit(),
        "versions": _bootstrap.library_versions(),
        "model": model.stats,
        "settings": {"ratio": args.ratio, "threshold_px": args.threshold_px,
                     "contrast": args.contrast, "surface_mm": args.surface_mm},
        "seconds": round(elapsed, 2),
        "metrics": table,
        "per_station": rows,
        "headline": {
            "stations": len(frames),
            "failures": failures,
            "translation_median_mm": table["translation_mm"]["median"],
            "rotation_median_deg": table["rotation_deg"]["median"],
            "add_median_mm": table["add_mm"]["median"],
            "add_recall_2pct": table["add_recall_2_pct_diameter"],
        },
    }
    (out / ("%s.json" % args.tag)).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    np.savez_compressed(
        out / ("%s.npz" % args.tag),
        frames=np.asarray(frames),
        estimates=np.asarray([e if e is not None else np.full((4, 4), np.nan)
                              for e in estimates]),
        truths=np.asarray(truths),
        solved=np.asarray([e is not None for e in estimates]),
    )
    say("\nwrote %s and %s" % (out / ("%s.json" % args.tag), out / ("%s.npz" % args.tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
