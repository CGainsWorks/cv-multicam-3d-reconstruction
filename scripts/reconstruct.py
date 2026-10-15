"""Sparse reconstruction of the turntable sweep, and its evaluation.

    python scripts/reconstruct.py --data data/turntable --out results

Reads the estimated rig if one exists in ``--out``, otherwise falls back to the
ground-truth rig and says so loudly -- reconstructing with the true rig and
quoting the number as if it came from the calibrated one would be the easiest
possible way to publish a flattering result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from multicam.evaluation.cad import (
    SurfaceDistance,
    completeness,
    register_to_cad,
    sparse_accuracy,
)
from multicam.evaluation.labels import OBJECT, composition, label_observations
from multicam.geometry import se3
from multicam.io.dataset import Dataset
from multicam.reconstruction import ReconstructionConfig, reconstruct
from multicam.types import RigCalibration


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--rig", default=None, help="rig yaml; default <out>/rig_estimated.yaml")
    p.add_argument("--truth-rig", action="store_true",
                   help="use the ground-truth rig instead of the calibrated one")
    p.add_argument("--stations", type=int, default=0, help="limit the sweep, 0 for all")
    p.add_argument("--contrast", type=float, default=0.025)
    p.add_argument("--no-second-pass", action="store_true")
    p.add_argument("--tag", default="sparse")
    p.add_argument("--quiet", action="store_true")
    return p


def load_rig(args, data, say):
    if args.truth_rig:
        say("using the GROUND-TRUTH rig (--truth-rig): this is an upper bound, not a result")
        return data.truth_rig, "truth"
    path = Path(args.rig) if args.rig else Path(args.out) / "rig_estimated.yaml"
    if path.exists():
        say("rig: %s (calibrated)" % path)
        return RigCalibration.load(path), str(path.as_posix())
    say("WARNING: %s does not exist, falling back to the ground-truth rig.  "
        "Run scripts/calibrate.py first for a result that means anything." % path)
    return data.truth_rig, "truth (fallback)"


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    data = Dataset(args.data)
    rig, rig_source = load_rig(args, data, say)

    frames = data.turntable_frames()
    if args.stations:
        frames = frames[: args.stations]
    images = {f: {c: data.turntable_image(f, c) for c in range(data.n_cameras)}
              for f in frames}

    config = ReconstructionConfig(
        contrast_threshold=args.contrast, second_pass=not args.no_second_pass
    )

    interactive = sys.stdout.isatty() and not args.quiet

    def progress(msg):
        if interactive:
            sys.stdout.write("\r  %-52s" % msg)
            sys.stdout.flush()

    t0 = time.time()
    rec = reconstruct(images, rig, config, progress=progress if interactive else None)
    if interactive:
        sys.stdout.write("\r%-54s\r" % "")
    elapsed = time.time() - t0

    say("\nSparse reconstruction")
    say("  %d stations, %d images" % (len(frames), len(frames) * data.n_cameras))
    say("  features per station: %s" % rec.stats["station_features"])
    say("  points per station:   %s" % rec.stats["station_points"])
    say("  descriptor matches %d -> %d after the epipolar gate (%.0f%% kept)"
        % (rec.stats["descriptor_matches"], rec.stats["epipolar_matches"],
           100.0 * rec.stats["epipolar_matches"] / max(rec.stats["descriptor_matches"], 1)))
    ratios = [v["inlier_ratio"] for v in rec.stats["alignments"].values()]
    say("  cross-station alignment inlier ratio: min %.2f, median %.2f, max %.2f"
        % (min(ratios), float(np.median(ratios)), max(ratios)))
    say("  tracks %d -> %d after requiring two or more stations"
        % (rec.stats["tracks_before_filter"], len(rec)))
    ba = rec.stats.get("bundle_second_pass", rec.stats["bundle"])
    say("  bundle adjustment: %d points, %d observations, rmse %.4f px "
        "(from %.4f), %d iterations, %s"
        % (ba["points"], ba["observations"], ba["rmse_px"], ba["initial_rmse_px"],
           ba["iterations"], ba["reason"]))
    say("  %.1f s total" % elapsed)

    # -- evaluation against the CAD model ----------------------------------
    cad = data.object_mesh
    T_world_object0 = data.T_world_object(frames[0])
    T_rig_world = data.T_rig_world
    # The bundle adjustment holds station 0's pose fixed, so the reconstruction
    # frame *is* the rig frame at station 0 -- exactly, by construction of the
    # gauge.  Ground truth therefore gives the transform into the CAD frame in
    # closed form, with no fitting at all, and that is the primary number.  ICP
    # is run afterwards as a diagnostic: how far it moves is how much of the
    # error is a rigid offset rather than shape.
    T_truth = np.linalg.inv(T_rig_world @ T_world_object0)

    masks = {s: {c: data.turntable_mask(f, c) for c in range(data.n_cameras)}
             for s, f in enumerate(frames)}
    labels, _ = label_observations(rec.observations, masks, len(rec))
    comp = composition(labels)
    say("\nWhat was reconstructed (ground-truth segmentation, evaluation only)")
    say("  %d points: %d on the part, %d on the turntable, %d background"
        % (comp["total"], comp["part"], comp["turntable"], comp["background"]))

    distance = SurfaceDistance(cad)
    on_part = labels == OBJECT
    points_truth = se3.transform(T_truth, rec.points)

    acc_all = sparse_accuracy(points_truth, cad, inlier_mm=5.0, distance=distance)
    acc_part = sparse_accuracy(points_truth[on_part], cad, inlier_mm=5.0, distance=distance)
    comp_part = completeness(cad, points_truth[on_part], threshold_mm=3.0)

    say("\nAgainst the CAD model, in the ground-truth frame (no fitting)")
    say("  part points (%d): median %.3f mm, mean %.3f mm, 90th pct %.3f mm, rms %.3f mm"
        % (int(on_part.sum()), acc_part["median_mm"], acc_part["mean_mm"],
           acc_part["p90_mm"], acc_part["rms_mm"]))
    say("  %.1f%% of part points within 5 mm of the surface"
        % (100.0 * acc_part["inlier_fraction"]))
    say("  surface completeness at 3 mm: %.1f%%" % (100.0 * comp_part["completeness"]))

    reg = register_to_cad(rec.points[on_part], cad, T_init_cad_from_recon=T_truth)
    acc_icp = sparse_accuracy(reg.points_in_cad, cad, inlier_mm=5.0, distance=distance)
    say("  after ICP (diagnostic): median %.3f mm, and ICP moved the pose by "
        "%.2f mm / %.3f deg" % (acc_icp["median_mm"], reg.correction_mm, reg.correction_deg))

    record = {
        "tag": args.tag,
        "dataset": str(Path(args.data).as_posix()),
        "rig_source": rig_source,
        "commit": _bootstrap.git_commit(),
        "versions": _bootstrap.library_versions(),
        "stations": len(frames),
        "seconds": round(elapsed, 2),
        "reconstruction": rec.stats,
        "composition": comp,
        "cad_accuracy_all_points": acc_all,
        "cad_accuracy_part": acc_part,
        "cad_completeness_part": comp_part,
        "cad_registration_icp": reg.summary(),
        "cad_accuracy_part_after_icp": acc_icp,
        "headline": {
            "points": int(len(rec)),
            "part_points": int(on_part.sum()),
            "reprojection_rmse_px": ba["rmse_px"],
            "cad_median_error_mm": acc_part["median_mm"],
            "cad_rms_error_mm": acc_part["rms_mm"],
            "cad_inlier_fraction_5mm": acc_part["inlier_fraction"],
            "cad_completeness_3mm": comp_part["completeness"],
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("%s.json" % args.tag)).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    np.savez_compressed(
        out / ("%s.npz" % args.tag),
        points=rec.points,
        points_cad=points_truth,
        labels=labels,
        descriptors=rec.descriptors,
        T_rig_object=np.asarray(rec.T_rig_object),
        T_cad_recon=T_truth,
        track_stations=rec.track_stations,
        track_views=rec.track_views,
        frames=np.asarray(rec.frames),
    )
    say("\nwrote %s and %s" % (out / ("%s.json" % args.tag), out / ("%s.npz" % args.tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
