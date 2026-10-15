"""Run COLMAP on the same images and compare it against our reconstruction.

    python scripts/colmap_baseline.py --data data/turntable --out results

COLMAP is the reference implementation for exactly this problem and it shares no
code with anything in ``src/multicam/reconstruction``.  Agreement between the
two is evidence that the geometry is right; disagreement means one of them has
a bug and is worth an afternoon.

The comparison is not symmetric and the report says so.  COLMAP solves for every
camera pose from scratch, which is a harder problem than ours -- we are handed
the calibrated rig -- and it recovers geometry only up to a similarity, so its
result has to be scaled before it can be compared to a metric one.  The scale
factor is reported, because it is the number that says what the rig calibration
was worth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from multicam.evaluation.cad import SurfaceDistance, completeness, sparse_accuracy
from multicam.geometry import se3
from multicam.io import colmap as colmap_io
from multicam.io.dataset import Dataset
from multicam.types import RigCalibration


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--workspace", default=None, help="default <repo>/colmap_workspace")
    p.add_argument("--sparse", default=None)
    p.add_argument("--rig", default=None)
    p.add_argument("--stations", type=int, default=0)
    p.add_argument("--max-image-size", type=int, default=1600)
    p.add_argument("--keep-workspace", action="store_true")
    p.add_argument("--tag", default="colmap")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print

    if not colmap_io.available():
        say("pycolmap is not installed, so the COLMAP baseline cannot run.")
        say("Install it with `pip install pycolmap` and re-run.  Every number "
            "this script would have produced is absent from the report rather "
            "than estimated.")
        record = {"tag": args.tag, "available": False,
                  "reason": "pycolmap is not installed"}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / ("%s.json" % args.tag)).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        return 0

    data = Dataset(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sparse_path = Path(args.sparse) if args.sparse else out / "sparse.npz"
    if not sparse_path.exists():
        raise SystemExit("%s not found; run scripts/reconstruct.py first" % sparse_path)
    ours = np.load(sparse_path)
    rig_path = Path(args.rig) if args.rig else out / "rig_estimated.yaml"
    rig = RigCalibration.load(rig_path) if rig_path.exists() else data.truth_rig

    frames = [int(f) for f in ours["frames"]]
    if args.stations:
        frames = frames[: args.stations]
    images = {f: {c: data.turntable_image(f, c) for c in range(data.n_cameras)}
              for f in frames}

    workspace = Path(args.workspace) if args.workspace else _bootstrap.ROOT / "colmap_workspace"
    say("running COLMAP over %d images in %s" % (len(frames) * data.n_cameras, workspace))
    result = colmap_io.run(images, workspace, max_image_size=args.max_image_size)

    say("  registered %d of %d images into %d model(s) in %.1f s"
        % (result.n_images_registered, result.n_images_total,
           result.stats.get("models", 0), result.seconds))
    if not len(result.points):
        say("  COLMAP produced no reconstruction; nothing to compare")
        (out / ("%s.json" % args.tag)).write_text(
            json.dumps({"tag": args.tag, "available": True,
                        "colmap": result.summary()}, indent=2, default=float),
            encoding="utf-8",
        )
        return 0

    say("  %d points, mean track length %.2f, mean reprojection error %.3f px"
        % (len(result.points), result.mean_track_length,
           result.mean_reprojection_error_px))

    # -- align COLMAP into our object frame --------------------------------
    T_rig_object = np.asarray(ours["T_rig_object"])
    reference_poses = {}
    for si, f in enumerate(frames):
        if si >= len(T_rig_object):
            break
        for c in range(data.n_cameras):
            reference_poses[(f, c)] = rig.T_cam_rig[c] @ T_rig_object[si]

    index = {k: tuple(v) for k, v in result.stats["image_index"].items()}
    T_align, scale, align_stats = colmap_io.align_by_camera_centres(
        result, reference_poses, index
    )
    say("  aligned to our frame using %d camera centres: scale %.6f, "
        "centre rmse %.3f mm (max %.3f mm)"
        % (align_stats["cameras_aligned"], scale, align_stats["centre_rmse_mm"],
           align_stats["centre_max_mm"]))

    points_ours_frame = colmap_io.apply_similarity(T_align, scale, result.points)

    # -- score both against the CAD model -----------------------------------
    cad = data.object_mesh
    distance = SurfaceDistance(cad)
    T_truth = np.linalg.inv(data.T_rig_world @ data.T_world_object(frames[0]))

    colmap_cad = se3.transform(T_truth, points_ours_frame)
    # COLMAP reconstructs the whole turning assembly, as we do, so the same
    # 6 mm band around the CAD surface is used to isolate the part in both.

    def on_part(points):
        return distance(points) * 1000.0 <= 6.0

    ours_cad = np.asarray(ours["points_cad"])
    labels = np.asarray(ours["labels"]) if "labels" in ours else None
    ours_part = ours_cad[labels == 0] if labels is not None else ours_cad[on_part(ours_cad)]
    colmap_part = colmap_cad[on_part(colmap_cad)]

    acc_ours = sparse_accuracy(ours_part, cad, inlier_mm=5.0, distance=distance)
    acc_colmap = sparse_accuracy(colmap_part, cad, inlier_mm=5.0, distance=distance)
    comp_ours = completeness(cad, ours_part, threshold_mm=3.0)
    comp_colmap = completeness(cad, colmap_part, threshold_mm=3.0)

    say("\nAgainst the CAD model, points within 6 mm of the surface")
    say("  %-8s %6d points  median %.3f mm  rms %.3f mm  completeness@3mm %.1f%%"
        % ("ours", acc_ours["points"], acc_ours["median_mm"], acc_ours["rms_mm"],
           100 * comp_ours["completeness"]))
    say("  %-8s %6d points  median %.3f mm  rms %.3f mm  completeness@3mm %.1f%%"
        % ("COLMAP", acc_colmap["points"], acc_colmap["median_mm"], acc_colmap["rms_mm"],
           100 * comp_colmap["completeness"]))

    # -- camera pose agreement ----------------------------------------------
    pose_errors = []
    for name, T_cam_colmap in result.image_poses.items():
        key = index.get(name)
        if key is None or tuple(key) not in reference_poses:
            continue
        # COLMAP's camera centre, brought into our frame.
        centre = colmap_io.apply_similarity(
            T_align, scale, se3.camera_centre(T_cam_colmap)[None, :]
        )[0]
        ours_centre = se3.camera_centre(reference_poses[tuple(key)])
        pose_errors.append(float(np.linalg.norm(centre - ours_centre)))
    pose_errors = np.asarray(pose_errors)
    say("\nCamera centre agreement over %d images: median %.3f mm, 90th pct %.3f mm"
        % (len(pose_errors), np.median(pose_errors) * 1000,
           np.quantile(pose_errors, 0.9) * 1000))

    record = {
        "tag": args.tag,
        "available": True,
        "dataset": str(Path(args.data).as_posix()),
        "commit": _bootstrap.git_commit(),
        "versions": _bootstrap.library_versions(),
        "pycolmap": colmap_io.version(),
        "colmap": {k: v for k, v in result.summary().items() if k != "image_index"},
        "alignment": {"scale": scale, **align_stats},
        "cad_accuracy": {"ours": acc_ours, "colmap": acc_colmap},
        "cad_completeness_3mm": {"ours": comp_ours, "colmap": comp_colmap},
        "camera_centre_agreement_mm": {
            "median": float(np.median(pose_errors) * 1000),
            "p90": float(np.quantile(pose_errors, 0.9) * 1000),
            "max": float(pose_errors.max() * 1000),
            "images": int(len(pose_errors)),
        },
        "caveats": [
            "COLMAP solves for every camera pose; ours is given the calibrated rig.",
            "COLMAP's reconstruction is metric only after the reported scale factor "
            "is applied; ours is metric from the rig calibration.",
            "Both were run on identical images.",
        ],
        "headline": {
            "colmap_scale": scale,
            "colmap_median_mm": acc_colmap["median_mm"],
            "ours_median_mm": acc_ours["median_mm"],
            "camera_centre_median_mm": float(np.median(pose_errors) * 1000),
        },
    }
    (out / ("%s.json" % args.tag)).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    np.savez_compressed(out / ("%s.npz" % args.tag), points_cad=colmap_cad,
                        T_align=T_align, scale=np.asarray([scale]))
    say("\nwrote %s" % (out / ("%s.json" % args.tag)))

    if not args.keep_workspace and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
        say("removed the COLMAP workspace (--keep-workspace to keep it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
