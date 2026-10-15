"""Calibrate the rig from a capture and report the residuals.

    python scripts/calibrate.py --data data/turntable --out results

Two stages, in the order you would do them in front of real hardware:

1. Per-camera intrinsics from that camera's own board sweep, cross-checked
   against ``cv2.calibrateCamera`` on identical detections.
2. Rig extrinsics from the shared board stations, initialised through a
   maximum-weight spanning tree over the camera co-observation graph and then
   bundle adjusted.

Because the capture is synthetic the ground-truth rig is on disk, so the report
carries both the *residual* -- which you would have on real hardware -- and the
*error against truth* -- which you would not.  Keeping both is the point: a
calibration can have a beautiful residual and be wrong, and this is where that
shows up.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from multicam.calibration import CharucoTarget, calibrate_extrinsics, calibrate_intrinsics
from multicam.calibration.extrinsic import compare_rigs
from multicam.io.dataset import Dataset
from multicam.sim.textures import charuco_board


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--huber", type=float, default=2.0,
                   help="robust kernel threshold in pixels")
    p.add_argument("--refine-intrinsics-jointly", action="store_true",
                   help="also free the intrinsics in the extrinsic solve, and report both")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = Dataset(args.data)
    cfg = data.meta["config"]
    target = CharucoTarget(
        charuco_board(
            squares_x=cfg["board_squares"][0], squares_y=cfg["board_squares"][1],
            square_length=cfg["square_length_m"],
        )
    )
    truth = data.truth_rig
    say = (lambda *a: None) if args.quiet else print

    record = {
        "dataset": str(Path(args.data).as_posix()),
        "commit": _bootstrap.git_commit(),
        "versions": _bootstrap.library_versions(),
        "board": {"squares": list(target.size), "square_length_m": target.square_length,
                  "corners": target.n_corners},
    }

    # -- intrinsics --------------------------------------------------------
    t0 = time.time()
    say("Per-camera intrinsics")
    intrinsics, cameras = [], []
    for c in range(data.n_cameras):
        detections = []
        for k, _ in enumerate(data.intrinsic_views(c)):
            det = target.detect(data.intrinsic_image(c, k), camera=c, frame=k)
            if det is not None:
                detections.append(det)
        result = calibrate_intrinsics(
            target, detections, data.meta["width"], data.meta["height"],
            name="cam%d" % c, huber_delta=args.huber,
        )
        intrinsics.append(result)
        cameras.append(result.camera)
        s = result.summary()
        t = truth.cameras[c]
        say("  cam%d  %2d/%2d views  rmse %.4f px (linear %.3f)  "
            "fx err %+.2f  cx err %+.2f  cy err %+.2f  | opencv agrees to %.1e px"
            % (c, len(result.T_cam_board), len(data.intrinsic_views(c)), s["rmse_px"],
               s["linear_rmse_px"], s["fx"] - t.fx, s["cx"] - t.cx, s["cy"] - t.cy,
               s.get("opencv_focal_diff_px", float("nan"))))

    record["intrinsics"] = [r.summary() for r in intrinsics]
    record["intrinsic_errors"] = [
        {
            "camera": c,
            "fx_error_px": intrinsics[c].camera.fx - truth.cameras[c].fx,
            "fy_error_px": intrinsics[c].camera.fy - truth.cameras[c].fy,
            "cx_error_px": intrinsics[c].camera.cx - truth.cameras[c].cx,
            "cy_error_px": intrinsics[c].camera.cy - truth.cameras[c].cy,
            "dist_error_max": float(np.max(np.abs(intrinsics[c].camera.dist
                                                  - truth.cameras[c].dist))),
        }
        for c in range(data.n_cameras)
    ]
    record["intrinsics_seconds"] = round(time.time() - t0, 2)

    # -- extrinsics --------------------------------------------------------
    t0 = time.time()
    say("\nRig extrinsics")
    detections = {}
    for frame in data.extrinsic_frames():
        for c in range(data.n_cameras):
            det = target.detect(data.extrinsic_image(frame, c), camera=c, frame=frame)
            if det is not None:
                detections[(frame, c)] = det
    seen = {}
    for (frame, c) in detections:
        seen[frame] = seen.get(frame, 0) + 1
    usable = sum(1 for v in seen.values() if v >= 2)
    say("  %d board detections over %d stations; %d stations seen by two or more cameras"
        % (len(detections), len(seen), usable))

    ext = calibrate_extrinsics(
        cameras, target, detections, huber_delta=args.huber,
        metadata={"dataset": str(Path(args.data).as_posix())},
    )
    say("  initial rmse %.4f px -> final %.4f px (max %.3f px) over %d observations"
        % (ext.initial_rmse_px, ext.rmse_px, ext.max_px, ext.n_observations))
    for c, v in sorted(ext.per_camera_rmse.items()):
        say("    cam%d rmse %.4f px" % (c, v))

    errors = compare_rigs(ext.rig, truth)
    say("  extrinsic error against truth:")
    for row in errors:
        say("    cam%d  baseline %6.1f mm  translation %5.2f mm  rotation %.4f deg"
            % (row["camera"], row["baseline_mm"], row["translation_error_mm"],
               row["rotation_error_deg"]))

    record["extrinsics"] = ext.summary()
    record["extrinsic_errors"] = errors
    record["extrinsics_seconds"] = round(time.time() - t0, 2)

    if args.refine_intrinsics_jointly:
        say("\n  also refining intrinsics inside the extrinsic solve")
        joint = calibrate_extrinsics(
            cameras, target, detections, huber_delta=args.huber, refine_intrinsics=True,
        )
        joint_errors = compare_rigs(joint.rig, truth)
        say("    rmse %.4f px (vs %.4f held fixed)" % (joint.rmse_px, ext.rmse_px))
        say("    mean translation error %.2f mm (vs %.2f)"
            % (float(np.mean([r["translation_error_mm"] for r in joint_errors])),
               float(np.mean([r["translation_error_mm"] for r in errors]))))
        record["extrinsics_joint_intrinsics"] = joint.summary()
        record["extrinsic_errors_joint_intrinsics"] = joint_errors

    # -- summary -----------------------------------------------------------
    record["headline"] = {
        "intrinsic_rmse_px": float(np.mean([r.rmse_px for r in intrinsics])),
        "intrinsic_max_focal_error_px": float(
            max(abs(e["fx_error_px"]) for e in record["intrinsic_errors"])
        ),
        "extrinsic_rmse_px": ext.rmse_px,
        "extrinsic_max_translation_error_mm": float(
            max(r["translation_error_mm"] for r in errors)
        ),
        "extrinsic_max_rotation_error_deg": float(
            max(r["rotation_error_deg"] for r in errors)
        ),
        "opencv_max_focal_disagreement_px": float(
            max(r.summary().get("opencv_focal_diff_px", 0.0) for r in intrinsics)
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ext.rig.save(out / "rig_estimated.yaml")
    (out / "calibration.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    say("\nwrote %s and %s" % (out / "rig_estimated.yaml", out / "calibration.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
