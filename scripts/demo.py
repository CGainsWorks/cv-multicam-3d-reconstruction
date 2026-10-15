"""The whole pipeline, small, from nothing, in about two minutes.

    python scripts/demo.py

Renders a miniature capture, calibrates the rig from it, reconstructs the part,
estimates its 6-DoF pose on held-out stations, and checks every number against
the ground truth it generated.  No dataset, no network weights, no downloads.

This is the smoke test and it is what CI runs.  It is deliberately *small* --
quarter resolution, a third of the stations, no supersampling -- so the numbers
it produces are worse than the ones in the README, which come from
``make experiments``.  The point of it is that a change which breaks the
estimator shows up as a failed demo the same day, not as a worse number weeks
later.

Thresholds are loose on purpose.  They are regression guards, not benchmarks:
tight enough that a sign error or a broken frame convention fails them,
loose enough that they do not fail because a library rounded differently.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from multicam.calibration import CharucoTarget, calibrate_extrinsics, calibrate_intrinsics
from multicam.calibration.extrinsic import compare_rigs
from multicam.evaluation.cad import SurfaceDistance, sparse_accuracy
from multicam.evaluation.labels import OBJECT, composition, label_observations
from multicam.evaluation.metrics import pose_error_table
from multicam.geometry import se3
from multicam.io.dataset import Dataset, DatasetWriter
from multicam.pose import build_model, estimate_pose, match_to_model
from multicam.reconstruction import ReconstructionConfig, detect, make_detector, reconstruct
from multicam.sim.capture import CaptureConfig, make_target_and_scene, plan_capture, render_capture
from multicam.sim.rig import RigSpec, build_rig, nominal_rig_pose
from multicam.sim.scene import default_scene

#: Regression guards.  See the module docstring: these are not targets, and
#: several of them are much looser than the numbers in the README.
#:
#: The reason is worth stating, because it is the most interesting thing the
#: demo measures.  A small capture **calibrates coarsely**: at this resolution
#: and this many board stations the rig extrinsics come out around 13 mm on a
#: 750 mm baseline, which is a 1.7 per cent error in the rig's scale.  That
#: error passes straight through to the pose, because a pose estimated against
#: a rig whose baselines are 1.7 per cent wrong lands 1.7 per cent of the
#: stand-off away -- about 11 mm at 0.66 m.  Which is, to within the noise,
#: exactly the pose error the demo reports.
#:
#: So the demo's pose number is not a pose-pipeline number, it is a calibration
#: number in disguise, and the guard is set accordingly.  The quantities that
#: *are* independent of the rig's absolute scale -- the reprojection residuals
#: and the reconstruction's accuracy against CAD -- are held tight, because
#: those are what would actually move if the geometry broke.
LIMITS = {
    "intrinsic_rmse_px": 0.60,
    "intrinsic_focal_error_px": 25.0,
    "extrinsic_rmse_px": 0.80,
    "extrinsic_translation_mm": 25.0,
    "extrinsic_rotation_deg": 3.0,
    "reconstruction_rmse_px": 0.60,
    "min_part_points": 250,
    "cad_median_mm": 1.5,
    "pose_translation_mm": 25.0,
    "pose_rotation_deg": 2.5,
    "min_pose_recall_10pct": 0.80,
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="JSON record; stdout only if omitted")
    p.add_argument("--data", default=None,
                   help="where to render the capture; a temp directory if omitted")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--intrinsic-views", type=int, default=18)
    p.add_argument("--extrinsic-stations", type=int, default=26)
    p.add_argument("--turntable-stations", type=int, default=10)
    p.add_argument("--pose-stations", type=int, default=10)
    p.add_argument("--pose-test", type=int, default=6)
    # Supersampling is not cosmetic here: an aliased ChArUco corner carries a
    # sub-pixel bias straight into the focal length, and turning this off
    # roughly doubles the calibration error.
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--keep", action="store_true", help="keep the rendered capture")
    p.add_argument("--no-check", action="store_true", help="report but do not assert")
    return p


def _render(root, args, say):
    config = CaptureConfig(
        n_intrinsic_views=args.intrinsic_views,
        n_extrinsic_stations=args.extrinsic_stations,
        n_turntable_stations=args.turntable_stations,
        n_pose_stations=args.pose_stations,
        n_pose_test=args.pose_test,
        supersample=args.supersample,
        seed=args.seed,
        depth_maps=False,
    )
    rig = build_rig(RigSpec(width=args.width, height=args.height, seed=args.seed * 7 + 1))
    scene = default_scene(seed=args.seed + 3, texture_size=512)
    target, board_sc, board_cell_sc, margin_m = make_target_and_scene(config, cell=scene)
    T_rig_world = nominal_rig_pose(rig)

    plan = plan_capture(rig, target, scene, config, T_rig_world)
    writer = DatasetWriter(root)
    counts = render_capture(writer, rig, scene, board_sc, plan, config, T_rig_world,
                            board_cell_sc=board_cell_sc)
    writer.write_meta({
        "format": "multicam-capture-v1",
        "created_by": "scripts/demo.py",
        "n_cameras": rig.n_cameras,
        "width": args.width, "height": args.height, "seed": args.seed,
        "config": {
            "intrinsic_views": config.n_intrinsic_views,
            "extrinsic_stations": config.n_extrinsic_stations,
            "turntable_stations": config.n_turntable_stations,
            "pose_stations": len(plan.pose),
            "pose_test": config.n_pose_test,
            "supersample": config.supersample,
            "board_squares": list(config.board_squares),
            "square_length_m": config.square_length_m,
            "board_margin_m": float(margin_m),
        },
        "counts": counts,
        "versions": _bootstrap.library_versions(),
    })
    say("  %d views in %.1f s (%.0f ms each)"
        % (counts["views"], counts["seconds"],
           1000 * counts["seconds"] / max(counts["views"], 1)))
    return counts


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = print
    started = time.time()
    record = {"limits": LIMITS, "versions": _bootstrap.library_versions(),
              "commit": _bootstrap.git_commit()}

    tmp = None
    if args.data:
        root = Path(args.data)
        root.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="multicam-demo-")
        root = Path(tmp.name)

    try:
        say("1/4  rendering a miniature capture at %dx%d" % (args.width, args.height))
        record["capture"] = _render(root, args, say)
        data = Dataset(root)
        truth = data.truth_rig
        cfg = data.meta["config"]
        target = CharucoTarget(
            __import__("multicam.sim.textures", fromlist=["charuco_board"]).charuco_board(
                squares_x=cfg["board_squares"][0], squares_y=cfg["board_squares"][1],
                square_length=cfg["square_length_m"],
            )
        )

        # -- 2. calibration -------------------------------------------------
        say("2/4  calibrating the rig")
        t0 = time.time()
        cameras, intrinsics = [], []
        for c in range(data.n_cameras):
            detections = [
                target.detect(data.intrinsic_image(c, k), camera=c, frame=k)
                for k in range(len(data.intrinsic_views(c)))
            ]
            result = calibrate_intrinsics(
                target, [d for d in detections if d is not None],
                data.meta["width"], data.meta["height"], name="cam%d" % c,
                opencv_check=False,
            )
            intrinsics.append(result)
            cameras.append(result.camera)

        board_detections = {}
        for frame in data.extrinsic_frames():
            for c in range(data.n_cameras):
                det = target.detect(data.extrinsic_image(frame, c), camera=c, frame=frame)
                if det is not None:
                    board_detections[(frame, c)] = det
        extrinsic = calibrate_extrinsics(cameras, target, board_detections)
        rig = extrinsic.rig
        errors = compare_rigs(rig, truth)

        intr_rmse = float(np.mean([r.rmse_px for r in intrinsics]))
        focal_err = float(max(abs(r.camera.fx - truth.cameras[i].fx)
                              for i, r in enumerate(intrinsics)))
        trans_err = float(max(r["translation_error_mm"] for r in errors))
        rot_err = float(max(r["rotation_error_deg"] for r in errors))
        say("  intrinsics rmse %.3f px, worst focal error %.2f px" % (intr_rmse, focal_err))
        say("  extrinsics rmse %.3f px, worst %.2f mm / %.3f deg"
            % (extrinsic.rmse_px, trans_err, rot_err))
        record["calibration"] = {
            "intrinsic_rmse_px": intr_rmse, "intrinsic_focal_error_px": focal_err,
            "extrinsic_rmse_px": extrinsic.rmse_px,
            "extrinsic_translation_mm": trans_err, "extrinsic_rotation_deg": rot_err,
            "seconds": round(time.time() - t0, 2),
        }

        # -- 3. reconstruction ----------------------------------------------
        say("3/4  reconstructing the part")
        t0 = time.time()
        frames = data.turntable_frames()
        images = {f: {c: data.turntable_image(f, c) for c in range(data.n_cameras)}
                  for f in frames}
        rec = reconstruct(images, rig, ReconstructionConfig())

        cad = data.object_mesh
        T_truth = np.linalg.inv(data.T_rig_world @ data.T_world_object(frames[0]))
        masks = {s: {c: data.turntable_mask(f, c) for c in range(data.n_cameras)}
                 for s, f in enumerate(frames)}
        labels, _ = label_observations(rec.observations, masks, len(rec))
        comp = composition(labels)
        distance = SurfaceDistance(cad)
        points_cad = se3.transform(T_truth, rec.points)
        accuracy = sparse_accuracy(points_cad[labels == OBJECT], cad, inlier_mm=5.0,
                                   distance=distance)
        ba = rec.stats.get("bundle_second_pass", rec.stats["bundle"])
        say("  %d points (%d on the part), reprojection %.3f px, CAD median %.3f mm"
            % (len(rec), comp["part"], ba["rmse_px"], accuracy["median_mm"]))
        record["reconstruction"] = {
            "points": int(len(rec)), "part_points": comp["part"],
            "reconstruction_rmse_px": ba["rmse_px"],
            "cad_median_mm": accuracy["median_mm"],
            "composition": comp, "seconds": round(time.time() - t0, 2),
        }

        # -- 4. pose ---------------------------------------------------------
        say("4/4  estimating 6-DoF pose on held-out stations")
        t0 = time.time()
        model = build_model(points_cad, rec.descriptors, cad, surface_mm=4.0,
                            distance=distance)
        detector = make_detector()
        _, test_frames = data.pose_split()
        T_rig_world = data.T_rig_world
        estimates, truths = [], []
        for f in test_frames:
            correspondences = [
                match_to_model(
                    detect(data.pose_image(f, c), detector, camera=c, frame=f), model
                )
                for c in range(data.n_cameras)
            ]
            result = estimate_pose(rig, correspondences, rng=np.random.default_rng(f))
            estimates.append(result.T_rig_object if result.ok else None)
            truths.append(T_rig_world @ data.T_world_object_pose(f))

        model_points, _ = cad.sample_surface(1024, np.random.default_rng(0))
        failures = sum(1 for e in estimates if e is None)
        table = pose_error_table(
            [e for e in estimates if e is not None],
            [t for e, t in zip(estimates, truths) if e is not None],
            model_points, cad.diameter, failures=failures,
        )
        say("  %d stations, %d failed: translation median %.3f mm, rotation %.4f deg"
            % (len(test_frames), failures, table["translation_mm"]["median"],
               table["rotation_deg"]["median"]))
        say("  ADD median %.3f mm, recall at 2%% of diameter %.0f%%"
            % (table["add_mm"]["median"], 100 * table["add_recall_2_pct_diameter"]))
        record["pose"] = {
            "stations": len(test_frames), "failures": failures,
            "pose_translation_mm": table["translation_mm"]["median"],
            "pose_rotation_deg": table["rotation_deg"]["median"],
            "min_pose_recall_10pct": table["add_recall_10_pct_diameter"],
            "metrics": table, "seconds": round(time.time() - t0, 2),
        }
    finally:
        if tmp is not None and not args.keep:
            tmp.cleanup()

    record["seconds"] = round(time.time() - started, 2)
    say("\ndemo finished in %.1f s" % record["seconds"])
    say("These are small-capture numbers.  The capture is a tenth the size of "
        "the real one and calibrates correspondingly coarsely, which is what "
        "the pose error below is mostly measuring.  `make data && make "
        "experiments` reproduces the README's numbers.")

    # -- the guards --------------------------------------------------------
    measured = {}
    for section in ("calibration", "reconstruction", "pose"):
        measured.update({k: v for k, v in record[section].items() if k in LIMITS})
    failures = []
    for key, limit in LIMITS.items():
        if key not in measured:
            continue
        value = measured[key]
        bad = value < limit if key.startswith("min_") else value > limit
        status = "FAIL" if bad else "ok"
        say("  %-28s %10.4f  %s %-8.3f  %s"
            % (key, value, "<" if key.startswith("min_") else ">", limit, status))
        if bad:
            failures.append("%s = %.4f, limit %.4f" % (key, value, limit))

    record["measured"] = measured
    record["failures"] = failures
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
        )
        say("wrote %s" % args.out)

    if failures and not args.no_check:
        say("\nDEMO FAILED:")
        for f in failures:
            say("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
