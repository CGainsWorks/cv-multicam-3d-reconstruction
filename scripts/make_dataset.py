"""Render a capture session to disk.

    python scripts/make_dataset.py --out data/turntable

There is no hardware behind this repository, so this script is where the images
come from.  It plans the three sessions -- per-camera intrinsic views, shared
extrinsic stations, a turntable sweep -- checks every pose geometrically before
paying to render it, and writes the ground truth alongside so that everything
downstream can be scored.

The ground-truth rig, masks and depth maps it writes are for **evaluation
only**.  The calibration and reconstruction read the images and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401
import numpy as np

from multicam.io.dataset import DatasetWriter
from multicam.sim.capture import CaptureConfig, make_target_and_scene, plan_capture, render_capture
from multicam.sim.rig import RigSpec, build_rig, nominal_rig_pose
from multicam.sim.scene import default_scene


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--intrinsic-views", type=int, default=22)
    p.add_argument("--extrinsic-stations", type=int, default=26)
    p.add_argument("--turntable-stations", type=int, default=18)
    p.add_argument("--pose-stations", type=int, default=140)
    p.add_argument("--pose-test", type=int, default=40)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-depth", action="store_true",
                   help="skip the ground-truth depth maps (about half the bytes)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = CaptureConfig(
        n_intrinsic_views=args.intrinsic_views,
        n_extrinsic_stations=args.extrinsic_stations,
        n_turntable_stations=args.turntable_stations,
        n_pose_stations=args.pose_stations,
        n_pose_test=args.pose_test,
        supersample=args.supersample,
        noise_sigma=args.noise,
        seed=args.seed,
        depth_maps=not args.no_depth,
    )

    rig = build_rig(RigSpec(n_cameras=args.cameras, width=args.width, height=args.height,
                            seed=args.seed * 7 + 1))
    scene = default_scene(seed=args.seed + 3)
    target, board_sc, board_cell_sc, margin_m = make_target_and_scene(config, cell=scene)
    T_rig_world = nominal_rig_pose(rig)

    t0 = time.time()
    plan = plan_capture(rig, target, scene, config, T_rig_world)
    if not args.quiet:
        print("planned: %d intrinsic views/camera, %d extrinsic stations, "
              "%d turntable stations, %d pose stations (%d held out)"
              % (len(plan.intrinsic[0]), len(plan.extrinsic), len(plan.turntable),
                 len(plan.pose), args.pose_test))
        seen = plan.rejected.get("extrinsic_cameras_per_station", [])
        if seen:
            print("         cameras per extrinsic station: mean %.2f, min %d"
                  % (float(np.mean(seen)), int(np.min(seen))))

    writer = DatasetWriter(args.out)
    # Carriage-return progress only makes sense on a terminal.  Piped to a file
    # or to CI's log it produces one enormous line, so it is simply off there.
    interactive = sys.stdout.isatty() and not args.quiet
    last = [0.0]

    def progress(msg):
        if not interactive:
            return
        now = time.time()
        if now - last[0] > 0.5:
            last[0] = now
            sys.stdout.write("\r  %-44s" % msg)
            sys.stdout.flush()

    counts = render_capture(writer, rig, scene, board_sc, plan, config, T_rig_world,
                            progress if interactive else None, board_cell_sc=board_cell_sc)
    if interactive:
        sys.stdout.write("\r%-46s\r" % "")

    writer.write_meta(
        {
            "format": "multicam-capture-v1",
            "created_by": "scripts/make_dataset.py",
            "commit": _bootstrap.git_commit(),
            "n_cameras": rig.n_cameras,
            "width": args.width,
            "height": args.height,
            "seed": args.seed,
            "config": {
                "intrinsic_views": config.n_intrinsic_views,
                "extrinsic_stations": config.n_extrinsic_stations,
                "turntable_stations": config.n_turntable_stations,
                "pose_stations": len(plan.pose),
                "pose_test": config.n_pose_test,
                "supersample": config.supersample,
                "noise_sigma": config.noise_sigma,
                "board_squares": list(config.board_squares),
                "square_length_m": config.square_length_m,
                "board_margin_m": float(margin_m),
                "turntable_wobble_deg": config.turntable_wobble_deg,
            },
            "object": {
                "name": scene.object_mesh.name,
                "diameter_m": float(scene.object_mesh.diameter),
                "faces": int(len(scene.object_mesh.faces)),
            },
            "counts": counts,
            "planning": {k: v for k, v in plan.rejected.items()
                         if not isinstance(v, list)},
            "versions": _bootstrap.library_versions(),
        }
    )

    if not args.quiet:
        print("wrote %d views to %s in %.1f s (%.0f ms/view)"
              % (counts["views"], args.out, time.time() - t0,
                 counts["seconds"] / max(counts["views"], 1) * 1000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
