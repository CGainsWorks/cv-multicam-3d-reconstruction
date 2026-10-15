"""Dense reconstruction: plane-sweep MVS, Open3D fusion, meshing, evaluation.

    python scripts/dense.py --data data/turntable --out results

Needs ``results/sparse.npz`` from ``scripts/reconstruct.py``: the sparse cloud
supplies the per-station object poses and the bounding volume the sweep is
restricted to.

Two clouds come out.  The **assembly** cloud is everything that turns -- the
part and the turntable it stands on -- and is what the mesh is built from,
because that is what the sensor actually saw.  The **part** cloud is the subset
whose pixels the ground-truth segmentation says belong to the part, and is what
the CAD numbers are computed on.  Splitting them is the only way to quote an
accuracy against a CAD model that does not include the fixture; using the
segmentation for it is a use of ground truth, it is confined to evaluation, and
it is labelled as such wherever the number appears.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from multicam.evaluation.cad import SurfaceDistance, chamfer, completeness, sparse_accuracy
from multicam.fusion import cross_check, fuse_depth_maps, poisson_mesh, save_mesh, sweep
from multicam.fusion.fuse import save_point_cloud
from multicam.geometry import se3
from multicam.io.dataset import Dataset
from multicam.types import RigCalibration


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--sparse", default=None, help="default <out>/sparse.npz")
    p.add_argument("--rig", default=None, help="default <out>/rig_estimated.yaml")
    p.add_argument("--station-step", type=int, default=2,
                   help="use every Nth station; 1 uses all of them")
    p.add_argument("--scale", type=float, default=0.5,
                   help="resolution the sweep runs at, relative to the capture")
    p.add_argument("--depths", type=int, default=96)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--min-score", type=float, default=0.55)
    p.add_argument("--voxel-mm", type=float, default=0.8)
    p.add_argument("--no-mesh", action="store_true")
    p.add_argument("--tag", default="dense")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    data = Dataset(args.data)

    sparse_path = Path(args.sparse) if args.sparse else Path(args.out) / "sparse.npz"
    if not sparse_path.exists():
        raise SystemExit(
            "%s not found.  Run scripts/reconstruct.py first: the dense stage "
            "needs the sparse reconstruction's station poses." % sparse_path
        )
    sparse = np.load(sparse_path)
    rig_path = Path(args.rig) if args.rig else Path(args.out) / "rig_estimated.yaml"
    rig = RigCalibration.load(rig_path) if rig_path.exists() else data.truth_rig

    frames = [int(f) for f in sparse["frames"]]
    T_rig_object = np.asarray(sparse["T_rig_object"])
    points = np.asarray(sparse["points"])

    # The sweep volume: the sparse cloud's own bounds, padded.  No ground truth
    # and no hand-tuned near and far plane.
    lo = points.min(axis=0) - 0.012
    hi = points.max(axis=0) + 0.012
    say("sweep volume from the sparse cloud: %s to %s mm"
        % (np.round(lo * 1000, 1).tolist(), np.round(hi * 1000, 1).tolist()))

    chosen = list(range(0, len(frames), max(1, args.station_step)))
    say("plane sweep over %d of %d stations, %d cameras each, at scale %.2f"
        % (len(chosen), len(frames), data.n_cameras, args.scale))

    interactive = sys.stdout.isatty() and not args.quiet
    t0 = time.time()
    depth_maps, sweep_stats = [], []
    for si in chosen:
        frame = frames[si]
        images = {c: data.turntable_image(frame, c) for c in range(data.n_cameras)}
        T_cam_object = [rig.T_cam_rig[c] @ T_rig_object[si] for c in range(data.n_cameras)]
        station_maps = []
        for c in range(data.n_cameras):
            sources = [(images[k], rig.cameras[k], T_cam_object[k])
                       for k in range(data.n_cameras) if k != c]
            dm = sweep(
                (images[c], rig.cameras[c], T_cam_object[c], si, c),
                sources, (lo, hi), n_depths=args.depths, window=args.window,
                min_score=args.min_score, scale=args.scale,
            )
            station_maps.append(dm)
            sweep_stats.append({"station": si, "camera": c, **dm.stats})
            if interactive:
                sys.stdout.write("\r  station %d camera %d: %d valid   "
                                 % (si, c, dm.stats.get("valid", 0)))
                sys.stdout.flush()
        depth_maps.extend(cross_check(station_maps))
    if interactive:
        sys.stdout.write("\r%-48s\r" % "")
    sweep_seconds = time.time() - t0

    valid_total = int(sum(int(d.valid.sum()) for d in depth_maps))
    removed = int(sum(d.stats.get("cross_check_removed", 0) for d in depth_maps))
    say("  %d depth maps, %d valid pixels (%d removed by the cross check), %.1f s"
        % (len(depth_maps), valid_total, removed, sweep_seconds))
    if not valid_total:
        raise SystemExit("the sweep produced nothing; try --min-score lower")

    # -- depth accuracy against the rendered ground truth ------------------
    depth_errors = []
    for dm in depth_maps:
        if not dm.valid.any():
            continue
        truth = data.turntable_depth(frames[dm.station], dm.camera)
        if args.scale != 1.0:
            import cv2

            truth = cv2.resize(truth, (dm.depth.shape[1], dm.depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        m = dm.valid & np.isfinite(truth)
        if m.any():
            depth_errors.append(np.abs(dm.depth[m] - truth[m]))
    depth_errors = np.concatenate(depth_errors) if depth_errors else np.zeros(0)
    depth_stats = {
        "pixels": int(len(depth_errors)),
        "median_mm": float(np.median(depth_errors) * 1000) if len(depth_errors) else float("nan"),
        "mean_mm": float(np.mean(depth_errors) * 1000) if len(depth_errors) else float("nan"),
        "p90_mm": (float(np.quantile(depth_errors, 0.9) * 1000)
                   if len(depth_errors) else float("nan")),
        "within_1mm": float(np.mean(depth_errors < 1e-3)) if len(depth_errors) else 0.0,
    }
    say("  depth error against the rendered truth: median %.3f mm, 90th pct %.3f mm, "
        "%.1f%% within 1 mm"
        % (depth_stats["median_mm"], depth_stats["p90_mm"], 100 * depth_stats["within_1mm"]))

    # -- split into assembly and part, then fuse ---------------------------
    part_maps = []
    for dm in depth_maps:
        if not dm.valid.any():
            part_maps.append(dm)
            continue
        mask = data.turntable_mask(frames[dm.station], dm.camera)
        if args.scale != 1.0:
            import cv2

            mask = cv2.resize(mask.astype(np.int16), (dm.depth.shape[1], dm.depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        keep = dm.valid & (mask == 0)
        part_maps.append(
            type(dm)(dm.station, dm.camera, np.where(keep, dm.depth, np.nan).astype(np.float32),
                     dm.score, keep, dm.camera_model, dm.T_cam_object, dm.stats)
        )

    voxel = args.voxel_mm * 1e-3
    assembly = fuse_depth_maps(depth_maps, voxel_m=voxel)
    part = fuse_depth_maps(part_maps, voxel_m=voxel)
    say("  fused: assembly %d points, part %d points (voxel %.1f mm)"
        % (len(assembly.points), len(part.points), args.voxel_mm))

    # -- evaluate the part cloud against CAD --------------------------------
    cad = data.object_mesh
    T_truth = np.linalg.inv(data.T_rig_world @ data.T_world_object(frames[0]))
    distance = SurfaceDistance(cad)
    part_cad = se3.transform(T_truth, part.points)
    acc = sparse_accuracy(part_cad, cad, inlier_mm=2.0, distance=distance)
    comp = completeness(cad, part_cad, threshold_mm=1.0)
    comp3 = completeness(cad, part_cad, threshold_mm=3.0)
    cham = chamfer(part_cad, cad, distance=distance)

    say("\nDense part cloud against the CAD model (ground-truth frame, no fitting)")
    say("  %d points: median %.3f mm, mean %.3f mm, 90th pct %.3f mm, rms %.3f mm"
        % (acc["points"], acc["median_mm"], acc["mean_mm"], acc["p90_mm"], acc["rms_mm"]))
    say("  %.1f%% within 2 mm of the surface" % (100 * acc["inlier_fraction"]))
    say("  completeness: %.1f%% of the CAD surface within 1 mm, %.1f%% within 3 mm"
        % (100 * comp["completeness"], 100 * comp3["completeness"]))
    say("  chamfer %.3f mm (accuracy %.3f, completeness %.3f)"
        % (cham["chamfer_mm"], cham["accuracy_mm"], cham["completeness_mm"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_point_cloud(out / ("%s_assembly.ply" % args.tag), assembly.points, assembly.normals)
    save_point_cloud(out / ("%s_part.ply" % args.tag), part.points, part.normals)

    mesh_stats = {}
    if not args.no_mesh:
        meshed = poisson_mesh(assembly)
        if meshed.has_mesh:
            save_mesh(out / ("%s_assembly_mesh.ply" % args.tag),
                      meshed.mesh_vertices, meshed.mesh_faces)
            mesh_stats = meshed.stats
            say("  meshed the assembly: %d vertices, %d faces"
                % (mesh_stats.get("mesh_vertices", 0), mesh_stats.get("mesh_faces", 0)))
        part_meshed = poisson_mesh(part)
        if part_meshed.has_mesh:
            save_mesh(out / ("%s_part_mesh.ply" % args.tag),
                      part_meshed.mesh_vertices, part_meshed.mesh_faces)
            mesh_stats["part"] = part_meshed.stats
            say("  meshed the part: %d vertices, %d faces"
                % (part_meshed.stats.get("mesh_vertices", 0),
                   part_meshed.stats.get("mesh_faces", 0)))

    record = {
        "tag": args.tag,
        "dataset": str(Path(args.data).as_posix()),
        "commit": _bootstrap.git_commit(),
        "versions": _bootstrap.library_versions(),
        "settings": {
            "station_step": args.station_step, "stations_used": len(chosen),
            "scale": args.scale, "depths": args.depths, "window": args.window,
            "min_score": args.min_score, "voxel_mm": args.voxel_mm,
        },
        "sweep_seconds": round(sweep_seconds, 2),
        "sweep": sweep_stats,
        "depth_accuracy": depth_stats,
        "fusion": {"assembly": assembly.stats, "part": part.stats},
        "mesh": mesh_stats,
        "cad_accuracy_part": acc,
        "cad_completeness_1mm": comp,
        "cad_completeness_3mm": comp3,
        "cad_chamfer": cham,
        "headline": {
            "part_points": int(len(part.points)),
            "cad_median_error_mm": acc["median_mm"],
            "cad_rms_error_mm": acc["rms_mm"],
            "completeness_1mm": comp["completeness"],
            "chamfer_mm": cham["chamfer_mm"],
            "depth_median_error_mm": depth_stats["median_mm"],
        },
    }
    (out / ("%s.json" % args.tag)).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    say("\nwrote %s and the point clouds beside it" % (out / ("%s.json" % args.tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
