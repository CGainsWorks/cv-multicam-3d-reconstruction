"""The sparse reconstruction, end to end.

    images -> per-station triangulation -> cross-station alignment ->
    global tracks -> bundle adjustment -> sparse cloud in the object frame

The object frame is defined as the rig frame at the reference station.  That
choice costs nothing -- the reconstruction is metric because the rig
calibration is metric -- and it means no ground truth is needed anywhere in
this file.  Expressing the result in the CAD model's own frame is a separate,
explicit registration step, in :mod:`multicam.evaluation`, so that the
registration error is reported rather than absorbed.

How the static background gets removed
--------------------------------------
It is not segmented out and no mask is used.  Features on the floor and the
walls triangulate perfectly well at each station -- they are real 3D points --
but between stations they contradict the part about what the motion was.
:func:`multicam.reconstruction.align.multi_motion_align` fits both motions and
keeps the one that is **not** the identity, on the principle that the part is
whatever moves relative to the rig.  Requiring a track to be seen at two or
more stations then drops the background entirely, because nothing links its
points across stations.  ``track_stations`` in the returned statistics is the
evidence: background points sit at exactly one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3
from .align import align_stations
from .bundle import bundle_adjust
from .features import make_detector
from .station import build_station


@dataclass
class ReconstructionConfig:
    ratio: float = 0.78
    #: Cross-station descriptor matching is a harder problem than within-station
    #: matching -- the viewpoint change is tens of degrees rather than tens of
    #: millimetres -- so it gets a looser ratio and leans on RANSAC instead.
    station_ratio: float = 0.85
    epipolar_px: float = 2.0
    max_reprojection_px: float = 1.5
    min_angle_deg: float = 2.0
    min_views: int = 2
    align_threshold_m: float = 0.004
    #: Which station offsets to attempt an alignment for.  1 chains the sweep;
    #: 2 and the wrap-around add redundancy so one bad link cannot break it.
    link_offsets: tuple = (1, 2)
    close_loop: bool = True
    #: Select the largest *moving* motion rather than the largest motion.  Off
    #: is what a single RANSAC does, and the ablation measures the difference.
    select_moving_motion: bool = True
    min_track_stations: int = 2
    huber_px: float = 2.0
    ba_iterations: int = 40
    outlier_px: float = 3.0
    second_pass: bool = True
    contrast_threshold: float = 0.025
    seed: int = 0


@dataclass
class SparseReconstruction:
    points: np.ndarray                  #: (N, 3) in the object frame
    descriptors: np.ndarray             #: (N, 128)
    T_rig_object: list                  #: per station
    observations: list                  #: (point, station, camera, xy)
    frames: list
    track_stations: np.ndarray = None   #: (N,) how many stations each point spans
    track_views: np.ndarray = None      #: (N,) how many images each point spans
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.points)

    def object_points_in_camera(self, station_index, camera, rig):
        T = rig.T_cam_rig[camera] @ self.T_rig_object[station_index]
        return se3.transform(T, self.points)


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return int(i)

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def reconstruct(images_by_frame, rig, config=None, reference=0, progress=None):
    """Run the whole sparse pipeline.

    ``images_by_frame`` is ``{frame_index: {camera_index: image}}``.
    """
    config = config or ReconstructionConfig()
    rng = np.random.default_rng(config.seed)
    frames = sorted(images_by_frame)
    n = len(frames)
    if n < 2:
        raise ValueError("the turntable sweep needs at least two stations")

    detector = make_detector(contrast_threshold=config.contrast_threshold)

    # -- 1. per-station triangulation --------------------------------------
    stations = []
    for k, f in enumerate(frames):
        if progress:
            progress("station %d/%d" % (k + 1, n))
        stations.append(
            build_station(
                images_by_frame[f], rig, frame=f, detector=detector,
                ratio=config.ratio, epipolar_px=config.epipolar_px,
                min_views=config.min_views,
                max_reprojection_px=config.max_reprojection_px,
                min_angle_deg=config.min_angle_deg,
            )
        )

    # -- 2. cross-station alignment ----------------------------------------
    edges = {}
    offsets = set(config.link_offsets)
    for a in range(n):
        for d in offsets:
            b = a + d
            if b >= n:
                continue
            edges[(a, b)] = None
    if config.close_loop and n > 2:
        edges[(0, n - 1)] = None

    alignments, motion_info = {}, {}
    for (a, b) in sorted(edges):
        if progress:
            progress("aligning %d-%d" % (a, b))
        result, pairs, info = align_stations(
            stations[a], stations[b], ratio=config.station_ratio,
            threshold=config.align_threshold_m, rng=rng,
            select_moving=config.select_moving_motion,
        )
        alignments[(a, b)] = (result, pairs)
        motion_info["%d-%d" % (a, b)] = info

    # Chain the links into a pose per station, over a maximum-inlier spanning
    # tree rather than the consecutive order.  A sweep has one weak link
    # somewhere -- a station where the part happens to present its least
    # textured face -- and chaining strictly in order propagates that link's
    # error into every station after it.  Expanding greedily by inlier count
    # routes around it instead.
    #
    # Frames, carefully.  ``alignments[(a, b)].T_a_b`` maps a point in
    # station b's rig frame into station a's, so chaining these gives
    # ``T_object_rig[s]``: station s's rig frame into the reference station's,
    # which *is* the object frame by definition of the gauge.  The bundle
    # adjuster wants the other direction -- object into rig, so that
    # ``T_cam_rig @ T_rig_object`` projects an object point -- so the inverse
    # is taken once, here, and named.
    T_object_rig = {reference: np.eye(4)}
    edge_weight = {
        k: int(v[0].inliers.sum()) if v[0].converged else -1 for k, v in alignments.items()
    }
    while len(T_object_rig) < n:
        best = None
        for (a, b), weight in edge_weight.items():
            if weight < 0:
                continue
            if a in T_object_rig and b not in T_object_rig:
                # T_object_rig[b] = T_object_rig[a] @ (b -> a)
                cand = (weight, b, T_object_rig[a] @ alignments[(a, b)][0].T_a_b)
            elif b in T_object_rig and a not in T_object_rig:
                cand = (weight, a, T_object_rig[b] @ se3.inverse(alignments[(a, b)][0].T_a_b))
            else:
                continue
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            missing = sorted(set(range(n)) - set(T_object_rig))
            raise RuntimeError(
                "stations %s could not be linked to the reference; the sweep is "
                "broken there.  Inlier counts per edge: %s"
                % (missing, {("%d-%d" % k): v for k, v in edge_weight.items()})
            )
        _, station, T = best
        T_object_rig[station] = T

    T_object_rig = [T_object_rig[s] for s in range(n)]
    T_rig_object = [se3.inverse(T) for T in T_object_rig]

    # -- 3. global tracks ---------------------------------------------------
    base, total = {}, 0
    for s, station in enumerate(stations):
        base[s] = total
        total += len(station)
    uf = _UnionFind(total)
    linked = 0
    for (a, b), (result, pairs) in alignments.items():
        if not result.converged or not len(pairs):
            continue
        for (ia, ib), keep in zip(pairs, result.inliers):
            if keep:
                uf.union(base[a] + int(ia), base[b] + int(ib))
                linked += 1

    groups = {}
    for s, station in enumerate(stations):
        for i in range(len(station)):
            groups.setdefault(uf.find(base[s] + i), []).append((s, i))

    points, descs, observations, track_stations, track_views = [], [], [], [], []
    dropped = {"one_station": 0, "duplicate_station": 0}
    for members in groups.values():
        station_ids = [s for s, _ in members]
        if len(set(station_ids)) < config.min_track_stations:
            dropped["one_station"] += 1
            continue
        if len(set(station_ids)) != len(station_ids):
            # Two points from one station merged into one track: a wrong link.
            dropped["duplicate_station"] += 1
            continue

        index = len(points)
        positions, views = [], 0
        best = max(members, key=lambda si: len(
            [o for o in stations[si[0]].observations if o[0] == si[1]]
        ))
        for s, i in members:
            # Station points are in station s's rig frame; T_object_rig takes
            # them to the object frame.
            positions.append(se3.transform(T_object_rig[s], stations[s].points[i]))
            for (p, c, xy) in stations[s].observations:
                if p == i:
                    observations.append((index, s, c, xy))
                    views += 1
        points.append(np.mean(positions, axis=0))
        descs.append(stations[best[0]].descriptors[best[1]])
        track_stations.append(len(members))
        track_views.append(views)

    if not points:
        raise RuntimeError("no track survived; nothing links the stations together")

    points = np.asarray(points, dtype=float)
    stats = {
        "stations": n,
        "station_points": [len(s) for s in stations],
        "station_features": [int(sum(s.stats["features"].values())) for s in stations],
        "descriptor_matches": int(sum(s.stats["descriptor_matches"] for s in stations)),
        "epipolar_matches": int(sum(s.stats["epipolar_matches"] for s in stations)),
        "alignments": {
            "%d-%d" % k: {
                "correspondences": int(v[0].n_correspondences),
                "inliers": int(v[0].inliers.sum()),
                "inlier_ratio": round(v[0].inlier_ratio, 4),
                "rmse_mm": None if not np.isfinite(v[0].rmse) else round(v[0].rmse * 1000, 3),
            }
            for k, v in sorted(alignments.items())
        },
        "motions": motion_info,
        "links_used": linked,
        "tracks_before_filter": len(groups),
        "tracks_dropped": dropped,
    }

    # -- 4. bundle adjustment ----------------------------------------------
    if progress:
        progress("bundle adjustment: %d points, %d observations"
                 % (len(points), len(observations)))
    ba = bundle_adjust(
        points, T_rig_object, observations, rig, reference=reference,
        huber_delta=config.huber_px, max_iterations=config.ba_iterations,
    )
    stats["bundle"] = ba.summary()

    points = ba.points
    T_rig_object = ba.T_rig_object

    if config.second_pass:
        # Drop observations the first pass could not explain, then re-solve.
        # The Huber kernel bounds an outlier's influence; it does not remove
        # it, and a track built on one bad link still drags its own position.
        keep_obs, per_point = [], {}
        residual = _observation_residuals(points, T_rig_object, observations, rig)
        for o, e in zip(observations, residual):
            if e < config.outlier_px:
                keep_obs.append(o)
                per_point[o[0]] = per_point.get(o[0], 0) + 1
        alive = {p for p, c in per_point.items() if c >= 2}
        keep_obs = [o for o in keep_obs if o[0] in alive]
        stats["outlier_pass"] = {
            "observations_before": len(observations),
            "observations_after": len(keep_obs),
            "points_before": int(len(points)),
            "points_after": len(alive),
        }
        if alive and len(keep_obs) > 12 * len(T_rig_object):
            remap = {p: i for i, p in enumerate(sorted(alive))}
            kept = np.array(sorted(alive), dtype=int)
            ba = bundle_adjust(
                points[kept], T_rig_object,
                [(remap[p], s, c, xy) for (p, s, c, xy) in keep_obs],
                rig, reference=reference, huber_delta=config.huber_px,
                max_iterations=config.ba_iterations,
            )
            points = ba.points
            T_rig_object = ba.T_rig_object
            observations = [(remap[p], s, c, xy) for (p, s, c, xy) in keep_obs]
            descs = [descs[i] for i in kept]
            track_stations = [track_stations[i] for i in kept]
            track_views = [track_views[i] for i in kept]
            stats["bundle_second_pass"] = ba.summary()

    return SparseReconstruction(
        points=points,
        descriptors=np.asarray(descs, dtype=np.float32).reshape(-1, 128),
        T_rig_object=T_rig_object,
        observations=observations,
        frames=frames,
        track_stations=np.asarray(track_stations, dtype=int),
        track_views=np.asarray(track_views, dtype=int),
        stats=stats,
    )


def _observation_residuals(points, T_rig_object, observations, rig):
    """Reprojection error in pixels, one per observation."""
    out = np.empty(len(observations))
    groups = {}
    for i, (p, s, c, _) in enumerate(observations):
        groups.setdefault((s, c), []).append(i)
    for (s, c), idx in groups.items():
        idx = np.asarray(idx)
        T = rig.T_cam_rig[c] @ T_rig_object[s]
        P = points[[observations[i][0] for i in idx]]
        uv = rig.cameras[c].project(se3.transform(T, P))
        meas = np.array([observations[i][3] for i in idx])
        out[idx] = np.linalg.norm(uv - meas, axis=1)
    return out
