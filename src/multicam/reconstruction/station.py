"""Per-station reconstruction: one instant, every camera.

A "station" is one turntable position.  At a station the rig is calibrated and
rigid, so the relative poses of all four cameras are known exactly and the only
unknown is the 3D position of each feature.  That makes a station a pure
multi-view triangulation problem with no pose estimation in it at all, which is
worth exploiting: it is fast, it has no gauge freedom, and it produces a small
metric point cloud per station that the cross-station alignment can then treat
as rigid-body data.

The alternative -- throwing all the images from all the stations into one
incremental SfM and letting it work out the poses -- would work, and is what
COLMAP does when it is run as the baseline in ``scripts/colmap_baseline.py``.
It also spends most of its effort rediscovering the rig geometry that was
measured properly in the calibration step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3, triangulation
from .features import detect, make_detector, match_views


@dataclass
class StationCloud:
    """Points triangulated from one synchronised multi-camera exposure.

    Everything is in the **rig frame** at that instant.
    """

    frame: int
    points: np.ndarray                  #: (N, 3) in the rig frame
    descriptors: np.ndarray             #: (N, 128) representative descriptor per point
    observations: list = field(default_factory=list)
    #: (N,) widest angle subtended by any camera pair, degrees
    angles: np.ndarray = None
    #: (N,) RMS reprojection error over the views that built the point, pixels
    errors: np.ndarray = None
    features: dict = field(default_factory=dict)     #: camera -> FeatureSet
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.points)

    def filtered(self, mask):
        mask = np.asarray(mask, dtype=bool)
        index = np.nonzero(mask)[0]
        remap = -np.ones(len(mask), dtype=int)
        remap[index] = np.arange(len(index))
        obs = [
            (remap[p], c, xy) for (p, c, xy) in self.observations if mask[p]
        ]
        return StationCloud(
            frame=self.frame,
            points=self.points[mask],
            descriptors=self.descriptors[mask],
            observations=obs,
            angles=None if self.angles is None else self.angles[mask],
            errors=None if self.errors is None else self.errors[mask],
            features=self.features,
            stats=dict(self.stats),
        )


def build_station(images, rig, frame=0, detector=None, ratio=0.78, epipolar_px=2.0,
                  min_views=2, max_reprojection_px=1.5, min_angle_deg=2.0,
                  masks=None):
    """Detect, match across the rig and triangulate, for one instant.

    ``masks`` is an optional ``{camera: bool image}`` restricting detection.
    The pipeline never passes it; it exists so an ablation can measure what a
    perfect segmentation would be worth.
    """
    detector = detector or make_detector()
    n_cameras = rig.n_cameras
    features = {}
    for c in range(n_cameras):
        m = None
        if masks is not None and c in masks:
            m = (np.asarray(masks[c]).astype(np.uint8) * 255)
        features[c] = detect(images[c], detector, mask=m, camera=c, frame=frame)

    # The rig frame is camera 0's frame, so the camera-from-rig extrinsics are
    # exactly the poses to triangulate with; no world pose is involved.
    poses = rig.T_cam_rig

    # Union-find over pairwise matches, linking a feature in one camera to the
    # same physical point seen in another.
    offsets, total = {}, 0
    for c in range(n_cameras):
        offsets[c] = total
        total += len(features[c])
    parent = np.arange(total)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    stats = {"pairs": {}, "descriptor_matches": 0, "epipolar_matches": 0}
    for a in range(n_cameras):
        for b in range(a + 1, n_cameras):
            pairs, s = match_views(
                features[a], features[b], rig.cameras[a], rig.cameras[b],
                poses[a], poses[b], ratio=ratio, epipolar_px=epipolar_px,
            )
            stats["pairs"]["%d-%d" % (a, b)] = s
            stats["descriptor_matches"] += s["descriptor"]
            stats["epipolar_matches"] += s["epipolar"]
            for i, j in pairs:
                union(offsets[a] + i, offsets[b] + j)

    # Gather groups, rejecting any that contain two features from one camera:
    # a physical point projects once per camera, so such a group is the result
    # of a chain of wrong matches and its triangulation would be meaningless.
    groups = {}
    for c in range(n_cameras):
        for i in range(len(features[c])):
            groups.setdefault(find(offsets[c] + i), []).append((c, i))

    points, descs, obs, angles, errors = [], [], [], [], []
    rejected = {"single_view": 0, "duplicate_camera": 0, "cheirality": 0,
                "angle": 0, "reprojection": 0}
    for members in groups.values():
        if len(members) < min_views:
            rejected["single_view"] += 1
            continue
        cams = [c for c, _ in members]
        if len(set(cams)) != len(cams):
            rejected["duplicate_camera"] += 1
            continue

        view_poses = [poses[c] for c, _ in members]
        xy = np.array([
            rig.cameras[c].undistort(features[c].xy[i: i + 1])[0] for c, i in members
        ])
        X = triangulation.triangulate_dlt(view_poses, xy)
        if not np.all(np.isfinite(X)):
            rejected["cheirality"] += 1
            continue
        X = triangulation.refine_point(view_poses, xy, X)
        if not np.all(np.isfinite(X)) or not triangulation.cheirality(view_poses, X).all():
            rejected["cheirality"] += 1
            continue

        angle = triangulation.triangulation_angle(view_poses, X)
        if angle < min_angle_deg:
            rejected["angle"] += 1
            continue

        residuals = []
        for (c, i), T in zip(members, view_poses):
            uv = rig.cameras[c].project(se3.transform(T, X))
            residuals.append(uv - features[c].xy[i])
        residuals = np.asarray(residuals)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        if rms > max_reprojection_px:
            rejected["reprojection"] += 1
            continue

        # The representative descriptor is the one from the strongest
        # detection.  Averaging descriptors across views blurs the very
        # distinctions the cross-station matching needs.
        best = max(members, key=lambda ci: features[ci[0]].response[ci[1]])
        index = len(points)
        points.append(X)
        descs.append(features[best[0]].descriptors[best[1]])
        angles.append(angle)
        errors.append(rms)
        for c, i in members:
            obs.append((index, c, features[c].xy[i]))

    stats.update(rejected)
    stats["features"] = {c: len(features[c]) for c in features}
    stats["points"] = len(points)
    return StationCloud(
        frame=frame,
        points=np.asarray(points, dtype=float).reshape(-1, 3),
        descriptors=np.asarray(descs, dtype=np.float32).reshape(-1, 128),
        observations=obs,
        angles=np.asarray(angles, dtype=float),
        errors=np.asarray(errors, dtype=float),
        features=features,
        stats=stats,
    )
