"""The types that cross module boundaries.

Chiefly :class:`RigCalibration`, which is the contract between the calibration
code that estimates it, the reconstruction that consumes it and the files on
disk.  Keeping it here rather than in ``multicam.calibration`` means the
reconstruction does not import the calibration solver just to read a pose.

Frame conventions, once, for the whole repository:

* ``T_a_b`` maps a point in frame ``b`` to frame ``a``.
* The **rig frame** is the reference camera's frame, so ``T_cam_rig[0]`` is the
  identity by construction.  Anchoring the rig to a physical camera rather than
  to an abstract frame removes a six-parameter gauge freedom from every solve
  that touches the extrinsics.
* The **world frame** is whatever the capture defines: for the turntable
  sequences it is the table, with ``z`` up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .geometry import PinholeCamera, se3


@dataclass
class RigCalibration:
    """Intrinsics and extrinsics for a rigid multi-camera rig."""

    cameras: list           #: list[PinholeCamera], one per camera
    T_cam_rig: list         #: list[(4, 4)], camera-from-rig; element 0 is identity
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T_cam_rig = [np.asarray(T, dtype=float).reshape(4, 4) for T in self.T_cam_rig]
        if len(self.cameras) != len(self.T_cam_rig):
            raise ValueError("one extrinsic per camera, got %d and %d"
                             % (len(self.cameras), len(self.T_cam_rig)))
        if not np.allclose(self.T_cam_rig[0], np.eye(4), atol=1e-9):
            raise ValueError(
                "the rig frame is the reference camera's frame, so T_cam_rig[0] "
                "must be the identity; got a matrix that is not"
            )

    def __len__(self):
        return len(self.cameras)

    @property
    def n_cameras(self):
        return len(self.cameras)

    def T_cam_world(self, index, T_rig_world):
        """Camera-from-world for camera ``index`` given the rig pose."""
        return self.T_cam_rig[index] @ np.asarray(T_rig_world, dtype=float).reshape(4, 4)

    def all_T_cam_world(self, T_rig_world):
        return [self.T_cam_world(i, T_rig_world) for i in range(self.n_cameras)]

    def baselines(self):
        """Distance in metres from the reference camera to each other camera."""
        centres = [se3.camera_centre(T) for T in self.T_cam_rig]
        return [float(np.linalg.norm(c - centres[0])) for c in centres]

    def to_dict(self):
        return {
            "format": "multicam-rig-v1",
            "reference_camera": 0,
            "cameras": [c.to_dict() for c in self.cameras],
            "T_cam_rig": [np.asarray(T, dtype=float).reshape(16).tolist()
                          for T in self.T_cam_rig],
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != "multicam-rig-v1":
            raise ValueError("unrecognised rig calibration format %r" % (d.get("format"),))
        return cls(
            cameras=[PinholeCamera.from_dict(c) for c in d["cameras"]],
            T_cam_rig=[np.asarray(T, dtype=float).reshape(4, 4) for T in d["T_cam_rig"]],
            metadata=d.get("metadata", {}),
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class Observation:
    """One 2D measurement: a keypoint in one image belonging to one track."""

    frame: int
    camera: int
    xy: np.ndarray          #: (2,) pixel coordinates, distorted (as measured)
    track: int = -1


@dataclass
class Reconstruction:
    """A sparse point cloud with the observations that produced it.

    ``points`` and ``point_ids`` are parallel; ``observations`` indexes into
    ``point_ids`` through :attr:`Observation.track`.  Keeping the ids explicit
    rather than relying on array position means points can be culled without
    renumbering every observation.
    """

    points: np.ndarray                   #: (N, 3) world (or object) coordinates
    point_ids: np.ndarray                #: (N,) int
    observations: list                   #: list[Observation]
    T_rig_world: dict = field(default_factory=dict)   #: frame index -> (4, 4)
    colours: np.ndarray = None           #: (N,) uint8 greyscale, may be None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.point_ids = np.asarray(self.point_ids, dtype=np.int64).reshape(-1)
        if len(self.points) != len(self.point_ids):
            raise ValueError("points and point_ids must agree in length")

    @property
    def n_points(self):
        return len(self.points)

    def track_lengths(self):
        counts = {}
        for o in self.observations:
            counts[o.track] = counts.get(o.track, 0) + 1
        return counts


def _plain(obj):
    """Convert numpy scalars and arrays so PyYAML emits readable numbers."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
