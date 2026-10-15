"""The on-disk capture layout.

One directory per capture session, with the three sessions kept separate
because they are used separately::

    <root>/
      meta.yaml                     configuration, seeds, library versions
      DATASET_CARD.md               provenance and licence
      truth/
        rig.yaml                    ground-truth intrinsics and extrinsics
        object.ply                  the CAD model
        poses.json                  rig pose, board poses, object poses
      intrinsics/
        cam0/000.png ...            one camera at a time
      extrinsics/
        000/cam0.png ...            all cameras triggered together
      turntable/
        000/cam0.png
        000/cam0_mask.png           ground-truth segmentation, evaluation only
        000/cam0_depth.npy          ground-truth depth, evaluation only

Nothing under ``data/`` is committed.  Every byte of it is reproducible from
``scripts/make_dataset.py`` and a seed, and committing a few hundred megabytes
of PNGs that a script regenerates would be committing the output of the build.

**The mask and depth files are ground truth and the pipeline never opens
them.**  Only :mod:`multicam.evaluation` does.  That separation is enforced by
a test which imports every pipeline module and asserts none of them reference
the ground-truth accessors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..sim.mesh import TriMesh
from ..types import RigCalibration

#: Mask value written for pixels belonging to no mesh.
BACKGROUND = 255


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError("cannot serialise %r" % (type(obj),))


def _dump_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )


@dataclass
class Frame:
    """One synchronised trigger: the same instant seen by every camera."""

    index: int
    images: dict            #: camera index -> (H, W) uint8
    masks: dict = None      #: camera index -> (H, W) int16, ground truth
    depths: dict = None     #: camera index -> (H, W) float32, ground truth


class DatasetWriter:
    """Builds a capture directory."""

    def __init__(self, root, meta=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})

    # -- truth -------------------------------------------------------------

    def write_truth(self, rig, object_mesh, poses):
        (self.root / "truth").mkdir(parents=True, exist_ok=True)
        rig.save(self.root / "truth" / "rig.yaml")
        object_mesh.save_ply(self.root / "truth" / "object.ply")
        _dump_json(self.root / "truth" / "poses.json", poses)

    def write_meta(self, extra=None):
        payload = dict(self.meta)
        payload.update(extra or {})
        (self.root / "meta.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    # -- images ------------------------------------------------------------

    def write_intrinsic_view(self, camera, index, image):
        d = self.root / "intrinsics" / ("cam%d" % camera)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / ("%03d.png" % index)), image)

    def write_extrinsic_view(self, index, camera, image):
        d = self.root / "extrinsics" / ("%03d" % index)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / ("cam%d.png" % camera)), image)

    def write_pose_view(self, index, camera, image, mask=None):
        self._write_view(self.root / "pose" / ("%03d" % index), camera, image, mask, None)

    def write_turntable_view(self, index, camera, image, mask=None, depth=None):
        self._write_view(self.root / "turntable" / ("%03d" % index), camera, image,
                         mask, depth)

    def _write_view(self, d, camera, image, mask, depth):
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / ("cam%d.png" % camera)), image)
        if mask is not None:
            m = np.asarray(mask)
            out = np.where(m < 0, BACKGROUND, np.clip(m, 0, BACKGROUND - 1)).astype(np.uint8)
            cv2.imwrite(str(d / ("cam%d_mask.png" % camera)), out)
        if depth is not None:
            dep = np.asarray(depth, dtype=np.float32)
            # Infinity does not survive float16; a sentinel that cannot be a
            # real depth does, and the reader turns it back into a NaN.
            dep = np.where(np.isfinite(dep), dep, 0.0)
            np.save(str(d / ("cam%d_depth.npy" % camera)), dep.astype(np.float16))


class Dataset:
    """Reads a capture directory written by :class:`DatasetWriter`."""

    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "meta.yaml").exists():
            raise FileNotFoundError(
                "%s does not look like a capture: no meta.yaml.  Run "
                "`python scripts/make_dataset.py` first." % (self.root,)
            )
        self.meta = yaml.safe_load((self.root / "meta.yaml").read_text(encoding="utf-8"))

    # -- truth (evaluation only) -------------------------------------------

    @property
    def truth_rig(self):
        return RigCalibration.load(self.root / "truth" / "rig.yaml")

    @property
    def object_mesh(self):
        return TriMesh.load_ply(self.root / "truth" / "object.ply", name="cad")

    @property
    def truth_poses(self):
        return json.loads((self.root / "truth" / "poses.json").read_text(encoding="utf-8"))

    def T_world_object(self, frame):
        return np.asarray(self.truth_poses["turntable"][int(frame)], dtype=float).reshape(4, 4)

    @property
    def T_rig_world(self):
        return np.asarray(self.truth_poses["rig"], dtype=float).reshape(4, 4)

    def turntable_mask(self, frame, camera):
        p = self.root / "turntable" / ("%03d" % frame) / ("cam%d_mask.png" % camera)
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(str(p))
        return np.where(m == BACKGROUND, -1, m.astype(np.int16)).astype(np.int16)

    def turntable_depth(self, frame, camera):
        p = self.root / "turntable" / ("%03d" % frame) / ("cam%d_depth.npy" % camera)
        d = np.load(str(p)).astype(np.float32)
        return np.where(d > 0, d, np.nan)

    # -- images ------------------------------------------------------------

    @property
    def n_cameras(self):
        return int(self.meta["n_cameras"])

    def intrinsic_views(self, camera):
        d = self.root / "intrinsics" / ("cam%d" % camera)
        return sorted(d.glob("*.png"))

    def intrinsic_image(self, camera, index):
        return _read_grey(self.root / "intrinsics" / ("cam%d" % camera) / ("%03d.png" % index))

    def extrinsic_frames(self):
        d = self.root / "extrinsics"
        return sorted(int(p.name) for p in d.iterdir() if p.is_dir())

    def extrinsic_image(self, frame, camera):
        return _read_grey(self.root / "extrinsics" / ("%03d" % frame) / ("cam%d.png" % camera))

    def turntable_frames(self):
        d = self.root / "turntable"
        return sorted(int(p.name) for p in d.iterdir() if p.is_dir())

    def turntable_image(self, frame, camera):
        return _read_grey(self.root / "turntable" / ("%03d" % frame) / ("cam%d.png" % camera))

    # -- the 6-DoF pose session -------------------------------------------

    def pose_frames(self):
        d = self.root / "pose"
        if not d.exists():
            return []
        return sorted(int(p.name) for p in d.iterdir() if p.is_dir())

    def pose_split(self):
        """``(train, test)`` frame indices.  Fixed by index, never by seed.

        The test split is the *last* ``n_pose_test`` stations.  Splitting by a
        fixed rule rather than a random draw means every run of every script
        agrees about which poses the refiner was allowed to see, including runs
        on different machines.
        """
        frames = self.pose_frames()
        n_test = int(self.meta.get("config", {}).get("pose_test", 0))
        if not n_test or n_test >= len(frames):
            return frames, frames
        return frames[:-n_test], frames[-n_test:]

    def pose_image(self, frame, camera):
        return _read_grey(self.root / "pose" / ("%03d" % frame) / ("cam%d.png" % camera))

    def pose_mask(self, frame, camera):
        p = self.root / "pose" / ("%03d" % frame) / ("cam%d_mask.png" % camera)
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(str(p))
        return np.where(m == BACKGROUND, -1, m.astype(np.int16)).astype(np.int16)

    def T_world_object_pose(self, frame):
        return np.asarray(self.truth_poses["pose"][int(frame)], dtype=float).reshape(4, 4)

    def turntable_frame(self, frame, with_truth=False):
        images = {c: self.turntable_image(frame, c) for c in range(self.n_cameras)}
        if not with_truth:
            return Frame(index=frame, images=images)
        return Frame(
            index=frame,
            images=images,
            masks={c: self.turntable_mask(frame, c) for c in range(self.n_cameras)},
            depths={c: self.turntable_depth(frame, c) for c in range(self.n_cameras)},
        )


def _read_grey(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img
