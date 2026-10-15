"""COLMAP, as an independent reference for the sparse reconstruction.

COLMAP is used through ``pycolmap``, which is the real thing as a library
rather than a reimplementation of it, and it shares no code with anything in
:mod:`multicam.reconstruction`.  That is the point: it is a referee, and a
referee that shared a bug with the thing it is judging would be worthless.

The comparison is not apples to apples and pretending otherwise would be
dishonest, so the differences are stated here and repeated in the README:

* **COLMAP does not know the rig.**  It solves for every camera pose
  independently by structure from motion.  Our pipeline is handed the
  calibrated extrinsics and only solves for the object's motion.  COLMAP is
  therefore doing a strictly harder problem and should be expected to do
  slightly worse on pose, and it is a fair check on whether the *shape* we
  recover is right.
* **COLMAP's reconstruction has no scale.**  Structure from motion recovers
  geometry up to a similarity.  Ours is metric because the rig calibration is
  metric.  So the comparison is made after a Sim(3) alignment, and the scale
  factor that alignment needed is itself reported -- it is the single number
  that says what the rig calibration bought.
* **The frames are different.**  COLMAP reconstructs in the frame of whichever
  image it chose to initialise from.

The turntable makes one more wrinkle.  COLMAP assumes a static scene and a
moving camera; here the camera is static and the scene turns.  The two are the
same problem viewed from the object's frame, so COLMAP reconstructs the turning
assembly correctly and treats the static background as outliers -- provided the
turning assembly is the dominant structure, which is exactly the condition the
tight framing was chosen to satisfy.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry import se3, so3


def available():
    """Is pycolmap importable?"""
    try:
        import pycolmap  # noqa: F401

        return True
    except Exception:
        return False


def version():
    try:
        import pycolmap

        return str(getattr(pycolmap, "__version__", "unknown"))
    except Exception:
        return None


@dataclass
class ColmapResult:
    points: np.ndarray                  #: (N, 3) in COLMAP's own frame
    image_poses: dict                   #: image name -> T_cam_colmap (4, 4)
    n_images_registered: int
    n_images_total: int
    mean_track_length: float
    mean_reprojection_error_px: float
    seconds: float
    stats: dict = field(default_factory=dict)

    def summary(self):
        return {
            "points": int(len(self.points)),
            "images_registered": int(self.n_images_registered),
            "images_total": int(self.n_images_total),
            "mean_track_length": self.mean_track_length,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "seconds": self.seconds,
            **self.stats,
        }


def export_images(images_by_frame, workspace, prefix="img"):
    """Write the sweep out as a flat directory of images for COLMAP.

    Returns ``{image name: (frame, camera)}``.  Names carry the frame and
    camera so the reconstruction can be mapped back afterwards.
    """
    import cv2

    workspace = Path(workspace)
    image_dir = workspace / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    index = {}
    for frame in sorted(images_by_frame):
        for camera in sorted(images_by_frame[frame]):
            name = "%s_f%03d_c%d.png" % (prefix, frame, camera)
            cv2.imwrite(str(image_dir / name), images_by_frame[frame][camera])
            index[name] = (frame, camera)
    return index, image_dir


def run(images_by_frame, workspace, camera_model="OPENCV", shared_camera=False,
        max_image_size=1600, quiet=True, max_models=3, min_model_size=12, seed=0):
    """Run the COLMAP incremental pipeline over a capture.

    ``shared_camera=False`` because the four cameras genuinely have different
    intrinsics; telling COLMAP they are the same camera would hand it a wrong
    model and make the comparison meaningless.
    """
    import time

    import pycolmap

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    index, image_dir = export_images(images_by_frame, workspace)

    database = workspace / "database.db"
    if database.exists():
        database.unlink()
    sparse_dir = workspace / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    if quiet:
        try:
            pycolmap.logging.minloglevel = 3
        except Exception:
            pass

    # COLMAP's incremental mapper is randomised, and on this sequence it is
    # *very* randomised: two runs with identical settings registered 50 and 15
    # of the 72 images.  That spread is itself the finding -- a turntable gives
    # structure from motion two irreconcilable rigid scenes and no stable
    # solution to settle on -- but an unseeded number has no business being
    # quoted in a README, so the seed is fixed and recorded.
    try:
        pycolmap.set_random_seed(int(seed))
    except Exception:
        pass

    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = int(max_image_size)
    pycolmap.extract_features(
        database_path=database, image_path=image_dir,
        camera_mode=pycolmap.CameraMode.PER_IMAGE if not shared_camera
        else pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extraction,
    )
    pycolmap.match_exhaustive(database_path=database)

    mapper = pycolmap.IncrementalPipelineOptions()
    # A turntable sequence is an awkward thing to hand a structure-from-motion
    # pipeline: the object moves, the room does not, and the two are
    # irreconcilable in any single rigid reconstruction.  COLMAP's response is
    # to split the images into separate models, and left to its default of 50 it
    # will keep re-initialising for well over an hour, producing a sixth and a
    # seventh fragment that register a handful of images each.
    #
    # Capping the model count bounds the run without changing what is measured:
    # the comparison uses the largest model, and the number of models it needed
    # is reported, because "COLMAP could not fit this in one model" is itself a
    # result worth having.
    mapper.max_num_models = int(max_models)
    mapper.min_model_size = int(min_model_size)

    maps = pycolmap.incremental_mapping(
        database_path=database, image_path=image_dir, output_path=sparse_dir,
        options=mapper,
    )
    seconds = time.time() - t0

    if not maps:
        return ColmapResult(np.zeros((0, 3)), {}, 0, len(index), 0.0, float("nan"),
                            seconds, {"reason": "COLMAP registered no model"})

    def _n_registered(model):
        n = model.num_reg_images
        return int(n() if callable(n) else n)

    # The largest model, if COLMAP split the images into several.
    best = max(maps.values(), key=_n_registered)
    points = np.array([p.xyz for p in best.points3D.values()], dtype=float)
    errors = [p.error for p in best.points3D.values() if p.error >= 0]
    tracks = [len(p.track.elements) for p in best.points3D.values()]

    poses = {}
    for image in best.images.values():
        if not image.has_pose:
            continue
        # cam_from_world is a property in some pycolmap builds and a method in
        # others; both appear in the wild and both are handled.
        rigid = image.cam_from_world
        if callable(rigid):
            rigid = rigid()
        R = np.asarray(rigid.rotation.matrix(), dtype=float)
        t = np.asarray(rigid.translation, dtype=float)
        poses[image.name] = se3.from_Rt(R, t)

    registered = best.num_reg_images
    if callable(registered):
        registered = registered()

    return ColmapResult(
        points=points,
        image_poses=poses,
        n_images_registered=int(registered),
        n_images_total=len(index),
        mean_track_length=float(np.mean(tracks)) if tracks else 0.0,
        mean_reprojection_error_px=float(np.mean(errors)) if errors else float("nan"),
        seconds=seconds,
        stats={"models": len(maps), "seed": int(seed),
               "max_models": int(max_models),
               "image_index": {k: list(v) for k, v in index.items()}},
    )


def umeyama(source, target, with_scale=True):
    """Similarity transform taking ``source`` onto ``target``.

    Returns ``(T, scale)`` with ``target ~= scale * R @ source + t``.  This is
    the standard closed form; the scale is what makes it usable on a structure
    from motion result, which is only determined up to one.
    """
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("umeyama needs at least three matched points")

    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    S = source - mu_s
    T = target - mu_t
    H = S.T @ T / len(source)
    U, D, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    R = so3.project_to_so3(R)

    if with_scale:
        var_s = float(np.mean(np.sum(S ** 2, axis=1)))
        scale = float(np.sum(D * np.array([1.0, 1.0, d]))) / max(var_s, 1e-30)
    else:
        scale = 1.0
    t = mu_t - scale * R @ mu_s
    return se3.from_Rt(R, t), scale


def apply_similarity(T, scale, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return scale * (points @ T[:3, :3].T) + T[:3, 3]


def align_by_camera_centres(colmap_result, reference_poses, index):
    """Sim(3) from COLMAP's frame to a reference frame, using camera centres.

    ``reference_poses`` maps ``(frame, camera)`` to the camera-from-reference
    pose that the calibrated pipeline believes.  Camera centres are used rather
    than the 3D points because the two reconstructions have no point
    correspondence -- they found their own features independently -- while the
    images are the same images and their centres are directly comparable.
    """
    src, dst = [], []
    for name, T_cam_colmap in colmap_result.image_poses.items():
        key = tuple(index[name]) if name in index else None
        if key is None or key not in reference_poses:
            continue
        src.append(se3.camera_centre(T_cam_colmap))
        dst.append(se3.camera_centre(reference_poses[key]))
    if len(src) < 3:
        raise RuntimeError(
            "only %d COLMAP images could be matched to reference poses" % len(src)
        )
    T, scale = umeyama(np.array(src), np.array(dst))
    residual = np.linalg.norm(
        apply_similarity(T, scale, np.array(src)) - np.array(dst), axis=1
    )
    return T, scale, {
        "cameras_aligned": len(src),
        "centre_rmse_mm": float(np.sqrt(np.mean(residual ** 2)) * 1000),
        "centre_max_mm": float(residual.max() * 1000),
    }
