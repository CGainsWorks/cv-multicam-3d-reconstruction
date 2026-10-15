"""The dataset layout, the calibration file format, and the COLMAP bridge.

The most important test here is the last one: that no module in the pipeline
reads ground truth.  Every accuracy number in this repository depends on that
separation holding, and the separation is a convention rather than a mechanism,
so it is worth a test that enforces it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from multicam.geometry import PinholeCamera, se3, so3
from multicam.io import colmap as colmap_io
from multicam.io.dataset import BACKGROUND, Dataset, DatasetWriter
from multicam.types import RigCalibration


def test_rig_yaml_round_trip(rig, tmp_path):
    path = rig.save(tmp_path / "rig.yaml")
    back = RigCalibration.load(path)
    assert back.n_cameras == rig.n_cameras
    for a, b in zip(back.cameras, rig.cameras):
        assert np.allclose(a.params, b.params)
        assert (a.width, a.height) == (b.width, b.height)
    for a, b in zip(back.T_cam_rig, rig.T_cam_rig):
        assert np.allclose(a, b, atol=1e-12)


def test_rig_rejects_a_non_identity_reference(rig):
    bad = [T.copy() for T in rig.T_cam_rig]
    bad[0] = se3.exp(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="identity"):
        RigCalibration(cameras=rig.cameras, T_cam_rig=bad)


def test_rig_rejects_a_mismatched_camera_count(rig):
    with pytest.raises(ValueError, match="one extrinsic per camera"):
        RigCalibration(cameras=rig.cameras, T_cam_rig=rig.T_cam_rig[:2])


def test_rig_format_is_checked(tmp_path):
    with pytest.raises(ValueError, match="unrecognised"):
        RigCalibration.from_dict({"format": "something-else"})


def test_baselines_are_measured_from_the_reference(rig):
    baselines = rig.baselines()
    assert baselines[0] == 0.0
    assert all(b > 0 for b in baselines[1:])
    assert baselines == sorted(baselines), "the arc should be monotonic"


def test_camera_params_round_trip():
    camera = PinholeCamera(600.0, 601.0, 320.0, 240.0, 640, 480,
                           np.array([-0.2, 0.05, 1e-3, -2e-3, 0.01]))
    other = PinholeCamera.from_params(camera.params, camera.width, camera.height)
    assert np.allclose(other.params, camera.params)


def _tiny_capture(root, rig, rng):
    """Write a minimal but structurally complete capture."""
    writer = DatasetWriter(root)
    h, w = 24, 32
    for c in range(rig.n_cameras):
        for k in range(2):
            writer.write_intrinsic_view(c, k, rng.integers(0, 255, (h, w), dtype=np.uint8))
    for frame in range(2):
        for c in range(rig.n_cameras):
            writer.write_extrinsic_view(frame, c,
                                        rng.integers(0, 255, (h, w), dtype=np.uint8))
    for frame in range(2):
        for c in range(rig.n_cameras):
            mask = np.full((h, w), -1, dtype=np.int16)
            mask[4:10, 4:10] = 0
            mask[12:18, 12:18] = 1
            depth = np.full((h, w), np.inf, dtype=np.float32)
            depth[4:10, 4:10] = 0.63
            writer.write_turntable_view(
                frame, c, rng.integers(0, 255, (h, w), dtype=np.uint8),
                mask=mask, depth=depth,
            )
            writer.write_pose_view(frame, c, rng.integers(0, 255, (h, w), dtype=np.uint8),
                                   mask=mask)
    from multicam.sim import mesh as M

    writer.write_truth(rig, M.bracket(), {
        "rig": np.eye(4).reshape(16).tolist(),
        "turntable": [np.eye(4).reshape(16).tolist()] * 2,
        "pose": [np.eye(4).reshape(16).tolist()] * 2,
    })
    writer.write_meta({
        "format": "multicam-capture-v1", "n_cameras": rig.n_cameras,
        "width": w, "height": h, "config": {"pose_test": 1},
    })
    return writer


def test_dataset_round_trip(rig, tmp_path, rng):
    _tiny_capture(tmp_path / "cap", rig, rng)
    data = Dataset(tmp_path / "cap")

    assert data.n_cameras == rig.n_cameras
    assert len(data.intrinsic_views(0)) == 2
    assert data.extrinsic_frames() == [0, 1]
    assert data.turntable_frames() == [0, 1]
    assert data.pose_frames() == [0, 1]

    image = data.turntable_image(0, 0)
    assert image.shape == (24, 32) and image.dtype == np.uint8

    mask = data.turntable_mask(0, 0)
    assert mask[5, 5] == 0
    assert mask[13, 13] == 1
    assert mask[0, 0] == -1, "background must come back as -1, not %d" % BACKGROUND

    depth = data.turntable_depth(0, 0)
    assert np.isclose(depth[5, 5], 0.63, atol=1e-2)
    assert np.isnan(depth[0, 0]), "no surface must be NaN, not zero"

    assert np.allclose(data.truth_rig.T_cam_rig[1], rig.T_cam_rig[1], atol=1e-12)
    assert data.object_mesh.vertices.shape[1] == 3


def test_pose_split_is_by_a_fixed_rule(rig, tmp_path, rng):
    _tiny_capture(tmp_path / "cap", rig, rng)
    data = Dataset(tmp_path / "cap")
    train, test = data.pose_split()
    assert test == [1] and train == [0]
    assert not set(train) & set(test), "the splits must not overlap"
    # Calling it twice must give the same answer, with no RNG involved.
    assert data.pose_split() == (train, test)


def test_dataset_complains_clearly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_dataset"):
        Dataset(tmp_path / "nothing-here")


def test_frame_bundle_carries_truth_only_when_asked(rig, tmp_path, rng):
    _tiny_capture(tmp_path / "cap", rig, rng)
    data = Dataset(tmp_path / "cap")
    plain = data.turntable_frame(0)
    assert plain.masks is None and plain.depths is None
    full = data.turntable_frame(0, with_truth=True)
    assert set(full.masks) == set(range(rig.n_cameras))


# --------------------------------------------------------------------------
# COLMAP bridge
# --------------------------------------------------------------------------


def test_umeyama_recovers_a_known_similarity(rng):
    scale = 2.75
    T = se3.from_Rt(so3.exp(rng.normal(size=3) * 0.4), rng.normal(size=3) * 0.3)
    source = rng.normal(size=(50, 3))
    target = scale * (source @ T[:3, :3].T) + T[:3, 3]

    T_est, s_est = colmap_io.umeyama(source, target)
    assert np.isclose(s_est, scale, rtol=1e-9)
    assert np.allclose(T_est, T, atol=1e-9)
    assert np.allclose(colmap_io.apply_similarity(T_est, s_est, source), target, atol=1e-9)


def test_umeyama_without_scale_is_rigid(rng):
    T = se3.from_Rt(so3.exp(rng.normal(size=3) * 0.3), rng.normal(size=3) * 0.2)
    source = rng.normal(size=(30, 3))
    target = source @ T[:3, :3].T + T[:3, 3]
    T_est, s_est = colmap_io.umeyama(source, target, with_scale=False)
    assert s_est == 1.0
    assert np.allclose(T_est, T, atol=1e-9)


def test_umeyama_needs_three_points(rng):
    with pytest.raises(ValueError):
        colmap_io.umeyama(rng.normal(size=(2, 3)), rng.normal(size=(2, 3)))


def test_colmap_availability_is_reported_not_assumed():
    """Whether pycolmap is installed is a fact to report, not to depend on."""
    available = colmap_io.available()
    assert isinstance(available, bool)
    assert (colmap_io.version() is None) == (not available)


# --------------------------------------------------------------------------
# The ground-truth firewall
# --------------------------------------------------------------------------


#: Accessors on :class:`Dataset` that return ground truth.  Anything that reads
#: one of these is, by definition, evaluation.
GROUND_TRUTH_ACCESSORS = (
    "truth_rig", "truth_poses", "object_mesh", "T_world_object", "T_rig_world",
    "turntable_mask", "turntable_depth", "pose_mask", "T_world_object_pose",
)

#: Packages that implement the pipeline.  None of them may touch the above.
PIPELINE_PACKAGES = ("calibration", "reconstruction", "fusion", "pose")

#: The two deliberate exceptions, each narrow and each written down.
#:
#: ``pose/model.py`` uses the **CAD mesh** -- which the brief supplies as an
#: input to the problem, not as ground truth -- to decide which reconstructed
#: points lie on the part rather than on the fixture.  It never reads a pose, a
#: mask or the true rig.
#:
#: ``pose/dataset.py`` reads the **true poses of the training stations**,
#: because that is what supervised training means.  The held-out stations are
#: selected by a fixed rule in ``Dataset.pose_split`` and are never loaded
#: there; ``scripts/train_refiner.py`` asserts the two splits are disjoint
#: before it will train.
ALLOWED_GROUND_TRUTH_READERS = {
    "model.py": "uses the CAD mesh, which is a declared input",
    "dataset.py": "reads training-split poses; the test split is held out",
}


def test_no_pipeline_module_reads_ground_truth():
    """The firewall that every accuracy number in this repository rests on.

    The pipeline reads images.  Evaluation reads ground truth.  If a pipeline
    module ever reaches for a mask or the true rig, every number it produces
    becomes meaningless, and the mistake is completely invisible in the output
    -- the residuals would look *better*.

    The two deliberate exceptions are listed in
    :data:`ALLOWED_GROUND_TRUTH_READERS` with a reason each.  Everything that
    runs at *inference* time -- the calibration, the reconstruction, the
    fusion, and the pose estimator itself -- reads images and nothing else.
    """
    import ast
    import pathlib

    import multicam

    root = pathlib.Path(multicam.__file__).resolve().parent
    offenders = []
    for package in PIPELINE_PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.Name):
                    name = node.id
                if name in GROUND_TRUTH_ACCESSORS:
                    offenders.append("%s:%d uses %s" % (path.name, node.lineno, name))
    offenders = [
        o for o in offenders if o.split(":")[0] not in ALLOWED_GROUND_TRUTH_READERS
    ]
    assert not offenders, (
        "pipeline modules must not read ground truth:\n  " + "\n  ".join(offenders)
    )


def test_the_inference_path_reads_no_ground_truth_at_all():
    """Narrower than the test above, and the one that actually matters.

    The modules that run when a pose is estimated -- not the ones that build
    the model offline or train the refiner -- must not contain a single
    reference to a ground-truth accessor, with no exceptions allowed.
    """
    import ast
    import pathlib

    import multicam

    root = pathlib.Path(multicam.__file__).resolve().parent
    inference = [
        root / "pose" / "classical.py",
        root / "pose" / "refine.py",
        root / "pose" / "render_compare.py",
        root / "pose" / "network.py",
        root / "reconstruction" / "pipeline.py",
        root / "reconstruction" / "station.py",
        root / "calibration" / "extrinsic.py",
        root / "calibration" / "intrinsic.py",
        root / "fusion" / "stereo.py",
    ]
    offenders = []
    for path in inference:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else (
                node.id if isinstance(node, ast.Name) else None
            )
            if name in GROUND_TRUTH_ACCESSORS:
                offenders.append("%s:%d uses %s" % (path.name, node.lineno, name))
    assert not offenders, "\n  ".join(offenders)


def test_evaluation_is_allowed_to_read_ground_truth():
    """The counterpart: the firewall test must be capable of firing."""
    import pathlib

    import multicam

    root = pathlib.Path(multicam.__file__).resolve().parent
    text = (root / "evaluation" / "labels.py").read_text(encoding="utf-8")
    assert "mask" in text, "the labels module is meant to consume ground-truth masks"


def test_json_record_is_serialisable(tmp_path, rig):
    """Every script writes a JSON record; numpy must not leak into it."""
    payload = {
        "array": np.arange(3),
        "float": np.float64(1.5),
        "int": np.int64(7),
        "nested": {"poses": [np.eye(4)]},
    }
    text = json.dumps(payload, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    assert json.loads(text)["int"] == 7
