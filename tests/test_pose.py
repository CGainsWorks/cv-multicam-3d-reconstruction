"""6-DoF pose: generalised PnP, the refiner's geometry, and the network's maths.

The network itself is not tested for accuracy here -- that is what
``scripts/ablate.py`` is for, and an accuracy assertion on an untrained network
would be meaningless.  What is tested is everything the network's correctness
depends on: that its SE(3) exponential agrees with the NumPy one it has to be
interchangeable with, that its loss is the metric it claims to be, that a zero
prediction is a no-op, and that the crops it is fed are geometrically aligned
with the images they came from.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam.evaluation.metrics import add_metric, adds_metric, pose_error_table, recall_at
from multicam.geometry import se3, so3
from multicam.pose import ModelRenderer, build_input, estimate_pose, refine_pose
from multicam.pose.classical import Correspondences
from multicam.pose.dataset import PerturbationSpec


def _correspondences(rig, T_rig_object, points, rng, noise_px=0.0, outliers=0.0,
                     margin=4.0):
    """Ideal 2D-3D correspondences for a known pose, optionally corrupted."""
    out = []
    for c in range(rig.n_cameras):
        T = rig.T_cam_rig[c] @ T_rig_object
        uv, valid = rig.cameras[c].project(se3.transform(T, points), return_valid=True)
        inside = valid & rig.cameras[c].in_image(uv, margin=margin)
        idx = np.nonzero(inside)[0]
        if not len(idx):
            out.append(Correspondences(c, np.zeros((0, 2)), np.zeros((0, 3))))
            continue
        xy = uv[idx].copy()
        if noise_px:
            xy = xy + rng.normal(scale=noise_px, size=xy.shape)
        obj = points[idx].copy()
        if outliers:
            n_bad = int(round(len(idx) * outliers))
            if n_bad:
                bad = rng.choice(len(idx), size=n_bad, replace=False)
                # A wrong match is a correct 2D point paired with the wrong 3D
                # point, which is exactly what a descriptor mismatch produces.
                obj[bad] = points[rng.choice(len(points), size=n_bad)]
        out.append(Correspondences(c, xy, obj))
    return out


@pytest.fixture(scope="module")
def model_points():
    from multicam.sim import mesh as M

    part = M.bracket()
    pts, _ = part.sample_surface(300, np.random.default_rng(0))
    return pts


@pytest.fixture(scope="module")
def true_pose():
    return se3.from_Rt(so3.exp([0.2, -0.15, 0.4]), [0.01, -0.02, 0.62])


def test_generalised_pnp_is_exact_without_noise(rig, model_points, true_pose):
    corr = _correspondences(rig, true_pose, model_points, np.random.default_rng(0))
    result = estimate_pose(rig, corr, threshold_px=1.0, rng=np.random.default_rng(0))
    assert result.ok
    t_err, r_err = se3.pose_error(result.T_rig_object, true_pose)
    assert t_err < 1e-5, t_err
    assert r_err < 1e-3, r_err


def test_generalised_pnp_survives_heavy_outliers(rig, model_points, true_pose):
    rng = np.random.default_rng(3)
    corr = _correspondences(rig, true_pose, model_points, rng, noise_px=0.4, outliers=0.45)
    result = estimate_pose(rig, corr, threshold_px=3.0, rng=np.random.default_rng(1))
    assert result.ok
    t_err, r_err = se3.pose_error(result.T_rig_object, true_pose)
    assert t_err < 1.5e-3, t_err
    assert r_err < 0.5, r_err
    assert result.n_inliers > 0.4 * result.n_correspondences


def test_all_cameras_contribute(rig, model_points, true_pose):
    """The estimate must use correspondences from more than the sampled camera."""
    corr = _correspondences(rig, true_pose, model_points, np.random.default_rng(0),
                            noise_px=0.3)
    result = estimate_pose(rig, corr, threshold_px=3.0, rng=np.random.default_rng(0))
    assert result.ok
    assert result.stats["cameras_used"] >= 3


def test_multi_camera_beats_a_single_camera(rig, model_points, true_pose):
    """The reason for solving jointly rather than averaging per-camera poses."""
    rng = np.random.default_rng(11)
    corr = _correspondences(rig, true_pose, model_points, rng, noise_px=1.2)

    joint = estimate_pose(rig, corr, threshold_px=4.0, rng=np.random.default_rng(0))
    assert joint.ok
    joint_err = se3.pose_error(joint.T_rig_object, true_pose)[0]

    singles = []
    for c in range(rig.n_cameras):
        one = [corr[c]]
        if len(one[0]) < 12:
            continue
        r = estimate_pose(rig, one, threshold_px=4.0, rng=np.random.default_rng(0))
        if r.ok:
            singles.append(se3.pose_error(r.T_rig_object, true_pose)[0])
    assert singles, "at least one camera should see enough points"
    assert joint_err < np.median(singles), (joint_err, np.median(singles))


def test_pose_refinement_improves_a_perturbed_start(rig, model_points, true_pose):
    rng = np.random.default_rng(5)
    corr = _correspondences(rig, true_pose, model_points, rng, noise_px=0.3)
    start = true_pose @ se3.exp(np.array([0.004, -0.003, 0.005, 0.02, -0.015, 0.01]))
    before = se3.pose_error(start, true_pose)[0]
    refined, result = refine_pose(rig, start, corr, huber_px=None)
    after = se3.pose_error(refined, true_pose)[0]
    assert result is not None and result.converged
    assert after < before / 5.0, (before, after)


def test_pose_refinement_reports_a_covariance(rig, model_points, true_pose):
    rng = np.random.default_rng(7)
    corr = _correspondences(rig, true_pose, model_points, rng, noise_px=0.5)
    result = estimate_pose(rig, corr, threshold_px=3.0, rng=np.random.default_rng(0))
    assert result.covariance is not None
    sigma = result.position_sigma_mm()
    assert sigma is not None and all(0 < s < 5.0 for s in sigma), sigma


def test_pose_estimation_refuses_when_starved(rig, model_points, true_pose):
    corr = [Correspondences(c, np.zeros((0, 2)), np.zeros((0, 3)))
            for c in range(rig.n_cameras)]
    result = estimate_pose(rig, corr)
    assert not result.ok
    assert "too few" in result.reason


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_add_is_zero_on_the_true_pose(model_points, true_pose):
    assert add_metric(true_pose, true_pose, model_points) < 1e-12
    assert adds_metric(true_pose, true_pose, model_points) < 1e-12


def test_add_reads_off_a_pure_translation(model_points, true_pose):
    shifted = true_pose @ se3.from_Rt(np.eye(3), [0.003, 0.0, 0.0])
    assert np.isclose(add_metric(shifted, true_pose, model_points), 0.003, atol=1e-9)


def test_adds_is_never_larger_than_add(model_points, true_pose, rng):
    for _ in range(10):
        perturbed = true_pose @ se3.exp(rng.normal(size=6) * 0.05)
        a = add_metric(perturbed, true_pose, model_points)
        s = adds_metric(perturbed, true_pose, model_points)
        assert s <= a + 1e-12


def test_recall_counts_failures_as_failures():
    assert recall_at([0.001, 0.002], 0.0015) == 0.5
    table = pose_error_table([], [], np.zeros((4, 3)), 0.2, failures=3)
    assert table["n"] == 3 and table["n_solved"] == 0
    assert table["add_recall_2_pct_diameter"] == 0.0


def test_pose_error_table_counts_the_failures_in_the_denominator(
    model_points, true_pose
):
    good = [true_pose] * 8
    table = pose_error_table(good, good, model_points, 0.187, failures=2)
    assert table["n"] == 10 and table["n_solved"] == 8
    # Eight perfect estimates out of ten attempts is 80 per cent recall.
    assert np.isclose(table["add_recall_2_pct_diameter"], 0.8)


# --------------------------------------------------------------------------
# The learned refiner's supporting machinery
# --------------------------------------------------------------------------


def test_torch_se3_exp_matches_numpy(rng):
    torch = pytest.importorskip("torch")
    from multicam.pose.network import se3_exp_torch

    twists = np.vstack([
        np.zeros(6),
        rng.normal(size=6) * 1e-8,
        rng.normal(size=6) * 0.01,
        rng.normal(size=6) * 0.5,
        rng.normal(size=6) * 2.0,
    ])
    ours = se3_exp_torch(torch.from_numpy(twists)).numpy()
    for i, xi in enumerate(twists):
        assert np.allclose(ours[i], se3.exp(xi), atol=1e-9), i


def test_torch_se3_exp_has_finite_gradients_at_zero():
    torch = pytest.importorskip("torch")
    from multicam.pose.network import se3_exp_torch

    twist = torch.zeros(1, 6, requires_grad=True)
    T = se3_exp_torch(twist)
    T.sum().backward()
    assert torch.isfinite(twist.grad).all()


def test_point_matching_loss_is_zero_for_the_right_twist(model_points, true_pose):
    torch = pytest.importorskip("torch")
    from multicam.pose.network import point_matching_loss

    start = true_pose @ se3.exp(np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.02]))
    correct = se3.log(se3.inverse(start) @ true_pose)

    loss = point_matching_loss(
        torch.from_numpy(correct[None].astype(np.float64)),
        torch.from_numpy(start[None].astype(np.float64)),
        torch.from_numpy(true_pose[None].astype(np.float64)),
        torch.from_numpy(model_points.astype(np.float64)),
    )
    assert float(loss) < 1e-9


def test_point_matching_loss_equals_add(model_points, true_pose):
    torch = pytest.importorskip("torch")
    from multicam.pose.network import point_matching_loss

    start = true_pose @ se3.exp(np.array([0.004, -0.002, 0.001, 0.01, 0.0, 0.03]))
    zero = torch.zeros(1, 6, dtype=torch.float64)
    loss = float(point_matching_loss(
        zero,
        torch.from_numpy(start[None]),
        torch.from_numpy(true_pose[None]),
        torch.from_numpy(model_points),
    ))
    assert np.isclose(loss, add_metric(start, true_pose, model_points), atol=1e-12)


def test_untrained_network_predicts_no_correction():
    """The head is zero initialised, so the first iteration must be a no-op."""
    torch = pytest.importorskip("torch")
    from multicam.pose.network import PoseRefineNet

    net = PoseRefineNet(n_cameras=4)
    out = net(torch.zeros(2, 4, 4, 96, 96))
    assert torch.allclose(out, torch.zeros_like(out))


def test_network_output_shape_and_camera_masking():
    torch = pytest.importorskip("torch")
    from multicam.pose.network import PoseRefineNet

    net = PoseRefineNet(n_cameras=4)
    crops = torch.randn(3, 4, 4, 96, 96)
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 3)
    assert net(crops).shape == (3, 6)
    assert net(crops, valid).shape == (3, 6)


def test_render_compare_crop_is_aligned_with_the_image(rig, scene):
    """The crop's own camera must project the model where the crop shows it.

    This is the geometric contract the whole render-and-compare idea rests on:
    if the observed crop and the rendered crop are not the same window of the
    same image, the network is being asked to compare two different pictures.
    """
    from multicam.sim.render import Rasteriser
    from multicam.sim.rig import nominal_rig_pose

    cad = scene.object_mesh
    renderer = ModelRenderer(cad)
    T_rig_world = nominal_rig_pose(rig)
    T_rig_object = T_rig_world @ scene.T_world_object

    ras = Rasteriser(scene.textures)
    images = {}
    for c in range(rig.n_cameras):
        r = ras.render(rig.cameras[c], rig.T_cam_world(c, T_rig_world),
                       scene.meshes_at(), supersample=1)
        images[c] = r.image

    crops, specs = build_input(renderer, images, rig, T_rig_object)
    assert crops.shape == (rig.n_cameras, 4, renderer.crop, renderer.crop)

    for c, spec in enumerate(specs):
        if not spec.valid:
            continue
        T_cam_object = rig.T_cam_rig[c] @ T_rig_object
        crop_camera = renderer.crop_camera(rig.cameras[c], spec)
        # The model centroid must project to the middle of the crop.
        uv = crop_camera.project(se3.transform(T_cam_object, cad.centre))
        assert np.allclose(uv, renderer.crop / 2.0, atol=1.5), (c, uv)
        # And the rendered mask must be non-empty and roughly centred.
        mask = crops[c, 1]
        assert mask.sum() > 100, c
        v, u = np.nonzero(mask > 0.5)
        assert abs(u.mean() - renderer.crop / 2.0) < renderer.crop * 0.25
        assert abs(v.mean() - renderer.crop / 2.0) < renderer.crop * 0.25


def test_perturbation_magnitudes_land_in_their_band(rng):
    spec = PerturbationSpec(0.002, 0.002, 1.0, 1.0)
    for _ in range(20):
        T = spec.sample(rng)
        t, r = se3.pose_error(T, np.eye(4))
        assert np.isclose(t, 0.002, atol=1e-9)
        assert np.isclose(r, 1.0, atol=1e-6)


def test_perturbation_covers_its_whole_range(rng):
    spec = PerturbationSpec(0.0002, 0.015, 0.05, 8.0)
    magnitudes = []
    for _ in range(400):
        t, _ = se3.pose_error(spec.sample(rng), np.eye(4))
        magnitudes.append(t)
    magnitudes = np.asarray(magnitudes)
    assert magnitudes.min() < 0.0006, "no easy samples were drawn"
    assert magnitudes.max() > 0.010, "no hard samples were drawn"
