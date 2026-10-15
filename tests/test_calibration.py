"""Calibration: homographies, Zhang's linear stage, the joint refinement, the rig.

The valuable tests here are the ones on *ideal synthetic data*, where the
answer is known exactly and any error is a bug rather than noise.  A
calibration that cannot recover the intrinsics from noise-free projections of a
board is broken, and no amount of averaging over real captures will reveal that
as clearly as one exact test.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam.calibration import (
    CharucoTarget,
    calibrate_extrinsics,
    estimate_homography,
    linear_intrinsics,
    pose_from_homography,
    refine_intrinsics,
)
from multicam.calibration.extrinsic import (
    average_poses,
    pairwise_relative_poses,
    spanning_tree_extrinsics,
)
from multicam.calibration.target import BoardDetection
from multicam.geometry import PinholeCamera, se3, so3


@pytest.fixture(scope="module")
def target():
    return CharucoTarget()


def _board_views(camera, target, n=14, seed=0, noise_px=0.0):
    """Ideal projections of the board at ``n`` poses."""
    rng = np.random.default_rng(seed)
    obj = target.object_points
    centre = obj.mean(axis=0)
    poses, images = [], []
    for _ in range(n):
        R = so3.exp(rng.normal(scale=0.35, size=3))
        depth = rng.uniform(0.35, 0.75)
        offset = np.array([rng.normal(scale=0.05), rng.normal(scale=0.05), depth])
        T = se3.from_Rt(R, offset - R @ centre)
        uv = camera.project(se3.transform(T, obj))
        if noise_px:
            uv = uv + rng.normal(scale=noise_px, size=uv.shape)
        poses.append(T)
        images.append(uv)
    return poses, images


# --------------------------------------------------------------------------
# Homography
# --------------------------------------------------------------------------


def test_homography_is_exact_on_a_plane(rng):
    H_true = np.array([[1.4, 0.15, 30.0], [-0.1, 1.25, -12.0], [3e-4, -2e-4, 1.0]])
    src = rng.uniform(-1.0, 1.0, size=(30, 2))
    h = np.c_[src, np.ones(len(src))] @ H_true.T
    dst = h[:, :2] / h[:, 2:]
    H = estimate_homography(src, dst)
    assert np.allclose(H / H[2, 2], H_true / H_true[2, 2], atol=1e-9)


def test_homography_needs_four_points(rng):
    with pytest.raises(ValueError):
        estimate_homography(rng.normal(size=(3, 2)), rng.normal(size=(3, 2)))


# --------------------------------------------------------------------------
# Zhang
# --------------------------------------------------------------------------


def test_linear_stage_recovers_the_intrinsics_exactly(target):
    """No distortion, no noise: the closed form should be exact."""
    camera = PinholeCamera(700.0, 690.0, 400.0, 300.0, 800, 600, np.zeros(5))
    poses, images = _board_views(camera, target, n=12)
    H = [estimate_homography(target.object_points[:, :2], uv) for uv in images]
    K, skew = linear_intrinsics(H)
    assert np.isclose(K[0, 0], camera.fx, atol=1e-6)
    assert np.isclose(K[1, 1], camera.fy, atol=1e-6)
    assert np.isclose(K[0, 2], camera.cx, atol=1e-6)
    assert np.isclose(K[1, 2], camera.cy, atol=1e-6)
    assert abs(skew) < 1e-6


def test_pose_from_homography_is_exact(target):
    camera = PinholeCamera(700.0, 690.0, 400.0, 300.0, 800, 600, np.zeros(5))
    poses, images = _board_views(camera, target, n=8, seed=3)
    for T_true, uv in zip(poses, images):
        H = estimate_homography(target.object_points[:, :2], uv)
        T = pose_from_homography(camera.K, H)
        t_err, r_err = se3.pose_error(T, T_true)
        assert t_err < 1e-9 and r_err < 1e-7


def test_linear_stage_refuses_too_few_views(target):
    camera = PinholeCamera(700.0, 690.0, 400.0, 300.0, 800, 600, np.zeros(5))
    _, images = _board_views(camera, target, n=2)
    H = [estimate_homography(target.object_points[:, :2], uv) for uv in images]
    with pytest.raises(ValueError):
        linear_intrinsics(H)


# --------------------------------------------------------------------------
# Joint refinement
# --------------------------------------------------------------------------


def test_refinement_recovers_distortion_from_a_wrong_start(target):
    """The whole point of the non-linear stage: the linear one cannot see distortion."""
    truth = PinholeCamera(
        700.0, 690.0, 400.0, 300.0, 800, 600,
        np.array([-0.24, 0.07, 0.0009, -0.0006, -0.012]),
    )
    poses, images = _board_views(truth, target, n=16, seed=5)
    obj = [target.object_points] * len(poses)

    # Start deliberately wrong: 5 per cent off in focal length, 8 px off in the
    # principal point, and no distortion at all.
    start = PinholeCamera(735.0, 725.0, 408.0, 292.0, 800, 600, np.zeros(5))
    start_poses = [T @ se3.exp(np.concatenate([np.full(3, 0.004), np.full(3, 0.01)]))
                   for T in poses]

    camera, refined, result = refine_intrinsics(
        start, start_poses, obj, images, huber_delta=None, max_iterations=120
    )
    assert result.converged, result.reason
    assert np.isclose(camera.fx, truth.fx, atol=0.05)
    assert np.isclose(camera.fy, truth.fy, atol=0.05)
    assert np.isclose(camera.cx, truth.cx, atol=0.05)
    assert np.isclose(camera.cy, truth.cy, atol=0.05)
    assert np.allclose(camera.dist, truth.dist, atol=2e-4)
    assert result.rmse < 1e-4
    for T, T_true in zip(refined, poses):
        t_err, r_err = se3.pose_error(T, T_true)
        assert t_err < 1e-5 and r_err < 1e-3


def test_refinement_is_robust_to_a_few_gross_outliers(target):
    """The Huber kernel is there for misdetected corners; check it works."""
    truth = PinholeCamera(700.0, 690.0, 400.0, 300.0, 800, 600,
                          np.array([-0.2, 0.05, 0.0, 0.0, 0.0]))
    poses, images = _board_views(truth, target, n=16, seed=9, noise_px=0.05)
    rng = np.random.default_rng(1)
    corrupted = []
    for uv in images:
        uv = uv.copy()
        # Two corners per view thrown 20 px off, which is what a mis-decoded
        # marker looks like.
        idx = rng.choice(len(uv), size=2, replace=False)
        uv[idx] += rng.normal(scale=20.0, size=(2, 2))
        corrupted.append(uv)

    start = PinholeCamera(720.0, 710.0, 404.0, 296.0, 800, 600, np.zeros(5))
    camera, _, result = refine_intrinsics(
        start, list(poses), [target.object_points] * len(poses), corrupted,
        huber_delta=2.0, max_iterations=120,
    )
    assert abs(camera.fx - truth.fx) < 3.0, camera.fx
    assert abs(camera.cx - truth.cx) < 3.0, camera.cx


# --------------------------------------------------------------------------
# Rig extrinsics
# --------------------------------------------------------------------------


def test_average_poses_is_the_mean_of_small_perturbations(rng):
    base = se3.exp(rng.normal(size=6) * 0.3)
    poses = [base @ se3.exp(rng.normal(size=6) * 0.01) for _ in range(40)]
    mean = average_poses(poses)
    t_err, r_err = se3.pose_error(mean, base)
    assert t_err < 0.004 and r_err < 0.4


def test_spanning_tree_chains_through_the_middle(rig):
    """Cameras 0 and 3 never co-observe, so the chain must go through 1 and 2."""
    T_true = rig.T_cam_rig
    board_poses = {}
    # Frames where only neighbouring pairs see the board.
    for frame, (a, b) in enumerate([(0, 1)] * 5 + [(1, 2)] * 5 + [(2, 3)] * 5):
        T_rig_board = se3.exp(np.array([0.01, 0.02, 0.5, 0.1, 0.05, 0.2]) * (frame + 1) * 0.1)
        board_poses[(frame, a)] = T_true[a] @ T_rig_board
        board_poses[(frame, b)] = T_true[b] @ T_rig_board

    pairwise = pairwise_relative_poses(board_poses)
    assert (0, 3) not in pairwise, "the fixture must not give a direct 0-3 edge"
    estimated = spanning_tree_extrinsics(4, pairwise, reference=0, min_shared=3)
    for i in range(4):
        t_err, r_err = se3.pose_error(estimated[i], T_true[i])
        assert t_err < 1e-9 and r_err < 1e-7, i


def test_spanning_tree_reports_an_unreachable_camera():
    poses = {}
    T = np.eye(4)
    for frame in range(5):
        poses[(frame, 0)] = T
        poses[(frame, 1)] = T
    pairwise = pairwise_relative_poses(poses)
    with pytest.raises(ValueError, match="not connected"):
        spanning_tree_extrinsics(3, pairwise, reference=0, min_shared=3)


def test_extrinsic_solve_recovers_a_known_rig(rig, target):
    """Synthetic board observations through a known rig, solved back."""
    rng = np.random.default_rng(4)
    obj = target.object_points
    centre = obj.mean(axis=0)

    detections, truth_board = {}, {}
    for frame in range(22):
        R = so3.exp(rng.normal(scale=0.30, size=3))
        C = np.array([rng.normal(scale=0.05), rng.normal(scale=0.05),
                      rng.uniform(0.45, 0.75)])
        T_rig_board = se3.from_Rt(R, C - R @ centre)
        truth_board[frame] = T_rig_board
        for c in range(rig.n_cameras):
            T = rig.T_cam_rig[c] @ T_rig_board
            uv, valid = rig.cameras[c].project(se3.transform(T, obj), return_valid=True)
            inside = valid & rig.cameras[c].in_image(uv, margin=2.0)
            if inside.sum() < 8:
                continue
            ids = np.nonzero(inside)[0]
            detections[(frame, c)] = BoardDetection(
                ids=ids, corners=uv[ids] + rng.normal(scale=0.08, size=(len(ids), 2)),
                n_markers=len(ids), camera=c, frame=frame,
            )

    seen = {}
    for (frame, c) in detections:
        seen[frame] = seen.get(frame, 0) + 1
    assert sum(1 for v in seen.values() if v >= 2) >= 8, "fixture gives too few shared views"

    result = calibrate_extrinsics(rig.cameras, target, detections, huber_delta=2.0)
    assert result.rmse_px < 0.3, result.rmse_px
    for c in range(rig.n_cameras):
        t_err, r_err = se3.pose_error(result.rig.T_cam_rig[c], rig.T_cam_rig[c])
        assert t_err < 0.004, (c, t_err)
        assert r_err < 0.35, (c, r_err)


def test_reference_camera_stays_at_the_identity(rig, target):
    """The gauge must be exactly fixed, not approximately."""
    rng = np.random.default_rng(6)
    obj = target.object_points
    detections = {}
    for frame in range(14):
        T_rig_board = se3.from_Rt(
            so3.exp(rng.normal(scale=0.25, size=3)),
            np.array([0.0, 0.0, 0.6]) + rng.normal(scale=0.04, size=3),
        )
        for c in range(rig.n_cameras):
            T = rig.T_cam_rig[c] @ T_rig_board
            uv, valid = rig.cameras[c].project(se3.transform(T, obj), return_valid=True)
            inside = valid & rig.cameras[c].in_image(uv, margin=2.0)
            if inside.sum() < 8:
                continue
            ids = np.nonzero(inside)[0]
            detections[(frame, c)] = BoardDetection(ids, uv[ids], len(ids), c, frame)
    result = calibrate_extrinsics(rig.cameras, target, detections)
    assert np.allclose(result.rig.T_cam_rig[0], np.eye(4), atol=1e-12)


def test_charuco_object_points_are_the_expected_grid(target):
    nx, ny = target.size
    obj = target.object_points
    assert len(obj) == (nx - 1) * (ny - 1)
    assert np.allclose(obj[:, 2], 0.0)
    step = target.square_length
    assert np.isclose(obj[1, 0] - obj[0, 0], step, atol=1e-6)
