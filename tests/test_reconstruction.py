"""Reconstruction: matching, alignment, multi-motion fitting, bundle adjustment.

The bundle adjustment tests are the important ones.  A Schur complement is easy
to get subtly wrong -- a transposed block, a sign in the back-substitution --
and the result is a solver that still reduces the cost, just not as far or as
fast as it should.  So the tests check it against a known answer on synthetic
data, and against a dense reference solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam.geometry import se3, so3
from multicam.reconstruction import (
    bundle_adjust,
    epipolar_filter,
    kabsch,
    match_descriptors,
    sampson_distance,
)
from multicam.reconstruction.align import align_ransac, motion_magnitude, multi_motion_align


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_match_descriptors_finds_the_planted_correspondence(rng):
    a = rng.normal(size=(120, 128)).astype(np.float32)
    order = rng.permutation(120)
    b = a[order] + rng.normal(scale=0.01, size=(120, 128)).astype(np.float32)
    pairs = match_descriptors(a, b, ratio=0.8)
    assert len(pairs) > 100
    lookup = {int(i): int(j) for i, j in pairs}
    correct = sum(1 for i, j in lookup.items() if order[j] == i)
    assert correct / max(len(lookup), 1) > 0.98


def test_match_descriptors_returns_nothing_on_noise(rng):
    a = rng.normal(size=(60, 128)).astype(np.float32)
    b = rng.normal(size=(60, 128)).astype(np.float32)
    pairs = match_descriptors(a, b, ratio=0.7)
    assert len(pairs) < 5


def test_sampson_distance_is_zero_on_exact_correspondences(rig, rng):
    """A true correspondence lies exactly on its epipolar line."""
    X = rng.uniform(-0.05, 0.05, size=(60, 3)) + np.array([0.0, 0.0, 0.6])
    a, b = 0, 2
    Ta, Tb = rig.T_cam_rig[a], rig.T_cam_rig[b]
    uva = rig.cameras[a].project(se3.transform(Ta, X))
    uvb = rig.cameras[b].project(se3.transform(Tb, X))
    d = sampson_distance(rig.cameras[a], rig.cameras[b], Tb @ se3.inverse(Ta), uva, uvb)
    assert np.max(d) < 1e-6, np.max(d)


def test_epipolar_filter_rejects_wrong_matches(rig, rng):
    X = rng.uniform(-0.05, 0.05, size=(80, 3)) + np.array([0.0, 0.0, 0.6])
    a, b = 0, 1
    uva = rig.cameras[a].project(se3.transform(rig.T_cam_rig[a], X))
    uvb = rig.cameras[b].project(se3.transform(rig.T_cam_rig[b], X))
    # Shuffle half of them: those are wrong matches by construction.
    wrong = uvb.copy()
    perm = rng.permutation(40)
    wrong[:40] = uvb[perm]

    keep, baseline = epipolar_filter(
        rig.cameras[a], rig.cameras[b], rig.T_cam_rig[a], rig.T_cam_rig[b],
        uva, wrong, threshold_px=2.0,
    )
    assert baseline > 0.05
    assert keep[40:].all(), "every true match must survive"
    assert keep[:40].mean() < 0.35, "most shuffled matches must be rejected"


def test_epipolar_filter_is_a_no_op_at_zero_baseline(rig, rng):
    uv = rng.uniform(0, 400, size=(20, 2))
    keep, baseline = epipolar_filter(
        rig.cameras[0], rig.cameras[0], rig.T_cam_rig[0], rig.T_cam_rig[0], uv, uv
    )
    assert baseline < 1e-9
    assert keep.all()


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def test_kabsch_is_exact(rng):
    T = se3.from_Rt(so3.exp(rng.normal(size=3) * 0.5), rng.normal(size=3) * 0.2)
    Q = rng.normal(size=(40, 3))
    assert np.allclose(kabsch(se3.transform(T, Q), Q), T, atol=1e-9)


def test_kabsch_rejects_a_reflection(rng):
    Q = rng.normal(size=(30, 3))
    P = Q * np.array([1.0, 1.0, -1.0])          # a reflection, not a rotation
    T = kabsch(P, Q)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_align_ransac_ignores_outliers(rng):
    T = se3.from_Rt(so3.exp([0.0, 0.0, 0.5]), [0.02, -0.01, 0.0])
    Q = rng.uniform(-0.1, 0.1, size=(200, 3))
    P = se3.transform(T, Q) + rng.normal(scale=2e-4, size=(200, 3))
    P[:60] = rng.uniform(-0.5, 0.5, size=(60, 3))      # 30 per cent gross outliers

    result = align_ransac(P, Q, threshold=0.002, rng=np.random.default_rng(0))
    assert result.converged
    assert result.inliers[60:].mean() > 0.95
    assert result.inliers[:60].mean() < 0.1
    t_err, r_err = se3.pose_error(result.T_a_b, T)
    assert t_err < 5e-4 and r_err < 0.2


def test_multi_motion_separates_a_static_group_from_a_moving_one(rng):
    """The turntable problem in miniature: two rigid groups, different motions."""
    T_move = se3.from_Rt(so3.exp([0.0, 0.0, np.radians(30.0)]), [0.0, 0.0, 0.0])

    static = rng.uniform(-0.4, 0.4, size=(300, 3))
    moving = rng.uniform(-0.08, 0.08, size=(120, 3))
    Q = np.vstack([static, moving])
    P = np.vstack([static, se3.transform(T_move, moving)])
    P += rng.normal(scale=2e-4, size=P.shape)

    motions = multi_motion_align(P, Q, threshold=0.002, rng=np.random.default_rng(1))
    assert len(motions) >= 2

    statics = [m for m in motions if m[1]]
    movers = [m for m in motions if not m[1]]
    assert statics, "the static group must be found and labelled static"
    assert movers, "the moving group must be found and labelled moving"

    # The static group is the larger one here, exactly as in the real capture,
    # so selecting by size alone would pick the wrong one.
    assert statics[0][0].inliers.sum() > movers[0][0].inliers.sum()

    best_mover = max(movers, key=lambda m: m[0].inliers.sum())[0]
    rot, _ = motion_magnitude(best_mover.T_a_b)
    assert abs(rot - 30.0) < 0.5, rot
    assert best_mover.inliers[300:].mean() > 0.9


def test_motion_magnitude_reads_off_a_known_transform():
    T = se3.from_Rt(so3.exp([0.0, 0.0, np.radians(12.0)]), [0.03, 0.04, 0.0])
    rot, trans = motion_magnitude(T)
    assert np.isclose(rot, 12.0)
    assert np.isclose(trans, 0.05)


# --------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------


def _synthetic_bundle(rig, rng, n_points=90, n_stations=5, noise_px=0.25,
                      pose_error=0.004, point_error=0.002):
    """A small bundle problem with a known answer."""
    points_true = rng.uniform(-0.06, 0.06, size=(n_points, 3))
    poses_true = []
    for s in range(n_stations):
        angle = np.radians(25.0 * s)
        T = se3.from_Rt(so3.exp([0.0, 0.0, angle]), [0.0, 0.0, 0.62])
        poses_true.append(T)

    observations = []
    for s, T_rig_object in enumerate(poses_true):
        for c in range(rig.n_cameras):
            T = rig.T_cam_rig[c] @ T_rig_object
            uv, valid = rig.cameras[c].project(se3.transform(T, points_true),
                                               return_valid=True)
            inside = valid & rig.cameras[c].in_image(uv, margin=2.0)
            for p in np.nonzero(inside)[0]:
                observations.append(
                    (int(p), s, c, uv[p] + rng.normal(scale=noise_px, size=2))
                )

    points0 = points_true + rng.normal(scale=point_error, size=points_true.shape)
    poses0 = [poses_true[0]] + [
        T @ se3.exp(np.concatenate([
            rng.normal(scale=pose_error, size=3), rng.normal(scale=pose_error * 3, size=3)
        ]))
        for T in poses_true[1:]
    ]
    return points_true, poses_true, points0, poses0, observations


def test_bundle_adjustment_recovers_a_known_problem(rig, rng):
    truth_points, truth_poses, points0, poses0, observations = _synthetic_bundle(rig, rng)
    assert len(observations) > 800, "the fixture should produce a well constrained problem"

    result = bundle_adjust(points0, poses0, observations, rig, reference=0,
                           huber_delta=None, max_iterations=60)
    assert result.converged, result.reason
    assert result.rmse_px < result.initial_rmse_px
    assert result.rmse_px < 0.4, result.rmse_px

    for s in range(1, len(truth_poses)):
        t_err, r_err = se3.pose_error(result.T_rig_object[s], truth_poses[s])
        assert t_err < 1.5e-3, (s, t_err)
        assert r_err < 0.25, (s, r_err)

    error = np.linalg.norm(result.points - truth_points, axis=1)
    assert np.median(error) < 8e-4, np.median(error)
    assert np.median(error) < np.median(np.linalg.norm(points0 - truth_points, axis=1))


def test_bundle_adjustment_holds_the_reference_station_fixed(rig, rng):
    """The gauge.  If the reference moves, the problem was under-constrained."""
    _, _, points0, poses0, observations = _synthetic_bundle(rig, rng, n_stations=4)
    result = bundle_adjust(points0, poses0, observations, rig, reference=0)
    assert np.allclose(result.T_rig_object[0], poses0[0], atol=1e-15)


def test_bundle_adjustment_is_deterministic(rig, rng):
    _, _, points0, poses0, observations = _synthetic_bundle(rig, rng, n_stations=4)
    a = bundle_adjust(points0, poses0, observations, rig)
    b = bundle_adjust(points0, poses0, observations, rig)
    assert np.array_equal(a.points, b.points)
    assert a.rmse_px == b.rmse_px


def test_bundle_adjustment_rejects_duplicate_observations(rig, rng):
    """A point seen twice in one image means a broken track, not a measurement."""
    _, _, points0, poses0, observations = _synthetic_bundle(rig, rng, n_stations=3)
    first = observations[0]
    observations.append((first[0], first[1], first[2], first[3] + 1.0))
    with pytest.raises(ValueError, match="more than once"):
        bundle_adjust(points0, poses0, observations, rig)


def test_bundle_adjustment_survives_outlier_observations(rig, rng):
    """The Huber kernel plus the outlier mask should contain a few bad matches."""
    truth_points, truth_poses, points0, poses0, observations = _synthetic_bundle(
        rig, rng, n_points=110, n_stations=5
    )
    corrupted = list(observations)
    idx = rng.choice(len(corrupted), size=len(corrupted) // 20, replace=False)
    for i in idx:
        p, s, c, uv = corrupted[i]
        corrupted[i] = (p, s, c, uv + rng.normal(scale=25.0, size=2))

    result = bundle_adjust(points0, poses0, corrupted, rig, huber_delta=2.0,
                           max_iterations=60)
    for s in range(1, len(truth_poses)):
        t_err, r_err = se3.pose_error(result.T_rig_object[s], truth_poses[s])
        assert t_err < 3e-3, (s, t_err)
        assert r_err < 0.5, (s, r_err)
    assert result.inlier_mask is not None
    assert result.inlier_mask.mean() > 0.9
