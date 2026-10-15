"""The foundation: Lie groups, the camera model and triangulation.

Every Jacobian in this file is checked against central finite differences.
That is the whole point of the file -- the bundle adjuster, the calibration
solve and the pose refinement all differentiate through these functions, and a
sign error in any of them produces a solver that converges slowly on easy data
and not at all on hard data, which is very hard to debug later and very easy to
catch here.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam.geometry import PinholeCamera, se3, so3, triangulation


def central_difference(f, x, eps=1e-6):
    """``df/dx`` by central differences, for a vector-valued ``f``."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(f(x), dtype=float))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.atleast_1d(f(x + dx)) - np.atleast_1d(f(x - dx))) / (2.0 * eps)
    return J


# --------------------------------------------------------------------------
# SO(3)
# --------------------------------------------------------------------------


def test_hat_is_the_cross_product(rng):
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(so3.hat(a) @ b, np.cross(a, b))


def test_hat_vee_round_trip(rng):
    for _ in range(20):
        w = rng.normal(size=3)
        assert np.allclose(so3.vee(so3.hat(w)), w)


def test_exp_produces_rotations(rng):
    for _ in range(50):
        R = so3.exp(rng.normal(size=3) * rng.uniform(0, 2.0))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.5, 2.0, 3.0])
def test_exp_log_round_trip_across_the_whole_range(scale, rng):
    """Including the two branches: the series near zero and the near-pi form."""
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        w = axis * scale
        assert np.allclose(so3.log(so3.exp(w)), w, atol=1e-9)


def test_log_near_pi_is_stable(rng):
    """theta -> pi is where the naive formula divides by a vanishing sine."""
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = float(rng.uniform(np.pi - 1e-6, np.pi))
        R = so3.exp(axis * theta)
        w = so3.log(R)
        # The axis sign is arbitrary exactly at pi, so compare the rotations.
        assert np.allclose(so3.exp(w), R, atol=1e-7)
        assert np.isclose(np.linalg.norm(w), theta, atol=1e-6)


def test_right_jacobian_matches_finite_differences(rng):
    """exp(w + dw) ~= exp(w) exp(Jr(w) dw) is the defining property."""
    for scale in (1e-8, 1e-3, 0.7, 2.5):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * scale
        analytic = so3.right_jacobian(w)
        numeric = central_difference(lambda v: so3.log(so3.exp(w).T @ so3.exp(v)), w, eps=1e-7)
        assert np.allclose(analytic, numeric, atol=1e-6), scale


def test_right_jacobian_inverse(rng):
    for scale in (1e-9, 1e-4, 1.0, 3.0):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * scale
        assert np.allclose(
            so3.right_jacobian(w) @ so3.right_jacobian_inv(w), np.eye(3), atol=1e-9
        )


def test_project_to_so3_repairs_a_perturbed_rotation(rng):
    R = so3.exp(rng.normal(size=3))
    dirty = R + rng.normal(size=(3, 3)) * 1e-3
    fixed = so3.project_to_so3(dirty)
    assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(fixed), 1.0)
    assert so3.angle_between(R, fixed) < 2e-3


def test_project_to_so3_rejects_a_reflection():
    M = np.diag([1.0, 1.0, -1.0])
    R = so3.project_to_so3(M)
    assert np.isclose(np.linalg.det(R), 1.0)


# --------------------------------------------------------------------------
# SE(3)
# --------------------------------------------------------------------------


def test_se3_exp_log_round_trip(rng):
    for scale in (0.0, 1e-9, 1e-4, 1.0):
        for _ in range(20):
            xi = rng.normal(size=6) * scale
            assert np.allclose(se3.log(se3.exp(xi)), xi, atol=1e-9)


def test_se3_inverse_and_compose(rng):
    for _ in range(20):
        T = se3.exp(rng.normal(size=6))
        assert np.allclose(se3.compose(T, se3.inverse(T)), np.eye(4), atol=1e-12)


def test_adjoint_identity(rng):
    """T exp(xi) == exp(Ad(T) xi) T is what makes right perturbations usable."""
    for _ in range(20):
        T = se3.exp(rng.normal(size=6))
        xi = rng.normal(size=6) * 0.1
        lhs = T @ se3.exp(xi)
        rhs = se3.exp(se3.adjoint(T) @ xi) @ T
        assert np.allclose(lhs, rhs, atol=1e-9)


def test_transform_matches_matrix_multiplication(rng):
    T = se3.exp(rng.normal(size=6))
    P = rng.normal(size=(17, 3))
    expected = (np.c_[P, np.ones(17)] @ T.T)[:, :3]
    assert np.allclose(se3.transform(T, P), expected)
    assert np.allclose(se3.transform(T, P[0]), expected[0])


def test_look_at_puts_the_target_on_the_optical_axis(rng):
    for _ in range(20):
        eye = rng.normal(size=3) * 2.0
        target = rng.normal(size=3) * 0.3
        T = se3.look_at(eye, target)
        p = se3.transform(T, target)
        assert p[2] > 0, "target must be in front of the camera"
        assert np.allclose(p[:2], 0.0, atol=1e-9), "target must be on the axis"
        assert np.allclose(se3.camera_centre(T), eye, atol=1e-9)


def test_look_at_survives_the_degenerate_up_vector():
    T = se3.look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert se3.transform(T, [0.0, 0.0, 0.0])[2] > 0


def test_pose_error_is_zero_on_itself(rng):
    T = se3.exp(rng.normal(size=6))
    t, r = se3.pose_error(T, T)
    assert t < 1e-12 and r < 1e-9


def test_pose_error_reads_off_a_known_perturbation():
    T = se3.exp(np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0]))
    dT = se3.from_Rt(so3.exp([0.0, 0.0, np.radians(5.0)]), [0.01, 0.0, 0.0])
    t, r = se3.pose_error(T @ dT, T)
    assert np.isclose(t, 0.01, atol=1e-9)
    assert np.isclose(r, 5.0, atol=1e-9)


def test_interpolate_hits_both_ends(rng):
    A, B = se3.exp(rng.normal(size=6)), se3.exp(rng.normal(size=6))
    assert np.allclose(se3.interpolate(A, B, 0.0), A, atol=1e-12)
    assert np.allclose(se3.interpolate(A, B, 1.0), B, atol=1e-9)


# --------------------------------------------------------------------------
# The camera
# --------------------------------------------------------------------------


def test_projection_matches_opencv(camera, rng):
    """Our model is OpenCV's, so it had better agree with OpenCV's."""
    cv2 = pytest.importorskip("cv2")
    P = np.c_[rng.uniform(-0.4, 0.4, 40), rng.uniform(-0.3, 0.3, 40), rng.uniform(1.0, 4.0, 40)]
    ours = camera.project(P)
    theirs, _ = cv2.projectPoints(
        P, np.zeros(3), np.zeros(3), camera.K, camera.dist.reshape(1, 5)
    )
    assert np.allclose(ours, theirs.reshape(-1, 2), atol=1e-9)


def test_point_jacobian_matches_finite_differences(camera, rng):
    P = np.c_[rng.uniform(-0.4, 0.4, 12), rng.uniform(-0.3, 0.3, 12), rng.uniform(1.0, 4.0, 12)]
    analytic = camera.project_jacobian_point(P)
    for i, X in enumerate(P):
        numeric = central_difference(lambda p: camera.project(p), X, eps=1e-7)
        assert np.allclose(analytic[i], numeric, rtol=1e-5, atol=1e-5), i


def test_parameter_jacobian_matches_finite_differences(camera, rng):
    P = np.c_[rng.uniform(-0.4, 0.4, 8), rng.uniform(-0.3, 0.3, 8), rng.uniform(1.0, 4.0, 8)]
    analytic = camera.project_jacobian_params(P)
    for i, X in enumerate(P):
        numeric = central_difference(
            lambda q: camera.with_params(q).project(X), camera.params, eps=1e-6
        )
        assert np.allclose(analytic[i], numeric, rtol=1e-4, atol=1e-4), i


def test_undistort_inverts_distort(camera, rng):
    uv = np.c_[rng.uniform(0, camera.width, 500), rng.uniform(0, camera.height, 500)]
    xy = camera.undistort(uv)
    P = np.c_[xy, np.ones(len(xy))]
    assert np.allclose(camera.project(P), uv, atol=1e-6)


def test_undistort_agrees_with_opencv_to_opencvs_own_accuracy(camera, rng):
    """We agree with ``cv2.undistortPoints``, and we converge further than it.

    The radtan model has no closed-form inverse, so both implementations
    iterate.  OpenCV stops after a fixed five iterations; ours runs to a
    tolerance.  Measured on this camera, our result round-trips through the
    forward model to about 1e-10 px and OpenCV's to about 1e-2 px, so the
    residual disagreement between the two is *OpenCV's* error, not ours.

    That is why the tolerance here is 1e-4 rather than 1e-9, and why the second
    half of the test is the one that actually pins our implementation down.
    """
    cv2 = pytest.importorskip("cv2")
    uv = np.c_[rng.uniform(20, camera.width - 20, 200), rng.uniform(20, camera.height - 20, 200)]
    ours = camera.undistort(uv)
    theirs = cv2.undistortPoints(
        uv.reshape(-1, 1, 2), camera.K, camera.dist.reshape(1, 5)
    ).reshape(-1, 2)
    assert np.allclose(ours, theirs, atol=1e-4)

    def round_trip(xy):
        return np.linalg.norm(camera.project(np.c_[xy, np.ones(len(xy))]) - uv, axis=1)

    ours_error = round_trip(ours).max()
    opencv_error = round_trip(theirs).max()
    assert ours_error < 1e-8, ours_error
    assert ours_error < opencv_error, (ours_error, opencv_error)


def test_projection_marks_points_behind_the_camera(camera):
    uv, valid = camera.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]), return_valid=True)
    assert not valid[0] and valid[1]
    assert np.isnan(uv[0]).all()


def test_rays_are_unit_and_point_at_their_pixel(camera, rng):
    uv = np.c_[rng.uniform(0, camera.width, 50), rng.uniform(0, camera.height, 50)]
    d = camera.rays(uv)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(camera.project(d * 3.0), uv, atol=1e-6)


def test_scaled_camera_projects_to_scaled_pixels(camera, rng):
    half = camera.scaled(0.5)
    P = np.c_[rng.uniform(-0.3, 0.3, 30), rng.uniform(-0.2, 0.2, 30), rng.uniform(1.5, 3.0, 30)]
    expected = (camera.project(P) + 0.5) * 0.5 - 0.5
    assert np.allclose(half.project(P), expected, atol=1e-9)


def test_camera_dict_round_trip(camera):
    other = PinholeCamera.from_dict(camera.to_dict())
    assert np.allclose(other.params, camera.params)
    assert (other.width, other.height) == (camera.width, camera.height)


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------


def _four_view_setup(X_world):
    poses = [
        se3.look_at([1.5, 0.0, 0.6], X_world),
        se3.look_at([0.0, 1.5, 0.6], X_world),
        se3.look_at([-1.5, 0.2, 0.6], X_world),
        se3.look_at([0.2, -1.5, 0.7], X_world),
    ]
    xy = []
    for T in poses:
        Xc = se3.transform(T, X_world)
        xy.append(Xc[:2] / Xc[2])
    return poses, np.array(xy)


def test_triangulation_is_exact_without_noise(rng):
    for _ in range(20):
        X = rng.uniform(-0.2, 0.2, 3)
        poses, xy = _four_view_setup(X)
        assert np.allclose(triangulation.triangulate_dlt(poses, xy), X, atol=1e-9)
        assert np.allclose(triangulation.triangulate_midpoint(poses, xy), X, atol=1e-9)


def test_refinement_beats_the_linear_solve_under_noise(rng):
    """The DLT minimises an algebraic error; Gauss-Newton minimises the real one."""
    dlt_err, ref_err = [], []
    for _ in range(60):
        X = rng.uniform(-0.2, 0.2, 3)
        poses, xy = _four_view_setup(X)
        noisy = xy + rng.normal(scale=1.5 / 600.0, size=xy.shape)
        X_dlt = triangulation.triangulate_dlt(poses, noisy)
        X_ref = triangulation.refine_point(poses, noisy, X_dlt)
        dlt_err.append(np.linalg.norm(X_dlt - X))
        ref_err.append(np.linalg.norm(X_ref - X))
    assert np.mean(ref_err) <= np.mean(dlt_err)
    # Sanity bound rather than a target.  With 1.5 px of noise at f = 600, a
    # baseline of about 2 m and a depth of about 1.5 m, the depth uncertainty
    # is roughly z^2 sigma / (f b) = 2.25 * 1.5 / (600 * 2) ~= 3 mm, so an
    # error of a few millimetres is the physics and anything much larger is a
    # bug.
    assert np.mean(ref_err) < 6e-3


def test_triangulation_angle_and_cheirality(rng):
    X = np.zeros(3)
    poses, _ = _four_view_setup(X)
    assert triangulation.triangulation_angle(poses, X) > 80.0
    assert triangulation.cheirality(poses, X).all()
    behind = np.array([0.0, 0.0, 10.0])
    assert not triangulation.cheirality(poses, behind).all()


def test_degenerate_triangulation_returns_nan():
    """Two identical views cannot triangulate and must say so, not guess."""
    T = se3.look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    X = triangulation.triangulate_midpoint([T, T], np.zeros((2, 2)))
    assert not np.all(np.isfinite(X))


def test_point_covariance_shrinks_with_more_views(rng):
    X = np.zeros(3)
    poses, _ = _four_view_setup(X)
    c2 = triangulation.point_covariance(poses[:2], X, sigma_px=1.0, focal=600.0)
    c4 = triangulation.point_covariance(poses, X, sigma_px=1.0, focal=600.0)
    assert np.trace(c4) < np.trace(c2)
