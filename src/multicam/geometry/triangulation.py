"""Multi-view triangulation.

Three routines that do the same job with different failure modes:

* :func:`triangulate_dlt` -- the linear direct-linear-transform solve.  Fast,
  closed form, and it minimises an algebraic error that is not the reprojection
  error, so it is biased when the rays are short or the noise is large.
* :func:`triangulate_midpoint` -- the least-squares closest point to a bundle of
  rays.  Better conditioned than the DLT when the baseline is small.
* :func:`refine_point` -- Gauss-Newton on the actual reprojection error, started
  from either of the above.  This is what the reconstruction uses.

Everything here works in *normalised, undistorted* coordinates, so distortion
is handled once at the front door by :meth:`PinholeCamera.undistort` and never
thought about again.

A triangulation is only as good as its parallax, so :func:`triangulation_angle`
is exported and the reconstruction gates on it.  A point seen by four cameras
one degree apart is not a measurement, however small its residual looks.
"""

from __future__ import annotations

import numpy as np

from . import se3


def triangulate_dlt(poses, xy):
    """Linear triangulation from ``N >= 2`` views.

    Parameters
    ----------
    poses : sequence of (4, 4)
        Camera-from-world poses.
    xy : (N, 2)
        Normalised undistorted observations, one per pose.

    Returns
    -------
    (3,) world point.
    """
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if poses.shape[0] != xy.shape[0]:
        raise ValueError("triangulate_dlt: %d poses but %d observations"
                         % (poses.shape[0], xy.shape[0]))
    if poses.shape[0] < 2:
        raise ValueError("triangulate_dlt needs at least two views")

    A = np.empty((2 * poses.shape[0], 4), dtype=float)
    for i, (T, (x, y)) in enumerate(zip(poses, xy)):
        P = T[:3, :]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    # Row normalisation.  Without it a view whose projection matrix happens to
    # carry a large translation dominates the algebraic cost for no good
    # reason; this is the cheap half of Hartley's normalisation and it matters.
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        # Point at infinity: no finite triangulation exists.
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_midpoint(poses, xy):
    """The point minimising the sum of squared distances to the viewing rays."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for T, (x, y) in zip(poses, xy):
        R, t = T[:3, :3], T[:3, 3]
        centre = -R.T @ t
        d = R.T @ np.array([x, y, 1.0])
        d /= np.linalg.norm(d)
        # (I - d d^T) projects onto the plane perpendicular to the ray; summing
        # those projectors is the normal equation of the closest-point problem.
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ centre
    if np.linalg.cond(A) > 1e12:
        return np.full(3, np.nan)
    return np.linalg.solve(A, b)


def refine_point(poses, xy, X0, iterations=10, tol=1e-12):
    """Gauss-Newton refinement of a world point on the reprojection error.

    Minimises ``sum_i || pi(T_i X) - xy_i ||^2`` in normalised coordinates.
    Levenberg damping is applied so that a point behind a camera, which makes
    the Jacobian explode, backs off instead of diverging.
    """
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    X = np.asarray(X0, dtype=float).reshape(3).copy()
    if not np.all(np.isfinite(X)):
        return X

    lam = 1e-6
    cost = _point_cost(poses, xy, X)
    for _ in range(iterations):
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for T, obs in zip(poses, xy):
            Xc = T[:3, :3] @ X + T[:3, 3]
            if Xc[2] < 1e-6:
                continue
            inv_z = 1.0 / Xc[2]
            proj = np.array([Xc[0] * inv_z, Xc[1] * inv_z])
            # d(proj)/d(Xc) then chain through d(Xc)/dX = R.
            J_norm = np.array(
                [[inv_z, 0.0, -proj[0] * inv_z], [0.0, inv_z, -proj[1] * inv_z]]
            )
            J = J_norm @ T[:3, :3]
            r = proj - obs
            H += J.T @ J
            g += J.T @ r
        if not np.any(H):
            break
        try:
            step = np.linalg.solve(H + lam * np.eye(3) * max(np.trace(H) / 3.0, 1e-12), -g)
        except np.linalg.LinAlgError:
            break
        X_new = X + step
        cost_new = _point_cost(poses, xy, X_new)
        if cost_new < cost:
            X, cost = X_new, cost_new
            lam = max(lam * 0.3, 1e-10)
            if float(np.linalg.norm(step)) < tol:
                break
        else:
            lam *= 10.0
            if lam > 1e8:
                break
    return X


def _point_cost(poses, xy, X):
    total = 0.0
    for T, obs in zip(poses, xy):
        Xc = T[:3, :3] @ X + T[:3, 3]
        if Xc[2] < 1e-6:
            return np.inf
        total += float(np.sum((Xc[:2] / Xc[2] - obs) ** 2))
    return total


def point_covariance(poses, X, sigma_px=1.0, focal=1.0):
    """First-order covariance of a triangulated point, in metres squared.

    ``sigma_px`` is the standard deviation of the *pixel* observation and
    ``focal`` converts it to normalised units.  The result is the inverse of
    the Gauss-Newton information matrix, which is the Cramer-Rao bound for this
    estimator under Gaussian noise; it is optimistic whenever the residuals are
    not actually Gaussian, which is stated wherever it is quoted.
    """
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    X = np.asarray(X, dtype=float).reshape(3)
    sigma = sigma_px / float(focal)
    H = np.zeros((3, 3))
    for T in poses:
        Xc = T[:3, :3] @ X + T[:3, 3]
        if Xc[2] < 1e-6:
            continue
        inv_z = 1.0 / Xc[2]
        p = Xc[:2] * inv_z
        J_norm = np.array([[inv_z, 0.0, -p[0] * inv_z], [0.0, inv_z, -p[1] * inv_z]])
        J = J_norm @ T[:3, :3]
        H += J.T @ J
    H /= sigma * sigma
    if np.linalg.cond(H) > 1e14:
        return np.full((3, 3), np.inf)
    return np.linalg.inv(H)


def triangulation_angle(poses, X):
    """The widest angle, in degrees, subtended at ``X`` by any pair of cameras.

    This is the honest measure of whether a point is triangulated or merely
    guessed.  Below about one degree the depth is essentially unconstrained
    however many views there are.
    """
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    X = np.asarray(X, dtype=float).reshape(3)
    dirs = []
    for T in poses:
        d = se3.camera_centre(T) - X
        n = np.linalg.norm(d)
        if n > 1e-9:
            dirs.append(d / n)
    if len(dirs) < 2:
        return 0.0
    D = np.stack(dirs)
    cos = np.clip(D @ D.T, -1.0, 1.0)
    # The diagonal is 1 by construction; take the smallest off-diagonal cosine.
    np.fill_diagonal(cos, 1.0)
    return float(np.degrees(np.arccos(cos.min())))


def cheirality(poses, X, min_depth=1e-3):
    """Boolean mask: is ``X`` in front of each camera by at least ``min_depth``?"""
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    X = np.asarray(X, dtype=float).reshape(3)
    depths = np.array([float(T[2, :3] @ X + T[2, 3]) for T in poses])
    return depths > min_depth
