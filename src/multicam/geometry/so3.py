"""The rotation group, written out rather than imported.

Everything in this repository that touches a rotation goes through here, so the
conventions are stated once:

* A rotation is stored as a 3x3 matrix ``R`` with ``R @ R.T == I`` and
  ``det(R) == +1``.
* ``exp`` takes a rotation vector -- axis times angle, in radians -- to a matrix
  and ``log`` takes it back.  Both use the closed form away from the origin and
  a truncated series near it, so a Jacobian evaluated at zero is finite rather
  than a division by zero.
* Perturbations are **right** perturbations: an update ``d`` acts as
  ``R <- R @ exp(d)``.  Every Jacobian below follows that choice.  Mixing left
  and right conventions is the single most common way to get a solver that
  almost converges, so the choice is made once and never varied.

The series thresholds are the usual ones: below ``EPS`` the fourth-order term
of the Taylor expansion is already below double precision relative to the
leading term, so truncating there costs nothing measurable.
"""

from __future__ import annotations

import numpy as np

#: Below this rotation angle (radians) the closed forms are replaced by their
#: Taylor expansions.  1e-8 squared is 1e-16, which is machine epsilon.
EPS = 1e-8


def hat(w):
    """The skew-symmetric matrix with ``hat(w) @ v == cross(w, v)``.

    Accepts a single ``(3,)`` vector or a batch ``(..., 3)`` and returns
    ``(3, 3)`` or ``(..., 3, 3)`` to match.
    """
    w = np.asarray(w, dtype=float)
    if w.shape[-1] != 3:
        raise ValueError("hat expects vectors of length 3, got shape %r" % (w.shape,))
    out = np.zeros(w.shape[:-1] + (3, 3), dtype=float)
    out[..., 0, 1] = -w[..., 2]
    out[..., 0, 2] = w[..., 1]
    out[..., 1, 0] = w[..., 2]
    out[..., 1, 2] = -w[..., 0]
    out[..., 2, 0] = -w[..., 1]
    out[..., 2, 1] = w[..., 0]
    return out


def vee(W):
    """Inverse of :func:`hat`.  The antisymmetric part is taken, not assumed."""
    W = np.asarray(W, dtype=float)
    if W.shape[-2:] != (3, 3):
        raise ValueError("vee expects 3x3 matrices, got shape %r" % (W.shape,))
    return np.stack(
        [
            0.5 * (W[..., 2, 1] - W[..., 1, 2]),
            0.5 * (W[..., 0, 2] - W[..., 2, 0]),
            0.5 * (W[..., 1, 0] - W[..., 0, 1]),
        ],
        axis=-1,
    )


def exp(w):
    """Rotation vector to rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = hat(w)
    if theta < EPS:
        # sin(t)/t -> 1 - t^2/6, (1-cos t)/t^2 -> 1/2 - t^2/24.  Keeping the
        # quadratic terms makes the truncation error O(t^4) rather than O(t^2),
        # which matters when this is called inside a finite-difference check.
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def log(R):
    """Rotation matrix to rotation vector, with the angle in ``[0, pi]``.

    The near-pi branch matters: there ``sin(theta)`` vanishes and the usual
    ``vee`` formula loses all its precision, so the axis is recovered from the
    symmetric part instead.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # Clipping guards against a trace a few ulp outside [-1, 3] on a matrix
    # that is orthonormal only to rounding.
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < EPS:
        # exp(w) ~= I + hat(w), so the antisymmetric part is w itself.
        return vee(R)
    if theta > np.pi - 1e-5:
        # R = I + sin(t) K + (1-cos t) K^2.  Near pi the symmetric part
        # (R + R.T)/2 = I + (1-cos t) K^2 carries the axis without cancellation.
        A = (R + R.T) * 0.5 - np.eye(3) * cos_theta
        # A = (1 - cos t) * (a a^T - I) + ... ; diag gives a_i^2 up to scale.
        diag = np.clip(np.diag(A) / (1.0 - cos_theta) + 1.0, 0.0, None)
        axis = np.sqrt(diag)
        # Signs from the antisymmetric part where it is still meaningful, else
        # from the off-diagonals of A.
        k = int(np.argmax(axis))
        if axis[k] > 0:
            row = A[k] / (1.0 - cos_theta)
            axis = row / axis[k]
        n = np.linalg.norm(axis)
        if n < EPS:
            return np.zeros(3)
        axis = axis / n
        # exp(theta*a) and exp(-theta*a) both fit the symmetric part; the
        # antisymmetric residual, tiny as it is, picks the branch.
        if float(vee(R) @ axis) < 0:
            axis = -axis
        return axis * theta
    return vee(R) * (theta / np.sin(theta))


def right_jacobian(w):
    """``Jr(w)`` with ``exp(w + dw) ~= exp(w) @ exp(Jr(w) @ dw)``."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = hat(w)
    if theta < EPS:
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    a = (1.0 - np.cos(theta)) / t2
    b = (theta - np.sin(theta)) / (t2 * theta)
    return np.eye(3) - a * K + b * (K @ K)


def right_jacobian_inv(w):
    """The inverse of :func:`right_jacobian`, in closed form."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = hat(w)
    if theta < EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5 * theta
    # (1/theta^2 - (1 + cos theta) / (2 theta sin theta)), written through
    # cot(theta/2) so there is one trig call rather than three.  The removable
    # singularity at theta = 0 is handled by the branch above.
    c = (1.0 - half * np.cos(half) / np.sin(half)) / (theta * theta)
    return np.eye(3) + 0.5 * K + c * (K @ K)


def project_to_so3(M):
    """Nearest rotation matrix in the Frobenius sense, by SVD.

    Used after any operation that leaves a matrix orthonormal only to rounding,
    for instance averaging or a long chain of multiplications.
    """
    M = np.asarray(M, dtype=float).reshape(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # A reflection sneaked in; flip the least significant singular
        # direction, which is the smallest possible correction.
        U[:, -1] *= -1
        R = U @ Vt
    return R


def angle_between(R_a, R_b):
    """Geodesic angle in radians between two rotations."""
    return float(np.linalg.norm(log(np.asarray(R_a).T @ np.asarray(R_b))))


def random_rotation(rng, max_angle=np.pi):
    """A rotation drawn with an axis uniform on the sphere and a bounded angle."""
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), EPS)
    angle = rng.uniform(-max_angle, max_angle)
    return exp(axis * angle)
