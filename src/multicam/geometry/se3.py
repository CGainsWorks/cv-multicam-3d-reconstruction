"""Rigid transforms.

A pose is a 4x4 homogeneous matrix.  The twist ordering is **translation
first**::

    xi = [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]

which is the ordering Ceres and COLMAP's own bundle adjuster do *not* use --
they put rotation first -- so the conversion happens once, in
``multicam.io.colmap``, and nowhere else.

Perturbations are right perturbations, matching :mod:`multicam.geometry.so3`::

    T <- T @ exp(delta)

Naming: ``T_a_b`` maps a point expressed in frame ``b`` into frame ``a``, so
``T_cam_world @ X_world == X_cam`` and ``T_a_c == T_a_b @ T_b_c`` reads left to
right with the inner frames cancelling.  Every pose in this repository is named
that way.
"""

from __future__ import annotations

import numpy as np

from . import so3


def identity():
    return np.eye(4)


def from_Rt(R, t):
    """Assemble a 4x4 pose from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def to_Rt(T):
    T = np.asarray(T, dtype=float)
    return T[..., :3, :3], T[..., :3, 3]


def inverse(T):
    """``inv(T)``, built from the transpose rather than a general solve."""
    R, t = to_Rt(np.asarray(T, dtype=float).reshape(4, 4))
    return from_Rt(R.T, -R.T @ t)


def compose(*poses):
    out = np.eye(4)
    for T in poses:
        out = out @ np.asarray(T, dtype=float).reshape(4, 4)
    return out


def transform(T, points):
    """Apply a pose to ``(3,)`` or ``(N, 3)`` points."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    P = np.asarray(points, dtype=float)
    single = P.ndim == 1
    P = np.atleast_2d(P)
    out = P @ T[:3, :3].T + T[:3, 3]
    return out[0] if single else out


def exp(xi):
    """Twist to pose.  ``xi`` is ``[rho(3), phi(3)]``."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    R = so3.exp(phi)
    V = _left_jacobian_so3(phi)
    return from_Rt(R, V @ rho)


def log(T):
    """Pose to twist.  Inverse of :func:`exp`."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R, t = to_Rt(T)
    phi = so3.log(R)
    V_inv = _left_jacobian_so3_inv(phi)
    return np.concatenate([V_inv @ t, phi])


def adjoint(T):
    """``Ad(T)`` with ``T @ exp(xi) == exp(Ad(T) @ xi) @ T``."""
    R, t = to_Rt(np.asarray(T, dtype=float).reshape(4, 4))
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = so3.hat(t) @ R
    A[3:, 3:] = R
    return A


def _left_jacobian_so3(phi):
    """``V`` in ``exp([rho, phi]) = [[R, V rho], [0, 1]]``."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = so3.hat(phi)
    if theta < so3.EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    a = (1.0 - np.cos(theta)) / t2
    b = (theta - np.sin(theta)) / (t2 * theta)
    return np.eye(3) + a * K + b * (K @ K)


def _left_jacobian_so3_inv(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = so3.hat(phi)
    if theta < so3.EPS:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5 * theta
    # (1 - (theta/2) cot(theta/2)) / theta^2, with cot written as cos/sin.  The
    # removable singularity at theta = 0 is handled by the branch above.
    c = (1.0 - half * np.cos(half) / np.sin(half)) / (theta * theta)
    return np.eye(3) - 0.5 * K + c * (K @ K)


def relative(T_a, T_b):
    """``inv(T_a) @ T_b``: the pose of ``b`` seen from ``a``."""
    return inverse(T_a) @ np.asarray(T_b, dtype=float).reshape(4, 4)


def pose_error(T_est, T_true):
    """Translation error in metres and rotation error in degrees.

    Both are measured on the *difference* pose ``inv(T_true) @ T_est``, so the
    rotation error is a geodesic angle and not a sum of Euler differences.
    """
    T_est = np.asarray(T_est, dtype=float).reshape(4, 4)
    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    d = inverse(T_true) @ T_est
    trans = float(np.linalg.norm(d[:3, 3]))
    rot = float(np.degrees(np.linalg.norm(so3.log(d[:3, :3]))))
    return trans, rot


def interpolate(T_a, T_b, alpha):
    """Constant-twist interpolation on the manifold, ``alpha`` in ``[0, 1]``."""
    return np.asarray(T_a).reshape(4, 4) @ exp(alpha * log(relative(T_a, T_b)))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """A camera-from-world pose looking from ``eye`` towards ``target``.

    Camera convention is the computer-vision one: ``+x`` right, ``+y`` down,
    ``+z`` forward along the optical axis.
    """
    eye = np.asarray(eye, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)

    forward = target - eye
    n = np.linalg.norm(forward)
    if n < so3.EPS:
        raise ValueError("look_at: eye and target coincide")
    forward /= n

    if abs(float(forward @ (up / np.linalg.norm(up)))) > 1.0 - 1e-6:
        # Degenerate: looking straight along the up axis.  Any right vector is
        # as good as any other, so pick one deterministically.
        up = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    # Rows are the camera axes expressed in world coordinates, which is exactly
    # R_cam_world.
    R_cam_world = np.stack([right, down, forward], axis=0)
    R_cam_world = so3.project_to_so3(R_cam_world)
    return from_Rt(R_cam_world, -R_cam_world @ eye)


def camera_centre(T_cam_world):
    """The camera position in world coordinates."""
    R, t = to_Rt(np.asarray(T_cam_world, dtype=float).reshape(4, 4))
    return -R.T @ t
