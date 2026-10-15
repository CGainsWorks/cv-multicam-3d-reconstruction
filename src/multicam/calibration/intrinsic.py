"""Per-camera intrinsic calibration: Zhang's method, then a proper refinement.

This is written out rather than delegated to ``cv2.calibrateCamera``, and the
reason is not purity.  The rig extrinsic solve in
:mod:`multicam.calibration.extrinsic` refines intrinsics and extrinsics
together against a shared set of board poses, and that solve needs the same
Jacobians, the same parameterisation and the same robust kernel as the
per-camera solve, or the two disagree in ways that are invisible until the
residuals stop making sense.  Having one implementation of the projection
Jacobian, checked against finite differences, and using it everywhere, is worth
more than the hundred lines it costs.

OpenCV is still used, in two places, and both are deliberate:

* ``cv2.findChessboardCorners`` and the ChArUco detector do the *detection*.
  Reimplementing sub-pixel corner refinement would be reimplementing something
  that is already correct.
* ``cv2.calibrateCamera`` is run alongside as an **independent referee** and
  the two answers are compared.  ``scripts/calibrate.py`` reports the
  disagreement, and it is quoted in the README.  A calibration that agrees with
  OpenCV to a hundredth of a pixel is a calibration; one that does not is a
  bug, and this is how the bug gets found.

The stages are the textbook ones.  A homography per view, a closed-form linear
solve for the image of the absolute conic, intrinsics extracted from it,
extrinsics from the homographies, and then a joint non-linear refinement over
everything with distortion switched on.  The linear stage cannot see distortion
at all, which is why its residual is around a pixel and the refined one is
around a tenth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import PinholeCamera, se3, so3
from ..optim import levenberg_marquardt


@dataclass
class IntrinsicResult:
    """A calibrated camera and everything needed to judge it."""

    camera: PinholeCamera
    T_cam_board: list                   #: one pose per accepted view
    view_ids: list                      #: frame index of each accepted view
    rmse_px: float
    max_px: float
    per_view_rmse: np.ndarray
    residuals: np.ndarray = None        #: (M, 2) signed pixel residuals
    linear_rmse_px: float = float("nan")
    covariance: np.ndarray = None
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        c = self.camera
        return {
            "camera": c.name,
            "views": len(self.T_cam_board),
            "corners": 0 if self.residuals is None else int(len(self.residuals)),
            "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
            "dist": [float(v) for v in c.dist],
            "rmse_px": self.rmse_px,
            "max_px": self.max_px,
            "linear_rmse_px": self.linear_rmse_px,
            **self.diagnostics,
        }


# --------------------------------------------------------------------------
# Homography
# --------------------------------------------------------------------------


def _normalisation(points):
    """Hartley normalisation: centroid at the origin, mean distance sqrt(2)."""
    mu = points.mean(axis=0)
    d = np.linalg.norm(points - mu, axis=1).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    Tn = np.array([[s, 0.0, -s * mu[0]], [0.0, s, -s * mu[1]], [0.0, 0.0, 1.0]])
    return Tn


def estimate_homography(src, dst):
    """DLT homography ``dst ~ H src`` for planar point pairs.

    Both sets are normalised first.  Without it the DLT's algebraic cost mixes
    terms of order 1 with terms of order 10^5 and the smallest singular vector
    is dominated by rounding.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if len(src) < 4:
        raise ValueError("a homography needs at least four points")

    Ts, Td = _normalisation(src), _normalisation(dst)
    s = (np.c_[src, np.ones(len(src))] @ Ts.T)[:, :2]
    d = (np.c_[dst, np.ones(len(dst))] @ Td.T)[:, :2]

    n = len(s)
    A = np.zeros((2 * n, 9))
    A[0::2, 0:2] = s
    A[0::2, 2] = 1.0
    A[0::2, 6:8] = -d[:, 0:1] * s
    A[0::2, 8] = -d[:, 0]
    A[1::2, 3:5] = s
    A[1::2, 5] = 1.0
    A[1::2, 6:8] = -d[:, 1:2] * s
    A[1::2, 8] = -d[:, 1]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


# --------------------------------------------------------------------------
# Zhang's linear stage
# --------------------------------------------------------------------------


def _v_ij(H, i, j):
    """Zhang's constraint row built from columns ``i`` and ``j`` of ``H``."""
    hi, hj = H[:, i], H[:, j]
    return np.array(
        [
            hi[0] * hj[0],
            hi[0] * hj[1] + hi[1] * hj[0],
            hi[1] * hj[1],
            hi[2] * hj[0] + hi[0] * hj[2],
            hi[2] * hj[1] + hi[1] * hj[2],
            hi[2] * hj[2],
        ]
    )


def linear_intrinsics(homographies):
    """Closed-form ``K`` from three or more homographies.

    Solves for the image of the absolute conic ``B = K^-T K^-1``, whose two
    constraints per view come from the orthonormality of the first two columns
    of the rotation.  Returns ``(K, skew)``; the skew is not part of the camera
    model and is returned purely as a conditioning diagnostic -- a well posed
    set of views gives a skew of a few hundredths of a pixel, and a large one
    means the board was never tilted enough.
    """
    if len(homographies) < 3:
        raise ValueError("Zhang's linear stage needs at least three views, got %d"
                         % len(homographies))
    V = np.zeros((2 * len(homographies), 6))
    for k, H in enumerate(homographies):
        V[2 * k] = _v_ij(H, 0, 1)
        V[2 * k + 1] = _v_ij(H, 0, 0) - _v_ij(H, 1, 1)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b

    den = B11 * B22 - B12 * B12
    if abs(den) < 1e-18 or abs(B11) < 1e-18:
        raise np.linalg.LinAlgError("degenerate board views: the conic solve is singular")

    cy = (B12 * B13 - B11 * B23) / den
    lam = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11
    if lam / B11 <= 0 or lam * B11 / den <= 0:
        raise np.linalg.LinAlgError("degenerate board views: no positive-definite conic")

    fx = np.sqrt(lam / B11)
    fy = np.sqrt(lam * B11 / den)
    skew = -B12 * fx * fx * fy / lam
    cx = skew * cy / fy - B13 * fx * fx / lam
    K = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, float(skew)


def pose_from_homography(K, H):
    """``T_cam_board`` from a board-to-image homography and a linear ``K``."""
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    # The two scale estimates differ by a per cent or so on real data; their
    # geometric mean is the standard compromise and is what OpenCV does too.
    lam1 = 1.0 / max(np.linalg.norm(K_inv @ h1), 1e-12)
    lam2 = 1.0 / max(np.linalg.norm(K_inv @ h2), 1e-12)
    lam = np.sqrt(lam1 * lam2)

    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    t = lam * (K_inv @ h3)
    if t[2] < 0:
        # The board is behind the camera, which means the homography came back
        # with the wrong overall sign.  Flip everything.
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = so3.project_to_so3(np.stack([r1, r2, r3], axis=1))
    return se3.from_Rt(R, t)


# --------------------------------------------------------------------------
# Joint refinement
# --------------------------------------------------------------------------


def refine_intrinsics(camera0, poses0, object_points, image_points, huber_delta=2.0,
                      fix_distortion=False, max_iterations=80):
    """Joint LM over the nine intrinsics and every board pose.

    ``object_points`` and ``image_points`` are lists, one entry per view, of
    ``(M_v, 3)`` board points and ``(M_v, 2)`` measured pixels.

    The state is ``(camera, [poses])`` and the solver retracts poses on the
    manifold, ``T <- T exp(delta)``.  See :mod:`multicam.optim` for why the
    poses are not simply parameterised by their logarithms.
    """
    n_views = len(poses0)
    counts = [len(p) for p in image_points]
    obj = [np.asarray(p, dtype=float).reshape(-1, 3) for p in object_points]
    img = [np.asarray(p, dtype=float).reshape(-1, 2) for p in image_points]
    total = int(sum(counts))
    # Pre-compute hat(X) per view: it is constant and rebuilding it inside
    # every Jacobian evaluation is most of the cost of the solve.
    hats = [so3.hat(o) for o in obj]

    def residual(state):
        cam, poses = state
        out = np.empty((total, 2))
        at = 0
        for v in range(n_views):
            Xc = se3.transform(poses[v], obj[v])
            out[at:at + counts[v]] = cam.project(Xc) - img[v]
            at += counts[v]
        return out.ravel()

    def jacobian(state):
        cam, poses = state
        J = np.zeros((2 * total, 9 + 6 * n_views))
        at = 0
        for v in range(n_views):
            R = poses[v][:3, :3]
            Xc = se3.transform(poses[v], obj[v])
            m = counts[v]
            rows = slice(2 * at, 2 * (at + m))

            J_uv_X = cam.project_jacobian_point(Xc)              # (m, 2, 3)
            J[rows, :9] = cam.project_jacobian_params(Xc).reshape(2 * m, 9)

            # Right perturbation T <- T exp(xi):
            #   dXc/drho = R,  dXc/dphi = -R hat(X)
            dX = np.empty((m, 3, 6))
            dX[:, :, :3] = R
            dX[:, :, 3:] = -R @ hats[v]
            J[rows, 9 + 6 * v: 15 + 6 * v] = (J_uv_X @ dX).reshape(2 * m, 6)
            at += m

        if fix_distortion:
            J[:, 4:9] = 0.0
        return J

    def retract(state, delta):
        cam, poses = state
        params = cam.params + delta[:9]
        if fix_distortion:
            params[4:] = cam.dist
        return (
            cam.with_params(params),
            [T @ se3.exp(delta[9 + 6 * v: 15 + 6 * v]) for v, T in enumerate(poses)],
        )

    result = levenberg_marquardt(
        residual, jacobian, (camera0, list(poses0)), retract=retract,
        max_iterations=max_iterations, huber_delta=huber_delta, compute_covariance=True,
    )
    camera, poses = result.state
    return camera, poses, result


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------


def calibrate_intrinsics(target, detections, width, height, name="cam",
                         huber_delta=2.0, min_views=4, opencv_check=True):
    """Full per-camera calibration from a list of :class:`BoardDetection`."""
    dets = [d for d in detections if d is not None and len(d) >= 6]
    if len(dets) < min_views:
        raise ValueError(
            "intrinsic calibration needs at least %d usable views, got %d"
            % (min_views, len(dets))
        )

    obj_pts = [target.points_for(d.ids) for d in dets]
    img_pts = [d.corners for d in dets]

    homographies, kept = [], []
    for k, (o, i) in enumerate(zip(obj_pts, img_pts)):
        if len(o) < 6:
            continue
        try:
            H = estimate_homography(o[:, :2], i)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(H)):
            continue
        homographies.append(H)
        kept.append(k)

    if len(homographies) < 3:
        raise ValueError("only %d usable homographies; cannot run the linear stage"
                         % len(homographies))

    K_lin, skew = linear_intrinsics(homographies)
    camera = PinholeCamera(
        fx=K_lin[0, 0], fy=K_lin[1, 1], cx=K_lin[0, 2], cy=K_lin[1, 2],
        width=width, height=height, dist=np.zeros(5), name=name,
    )
    poses = [pose_from_homography(K_lin, H) for H in homographies]
    obj_pts = [obj_pts[k] for k in kept]
    img_pts = [img_pts[k] for k in kept]
    dets = [dets[k] for k in kept]

    linear_res = np.concatenate(
        [camera.project(se3.transform(T, o)) - i for T, o, i in zip(poses, obj_pts, img_pts)]
    )
    linear_rmse = float(np.sqrt(np.mean(linear_res ** 2)))

    camera, poses, result = refine_intrinsics(
        camera, poses, obj_pts, img_pts, huber_delta=huber_delta
    )

    residuals = np.concatenate(
        [camera.project(se3.transform(T, o)) - i for T, o, i in zip(poses, obj_pts, img_pts)]
    )
    per_view = np.array(
        [
            float(np.sqrt(np.mean((camera.project(se3.transform(T, o)) - i) ** 2)))
            for T, o, i in zip(poses, obj_pts, img_pts)
        ]
    )

    diagnostics = {
        "linear_skew_px": skew,
        "lm_iterations": result.iterations,
        "lm_converged": bool(result.converged),
        "lm_reason": result.reason,
    }
    if opencv_check:
        diagnostics.update(_opencv_cross_check(obj_pts, img_pts, width, height, camera))

    return IntrinsicResult(
        camera=camera,
        T_cam_board=poses,
        view_ids=[d.frame for d in dets],
        rmse_px=float(np.sqrt(np.mean(residuals ** 2))),
        max_px=float(np.max(np.linalg.norm(residuals, axis=1))),
        per_view_rmse=per_view,
        residuals=residuals,
        linear_rmse_px=linear_rmse,
        covariance=result.covariance,
        diagnostics=diagnostics,
    )


def _opencv_cross_check(obj_pts, img_pts, width, height, ours):
    """Run ``cv2.calibrateCamera`` on the same data and report the difference.

    This is the referee.  It shares no code with anything above it, so
    agreement is evidence and disagreement is a bug in one of the two.
    """
    try:
        obj = [o.astype(np.float32).reshape(-1, 1, 3) for o in obj_pts]
        img = [i.astype(np.float32).reshape(-1, 1, 2) for i in img_pts]
        rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (width, height), None, None)
    except cv2.error as exc:
        return {"opencv_check": "failed: %s" % (str(exc).splitlines()[0],)}

    dist = np.asarray(dist, dtype=float).ravel()[:5]
    return {
        "opencv_rms_px": float(rms),
        "opencv_fx": float(K[0, 0]),
        "opencv_fy": float(K[1, 1]),
        "opencv_cx": float(K[0, 2]),
        "opencv_cy": float(K[1, 2]),
        "opencv_focal_diff_px": float(max(abs(K[0, 0] - ours.fx), abs(K[1, 1] - ours.fy))),
        "opencv_centre_diff_px": float(
            max(abs(K[0, 2] - ours.cx), abs(K[1, 2] - ours.cy))
        ),
        "opencv_dist_diff_max": float(np.max(np.abs(dist - ours.dist))),
    }
