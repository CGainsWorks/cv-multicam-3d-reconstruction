"""Classical 6-DoF pose: generalised PnP over the whole rig, then refinement.

The rig is calibrated and rigid, so the four cameras do not each estimate a
pose and then average.  There is **one** unknown -- the object's pose in the rig
frame -- and all four cameras' correspondences constrain it together.  That is a
generalised camera, and it matters for two reasons:

* The cameras span 76 degrees, so between them they see three sides of the
  part.  A single camera sees one side, and its pose estimate is
  correspondingly weak along the viewing direction.  Solving jointly turns
  four badly conditioned problems into one well conditioned one.
* A camera that sees only a handful of points still contributes them.  Averaging
  per-camera poses would either discard that camera or weight its noisy answer
  equally with a good one.

RANSAC draws its minimal sample from a *single* camera, because P3P needs three
correspondences in one central projection.  The hypothesis it produces is then
scored against every correspondence in every camera, which is what makes it a
generalised-camera RANSAC rather than four independent ones.

The refinement is Levenberg-Marquardt on the Huber-weighted reprojection error
over all inliers in all cameras, with the analytic Jacobian.  Covariance comes
from the Hessian at the solution and is first order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import se3, so3
from ..optim import levenberg_marquardt


@dataclass
class Correspondences:
    """2D-3D matches for one camera."""

    camera: int
    xy: np.ndarray                      #: (M, 2) measured pixels
    points: np.ndarray                  #: (M, 3) model points, object frame
    distances: np.ndarray = None        #: (M,) descriptor distance, for diagnostics

    def __len__(self):
        return len(self.xy)


@dataclass
class PoseResult:
    T_rig_object: np.ndarray = None
    inliers: dict = field(default_factory=dict)     #: camera -> boolean mask
    n_correspondences: int = 0
    n_inliers: int = 0
    rmse_px: float = float("nan")
    covariance: np.ndarray = None                   #: 6x6 in the tangent space
    iterations: int = 0
    converged: bool = False
    reason: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.T_rig_object is not None

    def T_cam_object(self, rig, camera):
        return rig.T_cam_rig[camera] @ self.T_rig_object

    def position_sigma_mm(self):
        """Standard deviation of the translation, from the covariance, in mm."""
        if self.covariance is None:
            return None
        return [float(np.sqrt(max(self.covariance[i, i], 0.0))) * 1000.0 for i in range(3)]

    def rotation_sigma_deg(self):
        if self.covariance is None:
            return None
        return [float(np.degrees(np.sqrt(max(self.covariance[i, i], 0.0))))
                for i in range(3, 6)]

    def summary(self):
        return {
            "solved": bool(self.ok),
            "correspondences": int(self.n_correspondences),
            "inliers": int(self.n_inliers),
            "inlier_ratio": (self.n_inliers / self.n_correspondences)
            if self.n_correspondences else 0.0,
            "rmse_px": self.rmse_px,
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "position_sigma_mm": self.position_sigma_mm(),
            "rotation_sigma_deg": self.rotation_sigma_deg(),
            **self.stats,
        }


def match_to_model(features, model, ratio=0.80, max_distance=None):
    """Descriptor matches from one image to the object model."""
    from ..reconstruction.features import match_descriptors

    pairs = match_descriptors(
        features.descriptors, model.descriptors, ratio=ratio, mutual=False,
        max_distance=max_distance,
    )
    if not len(pairs):
        return Correspondences(features.camera, np.zeros((0, 2)), np.zeros((0, 3)))
    return Correspondences(
        camera=features.camera,
        xy=features.xy[pairs[:, 0]],
        points=model.points[pairs[:, 1]],
    )


def _reproject(rig, T_rig_object, correspondences):
    """Reprojection error norms, one array per camera."""
    out = {}
    for corr in correspondences:
        if not len(corr):
            out[corr.camera] = np.zeros(0)
            continue
        T = rig.T_cam_rig[corr.camera] @ T_rig_object
        uv, valid = rig.cameras[corr.camera].project(
            se3.transform(T, corr.points), return_valid=True
        )
        d = np.full(len(corr), np.inf)
        if valid.any():
            d[valid] = np.linalg.norm(uv[valid] - corr.xy[valid], axis=1)
        out[corr.camera] = d
    return out


def refine_pose(rig, T_rig_object, correspondences, inliers=None, huber_px=2.0,
                max_iterations=60):
    """LM refinement of the rig-frame object pose over all cameras.

    Returns ``(T, OptimiseResult)``.  The perturbation is right-handed,
    ``T <- T exp(delta)``, matching everything else in the repository.
    """
    blocks = []
    for corr in correspondences:
        mask = None if inliers is None else inliers.get(corr.camera)
        xy = corr.xy if mask is None else corr.xy[mask]
        P = corr.points if mask is None else corr.points[mask]
        if len(xy):
            blocks.append((corr.camera, P, xy, so3.hat(P)))
    total = int(sum(len(b[2]) for b in blocks))
    if total < 4:
        return T_rig_object, None

    def residual(T):
        out = np.empty((total, 2))
        at = 0
        for camera, P, xy, _ in blocks:
            Tc = rig.T_cam_rig[camera] @ T
            m = len(xy)
            out[at:at + m] = rig.cameras[camera].project(se3.transform(Tc, P)) - xy
            at += m
        return np.nan_to_num(out.ravel(), nan=1e3, posinf=1e3, neginf=-1e3)

    def jacobian(T):
        J = np.zeros((2 * total, 6))
        at = 0
        for camera, P, xy, hatP in blocks:
            A = rig.T_cam_rig[camera]
            Tc = A @ T
            m = len(xy)
            Xc = se3.transform(Tc, P)
            J_uv_X = rig.cameras[camera].project_jacobian_point(Xc)
            # T <- T exp(xi):  dXc/drho = R_A R_T,  dXc/dphi = -R_A R_T hat(P)
            RR = A[:3, :3] @ T[:3, :3]
            dX = np.empty((m, 3, 6))
            dX[:, :, :3] = RR
            dX[:, :, 3:] = -RR @ hatP
            J[2 * at: 2 * (at + m)] = np.nan_to_num((J_uv_X @ dX).reshape(2 * m, 6))
            at += m
        return J

    result = levenberg_marquardt(
        residual, jacobian, np.array(T_rig_object, dtype=float),
        retract=lambda T, d: T @ se3.exp(d),
        max_iterations=max_iterations, huber_delta=huber_px, compute_covariance=True,
    )
    return result.state, result


def estimate_pose(rig, correspondences, threshold_px=3.0, max_iterations=600,
                  min_inliers=12, confidence=0.9999, huber_px=2.0, rng=None,
                  refine_passes=3, initial=None):
    """Generalised-camera PnP under RANSAC, then robust refinement.

    ``initial`` skips the RANSAC and refines from a supplied pose, which is how
    the ablation feeds the classical refiner a known-perturbed starting point.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    correspondences = [c for c in correspondences if len(c)]
    total = int(sum(len(c) for c in correspondences))
    if total < 6:
        return PoseResult(n_correspondences=total, reason="too few correspondences")

    best_T, best_count, best_inliers = None, 0, None
    tried = 0

    if initial is not None:
        best_T = np.array(initial, dtype=float).reshape(4, 4)
        errs = _reproject(rig, best_T, correspondences)
        best_inliers = {c.camera: errs[c.camera] < threshold_px for c in correspondences}
        best_count = int(sum(m.sum() for m in best_inliers.values()))
    else:
        # Cameras are drawn in proportion to how many correspondences they have,
        # so a camera with three matches is not sampled as often as one with
        # three hundred and the budget goes where the evidence is.
        weights = np.array([len(c) for c in correspondences], dtype=float)
        weights = weights / weights.sum()
        budget = int(max_iterations)
        while tried < budget:
            tried += 1
            ci = int(rng.choice(len(correspondences), p=weights))
            corr = correspondences[ci]
            if len(corr) < 4:
                continue
            idx = rng.choice(len(corr), size=4, replace=False)
            obj = corr.points[idx].astype(np.float64)
            img = corr.xy[idx].astype(np.float64)
            camera = rig.cameras[corr.camera]
            try:
                ok, rvecs, tvecs = cv2.solveP3P(
                    obj[:3], img[:3], camera.K, camera.dist.reshape(1, 5),
                    flags=cv2.SOLVEPNP_P3P,
                )
            except cv2.error:
                continue
            if not ok:
                continue
            for rvec, tvec in zip(rvecs, tvecs):
                R, _ = cv2.Rodrigues(rvec)
                T_cam_object = se3.from_Rt(R, np.asarray(tvec).ravel())
                T = se3.inverse(rig.T_cam_rig[corr.camera]) @ T_cam_object
                errs = _reproject(rig, T, correspondences)
                inliers = {c.camera: errs[c.camera] < threshold_px for c in correspondences}
                count = int(sum(m.sum() for m in inliers.values()))
                if count > best_count:
                    best_count, best_T, best_inliers = count, T, inliers
                    w = max(count / total, 1e-6)
                    if w < 1.0:
                        need = np.log1p(-confidence) / np.log1p(-(w ** 3))
                        budget = int(min(max_iterations, max(need, 40)))

    if best_T is None or best_count < min_inliers:
        return PoseResult(
            n_correspondences=total, n_inliers=best_count,
            reason="RANSAC found only %d inliers" % best_count,
            stats={"ransac_iterations": tried},
        )

    T = best_T
    opt = None
    for _ in range(int(refine_passes)):
        T_new, opt = refine_pose(rig, T, correspondences, best_inliers, huber_px=huber_px)
        if opt is None:
            break
        T = T_new
        errs = _reproject(rig, T, correspondences)
        new_inliers = {c.camera: errs[c.camera] < threshold_px for c in correspondences}
        new_count = int(sum(m.sum() for m in new_inliers.values()))
        if new_count == best_count:
            best_inliers = new_inliers
            break
        best_inliers, best_count = new_inliers, new_count

    errs = _reproject(rig, T, correspondences)
    inlier_err = np.concatenate(
        [errs[c.camera][best_inliers[c.camera]] for c in correspondences]
    ) if best_count else np.zeros(0)

    return PoseResult(
        T_rig_object=T,
        inliers=best_inliers,
        n_correspondences=total,
        n_inliers=best_count,
        rmse_px=float(np.sqrt(np.mean(inlier_err ** 2))) if len(inlier_err) else float("nan"),
        covariance=None if opt is None else opt.covariance,
        iterations=0 if opt is None else opt.iterations,
        converged=bool(opt is not None and opt.converged),
        reason="" if opt is not None else "refinement skipped",
        stats={
            "ransac_iterations": tried,
            "cameras_used": int(sum(1 for m in best_inliers.values() if m.any())),
            "inliers_per_camera": {str(c.camera): int(best_inliers[c.camera].sum())
                                   for c in correspondences},
        },
    )
