"""Bundle adjustment: sparse Levenberg-Marquardt with a Schur complement.

The problem is the usual one.  Minimise reprojection error over

* the 3D position of every point, three parameters each, of which there are
  thousands, and
* the pose of the object at every station, six parameters each, of which there
  are a dozen.

The normal equations for that are a matrix with tens of thousands of rows, and
forming it densely is out of the question -- but it has structure.  No point
sees another point, so the point-point block ``H_pp`` is block diagonal with
3x3 blocks, and inverting it is inverting a few thousand 3x3 matrices.  The
Schur complement

    S = H_ss - H_sp H_pp^-1 H_ps

is then a small dense matrix -- six parameters per station, so 72x72 for a
twelve-station sweep -- which is solved directly, and the point updates are
recovered by back-substitution.  That is the whole algorithm, and it turns a
problem that would need a sparse solver into one that needs ``numpy.linalg``.

Gauge
-----
The reference station's pose is held fixed.  Without it the whole
reconstruction can rotate and translate freely, ``S`` is singular to six
decimal places, and the damping quietly picks an arbitrary answer.  Scale is
*not* free: the rig calibration fixed it, and that is the whole reason for
calibrating a rig rather than running structure from motion.

Robustness
----------
A Huber kernel on the reprojection residuals, applied as IRLS in the
square-root form so the normal equations stay the correctly weighted ones.  The
outliers it is there for are cross-station mismatches that survived RANSAC, and
there are always a few.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3, so3
from ..optim import huber_cost, huber_weights


@dataclass
class BundleResult:
    points: np.ndarray
    T_rig_object: list                  #: one pose per station
    initial_rmse_px: float
    rmse_px: float
    max_px: float
    iterations: int
    converged: bool
    reason: str
    n_observations: int
    inlier_mask: np.ndarray = None
    history: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def summary(self):
        return {
            "points": int(len(self.points)),
            "observations": int(self.n_observations),
            "stations": len(self.T_rig_object),
            "initial_rmse_px": self.initial_rmse_px,
            "rmse_px": self.rmse_px,
            "max_px": self.max_px,
            "iterations": self.iterations,
            "converged": bool(self.converged),
            "reason": self.reason,
            "inliers": None if self.inlier_mask is None else int(self.inlier_mask.sum()),
            **{("seconds_" + k): v for k, v in self.timings.items()},
        }


def bundle_adjust(points, T_rig_object, observations, rig, reference=0, huber_delta=2.0,
                  max_iterations=40, ftol=1e-8, lam0=1e-4, verbose=False):
    """Refine points and station poses against every observation.

    Parameters
    ----------
    points : (N, 3)
        Initial 3D points, in the reference station's rig frame -- which is the
        object frame, by definition of the gauge.
    T_rig_object : list of (4, 4)
        Initial object pose per station.  ``T_rig_object[reference]`` is held
        fixed.
    observations : sequence of (point, station, camera, xy)
        ``xy`` is the measured pixel, distorted, as it came out of the detector.
    rig : RigCalibration
    """
    import time

    points = np.array(points, dtype=float).reshape(-1, 3)
    poses = [np.array(T, dtype=float).reshape(4, 4) for T in T_rig_object]
    n_points = len(points)
    n_stations = len(poses)

    obs_point = np.array([o[0] for o in observations], dtype=np.int64)
    obs_station = np.array([o[1] for o in observations], dtype=np.int64)
    obs_camera = np.array([o[2] for o in observations], dtype=np.int64)
    obs_xy = np.array([o[3] for o in observations], dtype=float).reshape(-1, 2)
    n_obs = len(obs_point)
    if not n_obs:
        raise ValueError("bundle adjustment needs observations")

    free = [s for s in range(n_stations) if s != reference]
    slot = {s: k for k, s in enumerate(free)}
    n_free = len(free)

    # Observations grouped by (station, camera): every group shares one
    # projection matrix and one camera model, so the Jacobian for a group is
    # one vectorised call rather than one call per observation.
    groups = {}
    for i in range(n_obs):
        groups.setdefault((int(obs_station[i]), int(obs_camera[i])), []).append(i)
    groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}

    # A point projects into a given image exactly once, so the point indices
    # within a group are unique.  That is what lets the accumulation below use
    # fancy-indexed ``+=`` instead of ``np.add.at``, which is an order of
    # magnitude faster.  It is checked rather than assumed: a duplicate would
    # silently drop terms from the normal equations.
    for key, idx in groups.items():
        p = obs_point[idx]
        if len(np.unique(p)) != len(p):
            raise ValueError(
                "station %d camera %d observes some point more than once; "
                "the track builder should have rejected that group" % key
            )

    def residuals(points, poses):
        out = np.empty((n_obs, 2))
        for (s, c), idx in groups.items():
            T = rig.T_cam_rig[c] @ poses[s]
            Xc = se3.transform(T, points[obs_point[idx]])
            out[idx] = rig.cameras[c].project(Xc) - obs_xy[idx]
        return out

    r0 = residuals(points, poses)
    initial_rmse = float(np.sqrt(np.mean(r0 ** 2)))

    t_start = time.time()
    timings = {"jacobian": 0.0, "schur": 0.0, "solve": 0.0}

    lam = float(lam0)
    r = r0
    cost = huber_cost(r.ravel(), huber_delta)
    history = [cost]
    converged, reason = False, "max_iterations"

    for iteration in range(int(max_iterations)):
        t0 = time.time()
        w = np.sqrt(huber_weights(r.ravel(), huber_delta)).reshape(n_obs, 2)

        # Blocks.  H_ss is dense over the free stations, H_pp is block diagonal
        # over the points, H_sp couples them.
        H_ss = np.zeros((6 * n_free, 6 * n_free))
        H_pp = np.zeros((n_points, 3, 3))
        H_sp = np.zeros((6 * n_free, n_points, 3))
        g_s = np.zeros(6 * n_free)
        g_p = np.zeros((n_points, 3))

        for (s, c), idx in groups.items():
            camera = rig.cameras[c]
            A = rig.T_cam_rig[c]
            B = poses[s]
            X = points[obs_point[idx]]
            m = len(idx)

            Xc = se3.transform(A @ B, X)
            J_uv_X = camera.project_jacobian_point(Xc)             # (m, 2, 3)

            # Point block: dXc/dX = R_A R_B
            RARB = A[:3, :3] @ B[:3, :3]
            J_point = J_uv_X @ RARB                                # (m, 2, 3)

            wi = w[idx]                                            # (m, 2)
            ri = r[idx]                                            # (m, 2)
            Jp_w = J_point * wi[:, :, None]
            rp_w = ri * wi

            cols = obs_point[idx]
            H_pp[cols] += np.einsum("mki,mkj->mij", Jp_w, Jp_w)
            g_p[cols] += np.einsum("mki,mk->mi", Jp_w, rp_w)

            if s == reference:
                continue

            # Station block: B <- B exp(xi) gives
            #   dXc/drho = R_A R_B,  dXc/dphi = -R_A R_B hat(X)
            dX = np.empty((m, 3, 6))
            dX[:, :, :3] = RARB
            dX[:, :, 3:] = -RARB @ so3.hat(X)
            J_pose = J_uv_X @ dX                                   # (m, 2, 6)
            Js_w = J_pose * wi[:, :, None]

            k = slot[s]
            rows = np.arange(6 * k, 6 * k + 6)
            H_ss[6 * k: 6 * k + 6, 6 * k: 6 * k + 6] += np.einsum("mki,mkj->ij", Js_w, Js_w)
            g_s[6 * k: 6 * k + 6] += np.einsum("mki,mk->i", Js_w, rp_w)
            H_sp[rows[:, None], cols[None, :]] += np.einsum("mki,mkj->imj", Js_w, Jp_w)
        timings["jacobian"] += time.time() - t0

        if np.max(np.abs(np.concatenate([g_s, g_p.ravel()]))) < 1e-14:
            converged, reason = True, "gradient"
            break

        stepped = False
        for _ in range(30):
            t0 = time.time()
            # Damping, Marquardt scaled: proportional to each block's own
            # diagonal, so a translation in metres and a rotation in radians
            # are damped in their own units rather than sharing one constant.
            diag3 = np.arange(3)
            H_pp_d = H_pp.copy()
            H_pp_d[:, diag3, diag3] += lam * np.clip(H_pp[:, diag3, diag3], 1e-12, None)
            H_ss_d = H_ss + lam * np.diag(np.clip(np.diag(H_ss), 1e-12, None))

            try:
                H_pp_inv = np.linalg.inv(H_pp_d)
            except np.linalg.LinAlgError:
                lam *= 10.0
                timings["schur"] += time.time() - t0
                continue

            # S = H_ss - H_sp H_pp^-1 H_ps  and  b = -g_s + H_sp H_pp^-1 g_p
            Y = np.einsum("anj,njk->ank", H_sp, H_pp_inv)          # (6F, N, 3)
            S = H_ss_d - np.einsum("anj,bnj->ab", Y, H_sp)
            b = -g_s + np.einsum("anj,nj->a", Y, g_p)
            timings["schur"] += time.time() - t0

            t0 = time.time()
            try:
                delta_s = np.linalg.solve(S, b) if n_free else np.zeros(0)
            except np.linalg.LinAlgError:
                lam *= 10.0
                timings["solve"] += time.time() - t0
                continue

            rhs_p = -g_p - np.einsum("anj,a->nj", H_sp, delta_s)
            delta_p = np.einsum("nij,nj->ni", H_pp_inv, rhs_p)
            timings["solve"] += time.time() - t0

            new_points = points + delta_p
            new_poses = list(poses)
            for s in free:
                new_poses[s] = poses[s] @ se3.exp(delta_s[6 * slot[s]: 6 * slot[s] + 6])

            r_new = residuals(new_points, new_poses)
            cost_new = huber_cost(r_new.ravel(), huber_delta)
            if np.isfinite(cost_new) and cost_new < cost:
                rel = (cost - cost_new) / max(cost, 1e-30)
                points, poses, r, cost = new_points, new_poses, r_new, cost_new
                lam = max(lam * 0.33, 1e-12)
                history.append(cost)
                stepped = True
                if verbose:
                    print("  ba it %2d  cost %.6e  rmse %.4f px  lam %.1e"
                          % (iteration, cost, float(np.sqrt(np.mean(r ** 2))), lam))
                if rel < ftol:
                    converged, reason = True, "ftol"
                break
            lam *= 10.0
            if lam > 1e14:
                break
        if converged:
            break
        if not stepped:
            converged, reason = False, "damping"
            break

    norms = np.linalg.norm(r, axis=1)
    return BundleResult(
        points=points,
        T_rig_object=poses,
        initial_rmse_px=initial_rmse,
        rmse_px=float(np.sqrt(np.mean(r ** 2))),
        max_px=float(norms.max()),
        iterations=len(history) - 1,
        converged=converged,
        reason=reason,
        n_observations=n_obs,
        inlier_mask=norms < (huber_delta if huber_delta else np.inf),
        history=history,
        timings={k: round(v, 3) for k, v in timings.items()} | {
            "total": round(time.time() - t_start, 3)
        },
    )
