"""Rig extrinsic calibration.

Given per-camera intrinsics and a set of board stations that several cameras
saw at the same instant, recover where the cameras are relative to one another.

Three stages, and the middle one is the one that stops this being a toy:

1. **Pairwise.**  For every frame in which cameras ``i`` and ``j`` both see the
   board, ``T_ci_cj = T_ci_board @ inv(T_cj_board)``.  Each such frame gives an
   independent estimate; they are averaged on the manifold, not by averaging
   matrix entries.
2. **A spanning tree over the camera graph.**  On a 104-degree arc the two end
   cameras may never see the board together, so there is no direct
   ``T_c0_c3`` measurement at all.  Chaining through the middle cameras is the
   only way to initialise it.  The graph is built with the *number of
   co-observations* as the edge weight and a maximum-weight spanning tree is
   taken, so the chain runs through the best-observed links rather than
   whichever happens to be found first.
3. **A joint bundle adjustment** over every extrinsic and every board pose at
   once, with a Huber kernel.  This is where the residuals that get reported
   come from.

The reference camera's pose is held at the identity throughout.  That is not a
convenience: without it the problem has a six-parameter gauge freedom, the
normal equations are singular, and the covariance is meaningless.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3, so3
from ..optim import levenberg_marquardt
from ..types import RigCalibration


@dataclass
class ExtrinsicResult:
    rig: RigCalibration
    T_rig_board: dict                   #: frame index -> (4, 4)
    rmse_px: float
    max_px: float
    per_camera_rmse: dict
    n_observations: int
    n_frames: int
    pairwise_counts: dict = field(default_factory=dict)
    pairwise_spread: dict = field(default_factory=dict)
    initial_rmse_px: float = float("nan")
    covariance: np.ndarray = None
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        return {
            "cameras": self.rig.n_cameras,
            "frames": self.n_frames,
            "observations": self.n_observations,
            "rmse_px": self.rmse_px,
            "max_px": self.max_px,
            "initial_rmse_px": self.initial_rmse_px,
            "per_camera_rmse_px": {str(k): float(v) for k, v in self.per_camera_rmse.items()},
            "baselines_mm": [b * 1000.0 for b in self.rig.baselines()],
            "pairwise_counts": {"%d-%d" % k: int(v) for k, v in self.pairwise_counts.items()},
            "pairwise_spread_mm": {"%d-%d" % k: float(v) * 1000.0
                                   for k, v in self.pairwise_spread.items()},
            **self.diagnostics,
        }


# --------------------------------------------------------------------------
# Averaging on the manifold
# --------------------------------------------------------------------------


def average_poses(poses):
    """Chordal mean of a set of rigid transforms.

    Rotations are averaged by summing the matrices and projecting the sum back
    onto SO(3) with an SVD, which is the closed-form minimiser of the sum of
    squared Frobenius distances.  Averaging Euler angles or quaternion
    components without care gives an answer that is not even a rotation, and
    for the small spreads seen here the chordal mean and the geodesic mean
    agree to well under the noise.
    """
    poses = list(poses)
    if not poses:
        raise ValueError("nothing to average")
    R = so3.project_to_so3(sum(np.asarray(T)[:3, :3] for T in poses))
    t = np.mean([np.asarray(T)[:3, 3] for T in poses], axis=0)
    return se3.from_Rt(R, t)


def pose_spread(poses, mean=None):
    """Root-mean-square translation deviation from the mean, in metres."""
    poses = list(poses)
    if len(poses) < 2:
        return 0.0
    mean = average_poses(poses) if mean is None else mean
    d = [np.linalg.norm(np.asarray(T)[:3, 3] - mean[:3, 3]) for T in poses]
    return float(np.sqrt(np.mean(np.square(d))))


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------


def pairwise_relative_poses(board_poses):
    """``{(i, j): [T_ci_cj, ...]}`` from ``{(frame, camera): T_cam_board}``.

    Only ``i < j`` keys are produced; the reverse is the inverse.
    """
    by_frame = defaultdict(dict)
    for (frame, cam), T in board_poses.items():
        by_frame[frame][cam] = T

    out = defaultdict(list)
    for frame, cams in by_frame.items():
        keys = sorted(cams)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                i, j = keys[a], keys[b]
                out[(i, j)].append(cams[i] @ se3.inverse(cams[j]))
    return dict(out)


def spanning_tree_extrinsics(n_cameras, pairwise, reference=0, min_shared=3):
    """Initial ``T_cam_rig`` for every camera by walking a spanning tree.

    Edges with fewer than ``min_shared`` co-observations are dropped before the
    walk: a relative pose from one or two shared views is dominated by the
    noise in a single board pose, and chaining through it poisons everything
    downstream of it.
    """
    edges = {}
    for (i, j), poses in pairwise.items():
        if len(poses) < min_shared:
            continue
        edges[(i, j)] = (len(poses), average_poses(poses))

    T_cam_rig = {reference: np.eye(4)}
    # Greedy maximum-weight expansion: repeatedly attach the unvisited camera
    # reachable over the best-observed edge.
    while len(T_cam_rig) < n_cameras:
        best = None
        for (i, j), (count, T_ci_cj) in edges.items():
            if i in T_cam_rig and j not in T_cam_rig:
                cand = (count, j, T_cam_rig[i], se3.inverse(T_ci_cj))
            elif j in T_cam_rig and i not in T_cam_rig:
                cand = (count, i, T_cam_rig[j], T_ci_cj)
            else:
                continue
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            missing = sorted(set(range(n_cameras)) - set(T_cam_rig))
            raise ValueError(
                "cameras %s are not connected to the reference by any edge with at "
                "least %d shared board views; capture more stations where they "
                "overlap" % (missing, min_shared)
            )
        # T_cam_rig[new] = T_new_known @ T_known_rig
        _, new, T_known_rig, T_new_known = best
        T_cam_rig[new] = T_new_known @ T_known_rig
    return [T_cam_rig[i] for i in range(n_cameras)]


def initial_board_poses(board_poses, T_cam_rig):
    """``T_rig_board`` per frame, averaged over whichever cameras saw it."""
    by_frame = defaultdict(list)
    for (frame, cam), T_cam_board in board_poses.items():
        by_frame[frame].append(se3.inverse(T_cam_rig[cam]) @ T_cam_board)
    return {f: average_poses(v) for f, v in by_frame.items()}


# --------------------------------------------------------------------------
# Joint refinement
# --------------------------------------------------------------------------


def calibrate_extrinsics(cameras, target, detections, board_poses=None, reference=0,
                         huber_delta=2.0, min_shared=3, refine_intrinsics=False,
                         max_iterations=80, metadata=None):
    """Estimate the rig extrinsics and report the residuals.

    Parameters
    ----------
    cameras : list of PinholeCamera
        Intrinsics, already calibrated per camera.
    target : CharucoTarget
    detections : dict
        ``{(frame, camera): BoardDetection}``.
    board_poses : dict, optional
        ``{(frame, camera): T_cam_board}``.  Computed by PnP if omitted.
    refine_intrinsics : bool
        Also free the nine intrinsic parameters of every camera.  Off by
        default: the per-camera calibration sees the board across the whole
        image and this one does not, so freeing them here usually trades a
        slightly lower residual for a worse camera.  ``scripts/calibrate.py``
        runs it both ways and reports the difference rather than asserting it.
    """
    n_cameras = len(cameras)
    if board_poses is None:
        board_poses = {}
        for key, det in detections.items():
            frame, cam = key
            T = target.estimate_pose(det, cameras[cam])
            if T is not None:
                board_poses[key] = T

    pairwise = pairwise_relative_poses(board_poses)
    counts = {k: len(v) for k, v in pairwise.items()}
    spread = {k: pose_spread(v) for k, v in pairwise.items()}

    T_cam_rig = spanning_tree_extrinsics(n_cameras, pairwise, reference, min_shared)
    T_rig_board = initial_board_poses(board_poses, T_cam_rig)
    frames = sorted(T_rig_board)
    frame_index = {f: k for k, f in enumerate(frames)}

    # Flatten the observations once.  Everything below indexes into these.
    obs_cam, obs_frame, obs_obj, obs_px = [], [], [], []
    for (frame, cam), det in sorted(detections.items()):
        if frame not in frame_index or det is None or len(det) < 4:
            continue
        obs_cam.append(cam)
        obs_frame.append(frame_index[frame])
        obs_obj.append(target.points_for(det.ids))
        obs_px.append(np.asarray(det.corners, dtype=float).reshape(-1, 2))
    if not obs_cam:
        raise ValueError("no usable board observations")

    counts_per_obs = [len(p) for p in obs_px]
    total = int(sum(counts_per_obs))
    n_free_cams = n_cameras - 1
    n_frames = len(frames)
    n_intr = 9 * n_cameras if refine_intrinsics else 0

    free = [i for i in range(n_cameras) if i != reference]
    free_slot = {i: k for k, i in enumerate(free)}
    n_params = 6 * n_free_cams + 6 * n_frames + n_intr
    hats = [so3.hat(o) for o in obs_obj]

    # State: (extrinsics, board poses, cameras).  Poses retract on the
    # manifold; the reference extrinsic is never touched, which is what fixes
    # the six-parameter gauge freedom.
    state0 = (list(T_cam_rig), [T_rig_board[f] for f in frames], list(cameras))

    def retract(state, delta):
        ext, boards, cams = state
        ext = [
            T @ se3.exp(delta[6 * free_slot[i]: 6 * free_slot[i] + 6]) if i in free_slot else T
            for i, T in enumerate(ext)
        ]
        base = 6 * n_free_cams
        boards = [
            T @ se3.exp(delta[base + 6 * k: base + 6 * k + 6]) for k, T in enumerate(boards)
        ]
        if refine_intrinsics:
            off = base + 6 * n_frames
            cams = [c.with_params(c.params + delta[off + 9 * i: off + 9 * i + 9])
                    for i, c in enumerate(cams)]
        return ext, boards, cams

    def residual(state):
        ext, boards, cams = state
        out = np.empty((total, 2))
        at = 0
        for k, cam in enumerate(obs_cam):
            T = ext[cam] @ boards[obs_frame[k]]
            Xc = se3.transform(T, obs_obj[k])
            m = counts_per_obs[k]
            out[at:at + m] = cams[cam].project(Xc) - obs_px[k]
            at += m
        return out.ravel()

    def jacobian(state):
        ext, boards, cams = state
        J = np.zeros((2 * total, n_params))
        base = 6 * n_free_cams
        at = 0
        for k, cam in enumerate(obs_cam):
            A = ext[cam]
            B = boards[obs_frame[k]]
            X = obs_obj[k]
            m = counts_per_obs[k]
            rows = slice(2 * at, 2 * (at + m))

            P = se3.transform(B, X)                       # board points in rig frame
            Xc = se3.transform(A, P)
            J_uv_X = cams[cam].project_jacobian_point(Xc)  # (m, 2, 3)

            if cam in free_slot:
                # A <- A exp(xi):  dXc/drho = R_A,  dXc/dphi = -R_A hat(P)
                dX = np.empty((m, 3, 6))
                dX[:, :, :3] = A[:3, :3]
                dX[:, :, 3:] = -A[:3, :3] @ so3.hat(P)
                c = 6 * free_slot[cam]
                J[rows, c: c + 6] = (J_uv_X @ dX).reshape(2 * m, 6)

            # B <- B exp(zeta):  dXc/drho = R_A R_B,  dXc/dphi = -R_A R_B hat(X)
            RARB = A[:3, :3] @ B[:3, :3]
            dX = np.empty((m, 3, 6))
            dX[:, :, :3] = RARB
            dX[:, :, 3:] = -RARB @ hats[k]
            c = base + 6 * obs_frame[k]
            J[rows, c: c + 6] = (J_uv_X @ dX).reshape(2 * m, 6)

            if refine_intrinsics:
                c = base + 6 * n_frames + 9 * cam
                J[rows, c: c + 9] = cams[cam].project_jacobian_params(Xc).reshape(2 * m, 9)
            at += m
        return J

    initial_rmse = float(np.sqrt(np.mean(residual(state0) ** 2)))
    result = levenberg_marquardt(
        residual, jacobian, state0, retract=retract, max_iterations=max_iterations,
        huber_delta=huber_delta, compute_covariance=True,
    )
    ext, boards, cams = result.state

    r = residual(result.state).reshape(-1, 2)
    per_camera = {}
    at = 0
    for k, cam in enumerate(obs_cam):
        m = counts_per_obs[k]
        per_camera.setdefault(cam, []).append(r[at:at + m])
        at += m
    per_camera_rmse = {
        c: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for c, v in per_camera.items()
    }

    # Re-express so the reference camera is exactly the identity even after the
    # solve has nudged everything; it is held fixed, so this is a no-op that
    # documents the invariant rather than a correction.
    ext = [T @ se3.inverse(ext[reference]) for T in ext]
    ext[reference] = np.eye(4)

    meta = dict(metadata or {})
    meta.update({"source": "multicam.calibration.extrinsic.calibrate_extrinsics",
                 "reference_camera": int(reference),
                 "refined_intrinsics": bool(refine_intrinsics)})
    rig = RigCalibration(cameras=cams, T_cam_rig=ext, metadata=meta)

    return ExtrinsicResult(
        rig=rig,
        T_rig_board={frames[k]: boards[k] for k in range(n_frames)},
        rmse_px=float(np.sqrt(np.mean(r ** 2))),
        max_px=float(np.max(np.linalg.norm(r, axis=1))),
        per_camera_rmse=per_camera_rmse,
        n_observations=total,
        n_frames=n_frames,
        pairwise_counts=counts,
        pairwise_spread=spread,
        initial_rmse_px=initial_rmse,
        covariance=result.covariance,
        diagnostics={
            "lm_iterations": result.iterations,
            "lm_converged": bool(result.converged),
            "lm_reason": result.reason,
        },
    )


def compare_rigs(estimated, truth):
    """Per-camera extrinsic error against a known rig.

    Both rigs are anchored at their own reference camera, so this measures what
    it should: the *relative* geometry, not an overall pose that no calibration
    can observe.
    """
    if estimated.n_cameras != truth.n_cameras:
        raise ValueError("rigs have different camera counts")
    rows = []
    for i in range(estimated.n_cameras):
        t_err, r_err = se3.pose_error(estimated.T_cam_rig[i], truth.T_cam_rig[i])
        ce = estimated.cameras[i]
        ct = truth.cameras[i]
        rows.append(
            {
                "camera": i,
                "translation_error_mm": t_err * 1000.0,
                "rotation_error_deg": r_err,
                "baseline_mm": float(np.linalg.norm(se3.camera_centre(truth.T_cam_rig[i]))) * 1e3,
                "fx_error_px": ce.fx - ct.fx,
                "fy_error_px": ce.fy - ct.fy,
                "cx_error_px": ce.cx - ct.cx,
                "cy_error_px": ce.cy - ct.cy,
                "dist_error_max": float(np.max(np.abs(ce.dist - ct.dist))),
            }
        )
    return rows
