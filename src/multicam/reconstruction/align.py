"""Rigid alignment of two 3D point sets, robustly.

Used to chain one turntable station to the next.  Each station is a metric
point cloud in the rig frame, so the motion between two stations is a rigid
transform and nothing else -- no scale, because the rig calibration already
fixed it.

This is where the static background gets thrown away, and it needs more than
one RANSAC to do it.  Features on the floor and the walls match beautifully
between stations, triangulate cleanly, and then insist the transform between
stations is the **identity**, while the features on the part insist it is a
30-degree rotation.  Two large self-consistent groups, two different motions.

A single RANSAC returns whichever is larger, and which one that is depends on
how much of the frame the part happens to occupy -- which is a property of the
lens, not of the algorithm.  So :func:`multi_motion_align` fits motions
*sequentially*: find the dominant one, remove its inliers, fit again, repeat.
The turntable group is then picked out by the only criterion that does not
require knowing the answer in advance: **the part is the thing that moves**.
The static group is the one whose transform is the identity, and it is
discarded however large it is.

The group sizes are returned so the choice is visible in the results rather
than assumed in a comment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import se3, so3


@dataclass
class AlignmentResult:
    T_a_b: np.ndarray               #: maps points in frame ``b`` into frame ``a``
    inliers: np.ndarray             #: boolean mask over the correspondences
    rmse: float                     #: inlier RMS distance after alignment, metres
    n_correspondences: int
    iterations: int
    converged: bool

    @property
    def inlier_ratio(self):
        return float(self.inliers.mean()) if len(self.inliers) else 0.0


def kabsch(P, Q, weights=None):
    """The rigid transform taking ``Q`` onto ``P``, in the least-squares sense.

    Returns ``T`` with ``T @ Q ~= P``.  Reflections are rejected by flipping
    the least significant singular direction, which is the smallest correction
    that restores a right-handed frame.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    if len(P) != len(Q) or len(P) < 3:
        raise ValueError("kabsch needs at least three matched points")

    w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-12)
    mu_p = (w[:, None] * P).sum(axis=0)
    mu_q = (w[:, None] * Q).sum(axis=0)

    H = (w[:, None] * (Q - mu_q)).T @ (P - mu_p)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return se3.from_Rt(so3.project_to_so3(R), mu_p - R @ mu_q)


def align_ransac(P, Q, threshold=0.004, max_iterations=2000, min_inliers=8,
                 confidence=0.9995, rng=None, refine=True):
    """Robust rigid alignment of corresponding point pairs ``(P[i], Q[i])``.

    ``threshold`` is a distance in metres.  4 mm is about ten times the
    triangulation noise on this data and about a twentieth of the part, which
    separates "the same physical point" from "a different point that matched".
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    n = len(P)
    rng = np.random.default_rng(0) if rng is None else rng

    if n < 3:
        return AlignmentResult(np.eye(4), np.zeros(n, bool), float("inf"), n, 0, False)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    iterations = int(max_iterations)
    it = 0
    while it < iterations:
        it += 1
        idx = rng.choice(n, size=3, replace=False)
        # Three collinear points leave the rotation about their common axis
        # unconstrained, so reject the sample rather than accept a random
        # answer for that degree of freedom.
        a, b, c = Q[idx]
        cross = np.cross(b - a, c - a)
        if np.linalg.norm(cross) < 1e-9:
            continue
        try:
            T = kabsch(P[idx], Q[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        d = np.linalg.norm(se3.transform(T, Q) - P, axis=1)
        inliers = d < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
            # Adaptive stopping: once a large consensus is found there is no
            # point drawing the full budget of samples.
            w = max(count / n, 1e-6)
            if w < 1.0:
                need = np.log1p(-confidence) / np.log1p(-(w ** 3))
                iterations = int(min(max_iterations, max(need, 30)))
            else:
                break

    if best_count < min_inliers:
        return AlignmentResult(np.eye(4), best_inliers, float("inf"), n, it, False)

    T = kabsch(P[best_inliers], Q[best_inliers])
    if refine:
        # One reweighting pass: re-fit on the consensus, re-select, re-fit.
        for _ in range(3):
            d = np.linalg.norm(se3.transform(T, Q) - P, axis=1)
            inliers = d < threshold
            if inliers.sum() < min_inliers:
                break
            T_new = kabsch(P[inliers], Q[inliers])
            if np.allclose(T_new, T, atol=1e-12):
                T = T_new
                best_inliers = inliers
                break
            T, best_inliers = T_new, inliers

    d = np.linalg.norm(se3.transform(T, Q[best_inliers]) - P[best_inliers], axis=1)
    return AlignmentResult(
        T_a_b=T, inliers=best_inliers, rmse=float(np.sqrt(np.mean(d ** 2))),
        n_correspondences=n, iterations=it, converged=True,
    )


def motion_magnitude(T):
    """``(rotation in degrees, translation in metres)`` of a transform."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return (
        float(np.degrees(np.linalg.norm(so3.log(T[:3, :3])))),
        float(np.linalg.norm(T[:3, 3])),
    )


def multi_motion_align(P, Q, threshold=0.004, max_motions=3, min_inliers=8,
                       static_rotation_deg=1.5, static_translation_m=0.004, rng=None):
    """Fit several rigid motions sequentially and return them, largest first.

    Each entry is ``(AlignmentResult, is_static)`` where a motion counts as
    static if it is within ``static_rotation_deg`` and ``static_translation_m``
    of the identity.  The thresholds are generous relative to the triangulation
    noise and small relative to a turntable step, so the classification is not
    delicate.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    rng = np.random.default_rng(0) if rng is None else rng

    remaining = np.ones(len(P), dtype=bool)
    motions = []
    for _ in range(int(max_motions)):
        if remaining.sum() < min_inliers:
            break
        index = np.nonzero(remaining)[0]
        result = align_ransac(
            P[index], Q[index], threshold=threshold, min_inliers=min_inliers, rng=rng
        )
        if not result.converged:
            break
        # Lift the inlier mask back into the full correspondence indexing so
        # callers can use it against their own arrays.
        full = np.zeros(len(P), dtype=bool)
        full[index[result.inliers]] = True
        result = AlignmentResult(
            T_a_b=result.T_a_b, inliers=full, rmse=result.rmse,
            n_correspondences=len(P), iterations=result.iterations, converged=True,
        )
        rot, trans = motion_magnitude(result.T_a_b)
        is_static = rot < static_rotation_deg and trans < static_translation_m
        motions.append((result, is_static))
        remaining &= ~full
    return motions


def align_stations(station_a, station_b, ratio=0.82, threshold=0.004, rng=None,
                   select_moving=True, max_motions=3):
    """Align station ``b`` onto station ``a`` through descriptor matching.

    Returns ``(AlignmentResult, pairs, info)``.  ``T_a_b`` maps a point in
    ``b``'s rig frame into ``a``'s.

    With ``select_moving`` the largest **non-static** motion is returned, which
    is the turntable.  With it off the largest motion of any kind is returned,
    which is what a single RANSAC would have given and is kept so the ablation
    in ``scripts/ablate.py`` can measure the difference.
    """
    from .features import match_descriptors

    pairs = match_descriptors(station_a.descriptors, station_b.descriptors, ratio=ratio)
    failed = AlignmentResult(np.eye(4), np.zeros(len(pairs), bool), float("inf"),
                             len(pairs), 0, False)
    if len(pairs) < 4:
        return failed, pairs, {"motions": []}

    P = station_a.points[pairs[:, 0]]
    Q = station_b.points[pairs[:, 1]]

    if not select_moving:
        return align_ransac(P, Q, threshold=threshold, rng=rng), pairs, {"motions": []}

    motions = multi_motion_align(P, Q, threshold=threshold, max_motions=max_motions, rng=rng)
    info = {
        "motions": [
            {
                "inliers": int(r.inliers.sum()),
                "rotation_deg": round(motion_magnitude(r.T_a_b)[0], 4),
                "translation_mm": round(motion_magnitude(r.T_a_b)[1] * 1000, 3),
                "static": bool(static),
                "rmse_mm": None if not np.isfinite(r.rmse) else round(r.rmse * 1000, 4),
            }
            for r, static in motions
        ]
    }
    moving = [(r, s) for r, s in motions if not s]
    if not moving:
        info["selected"] = None
        return failed, pairs, info
    best = max(moving, key=lambda rs: int(rs[0].inliers.sum()))[0]
    info["selected"] = int(best.inliers.sum())
    return best, pairs, info
