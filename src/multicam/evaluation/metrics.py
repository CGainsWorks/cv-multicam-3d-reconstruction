"""Pose and reprojection metrics.

The pose metrics are the ones the 6-DoF literature uses, and they are used here
for the reason they exist: **translation and rotation error separately are not
enough**.  A one-degree rotation error about an axis through the centre of a
40 mm part moves its surface by 0.35 mm; the same error on a 400 mm part moves
it by 3.5 mm.  ADD collapses both into the quantity that actually matters --
how far the part's own points end up from where they should be.

* :func:`add_metric` -- mean distance between corresponding model points under
  the estimated and true poses.  The right metric for an asymmetric part.
* :func:`adds_metric` -- the same but to the *nearest* model point rather than
  the corresponding one.  The right metric for a symmetric part, and the wrong
  one for an asymmetric part because it forgives a 180-degree error on
  something that is nearly symmetric.  Both are reported, and the gap between
  them is a direct measure of how much symmetry the part has.
* Recall at a fraction of the part diameter is the standard summary; 10 per
  cent is what BOP and the older LINEMOD work quote, and the tighter
  thresholds are here because 10 per cent of a 187 mm part is 18.7 mm, which
  is a very forgiving bar for a machine-vision cell.
"""

from __future__ import annotations

import numpy as np

from ..geometry import se3


def add_metric(T_est, T_true, model_points):
    """Average Distance of corresponding model points, in metres."""
    P = np.asarray(model_points, dtype=float).reshape(-1, 3)
    a = se3.transform(np.asarray(T_est, dtype=float).reshape(4, 4), P)
    b = se3.transform(np.asarray(T_true, dtype=float).reshape(4, 4), P)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def adds_metric(T_est, T_true, model_points):
    """Average Distance to the nearest model point (the symmetric variant)."""
    from scipy.spatial import cKDTree

    P = np.asarray(model_points, dtype=float).reshape(-1, 3)
    a = se3.transform(np.asarray(T_est, dtype=float).reshape(4, 4), P)
    b = se3.transform(np.asarray(T_true, dtype=float).reshape(4, 4), P)
    d, _ = cKDTree(b).query(a, k=1)
    return float(np.mean(d))


def recall_at(values, threshold):
    """Fraction of ``values`` at or below ``threshold``.  NaNs count as failures."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return 0.0
    return float(np.mean(np.nan_to_num(v, nan=np.inf) <= threshold))


def reprojection_error(camera, T_cam_object, object_points, measured_px):
    """RMS reprojection error in pixels for a pose hypothesis."""
    P = se3.transform(np.asarray(T_cam_object, dtype=float).reshape(4, 4), object_points)
    uv = camera.project(P)
    d = uv - np.asarray(measured_px, dtype=float).reshape(-1, 2)
    finite = np.isfinite(d).all(axis=1)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(d[finite] ** 2)))


def pose_error_table(estimates, truths, model_points, diameter, thresholds=(0.02, 0.05, 0.10),
                     failures=0):
    """Summarise a set of pose estimates against their ground truth.

    ``failures`` is the number of frames on which the estimator returned
    nothing at all.  They are counted in the recall denominator, because an
    estimator that refuses the hard cases and is perfect on the rest is not a
    better estimator, and reporting recall only over the successes is the
    standard way to make one look like it is.
    """
    adds, addss, trans, rot = [], [], [], []
    for T_est, T_true in zip(estimates, truths):
        if T_est is None:
            continue
        adds.append(add_metric(T_est, T_true, model_points))
        addss.append(adds_metric(T_est, T_true, model_points))
        t, r = se3.pose_error(T_est, T_true)
        trans.append(t)
        rot.append(r)

    n_total = len(adds) + int(failures)
    padded = adds + [np.inf] * int(failures)
    padded_s = addss + [np.inf] * int(failures)

    def stats(v, scale=1.0):
        v = np.asarray(v, dtype=float)
        if not len(v):
            return {"median": float("nan"), "mean": float("nan"), "p90": float("nan")}
        return {
            "median": float(np.median(v)) * scale,
            "mean": float(np.mean(v)) * scale,
            "p90": float(np.quantile(v, 0.9)) * scale,
        }

    out = {
        "n": int(n_total),
        "n_solved": int(len(adds)),
        "failures": int(failures),
        "add_mm": stats(adds, 1000.0),
        "adds_mm": stats(addss, 1000.0),
        "translation_mm": stats(trans, 1000.0),
        "rotation_deg": stats(rot),
        "diameter_mm": float(diameter) * 1000.0,
    }
    for t in thresholds:
        out["add_recall_%d_pct_diameter" % int(round(t * 100))] = recall_at(
            padded, t * float(diameter)
        )
        out["adds_recall_%d_pct_diameter" % int(round(t * 100))] = recall_at(
            padded_s, t * float(diameter)
        )
    for mm in (1.0, 2.0, 5.0):
        out["add_recall_%gmm" % mm] = recall_at(padded, mm * 1e-3)
    return out
