"""A Levenberg-Marquardt solver on a manifold, with a robust kernel.

Small and dense.  It is used by the calibration, where the parameter vector is
a couple of hundred long and a dense normal-equation solve costs nothing, and
by the pose refinement, where it is six parameters.  The sparse
Schur-complement solver the reconstruction needs is a different animal and
lives in :mod:`multicam.reconstruction.bundle`.

Two design choices are load bearing.

**The state is opaque and the step is a tangent vector.**  Callers hand in a
``retract(state, delta) -> state`` and a Jacobian taken with respect to a
*local* perturbation at the current state.  For a pose that means
``T <- T exp(delta)``, and the Jacobian is the derivative at ``delta = 0``.

The tempting alternative -- parameterise each pose by the six numbers of its
own logarithm and let the solver do ``x + delta`` -- is subtly wrong, because
``exp(x + dx)`` is not ``exp(x) exp(dx)``: they differ by the right Jacobian
``Jr(x)``, which is close to the identity only for small ``x``.  Board poses
half a metre out with half a radian of rotation are not small, and the effect
is a gradient pointing somewhere other than downhill.  The symptom is a solver
that takes one step, then damps itself to a standstill a long way from the
optimum, reports "converged: damping", and leaves a four-pixel residual that
looks almost plausible.

**The robust cost is evaluated explicitly.**  IRLS weights change with the
residuals, so comparing the weighted sum of squares before and after a step
compares two different functions.  :func:`huber_cost` is the actual objective
and is what the accept-or-reject test uses; the weights only ever build the
normal equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimiseResult:
    state: object                       #: the optimised state, in the caller's own type
    cost: float                         #: the robust cost at the solution
    rmse: float                         #: root mean square of the *unweighted* residuals
    iterations: int
    converged: bool
    reason: str
    history: list = field(default_factory=list)
    covariance: np.ndarray = None


def huber_cost(residuals, delta):
    """The Huber loss, summed.  This is the objective the solver descends."""
    r = np.asarray(residuals, dtype=float)
    if delta is None or not np.isfinite(delta) or delta <= 0:
        return 0.5 * float(np.sum(r * r))
    a = np.abs(r)
    quadratic = np.minimum(a, delta)
    linear = a - quadratic
    return float(np.sum(0.5 * quadratic * quadratic + delta * linear))


def huber_weights(residuals, delta):
    """IRLS weights for the Huber kernel, one per residual.

    ``delta`` is in the same units as the residuals -- pixels, everywhere it is
    used here.  Below it the weight is 1 and the kernel is exactly least
    squares; above it the weight falls as ``delta / |r|`` so the influence of a
    gross outlier grows linearly rather than quadratically.
    """
    if delta is None or not np.isfinite(delta) or delta <= 0:
        return np.ones_like(residuals)
    a = np.abs(residuals)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def levenberg_marquardt(residual_fn, jacobian_fn, state0, retract=None, max_iterations=80,
                        ftol=1e-10, xtol=1e-12, gtol=1e-12, huber_delta=None, lam0=1e-4,
                        verbose=False, compute_covariance=False, sigma=None):
    """Minimise the robust cost of ``residual_fn(state)``.

    Parameters
    ----------
    residual_fn : callable
        ``state -> (m,)`` residual vector.
    jacobian_fn : callable
        ``state -> (m, n)`` derivative with respect to a local perturbation.
    state0 : object
        Initial state.  Passed straight through to the callbacks.
    retract : callable, optional
        ``(state, delta) -> state``.  Defaults to ``state + delta``, which is
        correct when the state really is a plain vector.
    compute_covariance : bool
        Return ``sigma^2 (J^T W J)^-1`` at the solution, in the tangent space
        at the solution.  ``sigma`` defaults to the weighted residual RMS.

    Marquardt's scaling -- damping proportional to the diagonal of ``J^T J``
    rather than to the identity -- is used because the parameter blocks here
    have wildly different units.  A focal length in pixels and a distortion
    coefficient that is dimensionless and of order 0.1 cannot share an additive
    damping term sensibly.
    """
    if retract is None:
        def retract(state, delta):
            return np.asarray(state, dtype=float) + delta

    state = state0
    r = np.asarray(residual_fn(state), dtype=float).ravel()
    cost = huber_cost(r, huber_delta)

    lam = float(lam0)
    history = [cost]
    reason = "max_iterations"
    converged = False

    for iteration in range(int(max_iterations)):
        J = np.asarray(jacobian_fn(state), dtype=float)
        if J.shape[0] != r.size:
            raise ValueError("jacobian has %d rows, residual has %d" % (J.shape[0], r.size))

        w = np.sqrt(huber_weights(r, huber_delta))
        Jw = J * w[:, None]
        H = Jw.T @ Jw
        g = Jw.T @ (r * w)

        if np.max(np.abs(g)) < gtol:
            reason, converged = "gradient", True
            break

        diag = np.clip(np.diag(H).copy(), 1e-12, None)
        stepped = False
        for _ in range(40):
            try:
                delta = np.linalg.solve(H + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                if lam > 1e14:
                    break
                continue

            candidate = retract(state, delta)
            r_new = np.asarray(residual_fn(candidate), dtype=float).ravel()
            cost_new = huber_cost(r_new, huber_delta)

            if np.isfinite(cost_new) and cost_new < cost:
                rel = (cost - cost_new) / max(cost, 1e-30)
                state, r, cost = candidate, r_new, cost_new
                lam = max(lam * 0.33, 1e-12)
                history.append(cost)
                stepped = True
                if verbose:
                    print("  it %2d  cost %.6e  lam %.2e" % (iteration, cost, lam))
                if rel < ftol:
                    reason, converged = "ftol", True
                elif float(np.linalg.norm(delta)) < xtol:
                    reason, converged = "xtol", True
                break
            lam *= 10.0
            if lam > 1e14:
                break
        if converged:
            break
        if not stepped:
            reason, converged = "damping", False
            break

    rmse = float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0
    covariance = None
    if compute_covariance:
        J = np.asarray(jacobian_fn(state), dtype=float)
        w = np.sqrt(huber_weights(r, huber_delta))
        Jw = J * w[:, None]
        H = Jw.T @ Jw
        dof = max(r.size - J.shape[1], 1)
        # First order and optimistic: it assumes the model is right and the
        # residuals Gaussian, and neither is exactly true.  Labelled as such
        # wherever it is quoted.
        s2 = (float(np.sum((w * r) ** 2)) / dof) if sigma is None else float(sigma) ** 2
        try:
            covariance = s2 * np.linalg.pinv(H)
        except np.linalg.LinAlgError:
            covariance = None

    return OptimiseResult(
        state=state, cost=cost, rmse=rmse, iterations=len(history) - 1,
        converged=converged, reason=reason, history=history, covariance=covariance,
    )
