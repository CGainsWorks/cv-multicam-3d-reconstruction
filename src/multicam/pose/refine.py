"""Applying the trained refiner at inference time.

Iterative: render at the current hypothesis, predict a correction, apply it,
render again.  Two or three iterations is the usual sweet spot -- the first
does almost all the work, the second cleans up, and by the fourth the
prediction is inside the noise and iterating further just moves the answer
around.  ``scripts/ablate.py`` measures that curve rather than asserting it.

The refiner is a *refiner*.  It is never asked to find the part; it is handed a
pose from the classical stage and asked to improve it.  If the classical stage
fails, this returns the failure unchanged rather than inventing an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3
from .render_compare import build_input


def apply_correction(T_rig_object, twist, centroid):
    """Compose a predicted twist, in the rig-aligned frame at the object centre.

    The NumPy counterpart of
    :func:`multicam.pose.network.apply_correction_torch`, and it must stay
    identical to it: training composes one way and inference the other, and a
    mismatch would be a refiner that works in training and not in use.  A test
    checks the two against each other.
    """
    T = np.asarray(T_rig_object, dtype=float).reshape(4, 4)
    centre = T[:3, :3] @ np.asarray(centroid, dtype=float).reshape(3) + T[:3, 3]
    delta = se3.exp(np.asarray(twist, dtype=float).reshape(6))
    shifted = np.eye(4)
    shifted[:3, :3] = delta[:3, :3]
    shifted[:3, 3] = delta[:3, 3] + centre - delta[:3, :3] @ centre
    return shifted @ T


@dataclass
class RefinementTrace:
    """What each iteration did, for the ablation's convergence curves."""

    poses: list = field(default_factory=list)
    twists: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def steps(self):
        return len(self.twists)

    def step_magnitudes(self):
        """``(translation mm, rotation deg)`` of each applied correction."""
        out = []
        for twist in self.twists:
            T = se3.exp(twist)
            t, r = se3.pose_error(T, np.eye(4))
            out.append((t * 1000.0, r))
        return out


def refine(network, renderer, images, rig, T_rig_object, iterations=2, device="cpu",
           trace=False):
    """Run the learned refiner for ``iterations`` steps.

    Returns the refined pose, or ``(pose, RefinementTrace)`` if ``trace``.
    """
    import time

    import torch

    T = np.array(T_rig_object, dtype=float).reshape(4, 4)
    record = RefinementTrace(poses=[T.copy()])
    t0 = time.time()

    network.eval()
    with torch.no_grad():
        for _ in range(int(iterations)):
            crops, specs = build_input(renderer, images, rig, T)
            if not any(s.valid for s in specs):
                break
            valid = np.array([[s.valid for s in specs]], dtype=np.float32)
            batch = torch.from_numpy(crops[None]).to(device)
            twist = network(batch, torch.from_numpy(valid).to(device))
            twist = twist.cpu().numpy().reshape(6)
            T = apply_correction(T, twist, renderer.centre)
            record.twists.append(twist)
            record.poses.append(T.copy())

    record.seconds = time.time() - t0
    return (T, record) if trace else T


def refine_batch(network, renderer, images_by_frame, rig, poses, iterations=2,
                 device="cpu"):
    """Refine a whole set of frames.  ``poses`` maps frame -> pose or None."""
    out = {}
    for frame, T in poses.items():
        if T is None:
            out[frame] = None
            continue
        out[frame] = refine(
            network, renderer, images_by_frame[frame], rig, T,
            iterations=iterations, device=device,
        )
    return out
