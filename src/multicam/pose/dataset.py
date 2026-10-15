"""Training data for the pose refiner.

Each sample is: take a capture whose true pose is known, perturb that pose,
render the CAD model at the perturbed pose, and ask the network for the twist
that undoes the perturbation.

The perturbation distribution is the design decision that matters.  It has to
cover the errors the classical estimator actually makes -- otherwise the
refiner is trained on a problem it will never see -- and it has to reach
somewhat beyond them, or the refiner has no margin when the classical stage has
a bad day.  The defaults here are drawn from the measured classical error
distribution on this rig, widened by roughly a factor of three, and
``scripts/train_refiner.py`` prints both so the choice is visible.

Perturbations are drawn as a *log-uniform magnitude* with a uniform direction,
not as a Gaussian on each of the six components.  A Gaussian on six components
concentrates its samples in a shell at one particular magnitude, so the
network sees almost no easy examples and almost no hard ones; log-uniform
magnitude gives it the whole range it has to work over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import se3, so3
from .render_compare import build_input


@dataclass
class PerturbationSpec:
    """The range of initial errors the refiner is trained to correct."""

    #: The lower end is 1 mm and 0.3 degrees, not something much smaller.
    #:
    #: A 0.2 mm perturbation moves the part's silhouette by a fifth of a pixel
    #: at this stand-off.  There is no signal in such a sample -- the rendered
    #: silhouette and the observed one are the same picture -- and its target
    #: is a twist of essentially zero.  Filling a third of the training set
    #: with those teaches the refiner that predicting nothing is usually right,
    #: which is exactly the behaviour it got stuck in.
    #:
    #: 1 mm is a quarter of the classical estimator's median error on this rig,
    #: so the range still comfortably brackets what the refiner will be handed.
    min_translation_m: float = 0.001
    max_translation_m: float = 0.015
    min_rotation_deg: float = 0.3
    max_rotation_deg: float = 8.0

    def sample(self, rng):
        """A random ``T_perturbation`` applied on the right of the true pose."""
        t_mag = float(np.exp(rng.uniform(
            np.log(self.min_translation_m), np.log(self.max_translation_m)
        )))
        r_mag = float(np.exp(rng.uniform(
            np.log(np.radians(self.min_rotation_deg)),
            np.log(np.radians(self.max_rotation_deg)),
        )))
        axis = rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        direction = rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        return se3.from_Rt(so3.exp(axis * r_mag), direction * t_mag)

    def describe(self):
        return {
            "translation_m": [self.min_translation_m, self.max_translation_m],
            "rotation_deg": [self.min_rotation_deg, self.max_rotation_deg],
            "magnitude_sampling": "log-uniform, uniform direction",
        }


class RefinementDataset:
    """Render-and-compare samples, as a ``torch.utils.data.Dataset``.

    Images are held in memory as ``uint8``.  The whole training split is about
    two hundred megabytes and re-reading four PNGs per sample would otherwise
    dominate the epoch time.
    """

    def __init__(self, images_by_frame, poses_by_frame, rig, renderer, model_points,
                 perturbation=None, samples_per_frame=8, seed=0):
        self.frames = sorted(images_by_frame)
        self.images = images_by_frame
        self.poses = poses_by_frame
        self.rig = rig
        self.renderer = renderer
        self.model_points = np.asarray(model_points, dtype=np.float32).reshape(-1, 3)
        self.perturbation = perturbation or PerturbationSpec()
        self.samples_per_frame = int(samples_per_frame)
        self.seed = int(seed)

    def __len__(self):
        return len(self.frames) * self.samples_per_frame

    def __getitem__(self, index):
        frame = self.frames[index % len(self.frames)]
        # Deterministic per index, so a sample is the same whichever worker
        # produces it and an epoch is reproducible.
        rng = np.random.default_rng([self.seed, index])

        T_true = self.poses[frame]
        T_current = T_true @ self.perturbation.sample(rng)

        crops, specs = build_input(self.renderer, self.images[frame], self.rig, T_current)
        valid = np.array([s.valid for s in specs], dtype=np.float32)
        return {
            "crops": crops,
            "valid": valid,
            "T_current": T_current.astype(np.float32),
            "T_target": T_true.astype(np.float32),
            "frame": frame,
        }


def collate(batch):
    """Stack a list of samples into torch tensors."""
    import torch

    return {
        "crops": torch.from_numpy(np.stack([b["crops"] for b in batch])),
        "valid": torch.from_numpy(np.stack([b["valid"] for b in batch])),
        "T_current": torch.from_numpy(np.stack([b["T_current"] for b in batch])),
        "T_target": torch.from_numpy(np.stack([b["T_target"] for b in batch])),
        "frame": [b["frame"] for b in batch],
    }


def load_pose_session(data, frames, cameras=None):
    """Images and ground-truth rig-frame poses for a set of pose stations."""
    cameras = list(range(data.n_cameras)) if cameras is None else list(cameras)
    T_rig_world = data.T_rig_world
    images, poses = {}, {}
    for f in frames:
        images[f] = {c: data.pose_image(f, c) for c in cameras}
        poses[f] = T_rig_world @ data.T_world_object_pose(f)
    return images, poses
