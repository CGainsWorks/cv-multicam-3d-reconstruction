"""The pose-refinement network.

A small render-and-compare CNN, in the DeepIM lineage, adapted to a calibrated
multi-camera rig.

    per-camera crop (4 x 96 x 96)
        -> shared CNN trunk
        -> concatenate a learned per-camera embedding
        -> mean over cameras
        -> MLP
        -> a 6-vector twist, applied as T <- T exp(delta)

Three choices worth defending:

**One trunk, shared across cameras.**  The four cameras see the same part from
different sides and the alignment cue -- how far the rendered silhouette is
from the image structure -- is the same cue in each.  A separate trunk per
camera would quadruple the parameters and learn the same thing four times.

**A learned camera embedding, then mean pooling.**  The delta is predicted in
the *object* frame, and a given misalignment in camera 0's crop implies a
different object-frame correction than the same misalignment in camera 3's.
The network cannot know which camera a crop came from unless it is told, so it
is told, with an embedding.  Mean pooling then makes the head indifferent to
camera ordering and lets a camera drop out without changing the shape.

**The output is a twist, not a quaternion or a matrix.**  It composes on the
right, exactly as everywhere else in this repository, so the retraction is the
same one the classical refiner uses and the two are directly comparable.

The network is small on purpose -- about 0.6 M parameters -- because the honest
comparison is against a classical refiner with six free parameters and perfect
knowledge of the geometry.  A network large enough to memorise this dataset
would win the benchmark and prove nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The predicted twist is scaled by these before being exponentiated, so the
#: network's raw output is O(1) in both blocks.  Translation in metres,
#: rotation in radians, sized to the errors a classical estimate actually has.
TRANSLATION_SCALE = 0.010
ROTATION_SCALE = 0.10


def _block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class PoseRefineNet(nn.Module):
    """Predicts a pose correction from multi-camera render-and-compare crops."""

    def __init__(self, in_channels=4, n_cameras=4, width=32, embedding=16, crop=96):
        super().__init__()
        self.n_cameras = int(n_cameras)
        self.trunk = nn.Sequential(
            _block(in_channels, width, stride=2),        # 96 -> 48
            _block(width, width * 2, stride=2),          # 48 -> 24
            _block(width * 2, width * 4, stride=2),      # 24 -> 12
            _block(width * 4, width * 4, stride=2),      # 12 -> 6
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.camera_embedding = nn.Embedding(self.n_cameras, int(embedding))
        self.per_camera = nn.Sequential(
            nn.Linear(width * 4 + int(embedding), 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 6),
        )
        # Start at the identity correction.  A refiner whose first output is a
        # large random twist diverges before it learns anything, because the
        # next iteration's render is then of a pose that is worse than the one
        # it started from.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, crops, valid=None):
        """``crops`` is ``(B, C, ch, H, W)``; returns a ``(B, 6)`` twist.

        ``valid`` is an optional ``(B, C)`` mask; invalid cameras are excluded
        from the mean rather than contributing a zero-filled crop, which would
        otherwise pull the average towards whatever a blank input encodes.
        """
        b, c = crops.shape[:2]
        flat = crops.reshape(b * c, *crops.shape[2:])
        features = self.trunk(flat)

        ids = torch.arange(c, device=crops.device).repeat(b)
        features = torch.cat([features, self.camera_embedding(ids)], dim=1)
        features = self.per_camera(features).reshape(b, c, -1)

        if valid is None:
            pooled = features.mean(dim=1)
        else:
            w = valid.to(features.dtype).unsqueeze(-1)
            pooled = (features * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

        raw = self.head(pooled)
        scale = torch.tensor(
            [TRANSLATION_SCALE] * 3 + [ROTATION_SCALE] * 3,
            dtype=raw.dtype, device=raw.device,
        )
        return raw * scale

    @property
    def n_parameters(self):
        return int(sum(p.numel() for p in self.parameters()))


def se3_exp_torch(twist):
    """Batched SE(3) exponential, differentiable.

    Mirrors :func:`multicam.geometry.se3.exp` exactly, including the truncated
    series near zero -- which matters more here than in NumPy, because at
    convergence the predicted twist *is* near zero and the closed form's
    ``sin(theta)/theta`` would produce a NaN gradient rather than a large one.
    """
    rho, phi = twist[:, :3], twist[:, 3:]
    theta = phi.norm(dim=1, keepdim=True)
    small = theta < 1e-6

    K = torch.zeros(twist.shape[0], 3, 3, dtype=twist.dtype, device=twist.device)
    K[:, 0, 1] = -phi[:, 2]
    K[:, 0, 2] = phi[:, 1]
    K[:, 1, 0] = phi[:, 2]
    K[:, 1, 2] = -phi[:, 0]
    K[:, 2, 0] = -phi[:, 1]
    K[:, 2, 1] = phi[:, 0]
    K2 = torch.bmm(K, K)

    t = theta.clamp(min=1e-12)
    a = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(t) / t)
    b = torch.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - torch.cos(t)) / (t ** 2))
    c = torch.where(small, 1.0 / 6.0 - theta ** 2 / 120.0, (t - torch.sin(t)) / (t ** 3))

    eye = torch.eye(3, dtype=twist.dtype, device=twist.device).expand_as(K)
    R = eye + a.unsqueeze(-1) * K + b.unsqueeze(-1) * K2
    V = eye + b.unsqueeze(-1) * K + c.unsqueeze(-1) * K2

    T = torch.zeros(twist.shape[0], 4, 4, dtype=twist.dtype, device=twist.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = torch.bmm(V, rho.unsqueeze(-1)).squeeze(-1)
    T[:, 3, 3] = 1.0
    return T


def apply_correction_torch(T_current, twist, centroid):
    """Compose a predicted twist onto a pose, in the frame the twist lives in.

    The twist is expressed in a frame that sits **at the object's centre and is
    oriented like the rig**, so

        T_new = A exp(delta) A^-1 T,     A = translation(T @ centroid)

    This is the single most important choice in the whole refiner and it took
    three failed training runs to get right.

    The obvious alternatives are both unlearnable:

    * ``T exp(delta)`` puts the twist in the *object's* frame.  The network sees
      camera crops, so the map from "the silhouette is three pixels left in
      camera 0" to an object-frame correction depends on the object's current
      orientation -- a latent the network is never given.  On a turntable
      sequence, where the object only ever rotates about one axis, it would
      half work; on poses drawn from the full rotation group it cannot work at
      all, and the symptom is a training loss that does not move from the
      identity in any number of steps.
    * ``exp(delta) T`` puts it in the rig frame, which fixes the orientation
      problem but rotates the object about the *rig origin*, 0.6 m away.  A
      one-degree rotation then translates the part by 10 mm, so rotation and
      translation are coupled through a lever arm that varies with where the
      part is standing.

    Centring on the object and orienting with the rig removes both.  The map
    from image displacement to twist is then fixed per camera, which is exactly
    what the camera embedding can encode.
    """
    centre = torch.einsum("bij,j->bi", T_current[:, :3, :3], centroid) + T_current[:, :3, 3]
    delta = se3_exp_torch(twist)

    # A exp(delta) A^-1 with A a pure translation is just a shift of the
    # translation part, so it is written out rather than built as matrices.
    R_d = delta[:, :3, :3]
    t_d = delta[:, :3, 3]
    shifted = torch.zeros_like(delta)
    shifted[:, :3, :3] = R_d
    shifted[:, :3, 3] = t_d + centre - torch.einsum("bij,bj->bi", R_d, centre)
    shifted[:, 3, 3] = 1.0
    return torch.bmm(shifted, T_current)


def point_matching_loss(twist, T_current, T_target, model_points, centroid=None):
    """DeepIM's loss: how far the predicted pose moves the model's own points.

    Minimising the twist directly would weight three millimetres of translation
    against three degrees of rotation by an arbitrary constant.  This weights
    them by what they actually do to the part, which is what ADD measures and
    therefore what the result is scored on.
    """
    if centroid is None:
        predicted = torch.bmm(T_current, se3_exp_torch(twist))
    else:
        predicted = apply_correction_torch(T_current, twist, centroid)
    a = torch.einsum("bij,pj->bpi", predicted[:, :3, :3], model_points) + \
        predicted[:, :3, 3].unsqueeze(1)
    b = torch.einsum("bij,pj->bpi", T_target[:, :3, :3], model_points) + \
        T_target[:, :3, 3].unsqueeze(1)
    return (a - b).norm(dim=2).mean()


def load_network(path, device="cpu"):
    """Load a checkpoint written by :mod:`multicam.pose.train`."""
    blob = torch.load(str(path), map_location=device, weights_only=False)
    config = blob.get("config", {})
    net = PoseRefineNet(
        in_channels=config.get("in_channels", 4),
        n_cameras=config.get("n_cameras", 4),
        width=config.get("width", 32),
        crop=config.get("crop", 96),
    )
    net.load_state_dict(blob["state_dict"])
    net.to(device)
    net.eval()
    return net, blob


def count_parameters(net):
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def describe(net):
    return {
        "parameters": count_parameters(net),
        "translation_scale_m": TRANSLATION_SCALE,
        "rotation_scale_rad": ROTATION_SCALE,
        "architecture": "shared CNN trunk, camera embedding, mean pool, MLP head",
    }
