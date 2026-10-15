"""The rig the repository does not own.

Four cameras on an arc looking down at a turntable.  The geometry is chosen to
be the geometry of a real inspection cell rather than the geometry that makes
the maths easy:

* The cameras span about 100 degrees, not 360.  A rig you can actually build
  sits on one side of the part; the full circle comes from the turntable.
* The intrinsics differ between cameras.  Four nominally identical lenses off
  the same reel differ by a per cent or two in focal length and by several
  pixels in principal point, and a calibration routine that only works when
  they are identical is not a calibration routine.
* The distortion is real barrel distortion, around 6 per cent at the corner.
  Small enough to be plausible for a machine-vision lens, large enough that
  ignoring it costs several pixels at the edge of the frame.

Everything here produces *ground truth*.  The calibration module never sees it;
it sees rendered images of a board and has to recover these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import PinholeCamera, se3
from ..types import RigCalibration


@dataclass
class RigSpec:
    """The physical description of a rig, before any calibration exists."""

    n_cameras: int = 4
    width: int = 800
    height: int = 600
    #: Nominal horizontal field of view in degrees, before per-camera jitter.
    #:
    #: 32 degrees, not 58.  An inspection cell frames the *working volume*, not
    #: the room: at 58 degrees and this stand-off the 187 mm part covered three
    #: per cent of the frame, the detector found seven usable points on it
    #: against sixteen hundred on the walls, and the reconstruction was of the
    #: room.  At 32 degrees the turntable fills about eighty per cent of the
    #: frame width, which is what the lens on a real cell would be chosen to do.
    fov_deg: float = 32.0
    #: 0.6 m stand-off.  Further away and the reconstruction is measuring the
    #: renderer's antialiasing; much closer and the part leaves the frame as it
    #: turns.
    ring_radius: float = 0.60
    ring_height: float = 0.34
    #: Azimuth span of the arc the cameras sit on, in degrees.
    #:
    #: 76 degrees over four cameras is 25 degrees between neighbours.  Wider is
    #: better for triangulation geometry and worse for descriptor matching, and
    #: SIFT starts to fail somewhere past thirty degrees of out-of-plane
    #: rotation on a surface like this one.  25 keeps both usable.
    azimuth_span: float = 76.0
    azimuth_centre: float = 0.0
    target: tuple = (0.0, 0.0, 0.055)
    #: Standard deviation of the manufacturing scatter applied per camera.
    focal_jitter: float = 0.015
    centre_jitter_px: float = 3.5
    mount_jitter_mm: float = 2.0
    mount_jitter_deg: float = 0.6
    distortion: tuple = (-0.26, 0.075, 0.0009, -0.0006, -0.011)
    distortion_jitter: float = 0.12
    seed: int = 4242
    name_prefix: str = "cam"
    metadata: dict = field(default_factory=dict)


def build_rig(spec=None, seed=None):
    """Ground-truth :class:`RigCalibration` for a :class:`RigSpec`."""
    spec = spec or RigSpec()
    rng = np.random.default_rng(spec.seed if seed is None else seed)

    n = int(spec.n_cameras)
    if n < 2:
        raise ValueError("a rig needs at least two cameras")

    if n == 1:
        azimuths = np.array([spec.azimuth_centre])
    else:
        azimuths = np.radians(
            spec.azimuth_centre
            + np.linspace(-0.5 * spec.azimuth_span, 0.5 * spec.azimuth_span, n)
        )

    target = np.asarray(spec.target, dtype=float)
    cameras, T_cam_world = [], []
    for i, az in enumerate(azimuths):
        centre = np.array(
            [
                spec.ring_radius * np.cos(az),
                spec.ring_radius * np.sin(az),
                spec.ring_height,
            ]
        )
        T = se3.look_at(centre, target)

        # Mounting scatter: the arc is where the brackets were meant to put the
        # cameras, not where they are.  This is what the extrinsic calibration
        # has to find.
        dxi = np.concatenate(
            [
                rng.normal(scale=spec.mount_jitter_mm * 1e-3, size=3),
                rng.normal(scale=np.radians(spec.mount_jitter_deg), size=3),
            ]
        )
        T = T @ se3.exp(dxi)
        T_cam_world.append(T)

        f_nominal = 0.5 * spec.width / np.tan(0.5 * np.radians(spec.fov_deg))
        fx = f_nominal * (1.0 + rng.normal(scale=spec.focal_jitter))
        # Real sensors are square-pixel to well within a per cent; the small
        # residual aspect difference is kept because a calibration that assumes
        # fx == fy would otherwise never be tested.
        fy = fx * (1.0 + rng.normal(scale=0.002))
        cx = 0.5 * (spec.width - 1) + rng.normal(scale=spec.centre_jitter_px)
        cy = 0.5 * (spec.height - 1) + rng.normal(scale=spec.centre_jitter_px)
        dist = np.asarray(spec.distortion, dtype=float) * (
            1.0 + rng.normal(scale=spec.distortion_jitter, size=5)
        )
        cameras.append(
            PinholeCamera(fx, fy, cx, cy, spec.width, spec.height, dist,
                          name="%s%d" % (spec.name_prefix, i))
        )

    # Rebase into the rig frame: camera 0 is the rig.
    T_rig_world = T_cam_world[0]
    T_cam_rig = [T @ se3.inverse(T_rig_world) for T in T_cam_world]
    T_cam_rig[0] = np.eye(4)

    meta = dict(spec.metadata)
    meta.update(
        {
            "source": "ground truth from multicam.sim.rig.build_rig",
            "seed": int(spec.seed if seed is None else seed),
            "azimuths_deg": np.degrees(azimuths).tolist(),
            "ring_radius_m": spec.ring_radius,
            "ring_height_m": spec.ring_height,
        }
    )
    rig = RigCalibration(cameras=cameras, T_cam_rig=T_cam_rig, metadata=meta)
    # The rig pose that reproduces the world layout above, carried alongside so
    # the capture script does not have to recompute it.
    rig.metadata["T_rig_world_nominal"] = T_rig_world.reshape(16).tolist()
    return rig


def nominal_rig_pose(rig):
    """The world pose the rig was built at, from its own metadata."""
    T = rig.metadata.get("T_rig_world_nominal")
    if T is None:
        raise KeyError("this rig carries no nominal world pose")
    return np.asarray(T, dtype=float).reshape(4, 4)
