"""Planning and rendering a capture session.

Three sessions, mirroring what you would actually do in front of a real rig:

``intrinsics``
    One camera at a time, the board waved through its whole field of view at a
    range of depths and tilts.  Poses are planned **per camera** and are
    accepted only if the board actually lands inside that camera's frame, so
    the corners of the image -- where the distortion lives -- get covered.
``extrinsics``
    The board held still in the shared working volume, all cameras triggered
    together.  Poses are accepted only if at least two cameras see enough of
    the board, because a station only one camera sees contributes nothing to
    the relative geometry.
``turntable``
    The part on the table, stepped through a full revolution, all cameras
    triggered together.  This is the reconstruction and pose data.

Every pose is checked geometrically *before* anything is rendered.  Rendering
is the expensive step and rejecting a station after paying for four renders of
it would triple the cost of a capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3, so3
from .rig import nominal_rig_pose
from .render import Rasteriser
from .scene import board_in_scene, board_scene, turntable_poses
from .textures import charuco_board, charuco_texture, surface_texture


@dataclass
class CaptureConfig:
    """Everything that decides what a capture contains."""

    n_intrinsic_views: int = 22
    n_extrinsic_stations: int = 26
    n_turntable_stations: int = 18
    turntable_wobble_deg: float = 0.15
    #: Poses for the 6-DoF evaluation.  The last ``n_pose_test`` are the held
    #: out split; the rest are what the learned refiner may train on.  The
    #: split is by index and is fixed, so no result can quietly be reported on
    #: data the network saw.
    n_pose_stations: int = 140
    n_pose_test: int = 40
    #: Sensor read noise as a fraction of full scale.  0.004 is about 1 count
    #: in 8 bits, which is a quiet machine-vision camera at a sensible gain.
    noise_sigma: float = 0.004
    supersample: int = 2
    board_squares: tuple = (7, 5)
    #: 25 mm squares, so the printed board is 175 x 125 mm.  Sized to the
    #: working volume: the extrinsic session needs the whole board inside the
    #: frame of two or more cameras at once, and the frame is 380 mm across at
    #: the turntable.  A 40 mm board overflows it and no station is usable.
    square_length_m: float = 0.025
    seed: int = 11
    depth_maps: bool = True
    max_attempts_per_pose: int = 400


@dataclass
class CapturePlan:
    """Poses only.  Rendering is a separate step so plans can be tested cheaply."""

    intrinsic: dict = field(default_factory=dict)   #: camera -> [T_cam_board]
    extrinsic: list = field(default_factory=list)   #: [T_world_board]
    turntable: list = field(default_factory=list)   #: [T_world_object]
    pose: list = field(default_factory=list)        #: [T_world_object]
    board_margin_m: float = 0.0
    rejected: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def _board_corners(target):
    """The four outer corners of the printed board, in the board frame."""
    nx, ny = target.size
    sq = target.square_length
    return np.array(
        [[0.0, 0.0, 0.0], [nx * sq, 0.0, 0.0], [nx * sq, ny * sq, 0.0], [0.0, ny * sq, 0.0]]
    )


def _board_centre(target):
    nx, ny = target.size
    sq = target.square_length
    return np.array([nx * sq * 0.5, ny * sq * 0.5, 0.0])


def _visible_fraction(camera, T_cam_board, corners, margin=6.0):
    """How much of the board lands inside the frame, and how big it is.

    Returns ``(fraction_of_corners_inside, area_fraction_of_image)``.  Both
    gates matter: a board that is fully inside but occupies 2 per cent of the
    frame constrains nothing, and one that fills the frame but has three
    corners outside it gives the distortion model nothing at the edges.
    """
    pts = se3.transform(T_cam_board, corners)
    if np.any(pts[:, 2] < 0.05):
        return 0.0, 0.0
    uv, valid = camera.project(pts, return_valid=True)
    if not valid.all():
        return 0.0, 0.0
    inside = camera.in_image(uv, margin=margin)
    # Shoelace area of the projected quadrilateral.
    x, y = uv[:, 0], uv[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return float(inside.mean()), area / float(camera.width * camera.height)


def board_depth_range(camera, target, fill=(0.28, 0.80)):
    """Depths at which the board covers ``fill`` of the image width.

    Derived rather than hard-coded, because the right distance to hold a board
    depends entirely on the lens.  A range tuned for a 58-degree lens puts the
    board off the edge of the frame at every pose on a 32-degree one, the
    planner rejects all of them, and the failure surfaces as "only 0 intrinsic
    views planned" a long way from the cause.
    """
    nx, _ = target.size
    board_w = nx * target.square_length
    far = board_w * camera.fx / (fill[0] * camera.width)
    near = board_w * camera.fx / (fill[1] * camera.width)
    return float(near), float(far)


def plan_intrinsic_views(camera, target, n, rng, depth_range=None,
                         max_tilt_deg=42.0, min_area=0.045, max_attempts=400):
    """Board poses in one camera's own frame, spread over its field of view.

    The target image position is drawn from a low-discrepancy-ish grid over the
    frame rather than uniformly, so the corners are actually visited.  Uniform
    sampling plus a rejection test that demands the whole board be inside
    concentrates every accepted pose in the middle of the image, which is
    exactly where the distortion coefficients are least observable.
    """
    corners = _board_corners(target)
    centre = _board_centre(target)
    if depth_range is None:
        depth_range = board_depth_range(camera, target)

    # Nine target regions: centre, edges and corners, cycled so the count is
    # spread evenly however many views are asked for.
    regions = [(fx, fy) for fy in (0.22, 0.5, 0.78) for fx in (0.22, 0.5, 0.78)]

    poses, attempts = [], 0
    while len(poses) < n and attempts < max_attempts:
        attempts += 1
        rx, ry = regions[len(poses) % len(regions)]
        u = (rx + rng.normal(scale=0.06)) * camera.width
        v = (ry + rng.normal(scale=0.06)) * camera.height
        depth = float(rng.uniform(*depth_range))

        # Ray through the chosen pixel, so the board centre lands there.
        ray = camera.rays(np.array([[u, v]]))[0]
        C = ray * (depth / max(ray[2], 1e-9))

        tilt = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, size=2))
        roll = rng.uniform(-np.pi, np.pi)
        R = so3.exp([0.0, 0.0, roll]) @ so3.exp([tilt[0], 0.0, 0.0]) @ so3.exp([0.0, tilt[1], 0.0])
        T = se3.from_Rt(R, C - R @ centre)

        frac, area = _visible_fraction(camera, T, corners)
        if frac < 1.0 or area < min_area:
            continue
        # Reject a board seen almost edge-on: the detector will find it and the
        # corner positions will be dominated by the resampling.
        normal_cam = R @ np.array([0.0, 0.0, 1.0])
        view = C / max(np.linalg.norm(C), 1e-9)
        if abs(float(normal_cam @ view)) < 0.35:
            continue
        poses.append(T)
    return poses, attempts


def plan_extrinsic_stations(rig, target, n, rng, T_rig_world=None, min_cameras=2,
                            min_area=0.03, max_attempts=1200):
    """Board poses in the world frame that several cameras can all see."""
    T_rig_world = nominal_rig_pose(rig) if T_rig_world is None else T_rig_world
    T_cam_world = rig.all_T_cam_world(T_rig_world)
    corners = _board_corners(target)
    centre = _board_centre(target)

    # Aim the board at a point on the camera arc so it faces the rig rather
    # than the ceiling.
    arc_centres = np.array([se3.camera_centre(T) for T in T_cam_world])

    poses, attempts, seen_hist = [], 0, []
    while len(poses) < n and attempts < max_attempts:
        attempts += 1
        C = np.array(
            [
                rng.uniform(-0.11, 0.11),
                rng.uniform(-0.11, 0.11),
                rng.uniform(0.05, 0.30),
            ]
        )
        aim = arc_centres[rng.integers(len(arc_centres))] + rng.normal(scale=0.12, size=3)
        # The board's own +z points *out of the back* of the printed sheet --
        # that is what makes T_cam_board the identity for a board held square
        # to a camera, which is the convention the intrinsic session and
        # OpenCV's object points both use.  So to face a viewer at `aim` the
        # board's +z must point *away* from `aim`, hence the reflected target.
        # Aiming at `aim` directly renders the back of the sheet: the geometry
        # is right, the image is mirrored, and ArUco decodes nothing.
        R = se3.look_at(C, 2.0 * C - aim)[:3, :3].T
        R = R @ so3.exp([0.0, 0.0, rng.uniform(-np.pi, np.pi)])
        R = R @ so3.exp(np.radians(rng.normal(scale=11.0, size=3)))
        T_world_board = se3.from_Rt(R, C - R @ centre)

        seen = 0
        for cam, T_cw in zip(rig.cameras, T_cam_world):
            frac, area = _visible_fraction(cam, T_cw @ T_world_board, corners, margin=2.0)
            if frac >= 1.0 and area >= min_area:
                seen += 1
        if seen < min_cameras:
            continue
        poses.append(T_world_board)
        seen_hist.append(seen)
    return poses, attempts, seen_hist


def plan_pose_stations(rig, scene, n, rng, T_rig_world=None, min_cameras=3,
                       min_pixels=6000, max_attempts=4000):
    """Object poses for the 6-DoF evaluation: arbitrary orientation, in the volume.

    Unlike the turntable sweep, which only ever rotates the part about one
    axis, these are drawn from the full rotation group.  That is the case a
    pick cell actually faces, and it is the case that separates a pose
    estimator from a turntable-angle estimator.

    Acceptance is geometric and cheap: the part's bounding sphere must project
    inside at least ``min_cameras`` frames and cover at least ``min_pixels``
    there.  Rendering first and rejecting afterwards would cost four renders
    per rejected pose.
    """
    T_rig_world = nominal_rig_pose(rig) if T_rig_world is None else T_rig_world
    T_cam_world = rig.all_T_cam_world(T_rig_world)

    lo, hi = scene.object_mesh.bounds
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    # The disc top, so the part sits in the same place the scan saw it.
    base_z = float(np.asarray(scene.T_world_object)[2, 3])

    poses, attempts = [], 0
    while len(poses) < n and attempts < max_attempts:
        attempts += 1
        centre = np.array([
            rng.uniform(-0.045, 0.045),
            rng.uniform(-0.045, 0.045),
            base_z + rng.uniform(-0.015, 0.045),
        ])
        R = so3.random_rotation(rng)
        T = se3.from_Rt(R, centre)

        seen, pixels = 0, []
        for camera, T_cw in zip(rig.cameras, T_cam_world):
            Xc = se3.transform(T_cw, centre)
            if Xc[2] < 0.15:
                continue
            uv, valid = camera.project(Xc, return_valid=True)
            if not valid or not camera.in_image(uv, margin=radius * camera.fx / Xc[2]).all():
                continue
            # Projected area of the bounding sphere, as a proxy for how much of
            # the frame the part will occupy.
            r_px = radius * camera.fx / Xc[2]
            area = np.pi * r_px * r_px
            if area < min_pixels:
                continue
            seen += 1
            pixels.append(area)
        if seen < min_cameras:
            continue
        poses.append(T)
    return poses, attempts


def plan_capture(rig, target, scene, config, T_rig_world=None):
    """A full :class:`CapturePlan` for a rig, target and scene."""
    rng = np.random.default_rng(config.seed)
    T_rig_world = nominal_rig_pose(rig) if T_rig_world is None else T_rig_world

    plan = CapturePlan()
    rejected = {}
    for i, cam in enumerate(rig.cameras):
        poses, attempts = plan_intrinsic_views(
            cam, target, config.n_intrinsic_views, rng,
            max_attempts=config.max_attempts_per_pose,
        )
        if len(poses) < 6:
            raise RuntimeError(
                "only %d intrinsic views planned for camera %d after %d attempts"
                % (len(poses), i, attempts)
            )
        plan.intrinsic[i] = poses
        rejected["intrinsic_cam%d_attempts" % i] = attempts

    stations, attempts, seen = plan_extrinsic_stations(
        rig, target, config.n_extrinsic_stations, rng, T_rig_world
    )
    if len(stations) < 8:
        raise RuntimeError(
            "only %d extrinsic stations planned after %d attempts" % (len(stations), attempts)
        )
    plan.extrinsic = stations
    rejected["extrinsic_attempts"] = attempts
    rejected["extrinsic_cameras_per_station"] = seen

    plan.turntable = turntable_poses(
        scene, config.n_turntable_stations, wobble_deg=config.turntable_wobble_deg,
        seed=config.seed + 1,
    )

    pose_stations, attempts = plan_pose_stations(
        rig, scene, config.n_pose_stations, rng, T_rig_world
    )
    # The planner either produced what was asked for or it did not; failing on
    # a separate, larger minimum would reject a deliberately small capture --
    # the demo asks for ten stations on purpose -- for no reason.
    if len(pose_stations) < config.n_pose_stations:
        raise RuntimeError(
            "only %d of %d pose stations planned after %d attempts; the working "
            "volume or the visibility gates are too tight"
            % (len(pose_stations), config.n_pose_stations, attempts)
        )
    if config.n_pose_test >= len(pose_stations):
        raise RuntimeError(
            "the test split (%d) would take every one of the %d pose stations, "
            "leaving nothing to train on"
            % (config.n_pose_test, len(pose_stations))
        )
    plan.pose = pose_stations
    rejected["pose_attempts"] = attempts

    plan.rejected = rejected
    return plan


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def make_target_and_scene(config, cell=None, seed=None):
    """The ChArUco board plus the two scenes it gets rendered in.

    Returns ``(target, isolated, in_cell, margin_m)``.

    ``isolated`` is the board on a stand in front of a plain wall, which is the
    intrinsic session: one camera at a time, board waved about, nothing else in
    shot.  ``in_cell`` puts the same board inside the working volume with the
    floor, walls and turntable present, which is the extrinsic session -- shot
    in situ, exactly as you would in front of real hardware.  ``in_cell`` is
    ``None`` if no cell scene is supplied.
    """
    from ..calibration.target import CharucoTarget

    rng = np.random.default_rng(config.seed if seed is None else seed)
    board = charuco_board(
        squares_x=config.board_squares[0], squares_y=config.board_squares[1],
        square_length=config.square_length_m,
    )
    texture, margin_m = charuco_texture(board)
    backdrop = surface_texture(rng, 1024, contrast=0.42, brightness=0.44)
    isolated = board_scene(board, margin_m, texture, backdrop)
    in_cell = None if cell is None else board_in_scene(board, margin_m, texture, cell)
    return CharucoTarget(board), isolated, in_cell, margin_m


def render_view(rasteriser, camera, T_cam_world, meshes, config, rng):
    return rasteriser.render(
        camera, T_cam_world, meshes,
        noise_sigma=config.noise_sigma, rng=rng, supersample=config.supersample,
    )


def render_capture(writer, rig, scene, board_sc, plan, config, T_rig_world=None,
                   progress=None, board_cell_sc=None):
    """Render a whole :class:`CapturePlan` into a :class:`DatasetWriter`.

    Returns a dict of counts and timings.  The random stream is derived from
    ``(session, frame, camera)`` rather than drawn sequentially, so rendering a
    subset of a capture gives byte-identical images to rendering all of it --
    which is what makes a partial re-run comparable to a full one.
    """
    import time

    T_rig_world = nominal_rig_pose(rig) if T_rig_world is None else T_rig_world
    board_cell_sc = board_cell_sc if board_cell_sc is not None else board_sc
    board_ras = Rasteriser(board_sc.textures)
    cell_ras = Rasteriser(board_cell_sc.textures)
    scene_ras = Rasteriser(scene.textures)

    def stream(session, frame, camera):
        return np.random.default_rng([config.seed, session, frame, camera])

    counts = {"intrinsic": 0, "extrinsic": 0, "turntable": 0, "pose": 0}
    start = time.time()

    # -- intrinsics: one camera at a time, board posed in that camera's frame
    truth_intrinsic = {}
    for cam_index, poses in sorted(plan.intrinsic.items()):
        camera = rig.cameras[cam_index]
        truth_intrinsic[str(cam_index)] = []
        for k, T_cam_board in enumerate(poses):
            r = render_view(
                board_ras, camera, np.eye(4), board_sc.meshes_at(T_cam_board),
                config, stream(0, k, cam_index),
            )
            writer.write_intrinsic_view(cam_index, k, r.image)
            truth_intrinsic[str(cam_index)].append(np.asarray(T_cam_board).reshape(16).tolist())
            counts["intrinsic"] += 1
            if progress:
                progress("intrinsics cam%d %d/%d" % (cam_index, k + 1, len(poses)))

    # -- extrinsics: every camera triggered on the same board station
    truth_extrinsic = []
    for k, T_world_board in enumerate(plan.extrinsic):
        meshes = board_cell_sc.meshes_at(T_world_board)
        for cam_index, camera in enumerate(rig.cameras):
            r = render_view(
                cell_ras, camera, rig.T_cam_world(cam_index, T_rig_world), meshes,
                config, stream(1, k, cam_index),
            )
            writer.write_extrinsic_view(k, cam_index, r.image)
            counts["extrinsic"] += 1
        truth_extrinsic.append(np.asarray(T_world_board).reshape(16).tolist())
        if progress:
            progress("extrinsics %d/%d" % (k + 1, len(plan.extrinsic)))

    # -- turntable: the part, stepped through a revolution
    truth_turntable = []
    for k, T_world_object in enumerate(plan.turntable):
        meshes = scene.meshes_at(T_world_object)
        for cam_index, camera in enumerate(rig.cameras):
            r = render_view(
                scene_ras, camera, rig.T_cam_world(cam_index, T_rig_world), meshes,
                config, stream(2, k, cam_index),
            )
            writer.write_turntable_view(
                k, cam_index, r.image, mask=r.mask,
                depth=r.depth if config.depth_maps else None,
            )
            counts["turntable"] += 1
        truth_turntable.append(np.asarray(T_world_object).reshape(16).tolist())
        if progress:
            progress("turntable %d/%d" % (k + 1, len(plan.turntable)))

    # -- pose stations: the part at arbitrary orientations
    truth_pose = []
    for k, T_world_object in enumerate(plan.pose):
        meshes = scene.meshes_at(T_world_object)
        for cam_index, camera in enumerate(rig.cameras):
            r = render_view(
                scene_ras, camera, rig.T_cam_world(cam_index, T_rig_world), meshes,
                config, stream(3, k, cam_index),
            )
            writer.write_pose_view(k, cam_index, r.image, mask=r.mask)
            counts["pose"] = counts.get("pose", 0) + 1
        truth_pose.append(np.asarray(T_world_object).reshape(16).tolist())
        if progress:
            progress("pose %d/%d" % (k + 1, len(plan.pose)))

    writer.write_truth(
        rig,
        scene.object_mesh,
        {
            "rig": np.asarray(T_rig_world).reshape(16).tolist(),
            "intrinsic_board": truth_intrinsic,
            "extrinsic_board": truth_extrinsic,
            "turntable": truth_turntable,
            "pose": truth_pose,
        },
    )
    counts["seconds"] = round(time.time() - start, 2)
    counts["views"] = sum(v for k, v in counts.items() if k != "seconds")
    return counts
