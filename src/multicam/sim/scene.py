"""The scene: what the rig is pointed at.

A turntable cell.  A textured floor, two backdrop panels, a turntable disc and
the bracket standing on it.  The split that matters for the reconstruction is
between the geometry that **turns with the table** -- the disc and the part --
and the geometry that **stays still** -- the floor and the panels.

Reconstruction happens in the object frame, so the turning geometry is
stationary there and the static geometry is not.  Features on the floor and the
panels are therefore genuine outliers with a consistent appearance and an
inconsistent 3D position, which is precisely the failure mode a turntable rig
has in real life and precisely the one that a synthetic scene with a black
backdrop would hide.

``mask`` values in a render index this list, and index 0 is always the part, so
``mask == 0`` is the ground-truth object segmentation.  It is used for
evaluation and never by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3
from . import mesh as M
from . import textures as T


@dataclass
class Scene:
    """Meshes, textures and the default object pose."""

    object_mesh: M.TriMesh          #: the CAD model, in its own frame
    turning_meshes: list            #: extra meshes rigidly attached to the object
    static_meshes: list             #: world-frame geometry that does not turn
    textures: list
    T_world_object: np.ndarray
    metadata: dict = field(default_factory=dict)

    #: Index of the part in the list :meth:`meshes_at` returns.
    OBJECT_INDEX = 0

    def meshes_at(self, T_world_object=None):
        """The full mesh list for a given object pose, part first."""
        T = self.T_world_object if T_world_object is None else np.asarray(T_world_object)
        T = np.asarray(T, dtype=float).reshape(4, 4)
        out = [self.object_mesh.transformed(T)]
        out += [m.transformed(T) for m in self.turning_meshes]
        out += list(self.static_meshes)
        return out

    @property
    def n_turning(self):
        """How many meshes move with the part, the part included."""
        return 1 + len(self.turning_meshes)

    def turning_mask(self, mask):
        """Boolean image: which pixels belong to geometry fixed to the object."""
        return (mask >= 0) & (mask < self.n_turning)

    def object_mask(self, mask):
        return mask == self.OBJECT_INDEX


def default_scene(seed=7, texture_size=1024):
    """The scene every result in this repository is produced from.

    Texture repeat rates are deliberately low -- roughly one to two tiles
    across each surface.  A finely tiled texture looks more detailed and makes
    the matching problem *easier to get wrong*: every keypoint acquires a dozen
    near-identical rivals one tile away, the ratio test throws almost
    everything out, and what survives is biased towards the few unique
    features.  Measuring that would be measuring the texture generator.
    """
    rng = np.random.default_rng(seed)

    # The part is brighter and higher contrast than its surroundings, which is
    # what a machined aluminium part on a matte table actually looks like.  It
    # also stops the detector spending its whole feature budget on the
    # backdrop: with equal contrast everywhere, an nfeatures cap keeps the
    # strongest responses and almost none of them are on a part that occupies
    # three per cent of the frame.
    textures = [
        T.machined_texture(rng, texture_size),                                  # 0: the part
        T.surface_texture(rng, texture_size, contrast=0.46, brightness=0.42),   # 1: table top
        T.surface_texture(rng, texture_size, contrast=0.40, brightness=0.40),   # 2: floor
        T.surface_texture(rng, texture_size, contrast=0.38, brightness=0.46),   # 3: walls
    ]

    part = M.bracket(texture_id=0)
    # Stand the part on the disc: the disc top is at z = 0.020 in world, and
    # the part's own origin is at the centre of its bounding box.
    lo, hi = part.bounds
    stand_height = 0.020 + (hi[2] - lo[2]) * 0.5

    disc = M.cylinder(
        0.150, 0.020, (0.0, 0.0, 0.010 - stand_height), segments=64, axis=2,
        uv_scale=3.0, texture_id=1,
    )

    # A closed cell rather than two floating panels.  Any pixel that renders to
    # black is a pixel where the silhouette of the part, or of the table, is
    # trivially separable from the background, and every number that depends on
    # segmentation would be flattered by it.
    floor = M.plane((2.6, 2.6), (0.0, 0.0, 0.0), normal_axis=2, uv_scale=2.0,
                    texture_id=2, subdivisions=6)
    walls = []
    for axis, offset in ((0, -1.3), (0, 1.3), (1, -1.3), (1, 1.3)):
        centre = [0.0, 0.0, 0.75]
        centre[axis] = offset
        walls.append(
            M.plane((2.6, 1.5), tuple(centre), normal_axis=axis, uv_scale=1.6,
                    texture_id=3, subdivisions=5)
        )
    ceiling = M.plane((2.6, 2.6), (0.0, 0.0, 1.5), normal_axis=2, uv_scale=1.6,
                      texture_id=3, subdivisions=3)

    T_world_object = se3.from_Rt(np.eye(3), [0.0, 0.0, stand_height])
    # A little yaw and tilt so the part is not axis-aligned with the rig, which
    # would make several degenerate configurations accidentally well
    # conditioned.
    T_world_object = T_world_object @ se3.exp([0.0, 0.0, 0.0, 0.03, -0.02, 0.41])

    return Scene(
        object_mesh=part,
        turning_meshes=[disc],
        static_meshes=[floor] + walls + [ceiling],
        textures=textures,
        T_world_object=T_world_object,
        metadata={
            "seed": int(seed),
            "object": part.name,
            "object_diameter_m": part.diameter,
            "turntable_radius_m": 0.150,
        },
    )


def turntable_poses(scene, n, start_deg=0.0, total_deg=360.0, wobble_deg=0.0, seed=0):
    """Object poses for a turntable sweep.

    ``wobble_deg`` tilts the axis by a small random amount at each station.  A
    real turntable is not perfectly perpendicular to its bearing and the
    reconstruction should not depend on it being so; setting it to zero gives
    the ideal case and the difference between the two is worth knowing.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(scene.T_world_object, dtype=float).reshape(4, 4)
    poses = []
    for k in range(int(n)):
        angle = np.radians(start_deg + total_deg * k / max(int(n), 1))
        R = se3.exp([0.0, 0.0, 0.0, 0.0, 0.0, angle])
        if wobble_deg > 0:
            tilt = np.radians(rng.normal(scale=wobble_deg, size=2))
            R = R @ se3.exp([0.0, 0.0, 0.0, tilt[0], tilt[1], 0.0])
        # Rotate about the world z axis through the turntable centre, which is
        # the world origin, so the rotation pre-multiplies.
        poses.append(R @ base)
    return poses


def board_mesh(board, margin_m, subdivisions=2):
    """The calibration board as a quad **in the ChArUco object-point frame**.

    That frame is OpenCV's: the origin sits at the board's first corner, ``x``
    runs right along the printed image, ``y`` runs *down* it, and ``z`` is
    ``x`` cross ``y``.  Building the mesh directly in that frame is the whole
    point -- it means ``board.getChessboardCorners()`` gives 3D points in the
    same frame as the mesh, so a pose recovered from the detected corners is
    directly comparable to the pose the renderer was given, with no
    compensating flip anywhere.  The test that renders a board at a known pose
    and solves for it again is what holds that claim up.

    The texture is the whole printed sheet including its quiet margin, so the
    quad is bigger than the board by ``margin_m`` on every side.
    """
    nx, ny = board.getChessboardSize()
    sq = float(board.getSquareLength())
    board_w, board_h = nx * sq, ny * sq
    x0, y0 = -margin_m, -margin_m
    x1, y1 = board_w + margin_m, board_h + margin_m

    n = int(subdivisions)
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)

    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            faces.append([a, a + (n + 1), a + 1])
            faces.append([a + 1, a + (n + 1), a + (n + 1) + 1])
    faces = np.asarray(faces, dtype=np.int32)

    # u along +x, v along +y.  The generated board image has its first row at
    # the top, which is y = 0, so v maps to the row index without a flip.
    u = (verts[:, 0] - x0) / (x1 - x0)
    v = (verts[:, 1] - y0) / (y1 - y0)
    uv = np.stack([u[faces], v[faces]], axis=2)

    m = M.TriMesh(verts, faces, uv, np.zeros(len(faces), dtype=np.int32), "charuco_board")
    return m


def board_in_scene(board, margin_m, texture, base):
    """The calibration board placed inside an existing scene.

    Used for the rig extrinsic session, which in real life you shoot in situ:
    the board goes into the working volume with the cell exactly as it is for
    the parts.  Rendering it against black instead would give the ArUco
    detector's adaptive threshold a far easier problem than it will ever have
    on the shop floor.

    The board takes texture slot 0 and every mesh inherited from ``base`` has
    its texture index shifted up by one.
    """
    panel = board_mesh(board, margin_m)
    # The turntable furniture is defined in the object frame, so it has to be
    # placed before it can be treated as scenery; the walls and floor are
    # already in world coordinates.
    placed = [m.transformed(base.T_world_object) for m in base.turning_meshes]
    placed += list(base.static_meshes)
    shifted = [
        M.TriMesh(m.vertices.copy(), m.faces.copy(), m.uv.copy(), m.texture_id + 1, m.name)
        for m in placed
    ]
    return Scene(
        object_mesh=panel,
        turning_meshes=[],
        static_meshes=shifted,
        textures=[texture] + list(base.textures),
        T_world_object=np.eye(4),
        metadata={
            "kind": "calibration_board_in_cell",
            "board_size": [int(v) for v in board.getChessboardSize()],
            "square_length_m": float(board.getSquareLength()),
            "margin_m": float(margin_m),
        },
    )


def board_scene(board, margin_m, texture, backdrop_texture):
    """A minimal scene holding just the calibration board in front of a wall."""
    nx, ny = board.getChessboardSize()
    panel = board_mesh(board, margin_m)
    # A textured wall well behind the board.  Without it the board floats on
    # black, the detector has an unrealistically easy time and any exposure or
    # background effect on the corner refinement goes unmeasured.
    backdrop = M.plane((4.0, 4.0), (0.0, 0.0, 1.2), normal_axis=2, uv_scale=7.0,
                       texture_id=1, subdivisions=4)
    return Scene(
        object_mesh=panel,
        turning_meshes=[],
        static_meshes=[backdrop],
        textures=[texture, backdrop_texture],
        T_world_object=np.eye(4),
        metadata={
            "kind": "calibration_board",
            "board_size": [int(nx), int(ny)],
            "square_length_m": float(board.getSquareLength()),
            "margin_m": float(margin_m),
        },
    )
