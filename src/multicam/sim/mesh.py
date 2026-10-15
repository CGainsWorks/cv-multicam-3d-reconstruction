"""Triangle meshes, the procedural parts that stand in for a CAD model, and PLY.

There is no CAD package in the dependency list and no downloaded model, so the
"CAD ground truth" this repository evaluates against is generated here, in
closed form, from a handful of primitives.  That is a real limitation and it is
stated in the README: a procedurally generated bracket is a fair test of the
*geometry* of the pipeline and it is not a fair test of how the pipeline copes
with a scanned mesh full of holes and self-intersections.

Texture coordinates are stored **per face corner** rather than per vertex.  A
box has three vertices meeting at a corner with three different surface
normals, and per-vertex UVs would smear the texture across the seam; per-corner
UVs let each face carry its own planar projection and keep the seams sharp,
which matters because sharp texture is what the feature detector lives on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriMesh:
    """A triangle soup with per-corner texture coordinates.

    Attributes
    ----------
    vertices : (V, 3) float
    faces : (F, 3) int
    uv : (F, 3, 2) float
        Texture coordinate for each corner of each face, in ``[0, 1]``.
    texture_id : (F,) int
        Which texture in the scene's texture list each face samples from.
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    texture_id: np.ndarray
    name: str = "mesh"

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int32).reshape(-1, 3)
        self.uv = np.asarray(self.uv, dtype=np.float64).reshape(-1, 3, 2)
        self.texture_id = np.asarray(self.texture_id, dtype=np.int32).reshape(-1)
        if not (len(self.faces) == len(self.uv) == len(self.texture_id)):
            raise ValueError("faces, uv and texture_id must agree in length")

    # -- properties --------------------------------------------------------

    @property
    def face_normals(self):
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    @property
    def face_areas(self):
        v = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    @property
    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def centre(self):
        lo, hi = self.bounds
        return 0.5 * (lo + hi)

    @property
    def diameter(self):
        """Largest distance between any two vertices.

        Used to normalise the pose metrics: an ADD of 5 mm means something very
        different on a 40 mm part and on a 400 mm one, so the metrics are also
        reported as a fraction of this.
        """
        lo, hi = self.bounds
        # The bounding-box diagonal is an upper bound on the true diameter and
        # is O(V) rather than O(V^2).  For the convex-ish parts here they are
        # within a few per cent, and it is used only as a normaliser.
        return float(np.linalg.norm(hi - lo))

    # -- construction ------------------------------------------------------

    def transformed(self, T):
        T = np.asarray(T, dtype=float).reshape(4, 4)
        v = self.vertices @ T[:3, :3].T + T[:3, 3]
        return TriMesh(v, self.faces.copy(), self.uv.copy(), self.texture_id.copy(), self.name)

    def merged(self, other, texture_offset=0):
        return TriMesh(
            np.vstack([self.vertices, other.vertices]),
            np.vstack([self.faces, other.faces + len(self.vertices)]),
            np.vstack([self.uv, other.uv]),
            np.concatenate([self.texture_id, other.texture_id + texture_offset]),
            self.name,
        )

    def sample_surface(self, n, rng):
        """``n`` points drawn uniformly by area, with their normals.

        Area weighting matters: sampling vertices instead would over-represent
        the densely tessellated cylinder caps and make a chamfer distance look
        better than it is.
        """
        areas = self.face_areas
        total = areas.sum()
        if total <= 0:
            raise ValueError("mesh has no area")
        idx = rng.choice(len(areas), size=n, p=areas / total)
        v = self.vertices[self.faces[idx]]
        # sqrt(r1) is the standard change of variables that makes (u, v)
        # uniform over the triangle rather than bunched at one corner.
        r1 = np.sqrt(rng.random(n))[:, None]
        r2 = rng.random(n)[:, None]
        pts = (1 - r1) * v[:, 0] + r1 * (1 - r2) * v[:, 1] + r1 * r2 * v[:, 2]
        return pts, self.face_normals[idx]

    # -- PLY ---------------------------------------------------------------

    def save_ply(self, path):
        """Binary little-endian PLY, vertices and faces only.

        Written by hand rather than through Open3D so that the mesh files this
        repository produces do not depend on the Open3D version that happened
        to be installed.
        """
        v = self.vertices.astype("<f4")
        f = self.faces.astype("<i4")
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment written by multicam.sim.mesh\n"
            "element vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face %d\n"
            "property list uchar int vertex_indices\n"
            "end_header\n" % (len(v), len(f))
        )
        counts = np.full((len(f), 1), 3, dtype=np.uint8)
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(v.tobytes())
            # Interleave the per-face vertex count with the three indices.
            rows = np.concatenate([counts.view(np.uint8), f.view(np.uint8)], axis=1)
            fh.write(rows.tobytes())

    @staticmethod
    def load_ply(path, name="mesh"):
        with open(path, "rb") as fh:
            data = fh.read()
        end = data.index(b"end_header\n") + len(b"end_header\n")
        header = data[:end].decode("ascii")
        n_v = n_f = 0
        for line in header.splitlines():
            if line.startswith("element vertex"):
                n_v = int(line.split()[-1])
            elif line.startswith("element face"):
                n_f = int(line.split()[-1])
        if "binary_little_endian" not in header:
            raise ValueError("only the binary little-endian PLY this module writes is read back")
        off = end
        verts = np.frombuffer(data, dtype="<f4", count=n_v * 3, offset=off).reshape(-1, 3)
        off += n_v * 12
        rows = np.frombuffer(data, dtype=np.uint8, count=n_f * 13, offset=off).reshape(n_f, 13)
        faces = rows[:, 1:].copy().view("<i4").reshape(-1, 3)
        uv = np.zeros((n_f, 3, 2))
        return TriMesh(verts.astype(float), faces, uv, np.zeros(n_f, dtype=np.int32), name)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


#: A distinct texture offset for each of the six face orientations, in tiles.
#:
#: These are not cosmetic.  Without them the triplanar projection hands
#: *identical* texture coordinates to any two faces whose projections coincide,
#: and there are two such coincidences on any box:
#:
#: * A ``+z`` face and the ``-z`` face opposite it both project to ``(x, y)``,
#:   so the top and the bottom of a plate are texturally identical.
#: * A ``z`` face at ``(x0, y0)`` and a ``y`` face at ``(x0, z0 = y0)`` sample
#:   the same texel, so the two limbs of an L-bracket become identical under
#:   the swap ``y <-> z`` -- which is a 90 degree rotation about ``x``.
#:
#: The second one is the expensive one.  It gives the part an exact *appearance*
#: symmetry that its geometry does not have, the descriptor matcher cannot tell
#: the two limbs apart, and the pose estimator lands on the 90-degree
#: alternative on about fifteen per cent of frames -- with a healthy inlier
#: count and a 0.4 px residual, so nothing in the output looks wrong.  It is an
#: artefact of the texture generator and a real part would not have it.
#:
#: The offsets are irrational-ish fractions of a tile so that no two of the six
#: regions coincide under any axis swap, and the texture tiles, so an offset
#: lands on genuinely different noise.
_FACE_UV_OFFSETS = {
    (0, 1): (0.000, 0.000), (0, -1): (0.517, 0.283),
    (1, 1): (0.371, 0.613), (1, -1): (0.149, 0.829),
    (2, 1): (0.733, 0.191), (2, -1): (0.907, 0.457),
}


def _planar_uv(vertices, faces, scale):
    """Triplanar UVs: project each face onto the plane its normal points along.

    The projection axis is chosen per face from the dominant component of the
    normal, so a box gets six flat, undistorted projections and a cylinder gets
    a wrap that is only badly stretched exactly at the silhouette.  Each of the
    six orientations is then shifted into its own region of the texture; see
    :data:`_FACE_UV_OFFSETS` for why that matters a great deal.

    The result is in world units times ``scale`` and is deliberately *not*
    wrapped into ``[0, 1]``: the sampler tiles, and wrapping here would put a
    seam through the middle of every face that happens to straddle an integer.
    """
    v = vertices[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    axis = np.argmax(np.abs(n), axis=1)
    sign = np.sign(n[np.arange(len(faces)), axis])
    sign[sign == 0] = 1

    uv = np.zeros((len(faces), 3, 2))
    for a in range(3):
        other = [i for i in range(3) if i != a]
        for s in (1, -1):
            m = (axis == a) & (sign == s)
            if not m.any():
                continue
            offset = np.asarray(_FACE_UV_OFFSETS[(a, s)])
            uv[m] = v[m][:, :, other] * scale + offset
    return uv


def box(size, centre=(0.0, 0.0, 0.0), uv_scale=8.0, texture_id=0):
    """An axis-aligned box as 12 triangles, wound anticlockwise from outside."""
    sx, sy, sz = np.asarray(size, dtype=float) * 0.5
    cx, cy, cz = np.asarray(centre, dtype=float)
    v = np.array(
        [
            [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
        ]
    )
    f = np.array(
        [
            [0, 2, 1], [0, 3, 2],        # -z
            [4, 5, 6], [4, 6, 7],        # +z
            [0, 1, 5], [0, 5, 4],        # -y
            [2, 3, 7], [2, 7, 6],        # +y
            [1, 2, 6], [1, 6, 5],        # +x
            [0, 4, 7], [0, 7, 3],        # -x
        ]
    )
    return TriMesh(v, f, _planar_uv(v, f, uv_scale), np.full(len(f), texture_id))


def cylinder(radius, height, centre=(0.0, 0.0, 0.0), segments=48, axis=2,
             uv_scale=8.0, texture_id=0, caps=True):
    """A capped cylinder along ``axis`` (0=x, 1=y, 2=z)."""
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    lo, hi = -0.5 * height, 0.5 * height

    def place(xy, z):
        p = np.zeros((len(xy), 3))
        others = [i for i in range(3) if i != axis]
        p[:, others[0]] = xy[:, 0]
        p[:, others[1]] = xy[:, 1]
        p[:, axis] = z
        return p + np.asarray(centre, dtype=float)

    bottom = place(ring, lo)
    top = place(ring, hi)
    verts = [bottom, top]
    faces = []
    n = segments
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])
    if caps:
        c_lo, c_hi = place(np.zeros((1, 2)), lo), place(np.zeros((1, 2)), hi)
        verts += [c_lo, c_hi]
        i_lo, i_hi = 2 * n, 2 * n + 1
        for i in range(n):
            j = (i + 1) % n
            faces.append([i_lo, j, i])
            faces.append([i_hi, n + i, n + j])
    v = np.vstack(verts)
    f = np.array(faces, dtype=np.int32)
    return TriMesh(v, f, _planar_uv(v, f, uv_scale), np.full(len(f), texture_id))


def plane(size, centre=(0.0, 0.0, 0.0), normal_axis=2, uv_scale=1.0, texture_id=0,
          subdivisions=1):
    """A flat rectangle.  Subdivided so it can be lit and clipped sensibly."""
    sx, sy = np.asarray(size, dtype=float) * 0.5
    n = int(subdivisions)
    gx = np.linspace(-sx, sx, n + 1)
    gy = np.linspace(-sy, sy, n + 1)
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    others = [i for i in range(3) if i != normal_axis]
    v = np.zeros(((n + 1) ** 2, 3))
    v[:, others[0]] = XX.ravel()
    v[:, others[1]] = YY.ravel()
    v += np.asarray(centre, dtype=float)

    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    f = np.array(faces, dtype=np.int32)

    # UVs run over the whole rectangle, so a checkerboard texture maps to it
    # exactly once and the board geometry is recoverable from the UVs.
    uv = np.zeros((len(f), 3, 2))
    norm = np.stack([(v[:, others[0]] + sx) / (2 * sx), (v[:, others[1]] + sy) / (2 * sy)], axis=1)
    norm -= np.array([np.asarray(centre)[others[0]], np.asarray(centre)[others[1]]]) / \
        np.array([2 * sx, 2 * sy])
    for k in range(3):
        uv[:, k, :] = norm[f[:, k]] * uv_scale
    return TriMesh(v, f, uv, np.full(len(f), texture_id))


def triangular_prism(p0, p1, p2, thickness, axis=1, uv_scale=8.0, texture_id=0):
    """A prism from a triangle in the plane perpendicular to ``axis``."""
    others = [i for i in range(3) if i != axis]

    def lift(p, t):
        q = np.zeros(3)
        q[others[0]], q[others[1]] = p[0], p[1]
        q[axis] = t
        return q

    v = np.array(
        [lift(p, -0.5 * thickness) for p in (p0, p1, p2)]
        + [lift(p, 0.5 * thickness) for p in (p0, p1, p2)]
    )
    f = np.array(
        [
            [0, 1, 2], [3, 5, 4],
            [0, 3, 4], [0, 4, 1],
            [1, 4, 5], [1, 5, 2],
            [2, 5, 3], [2, 3, 0],
        ],
        dtype=np.int32,
    )
    return TriMesh(v, f, _planar_uv(v, f, uv_scale), np.full(len(f), texture_id))


#: Texture repeat rate for the part, in tiles per metre.  Low on purpose: at a
#: high rate the surface texture repeats several times across one face and the
#: descriptor matcher sees a forest of near-identical candidates, which is a
#: property of the texture generator and not of the algorithm being tested.
BRACKET_UV_SCALE = 4.0


def bracket(texture_id=0, scale=1.0):
    """The part the pose estimator is asked to find.

    A right-angle mounting bracket in metres: a 120 x 80 x 12 mm base, a
    120 x 12 x 70 mm upright, a 36 mm diameter boss on the upright and two
    gussets.  It is deliberately *not* rotationally symmetric -- a symmetric
    part makes ADD meaningless and ADD-S the only honest metric, and that
    ambiguity would hide exactly the errors this repository is trying to
    measure.  The asymmetry is checked by a test.
    """
    s = float(scale)
    uv = BRACKET_UV_SCALE / s
    base = box((0.120 * s, 0.080 * s, 0.012 * s), (0.0, 0.0, 0.006 * s), uv, texture_id)
    upright = box(
        (0.120 * s, 0.012 * s, 0.070 * s),
        (0.0, -0.034 * s, 0.047 * s),
        uv,
        texture_id,
    )
    boss = cylinder(
        0.018 * s, 0.020 * s, (0.030 * s, -0.038 * s, 0.055 * s), 40, axis=1,
        uv_scale=uv, texture_id=texture_id,
    )
    gusset_a = triangular_prism(
        (-0.050 * s, 0.012 * s), (-0.050 * s, 0.070 * s), (0.006 * s, 0.012 * s),
        0.010 * s, axis=0, uv_scale=uv, texture_id=texture_id,
    )
    gusset_b = triangular_prism(
        (0.050 * s, 0.012 * s), (0.050 * s, 0.050 * s), (0.006 * s, 0.012 * s),
        0.010 * s, axis=0, uv_scale=uv, texture_id=texture_id,
    )
    # gusset prisms are built in the (y, z) plane; move them onto the upright.
    gusset_a = gusset_a.transformed(_translation([-0.045 * s, -0.028 * s, 0.0]))
    gusset_b = gusset_b.transformed(_translation([0.045 * s, -0.028 * s, 0.0]))

    m = base.merged(upright).merged(boss).merged(gusset_a).merged(gusset_b)
    m.name = "bracket"
    # Put the origin at the centroid of the bounding box so that a rotation
    # error and a translation error are not entangled through a long lever arm.
    return m.transformed(_translation(-m.centre))


def _translation(t):
    T = np.eye(4)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T
