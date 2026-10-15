"""A software rasteriser.

This exists because the repository has no cameras.  It is a z-buffered
triangle rasteriser with near-plane clipping, perspective-correct texture
interpolation and flat Lambertian shading, written in NumPy.  It is not a
renderer anyone should use for graphics; it is a *measurement device* whose
ground truth is exact, which is the property the rest of the repository needs.

Two details are worth stating because getting either wrong makes the whole
pipeline quietly optimistic:

* **Distortion is applied by resampling, not by bending triangles.**  The scene
  is rasterised through an ideal pinhole onto a canvas large enough to cover
  every source pixel the distortion map asks for, then warped.  Rasterising
  distorted vertices directly and interpolating linearly between them would
  produce an image whose distortion is right at the vertices and wrong in
  between, and the calibration would then recover the distortion of the
  renderer rather than of the model.
* **Interpolation is perspective correct.**  Screen-space linear interpolation
  of texture coordinates across a triangle that is steeply inclined -- the
  ground plane, most of the time -- produces the familiar swimming artefact,
  and the feature detector would happily match those artefacts between views.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import PinholeCamera
from .textures import sample_bilinear

#: Triangles are clipped against this depth before projection.  Not zero: a
#: vertex at z=1e-12 projects to coordinates around 1e12 and turns the bounding
#: box computation into an out-of-memory error.
NEAR = 1e-3


@dataclass
class RenderResult:
    """What one call to :meth:`Rasteriser.render` produces.

    ``depth`` is camera-frame *z*, not range along the ray, because that is
    what the plane-sweep and the fusion both want.
    """

    image: np.ndarray       #: (H, W) uint8 greyscale
    depth: np.ndarray       #: (H, W) float32, +inf where nothing was hit
    mask: np.ndarray        #: (H, W) int16 index into the mesh list, -1 for background

    @property
    def valid(self):
        return np.isfinite(self.depth)


class Rasteriser:
    """Renders a list of :class:`TriMesh` through a :class:`PinholeCamera`."""

    def __init__(self, textures, light_dir=(0.35, 0.45, -0.82), ambient=0.55,
                 diffuse=0.45, exposure=1.0):
        self.textures = [np.asarray(t, dtype=np.float32) for t in textures]
        d = np.asarray(light_dir, dtype=float)
        self.light_dir = d / np.linalg.norm(d)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.exposure = float(exposure)
        # The distortion maps depend only on the camera, and building them
        # means running the fixed-point undistortion over every pixel of the
        # frame.  Uncached that was forty per cent of the cost of a render, and
        # a capture renders the same four cameras a few hundred times.
        self._distortion_cache = {}

    # -- public ------------------------------------------------------------

    def render(self, camera, T_cam_world, meshes, apply_distortion=True,
               noise_sigma=0.0, rng=None, supersample=1):
        """Render ``meshes`` seen from ``T_cam_world``.

        ``supersample`` renders at N times the linear resolution and averages
        down, which is the cheapest honest antialiasing available and matters
        for the calibration board: an aliased checker corner has a sub-pixel
        bias that shows up directly in the calibration residual.
        """
        meshes = [m for m in meshes if len(m.faces)]
        if not meshes:
            raise ValueError("nothing to render")

        ss = max(1, int(supersample))
        distorting = apply_distortion and bool(np.any(camera.dist))

        if distorting:
            pinhole, map_x, map_y = self._distortion_setup(camera)
        else:
            # Strip the coefficients rather than trusting the caller: rasterising
            # through a distorting projection bends the edges of every triangle
            # and is exactly the mistake the module docstring warns about.
            pinhole = camera.with_params(np.concatenate([camera.params[:4], np.zeros(5)]))
            map_x = map_y = None

        work = pinhole.scaled(ss) if ss > 1 else pinhole
        colour, depth, mask = self._rasterise(work, T_cam_world, meshes)

        if ss > 1:
            colour = cv2.resize(
                colour, (pinhole.width, pinhole.height), interpolation=cv2.INTER_AREA
            )
            # Depth and mask are nearest-sampled: averaging a depth across a
            # silhouette would invent a surface floating between two objects.
            depth = depth[ss // 2::ss, ss // 2::ss][: pinhole.height, : pinhole.width]
            mask = mask[ss // 2::ss, ss // 2::ss][: pinhole.height, : pinhole.width]

        if distorting:
            colour = cv2.remap(
                colour, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            depth = cv2.remap(
                depth, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.inf,
            )
            mask = cv2.remap(
                mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0,
            ).astype(np.int16)

        colour = colour * self.exposure
        if noise_sigma > 0:
            if rng is None:
                raise ValueError("noise_sigma requires an rng")
            colour = colour + rng.normal(scale=noise_sigma, size=colour.shape)
        image = np.clip(colour * 255.0, 0, 255).astype(np.uint8)
        return RenderResult(image=image, depth=depth.astype(np.float32), mask=mask)

    # -- internals ---------------------------------------------------------

    def _distortion_setup(self, camera):
        """A padded pinhole camera plus the maps that warp it into ``camera``.

        ``cv2.remap`` reads source pixel ``(map_x, map_y)`` for each
        destination pixel.  The destination is the *measured*, distorted image
        and the source is the ideal pinhole render, so the maps are
        :meth:`PinholeCamera.distort_maps` -- the undistortion of each output
        pixel -- shifted by however far outside the nominal image they reach.

        Using ``undistort_maps`` here instead is the obvious mistake and it is
        nearly silent: it produces an image with the distortion applied in the
        wrong direction, which still looks like a plausible lens, still
        calibrates to a self-consistent model, and disagrees with the ground
        truth by roughly twice the true distortion.
        """
        key = (tuple(np.round(camera.params, 12)), camera.width, camera.height)
        hit = self._distortion_cache.get(key)
        if hit is not None:
            return hit

        map_x, map_y = camera.distort_maps()
        x0 = int(np.floor(min(map_x.min(), 0.0))) - 1
        y0 = int(np.floor(min(map_y.min(), 0.0))) - 1
        x1 = int(np.ceil(max(map_x.max(), camera.width - 1.0))) + 1
        y1 = int(np.ceil(max(map_y.max(), camera.height - 1.0))) + 1

        pinhole = PinholeCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx - x0, cy=camera.cy - y0,
            width=x1 - x0 + 1, height=y1 - y0 + 1,
            dist=np.zeros(5), name=camera.name,
        )
        out = (pinhole, (map_x - x0).astype(np.float32), (map_y - y0).astype(np.float32))
        self._distortion_cache[key] = out
        return out

    def _rasterise(self, camera, T_cam_world, meshes):
        H, W = camera.height, camera.width
        colour = np.zeros((H, W), dtype=np.float32)
        depth = np.full((H, W), np.inf, dtype=np.float32)
        mask = np.full((H, W), -1, dtype=np.int16)

        R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
        light_cam = R @ self.light_dir
        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy

        for mesh_index, mesh in enumerate(meshes):
            verts_cam = mesh.vertices @ R.T + t
            uv = mesh.uv                            # (F, 3, 2)
            normals_cam = mesh.face_normals @ R.T   # (F, 3)

            # Every vertex is projected once, vectorised, rather than three at
            # a time inside the triangle loop.  On a 600-triangle scene that
            # alone is most of the Python overhead.
            z = verts_cam[:, 2]
            safe_z = np.where(z > NEAR, z, 1.0)
            px_all = np.empty((len(verts_cam), 2), dtype=np.float64)
            px_all[:, 0] = fx * verts_cam[:, 0] / safe_z + cx
            px_all[:, 1] = fy * verts_cam[:, 1] / safe_z + cy

            # Flat Lambertian.  abs() rather than clamp-to-zero: the meshes are
            # built from primitives whose winding is not guaranteed consistent
            # after merging, and a two-sided light is a better approximation
            # than a face that goes black because its normal is inside out.
            lambert = self.ambient + self.diffuse * np.abs(normals_cam @ light_cam)

            faces = mesh.faces
            in_front = z > NEAR
            face_in_front = in_front[faces]
            n_in = face_in_front.sum(axis=1)

            for f in np.nonzero(n_in > 0)[0]:
                idx = faces[f]
                if n_in[f] == 3:
                    self._fill(
                        camera, colour, depth, mask, verts_cam[idx], px_all[idx], uv[f],
                        float(lambert[f]), int(mesh.texture_id[f]), mesh_index,
                    )
                    continue
                # Straddles the near plane: clip, then project the pieces.
                for v3, uv3 in _clip_near(verts_cam[idx], uv[f]):
                    p3 = np.empty((3, 2))
                    p3[:, 0] = fx * v3[:, 0] / v3[:, 2] + cx
                    p3[:, 1] = fy * v3[:, 1] / v3[:, 2] + cy
                    self._fill(
                        camera, colour, depth, mask, v3, p3, uv3,
                        float(lambert[f]), int(mesh.texture_id[f]), mesh_index,
                    )
        return colour, depth, mask

    def _fill(self, camera, colour, depth, mask, verts_cam, px, uv, shade, texture_id,
              mesh_index):
        """Rasterise one triangle, already projected, into the buffers."""
        if not np.all(np.isfinite(px)):
            return
        H, W = camera.height, camera.width

        x_lo = max(int(np.floor(px[:, 0].min())), 0)
        x_hi = min(int(np.ceil(px[:, 0].max())) + 1, W)
        y_lo = max(int(np.floor(px[:, 1].min())), 0)
        y_hi = min(int(np.ceil(px[:, 1].max())) + 1, H)
        if x_hi <= x_lo or y_hi <= y_lo:
            return

        (ax, ay), (bx, by), (cx, cy) = px
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-12:
            return  # degenerate on screen

        # Broadcast a row vector against a column vector rather than
        # materialising a full coordinate grid: same arithmetic, a third of the
        # memory traffic, and the big background triangles are memory bound.
        xs = np.arange(x_lo, x_hi, dtype=np.float32)[None, :]
        ys = np.arange(y_lo, y_hi, dtype=np.float32)[:, None]
        inv_denom = np.float32(1.0 / denom)
        l0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) * inv_denom
        l1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) * inv_denom
        l2 = np.float32(1.0) - l0 - l1
        # A half-pixel of slack closes the hairline cracks between adjacent
        # triangles that exact edge tests leave behind.
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not inside.any():
            return

        l0, l1, l2 = l0[inside], l1[inside], l2[inside]
        inv_z = np.stack([1.0 / verts_cam[i, 2] for i in range(3)])
        w = l0 * inv_z[0] + l1 * inv_z[1] + l2 * inv_z[2]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        z = 1.0 / w

        sub_depth = depth[y_lo:y_hi, x_lo:x_hi]
        closer = z < sub_depth[inside]
        if not closer.any():
            return

        # Perspective-correct UV: interpolate uv/z linearly, then multiply by z.
        uvz = (
            l0[:, None] * (uv[0] * inv_z[0])
            + l1[:, None] * (uv[1] * inv_z[1])
            + l2[:, None] * (uv[2] * inv_z[2])
        ) * z[:, None]

        texture = self.textures[texture_id % len(self.textures)]
        value = sample_bilinear(texture, uvz[closer]) * shade

        idx = np.nonzero(inside)
        rows = idx[0][closer] + y_lo
        cols = idx[1][closer] + x_lo
        depth[rows, cols] = z[closer]
        colour[rows, cols] = value
        mask[rows, cols] = mesh_index


def _clip_near(verts_cam, uv):
    """Sutherland-Hodgman clip of one triangle against ``z >= NEAR``.

    Yields ``(vertices, uv)`` pairs -- zero, one or two triangles.  The ground
    plane in the scene extends behind every camera, and without this it
    projects to coordinates in the millions and takes the bounding box with it.
    """
    z = verts_cam[:, 2]
    inside = z > NEAR
    n_in = int(inside.sum())
    if n_in == 0:
        return
    if n_in == 3:
        yield verts_cam, uv
        return

    poly_v, poly_uv = [], []
    for i in range(3):
        j = (i + 1) % 3
        vi, vj = verts_cam[i], verts_cam[j]
        if inside[i]:
            poly_v.append(vi)
            poly_uv.append(uv[i])
        if inside[i] != inside[j]:
            a = (NEAR - vi[2]) / (vj[2] - vi[2])
            poly_v.append(vi + a * (vj - vi))
            poly_uv.append(uv[i] + a * (uv[j] - uv[i]))

    poly_v = np.asarray(poly_v)
    poly_uv = np.asarray(poly_uv)
    for k in range(1, len(poly_v) - 1):
        yield (
            np.stack([poly_v[0], poly_v[k], poly_v[k + 1]]),
            np.stack([poly_uv[0], poly_uv[k], poly_uv[k + 1]]),
        )
