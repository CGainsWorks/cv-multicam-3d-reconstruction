"""Render-and-compare inputs for the learned pose refiner.

The refiner is given, for each camera, a crop around where the part *should* be
under the current pose hypothesis, containing:

0. the observed image,
1. the rendered silhouette of the CAD model at the hypothesis,
2. the rendered depth at the hypothesis, normalised about the hypothesis's own
   centroid depth,
3. the gradient magnitude of the observed image.

What is deliberately **not** in there is any rendering of the part's
*appearance*.  The CAD model is a declared input to this problem -- the brief
asks for evaluation against known CAD -- so its geometry is fair game.  Its
appearance is not: a CAD model has none, and rendering the simulator's own
surface texture into the network's input would be handing it the answer key and
would not transfer to a real part for a moment.  So the refiner aligns
silhouette and depth against image structure, which is what it would have to do
on real hardware.

The crop is centred on the projected centroid of the hypothesis and scaled by
the projected size of the model, so the part occupies a consistent fraction of
the crop whatever its distance.  Without that normalisation the network has to
learn a size-to-depth mapping that the geometry already determines.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import PinholeCamera, se3
from ..sim.render import Rasteriser

#: Side of the square crop handed to the network, in pixels.
CROP = 96

#: How much bigger than the model's projected extent the crop is.  1.6 leaves
#: enough context that a hypothesis which is wrong by a few millimetres still
#: has the true silhouette inside the crop.
CROP_MARGIN = 1.6


@dataclass
class CropSpec:
    """Where a crop was taken from, so a prediction can be put back."""

    x0: float
    y0: float
    scale: float                 #: crop pixels per source pixel
    camera: int
    valid: bool = True


class ModelRenderer:
    """Renders the CAD model's silhouette and depth into pose-centred crops."""

    def __init__(self, mesh, crop=CROP, margin=CROP_MARGIN):
        self.mesh = mesh
        self.crop = int(crop)
        self.margin = float(margin)
        # A flat white texture: only the mask and depth are used, so shading
        # detail would be wasted work.
        self.rasteriser = Rasteriser([np.ones((4, 4), dtype=np.float32)],
                                     ambient=1.0, diffuse=0.0)
        lo, hi = mesh.bounds
        self.radius = 0.5 * float(np.linalg.norm(hi - lo))
        self.centre = mesh.centre

    def crop_spec(self, camera, T_cam_object, image_shape):
        """Where to crop, given a pose hypothesis.

        The window is in the camera's **undistorted** pixel space, because
        that is the space the crop camera is ideal in.
        """
        centre_cam = se3.transform(T_cam_object, self.centre)
        if centre_cam[2] < 1e-3:
            return CropSpec(0.0, 0.0, 1.0, -1, valid=False)
        # Ideal projection: no distortion, so the crop grid is a plain scaling
        # of the pinhole image plane.
        uv = np.array([
            camera.fx * centre_cam[0] / centre_cam[2] + camera.cx,
            camera.fy * centre_cam[1] / centre_cam[2] + camera.cy,
        ])
        if not np.all(np.isfinite(uv)):
            return CropSpec(0.0, 0.0, 1.0, -1, valid=False)
        radius_px = self.radius * camera.fx / centre_cam[2]
        half = max(radius_px * self.margin, 8.0)
        scale = self.crop / (2.0 * half)
        return CropSpec(float(uv[0] - half), float(uv[1] - half), float(scale), -1, True)

    def crop_camera(self, camera, spec):
        """An **ideal** camera that projects directly into the crop's pixel grid.

        The crop is defined in the source camera's *undistorted* pixel space
        and the observed image is undistorted as it is sampled, so the crop
        camera carries no distortion coefficients.  That keeps the rendered
        model and the observed image in the same geometry without the renderer
        having to warp a 96-pixel window through the full distortion machinery.

        Built explicitly rather than by composing ``cropped`` and ``scaled``:
        those two each change the image size, so composing them produced a
        camera of size ``crop * scale`` and the per-camera tensors then refused
        to stack.
        """
        return PinholeCamera(
            fx=camera.fx * spec.scale,
            fy=camera.fy * spec.scale,
            cx=(camera.cx - spec.x0) * spec.scale,
            cy=(camera.cy - spec.y0) * spec.scale,
            width=self.crop,
            height=self.crop,
            dist=np.zeros(5),
            name=camera.name,
        )

    def render(self, camera, T_cam_object, spec):
        """``(mask, depth)`` of the model at the hypothesis, in crop pixels."""
        if not spec.valid:
            z = np.zeros((self.crop, self.crop), dtype=np.float32)
            return z, np.full((self.crop, self.crop), np.nan, dtype=np.float32)
        crop_camera = self.crop_camera(camera, spec)
        mesh = self.mesh.transformed(T_cam_object)
        result = self.rasteriser.render(
            crop_camera, np.eye(4), [mesh], apply_distortion=False, supersample=1
        )
        mask = (result.mask >= 0).astype(np.float32)
        depth = np.where(np.isfinite(result.depth), result.depth, np.nan).astype(np.float32)
        return mask, depth

    def observed_crop(self, image, camera, spec):
        """The observed image, undistorted and resampled into the crop grid.

        Each crop pixel is turned into a ray by the ideal crop camera, that ray
        is distorted by the *source* camera's model, and the source image is
        sampled there.  Only the forward distortion is needed, so this is a
        closed-form map with no iteration.
        """
        if not spec.valid:
            return np.zeros((self.crop, self.crop), dtype=np.float32)
        crop_camera = self.crop_camera(camera, spec)
        v, u = np.meshgrid(np.arange(self.crop, dtype=np.float64),
                           np.arange(self.crop, dtype=np.float64), indexing="ij")
        x = (u.ravel() - crop_camera.cx) / crop_camera.fx
        y = (v.ravel() - crop_camera.cy) / crop_camera.fy
        d = camera.distort_normalised(np.stack([x, y], axis=1))
        map_x = (camera.fx * d[:, 0] + camera.cx).reshape(self.crop, self.crop)
        map_y = (camera.fy * d[:, 1] + camera.cy).reshape(self.crop, self.crop)
        out = cv2.remap(
            image, map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return out.astype(np.float32) / 255.0

    def channels(self, image, camera, T_cam_object):
        """The four input channels for one camera, plus the crop spec."""
        spec = self.crop_spec(camera, T_cam_object, image.shape)
        observed = self.observed_crop(image, camera, spec)
        mask, depth = self.render(camera, T_cam_object, spec)

        # Depth relative to the hypothesis's own centroid depth, in units of
        # the model radius.  Absolute depth would make the network's job depend
        # on where the part happens to be standing.
        centre_cam = se3.transform(T_cam_object, self.centre)
        z0 = float(centre_cam[2]) if centre_cam[2] > 1e-3 else 1.0
        rel = np.where(np.isfinite(depth), (depth - z0) / max(self.radius, 1e-6), 0.0)
        rel = np.clip(rel, -3.0, 3.0).astype(np.float32)

        gx = cv2.Sobel(observed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(observed, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = np.clip(grad * 2.0, 0.0, 1.0).astype(np.float32)

        return np.stack([observed, mask, rel, grad], axis=0), spec


def build_input(renderer, images, rig, T_rig_object, cameras=None):
    """Stack the per-camera channels for one pose hypothesis.

    Returns ``(tensor (C, 4, H, W), specs)``.  Cameras where the hypothesis
    falls outside the frame still contribute a slot, filled with zeros, so the
    tensor shape does not depend on the hypothesis -- which it must not, or a
    batch cannot be formed.
    """
    cameras = list(range(rig.n_cameras)) if cameras is None else list(cameras)
    stacks, specs = [], []
    for c in cameras:
        T_cam_object = rig.T_cam_rig[c] @ T_rig_object
        ch, spec = renderer.channels(images[c], rig.cameras[c], T_cam_object)
        spec.camera = c
        stacks.append(ch)
        specs.append(spec)
    return np.stack(stacks, axis=0).astype(np.float32), specs
