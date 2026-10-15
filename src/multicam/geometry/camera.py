"""The pinhole camera and its distortion model, with analytic Jacobians.

The projection model is OpenCV's: a pinhole with a five-parameter
radial-tangential distortion ``(k1, k2, p1, p2, k3)``.  Choosing OpenCV's exact
convention rather than a private one means the intrinsics this repository
estimates can be handed straight to ``cv2.undistort``, to COLMAP's
``OPENCV`` camera model and back, without a translation layer that could quietly
transpose something.

Both Jacobians -- with respect to the 3D point and with respect to the nine
intrinsic parameters -- are derived by hand here and checked against central
finite differences in ``tests/test_geometry.py``.  That test is the reason to
trust the bundle adjuster: a solver with a subtly wrong Jacobian still
converges on easy problems and quietly fails on hard ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Order of the distortion vector, matching ``cv2``'s ``distCoeffs``.
DIST_ORDER = ("k1", "k2", "p1", "p2", "k3")

#: Order of the full intrinsic parameter vector used by the calibration
#: bundle adjustment.  Fixed here so every caller agrees.
PARAM_ORDER = ("fx", "fy", "cx", "cy") + DIST_ORDER


@dataclass
class PinholeCamera:
    """Intrinsics for one camera.

    ``width`` and ``height`` are carried along because the renderer, the
    undistortion map and the plane-sweep all need them, and a calibration file
    without an image size is a calibration file you cannot use.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    name: str = "cam"

    def __post_init__(self):
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)
        if self.dist.size < 5:
            self.dist = np.pad(self.dist, (0, 5 - self.dist.size))
        elif self.dist.size > 5:
            raise ValueError("only the 5-parameter radtan model is supported")
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.width = int(self.width)
        self.height = int(self.height)

    # -- basic accessors ---------------------------------------------------

    @property
    def K(self):
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    @property
    def params(self):
        """The nine free parameters, in :data:`PARAM_ORDER`."""
        return np.concatenate([[self.fx, self.fy, self.cx, self.cy], self.dist])

    @classmethod
    def from_params(cls, params, width, height, name="cam"):
        p = np.asarray(params, dtype=float).reshape(9)
        return cls(p[0], p[1], p[2], p[3], width, height, p[4:], name)

    def with_params(self, params):
        return PinholeCamera.from_params(params, self.width, self.height, self.name)

    def to_dict(self):
        return {
            "name": self.name,
            "model": "OPENCV_RADTAN5",
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist": [float(v) for v in self.dist],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            fx=d["fx"],
            fy=d["fy"],
            cx=d["cx"],
            cy=d["cy"],
            width=d["width"],
            height=d["height"],
            dist=np.asarray(d.get("dist", np.zeros(5)), dtype=float),
            name=d.get("name", "cam"),
        )

    # -- projection --------------------------------------------------------

    def distort_normalised(self, xy):
        """Apply the radtan model to normalised image coordinates."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        x, y = xy[:, 0], xy[:, 1]
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([xd, yd], axis=1)

    def project(self, points_cam, return_valid=False):
        """Project camera-frame points to pixels.

        Points at or behind the principal plane have no projection.  Rather
        than emit a NaN that propagates silently, they are marked invalid and
        the caller decides; ``return_valid=True`` asks for that mask.
        """
        P = np.asarray(points_cam, dtype=float)
        single = P.ndim == 1
        P = np.atleast_2d(P)
        z = P[:, 2]
        valid = z > 1e-9
        # Substitute a harmless z for the invalid rows so the arithmetic below
        # does not warn; those rows are masked out afterwards.
        z_safe = np.where(valid, z, 1.0)
        xy = np.stack([P[:, 0] / z_safe, P[:, 1] / z_safe], axis=1)
        d = self.distort_normalised(xy)
        uv = np.stack([self.fx * d[:, 0] + self.cx, self.fy * d[:, 1] + self.cy], axis=1)
        uv[~valid] = np.nan
        if single:
            return (uv[0], bool(valid[0])) if return_valid else uv[0]
        return (uv, valid) if return_valid else uv

    def in_image(self, uv, margin=0.0):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 0] < self.width - margin)
            & (uv[:, 1] < self.height - margin)
        )

    # -- Jacobians ---------------------------------------------------------

    def _distortion_jacobian(self, x, y):
        """``d(xd, yd) / d(x, y)`` at one or many normalised points."""
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        # d(radial)/d(r2), then chain through d(r2)/dx = 2x.
        drad_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 * r2
        drad_dx = drad_dr2 * 2.0 * x
        drad_dy = drad_dr2 * 2.0 * y

        J = np.empty(x.shape + (2, 2), dtype=float)
        J[..., 0, 0] = radial + x * drad_dx + 2.0 * p1 * y + 6.0 * p2 * x
        J[..., 0, 1] = x * drad_dy + 2.0 * p1 * x + 2.0 * p2 * y
        J[..., 1, 0] = y * drad_dx + 2.0 * p1 * x + 2.0 * p2 * y
        J[..., 1, 1] = radial + y * drad_dy + 6.0 * p1 * y + 2.0 * p2 * x
        return J

    def project_jacobian_point(self, points_cam):
        """``d(u, v) / d(X, Y, Z)``, shape ``(N, 2, 3)``."""
        P = np.atleast_2d(np.asarray(points_cam, dtype=float))
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        inv_z = 1.0 / Z
        x, y = X * inv_z, Y * inv_z

        # d(x, y) / d(X, Y, Z)
        J_norm = np.zeros((P.shape[0], 2, 3), dtype=float)
        J_norm[:, 0, 0] = inv_z
        J_norm[:, 0, 2] = -x * inv_z
        J_norm[:, 1, 1] = inv_z
        J_norm[:, 1, 2] = -y * inv_z

        J_dist = self._distortion_jacobian(x, y)
        F = np.zeros((P.shape[0], 2, 2), dtype=float)
        F[:, 0, 0] = self.fx
        F[:, 1, 1] = self.fy
        return F @ J_dist @ J_norm

    def project_jacobian_params(self, points_cam):
        """``d(u, v) / d(params)``, shape ``(N, 2, 9)``, in :data:`PARAM_ORDER`."""
        P = np.atleast_2d(np.asarray(points_cam, dtype=float))
        inv_z = 1.0 / P[:, 2]
        x, y = P[:, 0] * inv_z, P[:, 1] * inv_z
        r2 = x * x + y * y
        d = self.distort_normalised(np.stack([x, y], axis=1))

        J = np.zeros((P.shape[0], 2, 9), dtype=float)
        J[:, 0, 0] = d[:, 0]          # du/dfx
        J[:, 1, 1] = d[:, 1]          # dv/dfy
        J[:, 0, 2] = 1.0              # du/dcx
        J[:, 1, 3] = 1.0              # dv/dcy
        # Distortion terms: the derivative of the normalised distorted point
        # scaled by the focal length.
        J[:, 0, 4] = self.fx * x * r2
        J[:, 0, 5] = self.fx * x * r2 * r2
        J[:, 0, 6] = self.fx * 2.0 * x * y
        J[:, 0, 7] = self.fx * (r2 + 2.0 * x * x)
        J[:, 0, 8] = self.fx * x * r2 * r2 * r2
        J[:, 1, 4] = self.fy * y * r2
        J[:, 1, 5] = self.fy * y * r2 * r2
        J[:, 1, 6] = self.fy * (r2 + 2.0 * y * y)
        J[:, 1, 7] = self.fy * 2.0 * x * y
        J[:, 1, 8] = self.fy * y * r2 * r2 * r2
        return J

    # -- inverse -----------------------------------------------------------

    def undistort(self, uv, iterations=20, tol=1e-12):
        """Pixels to *undistorted normalised* coordinates.

        The radtan model has no closed-form inverse, so this is the standard
        fixed-point iteration.  It is written out rather than delegated to
        ``cv2.undistortPoints`` so that the convergence behaviour is visible
        and so the function works on a camera whose parameters are mid-solve.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        k1, k2, p1, p2, k3 = self.dist
        if not np.any(self.dist):
            return np.stack([x, y], axis=1)
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x_new = (xd - dx) / radial
            y_new = (yd - dy) / radial
            step = max(float(np.max(np.abs(x_new - x))), float(np.max(np.abs(y_new - y))))
            x, y = x_new, y_new
            if step < tol:
                break
        return np.stack([x, y], axis=1)

    def rays(self, uv):
        """Unit direction vectors in the camera frame for the given pixels."""
        xy = self.undistort(uv)
        d = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def _pixel_grid(self, width=None, height=None):
        w = self.width if width is None else int(width)
        h = self.height if height is None else int(height)
        v, u = np.meshgrid(
            np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
        )
        return u, v, w, h

    def undistort_maps(self):
        """``(map_x, map_y)`` for ``cv2.remap`` **distorted image -> ideal image**.

        This is OpenCV's ``initUndistortRectifyMap`` direction: each pixel of
        the ideal output is told where to read from in the distorted input, and
        that address is the *forward* distortion of the output pixel's ray.
        The output camera keeps the same focal lengths and principal point, so
        an ideal pixel and a distorted one refer to the same ray and the
        intrinsics need no adjustment.

        See :meth:`distort_maps` for the other direction.  Confusing the two is
        easy, silent, and produces an image with exactly the wrong sign of
        distortion; the renderer needs the other one.
        """
        u, v, w, h = self._pixel_grid()
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d = self.distort_normalised(np.stack([x.ravel(), y.ravel()], axis=1))
        map_x = (self.fx * d[:, 0] + self.cx).reshape(h, w).astype(np.float32)
        map_y = (self.fy * d[:, 1] + self.cy).reshape(h, w).astype(np.float32)
        return map_x, map_y

    def distort_maps(self, width=None, height=None):
        """``(map_x, map_y)`` for ``cv2.remap`` **ideal image -> distorted image**.

        Each pixel of the distorted output is told where to read from in the
        ideal input.  The output pixel is a *measured* pixel, so the ray it
        corresponds to is found by **undistorting** it, and the ideal image
        coordinate of that ray is then the pinhole projection.

        With barrel distortion the addresses run outside the nominal image --
        the corners of the distorted frame see rays that an ideal camera of the
        same focal length would put further out still -- so the caller has to
        render onto a canvas big enough to contain the range this returns.
        """
        u, v, w, h = self._pixel_grid(width, height)
        uv = np.stack([u.ravel(), v.ravel()], axis=1)
        xy = self.undistort(uv)
        map_x = (self.fx * xy[:, 0] + self.cx).reshape(h, w).astype(np.float32)
        map_y = (self.fy * xy[:, 1] + self.cy).reshape(h, w).astype(np.float32)
        return map_x, map_y

    def cropped(self, x0, y0, width, height):
        """The same camera restricted to a window of its own image.

        Only the principal point moves.  Used by the pose refiner, which works
        on a crop around the object rather than the whole frame, and needs a
        camera that still projects correctly into that crop.
        """
        return PinholeCamera(
            fx=self.fx, fy=self.fy,
            cx=self.cx - float(x0), cy=self.cy - float(y0),
            width=int(width), height=int(height),
            dist=self.dist.copy(), name=self.name,
        )

    def scaled(self, factor):
        """The same camera looking at an image resized by ``factor``.

        Distortion coefficients are dimensionless in normalised coordinates and
        so are unchanged; only the focal length and principal point move.
        """
        return PinholeCamera(
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=(self.cx + 0.5) * factor - 0.5,
            cy=(self.cy + 0.5) * factor - 0.5,
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
            dist=self.dist.copy(),
            name=self.name,
        )
