"""Plane-sweep multi-view stereo.

For each reference camera, hypothesise a depth for every pixel, warp the other
cameras into the reference view under that hypothesis, and score the agreement
with zero-mean normalised cross-correlation.  The depth that scores best wins.

Three things make it practical rather than merely correct:

**The sweep is bounded by the sparse reconstruction.**  The depth range for a
pixel is where its ray enters and leaves the bounding box of the sparse point
cloud, not some global near and far plane.  That is a factor of ten fewer
hypotheses per pixel, and -- more importantly -- it is what removes the
background.  A pixel whose ray never enters the box is not swept at all.  No
segmentation is involved and no ground truth: the sparse reconstruction already
found where the part is, so the dense stage is told.

**The sweep is uniform in inverse depth**, not in depth.  Depth resolution from
a stereo pair is uniform in disparity, so uniform depth sampling wastes
hypotheses near the camera and under-samples far away.

**ZNCC is computed with box filters over the whole image at once.**  A per-pixel
window loop in Python would be minutes per depth map; as a difference of box
filters it is milliseconds per hypothesis.  Zero-mean and normalised so that
the two cameras' different exposures and the shading difference between two
viewpoints do not count against a correct match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import se3


@dataclass
class DepthMap:
    """A depth map in one camera's frame, with the confidence that produced it."""

    station: int
    camera: int
    depth: np.ndarray            #: (H, W) float32, NaN where no hypothesis won
    score: np.ndarray            #: (H, W) float32, best ZNCC
    valid: np.ndarray            #: (H, W) bool
    camera_model: object = None
    T_cam_object: np.ndarray = None
    stats: dict = field(default_factory=dict)

    def points(self, subsample=1):
        """Unproject the valid pixels into the object frame."""
        v, u = np.nonzero(self.valid)
        if subsample > 1:
            v, u = v[::subsample], u[::subsample]
        if not len(v):
            return np.zeros((0, 3)), np.zeros(0)
        d = self.depth[v, u]
        xy = self.camera_model.undistort(np.stack([u, v], axis=1).astype(float))
        pts_cam = np.stack([xy[:, 0] * d, xy[:, 1] * d, d], axis=1)
        return se3.transform(se3.inverse(self.T_cam_object), pts_cam), self.score[v, u]


def _ray_box_range(camera, T_cam_object, lo, hi, shape):
    """Per-pixel depth interval where the ray is inside an object-frame box.

    Standard slab intersection, done in the object frame with the ray written
    as ``origin + t * direction`` and then converted back to camera-frame *z*,
    which is what the sweep parameterises.
    """
    h, w = shape
    v, u = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64),
                       indexing="ij")
    uv = np.stack([u.ravel(), v.ravel()], axis=1)
    xy = camera.undistort(uv)
    # Direction with z = 1, so the parameter t is exactly camera-frame depth.
    dir_cam = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)

    T_object_cam = se3.inverse(T_cam_object)
    R = T_object_cam[:3, :3]
    origin = T_object_cam[:3, 3]
    dirs = dir_cam @ R.T

    t_near = np.full(len(dirs), -np.inf)
    t_far = np.full(len(dirs), np.inf)
    for axis in range(3):
        d = dirs[:, axis]
        o = origin[axis]
        parallel = np.abs(d) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (lo[axis] - o) / d
            t1 = (hi[axis] - o) / d
        lo_t = np.minimum(t0, t1)
        hi_t = np.maximum(t0, t1)
        # A ray parallel to a slab either misses it entirely or is unconstrained
        # by it; encode the miss as an empty interval.
        outside = parallel & ((o < lo[axis]) | (o > hi[axis]))
        lo_t = np.where(parallel, -np.inf, lo_t)
        hi_t = np.where(parallel, np.inf, hi_t)
        lo_t = np.where(outside, 1.0, lo_t)
        hi_t = np.where(outside, -1.0, hi_t)
        t_near = np.maximum(t_near, lo_t)
        t_far = np.minimum(t_far, hi_t)

    hit = (t_far > t_near) & (t_far > 0)
    t_near = np.maximum(t_near, 1e-3)
    return (
        t_near.reshape(h, w).astype(np.float32),
        t_far.reshape(h, w).astype(np.float32),
        hit.reshape(h, w),
    )


def _zncc(a, b, window):
    """ZNCC of two images over a square window, via box filters."""
    k = (int(window), int(window))
    mean_a = cv2.boxFilter(a, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_aa = cv2.boxFilter(a * a, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_bb = cv2.boxFilter(b * b, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_ab = cv2.boxFilter(a * b, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)

    cov = mean_ab - mean_a * mean_b
    var_a = np.maximum(mean_aa - mean_a * mean_a, 0.0)
    var_b = np.maximum(mean_bb - mean_b * mean_b, 0.0)
    # A window with no texture in either image has zero variance and no
    # meaningful correlation; scoring it zero rather than dividing by zero
    # means it never wins a hypothesis.
    denom = np.sqrt(var_a * var_b)
    out = np.where(denom > 1e-8, cov / np.maximum(denom, 1e-12), 0.0)
    return out.astype(np.float32)


def sweep(reference, sources, bounds, n_depths=96, window=7, min_score=0.55,
          min_views=2, scale=1.0, sub_pixel=True):
    """Plane-sweep MVS for one reference view.

    Parameters
    ----------
    reference : (image, camera, T_cam_object, station, index)
    sources : list of (image, camera, T_cam_object)
    bounds : (lo, hi)
        Axis-aligned bounds of the object, in the object frame, in metres.
    min_views : int
        How many source views must agree, at the winning depth, for a pixel to
        be kept.  Two is the minimum that means anything: a single source view
        can be fooled by a repeated texture along the epipolar line, and two
        views at different angles rarely are.
    """
    image, camera, T_cam_object, station, index = reference
    if scale != 1.0:
        camera = camera.scaled(scale)
        image = cv2.resize(image, (camera.width, camera.height),
                           interpolation=cv2.INTER_AREA)
        sources = [
            (cv2.resize(im, (cam.scaled(scale).width, cam.scaled(scale).height),
                        interpolation=cv2.INTER_AREA), cam.scaled(scale), T)
            for im, cam, T in sources
        ]

    ref = image.astype(np.float32) / 255.0
    h, w = ref.shape
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    t_near, t_far, hit = _ray_box_range(camera, T_cam_object, lo, hi, (h, w))

    if not hit.any():
        return DepthMap(station, index, np.full((h, w), np.nan, np.float32),
                        np.zeros((h, w), np.float32), np.zeros((h, w), bool),
                        camera, T_cam_object, {"reason": "no ray meets the object bounds"})

    # Pixel rays, in the reference camera frame, with unit z.
    v_grid, u_grid = np.meshgrid(np.arange(h, dtype=np.float64),
                                 np.arange(w, dtype=np.float64), indexing="ij")
    xy = camera.undistort(np.stack([u_grid.ravel(), v_grid.ravel()], axis=1))
    dir_cam = np.concatenate([xy, np.ones((len(xy), 1))], axis=1).reshape(h, w, 3)

    support = np.zeros((h, w), np.int16)

    # Uniform in inverse depth between each pixel's own near and far.
    inv_near = 1.0 / np.maximum(t_near, 1e-4)
    inv_far = 1.0 / np.maximum(t_far, 1e-4)
    alphas = np.linspace(0.0, 1.0, int(n_depths), dtype=np.float32)

    # The whole cost volume is kept.  It is the only way to tell a genuine
    # peak from a plateau, and at the resolutions this runs at it is tens of
    # megabytes.
    score_stack = np.empty((int(n_depths), h, w), np.float32)
    depth_stack = np.empty((int(n_depths), h, w), np.float32)

    for k, alpha in enumerate(alphas):
        inv_d = inv_near * (1.0 - alpha) + inv_far * alpha
        depth = (1.0 / np.maximum(inv_d, 1e-9)).astype(np.float32)
        depth_stack[k] = depth
        pts_cam = dir_cam * depth[:, :, None]

        scores, count = np.zeros((h, w), np.float32), np.zeros((h, w), np.int16)
        for src_image, src_camera, T_src in sources:
            T_src_ref = T_src @ se3.inverse(T_cam_object)
            pts_src = pts_cam @ T_src_ref[:3, :3].T + T_src_ref[:3, 3]
            uv, ok = src_camera.project(pts_src.reshape(-1, 3), return_valid=True)
            map_x = uv[:, 0].reshape(h, w).astype(np.float32)
            map_y = uv[:, 1].reshape(h, w).astype(np.float32)
            inside = (
                ok.reshape(h, w)
                & (map_x >= 0) & (map_x < src_camera.width - 1)
                & (map_y >= 0) & (map_y < src_camera.height - 1)
            )
            warped = cv2.remap(
                src_image.astype(np.float32) / 255.0, np.nan_to_num(map_x),
                np.nan_to_num(map_y), interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            s = _zncc(ref, warped, window)
            s = np.where(inside, s, 0.0)
            scores += s
            count += inside.astype(np.int16)

        # Average over the views that actually saw the hypothesis, so a pixel
        # visible in one camera is not penalised against one visible in three.
        avg = np.where(count > 0, scores / np.maximum(count, 1), -2.0).astype(np.float32)
        avg = np.where(count >= min_views, avg, -2.0).astype(np.float32)
        score_stack[k] = avg
        support = np.maximum(support, count)

    rows, cols = np.mgrid[0:h, 0:w]
    arg = np.argmax(score_stack, axis=0)
    best = score_stack[arg, rows, cols]
    best_depth = depth_stack[arg, rows, cols]

    # Ambiguity rejection, done properly.  The runner-up has to be a *different
    # peak*, not the hypothesis next door: a cost volume is smooth, so the
    # depths either side of the winner always score almost as well, and
    # comparing against them rejects every pixel in the image.  That is exactly
    # what an earlier version of this did -- it returned four valid pixels out
    # of twenty thousand and looked like a matching failure.
    exclusion = max(2, int(n_depths) // 32)
    masked = score_stack.copy()
    lo_k = np.maximum(arg - exclusion, 0)
    hi_k = np.minimum(arg + exclusion, int(n_depths) - 1)
    depth_index = np.arange(int(n_depths))[:, None, None]
    masked[(depth_index >= lo_k[None]) & (depth_index <= hi_k[None])] = -2.0
    second = masked.max(axis=0)

    margin = best - second
    ambiguous = margin < 0.02
    valid = (
        hit & (best >= min_score) & (support >= min_views)
        & np.isfinite(best_depth) & ~ambiguous
    )

    if sub_pixel:
        # Parabola through the winning hypothesis and its two neighbours, in
        # inverse depth, which is the variable the sweep is uniform in.
        k0 = np.clip(arg, 1, int(n_depths) - 2)
        s_minus = score_stack[k0 - 1, rows, cols]
        s_zero = score_stack[k0, rows, cols]
        s_plus = score_stack[k0 + 1, rows, cols]
        denom = s_minus - 2.0 * s_zero + s_plus
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 0.5 * (s_minus - s_plus) / denom
        shift = np.clip(np.where(np.abs(denom) > 1e-6, raw, 0.0), -0.5, 0.5)
        inv_minus = 1.0 / np.maximum(depth_stack[k0 - 1, rows, cols], 1e-9)
        inv_plus = 1.0 / np.maximum(depth_stack[k0 + 1, rows, cols], 1e-9)
        inv_zero = 1.0 / np.maximum(depth_stack[k0, rows, cols], 1e-9)
        step = 0.5 * (inv_plus - inv_minus)
        refined = 1.0 / np.maximum(inv_zero + shift * step, 1e-9)
        # Only where the parabola really is a maximum, otherwise keep the
        # discrete pick.
        use = (denom < 0) & (arg > 0) & (arg < int(n_depths) - 1)
        best_depth = np.where(use, refined.astype(np.float32), best_depth)

    stats = {
        "pixels": int(hit.sum()),
        "valid": int(valid.sum()),
        "rejected_score": int((hit & (best < min_score)).sum()),
        "rejected_support": int((hit & (support < min_views)).sum()),
        "rejected_ambiguous": int((hit & ambiguous).sum()),
        "mean_score": float(best[valid].mean()) if valid.any() else float("nan"),
        "depths": int(n_depths),
        "window": int(window),
        "scale": float(scale),
    }
    depth_out = np.where(valid, best_depth, np.nan).astype(np.float32)
    return DepthMap(station, index, depth_out, best.astype(np.float32), valid,
                    camera, T_cam_object, stats)


def cross_check(depth_maps, tolerance_m=0.0015, min_agree=1):
    """Keep only depths that other views independently agree with.

    Each valid pixel is unprojected, projected into every other depth map, and
    kept if at least ``min_agree`` of them report a depth within
    ``tolerance_m`` at that pixel.  This is the standard left-right check
    generalised to N views, and it removes the streaks that a plane sweep
    leaves along occlusion boundaries, where the winning hypothesis is
    consistent with the reference image and with nothing else.
    """
    out = []
    for dm in depth_maps:
        if not dm.valid.any():
            out.append(dm)
            continue
        pts_object, _ = dm.points()
        v, u = np.nonzero(dm.valid)
        agree = np.zeros(len(v), dtype=np.int16)
        for other in depth_maps:
            if other is dm or not other.valid.any():
                continue
            pts_cam = se3.transform(other.T_cam_object, pts_object)
            uv, ok = other.camera_model.project(pts_cam, return_valid=True)
            uu = np.round(uv[:, 0]).astype(int)
            vv = np.round(uv[:, 1]).astype(int)
            inside = (
                ok & (uu >= 0) & (uu < other.camera_model.width)
                & (vv >= 0) & (vv < other.camera_model.height)
            )
            if not inside.any():
                continue
            probe = np.zeros(len(v), dtype=bool)
            idx = np.nonzero(inside)[0]
            other_depth = other.depth[vv[idx], uu[idx]]
            probe[idx] = np.abs(other_depth - pts_cam[idx, 2]) < tolerance_m
            agree += probe.astype(np.int16)

        keep = agree >= min_agree
        valid = np.zeros_like(dm.valid)
        valid[v[keep], u[keep]] = True
        stats = dict(dm.stats)
        stats["cross_checked"] = int(valid.sum())
        stats["cross_check_removed"] = int(dm.valid.sum() - valid.sum())
        out.append(
            DepthMap(dm.station, dm.camera, np.where(valid, dm.depth, np.nan).astype(np.float32),
                     dm.score, valid, dm.camera_model, dm.T_cam_object, stats)
        )
    return out
