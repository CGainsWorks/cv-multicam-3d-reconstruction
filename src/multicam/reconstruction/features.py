"""Feature detection, description and matching.

SIFT, through OpenCV.  Reimplementing a scale-space detector would demonstrate
nothing that the bundle adjuster does not demonstrate better, and OpenCV's is
correct.  What is written here is everything around it: the ratio test, the
mutual-consistency check and -- the part that matters on a *calibrated* rig --
the epipolar gate.

The epipolar gate is the point of having a calibrated rig at all.  Two cameras
whose relative pose is known constrain a correspondence to a line, so a match
that is off that line by more than the calibration residual is wrong however
good its descriptor distance was.  On this data it removes roughly a third of
the matches that survive the ratio test, and almost all of those are on the
repetitive wall texture, where the descriptor genuinely cannot tell one tile
from another and only the geometry can.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import se3, so3


@dataclass
class FeatureSet:
    """Keypoints and descriptors for one image."""

    xy: np.ndarray              #: (N, 2) pixel coordinates, distorted (as measured)
    descriptors: np.ndarray     #: (N, 128) float32
    scale: np.ndarray           #: (N,) keypoint size
    angle: np.ndarray           #: (N,) degrees
    response: np.ndarray        #: (N,) detector response
    camera: int = -1
    frame: int = -1

    def __len__(self):
        return len(self.xy)

    def subset(self, index):
        index = np.asarray(index, dtype=int)
        return FeatureSet(
            xy=self.xy[index], descriptors=self.descriptors[index],
            scale=self.scale[index], angle=self.angle[index],
            response=self.response[index], camera=self.camera, frame=self.frame,
        )


def make_detector(n_features=0, contrast_threshold=0.025, edge_threshold=12.0,
                  n_octave_layers=3, sigma=1.6):
    """A SIFT detector tuned for this data.

    ``contrast_threshold`` is below OpenCV's 0.04 default on purpose.  The part
    occupies about three per cent of the frame and is a machined surface rather
    than a poster; at the default threshold the detector spends its budget on
    the wall and returns a few dozen features on the object, which is not
    enough to reconstruct it.  ``n_features=0`` means no cap, so the ranking
    never gets a chance to prefer the background either.
    """
    return cv2.SIFT_create(
        nfeatures=int(n_features), nOctaveLayers=int(n_octave_layers),
        contrastThreshold=float(contrast_threshold), edgeThreshold=float(edge_threshold),
        sigma=float(sigma),
    )


def detect(image, detector=None, mask=None, camera=-1, frame=-1, root_sift=True):
    """Detect and describe.

    ``root_sift`` applies the Hellinger normalisation of Arandjelovic and
    Zisserman: L1-normalise the descriptor and take the square root, so that
    the Euclidean distance between two descriptors becomes the Hellinger
    distance between the original histograms.  It costs two lines, needs no
    retraining and measurably improves the match count on this data; there is
    no reason not to.
    """
    detector = detector or make_detector()
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kps, desc = detector.detectAndCompute(image, mask)
    if desc is None or not len(kps):
        return FeatureSet(
            xy=np.zeros((0, 2)), descriptors=np.zeros((0, 128), np.float32),
            scale=np.zeros(0), angle=np.zeros(0), response=np.zeros(0),
            camera=camera, frame=frame,
        )
    desc = np.asarray(desc, dtype=np.float32)
    if root_sift:
        desc = desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-7)
        desc = np.sqrt(desc)
    return FeatureSet(
        xy=np.array([k.pt for k in kps], dtype=np.float64),
        descriptors=desc,
        scale=np.array([k.size for k in kps], dtype=np.float64),
        angle=np.array([k.angle for k in kps], dtype=np.float64),
        response=np.array([k.response for k in kps], dtype=np.float64),
        camera=camera, frame=frame,
    )


def match_descriptors(a, b, ratio=0.78, mutual=True, max_distance=None):
    """Ratio-test matches between two descriptor sets.

    Returns ``(M, 2)`` index pairs.  ``mutual`` additionally requires that each
    match is the best one in both directions, which is cheap and removes the
    many-to-one matches that a repetitive texture produces.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or len(b) < 2:
        return np.zeros((0, 2), dtype=int)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    pairs = []
    for m in forward:
        if len(m) < 2:
            continue
        if m[0].distance < ratio * m[1].distance:
            if max_distance is None or m[0].distance < max_distance:
                pairs.append((m[0].queryIdx, m[0].trainIdx))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    pairs = np.array(pairs, dtype=int)

    if mutual:
        back = matcher.knnMatch(b, a, k=1)
        best = {m[0].queryIdx: m[0].trainIdx for m in back if len(m)}
        keep = np.array([best.get(j, -1) == i for i, j in pairs])
        pairs = pairs[keep]
    return pairs


def sampson_distance(camera_a, camera_b, T_b_a, xy_a, xy_b):
    """Sampson error in pixels for correspondences under a known relative pose.

    Working in normalised undistorted coordinates keeps the essential matrix
    honest; the result is scaled back by the focal length so the threshold the
    caller passes is in pixels and means what it says.
    """
    xa = camera_a.undistort(xy_a)
    xb = camera_b.undistort(xy_b)
    R, t = T_b_a[:3, :3], T_b_a[:3, 3]
    E = so3.hat(t) @ R

    ha = np.c_[xa, np.ones(len(xa))]
    hb = np.c_[xb, np.ones(len(xb))]
    Ex_a = ha @ E.T                    # epipolar lines in image b
    Et_x_b = hb @ E                    # epipolar lines in image a
    num = np.sum(hb * Ex_a, axis=1) ** 2
    den = Ex_a[:, 0] ** 2 + Ex_a[:, 1] ** 2 + Et_x_b[:, 0] ** 2 + Et_x_b[:, 1] ** 2
    d = np.sqrt(num / np.maximum(den, 1e-20))
    focal = 0.5 * (camera_a.fx + camera_b.fx)
    return d * focal


def epipolar_filter(camera_a, camera_b, T_cam_a, T_cam_b, xy_a, xy_b, threshold_px=2.0):
    """Boolean mask of matches consistent with a known relative pose.

    Degenerate when the two cameras are at the same centre -- the essential
    matrix is then zero and every match passes -- so that case returns
    everything and says so through the returned baseline.
    """
    T_b_a = T_cam_b @ se3.inverse(T_cam_a)
    baseline = float(np.linalg.norm(T_b_a[:3, 3]))
    if baseline < 1e-6:
        return np.ones(len(xy_a), dtype=bool), baseline
    d = sampson_distance(camera_a, camera_b, T_b_a, xy_a, xy_b)
    return d < float(threshold_px), baseline


def match_views(features_a, features_b, camera_a=None, camera_b=None,
                T_cam_a=None, T_cam_b=None, ratio=0.78, epipolar_px=2.0):
    """Ratio-tested, mutually-consistent, optionally epipolar-gated matches.

    Returns ``(pairs, stats)``.
    """
    pairs = match_descriptors(features_a.descriptors, features_b.descriptors, ratio=ratio)
    stats = {"descriptor": int(len(pairs)), "epipolar": int(len(pairs)), "baseline_m": None}
    if len(pairs) and camera_a is not None and T_cam_a is not None:
        keep, baseline = epipolar_filter(
            camera_a, camera_b, T_cam_a, T_cam_b,
            features_a.xy[pairs[:, 0]], features_b.xy[pairs[:, 1]],
            threshold_px=epipolar_px,
        )
        pairs = pairs[keep]
        stats["epipolar"] = int(len(pairs))
        stats["baseline_m"] = baseline
    return pairs, stats
