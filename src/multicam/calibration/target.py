"""The calibration target: a ChArUco board and its detector.

ChArUco rather than a plain checkerboard for one reason that matters on a
multi-camera rig: **partial views are usable**.  Four cameras on an arc cannot
all see the whole board at once, and a plain checkerboard detector returns all
corners or none.  ChArUco identifies each corner through the ArUco markers
around it, so a camera that sees half the board contributes that half, with its
corner indices known, and the extrinsic solve gets the co-observations it needs
without the board having to be waved into a pose that suits every camera at
once.

The detection is OpenCV's, not a reimplementation.  What is written here is the
part that is specific to this pipeline: the conversion from detected corner ids
to board-frame 3D points, the quality gates, and the sub-pixel refinement
settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..sim.textures import charuco_board  # noqa: F401  (re-exported for callers)


@dataclass
class BoardDetection:
    """Corners found in one image.

    ``ids`` index :meth:`CharucoTarget.object_points`, so ``object_points[ids]``
    and ``corners`` are the 3D-2D correspondence pair.
    """

    ids: np.ndarray            #: (M,) int, chessboard corner indices
    corners: np.ndarray        #: (M, 2) float, sub-pixel pixel coordinates
    n_markers: int             #: how many ArUco markers were decoded
    camera: int = -1
    frame: int = -1

    def __len__(self):
        return len(self.ids)

    @property
    def spread(self):
        """Diagonal of the corner bounding box, in pixels.

        A detection where every corner sits in a small patch constrains the
        board pose badly however many corners it has, so this is one of the
        gates.
        """
        if len(self.corners) < 2:
            return 0.0
        lo, hi = self.corners.min(axis=0), self.corners.max(axis=0)
        return float(np.linalg.norm(hi - lo))


class CharucoTarget:
    """Wraps a ``cv2.aruco.CharucoBoard`` with detection and geometry."""

    def __init__(self, board=None, min_corners=8, min_spread_px=60.0):
        self.board = board if board is not None else charuco_board()
        self.min_corners = int(min_corners)
        self.min_spread_px = float(min_spread_px)

        params = cv2.aruco.DetectorParameters()
        # Corner refinement is what takes the detection from integer-ish to
        # roughly a tenth of a pixel, and a tenth of a pixel is the difference
        # between a calibration residual of 0.1 px and one of 0.4 px.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 60
        params.cornerRefinementMinAccuracy = 0.01
        # Default is 4, which rejects markers seen at a steep angle -- exactly
        # the views that constrain the focal length best.
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 6
        params.polygonalApproxAccuracyRate = 0.05

        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        self.detector = cv2.aruco.CharucoDetector(
            self.board, charuco_params, params, cv2.aruco.RefineParameters()
        )

    # -- geometry ----------------------------------------------------------

    @property
    def object_points(self):
        """(N, 3) chessboard corners in the board frame, in metres."""
        return np.asarray(self.board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)

    @property
    def square_length(self):
        return float(self.board.getSquareLength())

    @property
    def size(self):
        return tuple(int(v) for v in self.board.getChessboardSize())

    @property
    def n_corners(self):
        return len(self.object_points)

    def points_for(self, ids):
        return self.object_points[np.asarray(ids, dtype=int).ravel()]

    # -- detection ---------------------------------------------------------

    def detect(self, image, camera=-1, frame=-1):
        """Detect the board in a greyscale image.  ``None`` if it fails a gate."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, marker_corners, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < self.min_corners:
            return None
        det = BoardDetection(
            ids=np.asarray(ids, dtype=int).ravel(),
            corners=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
            n_markers=0 if marker_corners is None else len(marker_corners),
            camera=int(camera),
            frame=int(frame),
        )
        if det.spread < self.min_spread_px:
            return None
        return det

    def estimate_pose(self, detection, camera):
        """Board-from-camera pose by PnP on the detected corners.

        Returns ``T_cam_board`` or ``None``.  ``SOLVEPNP_IPPE`` is the planar
        solver: the target is a plane, and the general iterative solver on a
        planar target is both slower and less reliable about which of the two
        mirror solutions it lands on.
        """
        from ..geometry import se3

        if detection is None or len(detection) < 4:
            return None
        obj = self.points_for(detection.ids).astype(np.float64)
        img = detection.corners.astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, camera.K, camera.dist.reshape(1, 5), flags=cv2.SOLVEPNP_IPPE
        )
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(
            obj, img, camera.K, camera.dist.reshape(1, 5), rvec, tvec
        )
        R, _ = cv2.Rodrigues(rvec)
        return se3.from_Rt(R, tvec.ravel())
