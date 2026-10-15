"""The object model the pose estimator matches against.

A set of 3D points in the CAD model's own frame, each carrying the descriptor
that was observed there.  It is built from the sparse reconstruction, not from
the CAD file: a CAD model has geometry and no appearance, and what a descriptor
matcher needs is appearance.  This is the standard way a markerless
model-based pose pipeline is bootstrapped -- scan the part once, register the
scan to the drawing, and use the scan as the appearance model.

Points are kept only if they lie on the part.  The turntable disc is rigidly
attached to the part during the scan and reconstructs perfectly well, but it is
not the part, and a pose estimator that locks onto the fixture is not solving
the problem.  The filter is a distance-to-CAD-surface threshold, which uses the
CAD model and no ground-truth pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectModel:
    """3D points with descriptors, in the CAD frame."""

    points: np.ndarray                  #: (N, 3) metres, CAD frame
    descriptors: np.ndarray             #: (N, 128) float32, RootSIFT
    diameter: float = 0.0
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.points)

    def save(self, path):
        np.savez_compressed(
            str(path), points=self.points, descriptors=self.descriptors,
            diameter=np.asarray([self.diameter]),
        )
        return path

    @staticmethod
    def load(path):
        d = np.load(str(path))
        return ObjectModel(
            points=d["points"], descriptors=d["descriptors"],
            diameter=float(d["diameter"][0]),
        )


def build_model(points_in_cad, descriptors, cad, surface_mm=4.0, min_points=50,
                distance=None):
    """Keep the reconstructed points that lie on the CAD surface.

    ``surface_mm`` is the band around the surface a point must fall in.  4 mm
    is several times the reconstruction's own error and far less than the
    distance from the part to the turntable it stands on, so the threshold is
    not delicate -- which is checked, by reporting how many points each side
    of it there were.
    """
    from ..evaluation.cad import SurfaceDistance

    points_in_cad = np.asarray(points_in_cad, dtype=float).reshape(-1, 3)
    descriptors = np.asarray(descriptors, dtype=np.float32).reshape(len(points_in_cad), -1)

    distance = distance or SurfaceDistance(cad)
    d = distance(points_in_cad) * 1000.0
    keep = d <= float(surface_mm)
    if keep.sum() < min_points:
        raise RuntimeError(
            "only %d of %d reconstructed points lie within %.1f mm of the CAD "
            "surface; the reconstruction or its registration is wrong"
            % (int(keep.sum()), len(points_in_cad), surface_mm)
        )
    return ObjectModel(
        points=points_in_cad[keep],
        descriptors=descriptors[keep],
        diameter=float(cad.diameter),
        stats={
            "candidates": int(len(points_in_cad)),
            "on_surface": int(keep.sum()),
            "rejected": int((~keep).sum()),
            "surface_threshold_mm": float(surface_mm),
            "median_distance_mm": float(np.median(d[keep])) if keep.any() else float("nan"),
        },
    )
