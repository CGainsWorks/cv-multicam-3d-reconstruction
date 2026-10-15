"""Evaluation: reprojection error, pose error and comparison against CAD.

Everything in this package is allowed to read ground truth.  Nothing outside it
is.
"""

from .cad import (  # noqa: F401
    Registration,
    SurfaceDistance,
    chamfer,
    completeness,
    register_to_cad,
    sparse_accuracy,
    to_open3d_mesh,
)
from .labels import (  # noqa: F401
    BACKGROUND,
    CLASS_NAMES,
    FIXTURE,
    OBJECT,
    composition,
    label_observations,
)
from .metrics import (  # noqa: F401
    add_metric,
    adds_metric,
    pose_error_table,
    recall_at,
    reprojection_error,
)

__all__ = [
    "register_to_cad", "Registration", "sparse_accuracy", "completeness", "chamfer",
    "SurfaceDistance", "to_open3d_mesh",
    "add_metric", "adds_metric", "pose_error_table", "recall_at", "reprojection_error",
    "label_observations", "composition", "OBJECT", "FIXTURE", "BACKGROUND", "CLASS_NAMES",
]
