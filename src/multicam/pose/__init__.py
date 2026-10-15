"""6-DoF pose: classical generalised PnP, and the learned refiner ablated against it."""

from .classical import (  # noqa: F401
    Correspondences,
    PoseResult,
    estimate_pose,
    match_to_model,
    refine_pose,
)
from .model import ObjectModel, build_model  # noqa: F401
from .render_compare import CROP, ModelRenderer, build_input  # noqa: F401

__all__ = [
    "estimate_pose", "refine_pose", "match_to_model", "PoseResult", "Correspondences",
    "ObjectModel", "build_model",
    "ModelRenderer", "build_input", "CROP",
]
