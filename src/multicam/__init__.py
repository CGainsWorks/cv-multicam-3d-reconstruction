"""Multi-camera 3D reconstruction and 6-DoF pose estimation.

The package is laid out along the pipeline:

``multicam.geometry``
    SO(3), SE(3), the pinhole-plus-radtan camera and triangulation.
``multicam.sim``
    The rig, the object meshes and the software rasteriser that stands in for
    the hardware this repository does not have.
``multicam.calibration``
    Target detection, per-camera intrinsics and rig extrinsics.
``multicam.reconstruction``
    Features, matching, track building, sparse reconstruction and the bundle
    adjuster.
``multicam.fusion``
    Plane-sweep multi-view stereo and the Open3D fusion and meshing on top.
``multicam.pose``
    Classical PnP with a robust refinement, and the learned refiner that is
    ablated against it.
``multicam.evaluation``
    Pose error, reprojection error and comparison against the CAD model.
``multicam.io``
    The on-disk dataset layout and the COLMAP bridge.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
