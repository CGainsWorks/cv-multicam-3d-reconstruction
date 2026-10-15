"""The simulator that stands in for hardware.

Meshes, procedural textures, a software rasteriser, the rig geometry and the
turntable scene.  Everything quantitative in this repository traces back to
here, which is a limitation and is stated as one.
"""

from . import mesh, render, rig, scene, textures  # noqa: F401

__all__ = ["mesh", "render", "rig", "scene", "textures"]
