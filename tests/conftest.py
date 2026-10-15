"""Shared fixtures.

``src/`` is put on the path here rather than installing the package, so a clone
plus ``pip install -r requirements.txt`` is enough to run the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multicam.geometry import PinholeCamera  # noqa: E402
from multicam.sim.rig import RigSpec, build_rig  # noqa: E402
from multicam.sim.scene import Scene, default_scene  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    """A generator seeded once per session.

    Tests that need independent draws call ``np.random.default_rng(seed)``
    themselves; this one exists so that the many tests which just want *some*
    numbers do not each invent a seed.
    """
    return np.random.default_rng(20260822)


@pytest.fixture(scope="session")
def camera():
    """A camera with distortion large enough that ignoring it would show."""
    return PinholeCamera(
        fx=612.0, fy=608.0, cx=317.5, cy=239.0, width=640, height=480,
        dist=np.array([-0.28, 0.09, 0.0012, -0.0008, -0.014]),
    )


@pytest.fixture(scope="session")
def rig():
    """The four-camera rig every other module is built around."""
    return build_rig(RigSpec(n_cameras=4, width=640, height=480))


@pytest.fixture(scope="session")
def scene() -> Scene:
    return default_scene(seed=7)
