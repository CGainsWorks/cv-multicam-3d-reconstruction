"""Put ``src/`` on the path.

Imported for its side effect by every script.  The alternative is asking the
user to ``pip install -e .`` before anything works, which is one more step
between a clone and a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def default_data_root():
    return ROOT / "data"


def default_results_root():
    return ROOT / "results"


def library_versions():
    """Versions of everything a number could depend on, for the record files."""
    import platform

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "cv2", "open3d", "torch", "pycolmap"):
        try:
            module = __import__(name)
            # str() rather than the attribute itself: torch reports a
            # TorchVersion, which is a str subclass that PyYAML refuses to
            # represent, and the failure lands at the end of a long render.
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    return out


def git_commit():
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
