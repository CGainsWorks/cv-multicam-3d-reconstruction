"""Synchronised multi-camera capture.

``multicam.capture.sync``
    Frame sets, the timestamp synchronisation check and the capture loop.  Pure
    Python; no camera library.
``multicam.capture.genicam``
    The GenICam / GigE Vision backend, through Aravis.  Needs Aravis and
    PyGObject (``requirements-genicam.txt``), so it is imported on demand.
``multicam.capture.fakecam``
    Aravis's simulated GigE Vision cameras, run as local processes for the
    tests and for ``scripts/genicam_demo.py``.  **Simulated**: nothing in this
    repository has been run against a real camera.

Not part of the pipeline: nothing under :mod:`multicam.calibration`,
:mod:`multicam.reconstruction`, :mod:`multicam.fusion` or :mod:`multicam.pose`
imports it.  The pipeline reads images from the capture layout; whether
they came from the renderer or over GigE Vision is invisible to it, which is
the point.
"""

from .sync import (  # noqa: F401
    DEFAULT_SYNC_TOLERANCE_MS,
    CaptureBackend,
    FrameLost,
    FrameSet,
    Turntable,
    capture_session,
    check_sync,
)
