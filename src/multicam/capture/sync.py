"""Frame sets, the synchronisation check and the capture loop.

What a real rig has to get right
--------------------------------

**Hardware trigger, not software.**  Calling ``grab()`` on four cameras in a
Python loop gives exposures spread over tens of milliseconds, which on a part
moving at 100 mm/s is several millimetres of motion between the first camera
and the last.  Every pose this repository estimates assumes the views are
simultaneous.  So: one trigger source, wired to every camera; the software
only *arms* them.

**Verify the synchronisation, do not assume it.**  Cameras report a timestamp
per frame; the check is that the spread across a frame set is below a
threshold; a set that fails it is dropped rather than quietly used.
:func:`check_sync` does that.  For the timestamps of different cameras to be
comparable at all their clocks must agree, which on GigE Vision means PTP
(IEEE 1588); :class:`multicam.capture.genicam.AravisBackend` enables it where
the camera offers it and records whether it did.

**Fixed exposure, gain and white balance.**  Auto-anything changes the
appearance between cameras and between frames; the descriptor matching
pays for it.  The settings go in the capture log.

**Record the trigger and the turntable position.**  A capture whose frames
cannot be related to the table position afterwards is much less useful.  On a
stationary table a frame set that fails the check can be re-triggered at the
same station, which :func:`capture_session` does up to ``max_attempts`` times.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Largest acceptable spread of exposure timestamps within one frame set, in
#: milliseconds.  200 microseconds is comfortable for a wired trigger and
#: impossible for software triggers fired one camera at a time over a network,
#: which is the point.
DEFAULT_SYNC_TOLERANCE_MS = 0.2


class FrameLost(RuntimeError):
    """A trigger produced an incomplete frame set: a frame timed out or arrived
    with packets missing.  The capture loop treats it like a frame set that
    failed the synchronisation check: dropped and re-triggered."""


@dataclass
class FrameSet:
    """One trigger: an image and a timestamp from every camera."""

    index: int
    images: dict                      #: camera index -> ndarray
    timestamps: dict                  #: camera index -> seconds, camera clock
    metadata: dict = field(default_factory=dict)

    @property
    def spread_ms(self):
        return check_sync(self.timestamps, float("inf"))[1]


def check_sync(timestamps, tolerance_ms=DEFAULT_SYNC_TOLERANCE_MS):
    """Is a frame set simultaneous enough to use?

    ``timestamps`` maps camera to seconds on a clock the cameras share.
    Returns ``(ok, spread_ms)``.  Pure arithmetic over the timestamps, so it is
    tested without a camera; it is the same function whatever the backend.
    """
    values = [float(v) for v in timestamps.values()]
    if len(values) < 2:
        return True, 0.0
    spread = (max(values) - min(values)) * 1000.0
    return spread <= float(tolerance_ms), spread


class CaptureBackend:
    """What a camera backend has to provide."""

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def trigger(self, index=0):
        """Fire one exposure on every camera and return a :class:`FrameSet`."""
        raise NotImplementedError

    @property
    def n_cameras(self):
        raise NotImplementedError

    def describe(self):
        """Settings and device identities, for the capture log."""
        return {}


class Turntable:
    """A stepper-driven turntable.  Interface only; no motion control here."""

    def __init__(self, n_stations, total_deg=360.0):
        self.n_stations = int(n_stations)
        self.total_deg = float(total_deg)

    def angle_deg(self, index):
        return self.total_deg * index / max(self.n_stations, 1)

    def move_to(self, index):
        raise NotImplementedError(
            "Connect this to your motion controller.  It must block until the "
            "table has settled: a capture taken while the table is still "
            "ringing has motion blur that no amount of bundle adjustment will "
            "recover from."
        )


def _default_frame_path(session):
    def path(station, kept, camera):
        return Path(session) / ("%03d" % kept) / ("cam%d.png" % camera)
    return path


def capture_session(backend, out_dir, n_frames, tolerance_ms=DEFAULT_SYNC_TOLERANCE_MS,
                    session="turntable", turntable=None, settle_s=0.4, max_attempts=1,
                    frame_path=None, log_name=None, say=None):
    """Capture ``n_frames`` stations and write them in the dataset layout.

    At each station the turntable (if any) is moved and the cameras are
    triggered.  A frame set whose exposures are not simultaneous enough is
    **dropped** and re-triggered at the same station up to ``max_attempts``
    times in all.  Keeping it would be the easiest way to produce a capture
    that looks fine and reconstructs wrongly.  A frame set with a lost or
    incomplete frame (:class:`FrameLost`) is treated the same way.  Every drop
    and every loss is counted in the log.

    ``frame_path(station, kept, camera)`` gives the path, relative to
    ``out_dir``, an image is written to, or ``None`` to not write that camera's
    image for that station.  The default is ``<session>/<kept>/cam<camera>.png``.

    Returns the capture log as a dict; it is also written to
    ``<out_dir>/<session>/capture_log.json`` unless ``log_name`` says otherwise.
    """
    import cv2

    warn = say or (lambda msg: print(msg, file=sys.stderr))
    out_dir = Path(out_dir)
    frame_path = frame_path or _default_frame_path(session)
    log_path = out_dir / (log_name or (Path(session) / "capture_log.json"))

    kept, dropped, lost, missed, records = 0, 0, 0, 0, []
    backend.open()
    try:
        description = backend.describe()
        for station in range(int(n_frames)):
            if turntable is not None:
                turntable.move_to(station)
                if settle_s:
                    time.sleep(settle_s)

            for attempt in range(1, int(max_attempts) + 1):
                try:
                    frames = backend.trigger(index=station)
                except FrameLost as exc:
                    ok = False
                    lost += 1
                    warn("station %d attempt %d lost: %s" % (station, attempt, exc))
                    continue
                ok, spread = check_sync(frames.timestamps, tolerance_ms)
                if ok:
                    break
                dropped += 1
                warn("station %d attempt %d dropped: exposures spread over %.3f ms "
                     "(limit %.3f)" % (station, attempt, spread, tolerance_ms))
            if not ok:
                missed += 1
                continue

            written = {}
            for camera, image in frames.images.items():
                rel = frame_path(station, kept, camera)
                if rel is None:
                    continue
                path = out_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image):
                    raise OSError("could not write %s" % path)
                written[str(camera)] = Path(rel).as_posix()
            records.append({
                "frame": kept,
                "station": station,
                "attempts": attempt,
                "spread_ms": spread,
                "timestamps": {str(k): float(v) for k, v in frames.timestamps.items()},
                "metadata": frames.metadata,
                "written": written,
                "turntable_deg": None if turntable is None else turntable.angle_deg(station),
            })
            kept += 1
    finally:
        backend.close()

    spreads = [r["spread_ms"] for r in records]
    log = {
        "session": session,
        "cameras": backend.n_cameras,
        "backend": description,
        "stations": int(n_frames),
        "frames_kept": kept,
        "frame_sets_dropped": dropped,
        "frame_sets_lost": lost,
        "stations_missed": missed,
        "max_attempts": int(max_attempts),
        "sync_tolerance_ms": tolerance_ms,
        "spread_ms": {
            "max": max(spreads) if spreads else None,
            "median": sorted(spreads)[len(spreads) // 2] if spreads else None,
        },
        "frames": records,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(log, fh, indent=2)
        fh.write("\n")
    return log
