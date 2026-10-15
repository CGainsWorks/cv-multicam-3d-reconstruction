"""Synchronised multi-camera capture.

**This script has never been run.  There is no hardware behind this
repository.**

That is stated at the top rather than buried, because the rest of the
repository is tested and this is not.  It is written out because the capture
layout, the trigger discipline and the metadata are part of the design, and
because the thing that makes a multi-camera rig work or not work is almost
never the reconstruction -- it is whether the four exposures really happened at
the same instant.  What the simulator provides for free is exactly what is
hardest to get right in a real cell, so the assumptions it makes are at least
written down here.

What a real implementation has to get right
-------------------------------------------

**Hardware trigger, not software.**  Calling ``grab()`` on four cameras in a
Python loop gives exposures spread over tens of milliseconds, which on a part
moving at 100 mm/s is several millimetres of motion between the first camera
and the last.  Every pose this repository estimates assumes the four views are
simultaneous.  So: one trigger source, wired to all four cameras, and the
software only *arms* them.

**Verify the synchronisation, do not assume it.**  Cameras report a timestamp;
the check is that the spread across a frame set is below a threshold, and a set
that fails it is dropped rather than quietly used.  :func:`check_sync` does
that and is the one function here with a unit test, because it is pure
arithmetic over timestamps and needs no camera.

**Fixed exposure, gain and white balance.**  Auto-anything changes the
appearance between cameras and between frames, and the descriptor matching
pays for it.  The settings go in the metadata.

**Record the trigger and the turntable angle.**  A capture whose frames cannot
be related to the table position afterwards is much less useful.

Backends
--------
``--backend opencv`` uses ``cv2.VideoCapture``, which is what a USB webcam rig
would use and which **cannot** be hardware triggered; it is there for a bench
test and it warns.  ``--backend harvesters`` sketches the GenICam path, which is
what a real machine-vision rig would use.  Neither is exercised.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: Largest acceptable spread of exposure timestamps within one frame set, in
#: milliseconds.  200 microseconds is comfortable for a wired trigger and
#: impossible for a software one, which is the point.
DEFAULT_SYNC_TOLERANCE_MS = 0.2


@dataclass
class FrameSet:
    """One trigger: an image and a timestamp from every camera."""

    index: int
    images: dict                      #: camera index -> ndarray
    timestamps: dict                  #: camera index -> seconds, camera clock
    metadata: dict = field(default_factory=dict)

    @property
    def spread_ms(self):
        if len(self.timestamps) < 2:
            return 0.0
        values = list(self.timestamps.values())
        return (max(values) - min(values)) * 1000.0


def check_sync(timestamps, tolerance_ms=DEFAULT_SYNC_TOLERANCE_MS):
    """Is a frame set simultaneous enough to use?

    Returns ``(ok, spread_ms)``.  Pure arithmetic, so this is the one part of
    the module that is tested.
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

    def trigger(self):
        """Fire one exposure on every camera and return a :class:`FrameSet`."""
        raise NotImplementedError

    @property
    def n_cameras(self):
        raise NotImplementedError


class OpenCVBackend(CaptureBackend):
    """``cv2.VideoCapture``.  Software triggered, and therefore not synchronised.

    Usable for a bench test of the plumbing and nothing else.  It warns on
    every open, because a rig that silently captures unsynchronised frames
    produces plausible images and a reconstruction that is wrong in a way no
    residual will reveal.
    """

    def __init__(self, device_ids, width=1280, height=1024, exposure=None, gain=None):
        self.device_ids = list(device_ids)
        self.width, self.height = int(width), int(height)
        self.exposure, self.gain = exposure, gain
        self.captures = []

    @property
    def n_cameras(self):
        return len(self.device_ids)

    def open(self):
        import cv2

        print("WARNING: the OpenCV backend is software triggered.  Exposures will "
              "be spread over tens of milliseconds and nothing downstream is "
              "valid for a moving part.  Use a hardware trigger for real work.",
              file=sys.stderr)
        for device in self.device_ids:
            cap = cv2.VideoCapture(int(device))
            if not cap.isOpened():
                raise RuntimeError("camera %s did not open" % device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Auto-exposure off, so the appearance does not drift between
            # frames and between cameras.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            if self.exposure is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
            if self.gain is not None:
                cap.set(cv2.CAP_PROP_GAIN, float(self.gain))
            self.captures.append(cap)
        return self

    def close(self):
        for cap in self.captures:
            cap.release()
        self.captures = []

    def trigger(self, index=0):
        import cv2

        # Two passes: ask every camera to grab, then retrieve.  It does not
        # make the exposures simultaneous, it only stops the decode time of
        # camera 0 from delaying the grab of camera 3.
        stamps = {}
        for i, cap in enumerate(self.captures):
            cap.grab()
            stamps[i] = time.perf_counter()
        images = {}
        for i, cap in enumerate(self.captures):
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError("camera %d dropped a frame" % i)
            images[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return FrameSet(index=index, images=images, timestamps=stamps,
                        metadata={"backend": "opencv", "synchronised": False})


class HarvestersBackend(CaptureBackend):
    """GenICam through ``harvesters``: the path a real rig would take.

    Sketched, not implemented, because there is no camera to develop it
    against and a half-working camera driver is worse than an honest stub.
    The configuration below is the part that matters and is what would need
    checking first on real hardware.
    """

    #: The features that have to be set for a triggered capture.  Names are
    #: from the GenICam Standard Features Naming Convention.
    REQUIRED_FEATURES = {
        "TriggerMode": "On",
        "TriggerSource": "Line1",
        "TriggerSelector": "FrameStart",
        "TriggerActivation": "RisingEdge",
        "ExposureAuto": "Off",
        "GainAuto": "Off",
        "BalanceWhiteAuto": "Off",
        "AcquisitionMode": "Continuous",
    }

    def __init__(self, cti_path, serials, exposure_us=4000, gain_db=0.0):
        self.cti_path = cti_path
        self.serials = list(serials)
        self.exposure_us = float(exposure_us)
        self.gain_db = float(gain_db)

    @property
    def n_cameras(self):
        return len(self.serials)

    def open(self):
        raise NotImplementedError(
            "The GenICam backend is not implemented: this repository has no "
            "cameras to develop it against, and shipping an untested camera "
            "driver as though it worked would be worse than saying so.\n\n"
            "To implement it: install `harvesters`, point --cti at the "
            "vendor's GenTL producer .cti file, open one ImageAcquirer per "
            "serial, set %s on each, start acquisition on all of them, and "
            "then fire the external trigger.  Read the device timestamp off "
            "each buffer and pass it through check_sync()."
            % sorted(self.REQUIRED_FEATURES)
        )

    def close(self):
        pass

    def trigger(self, index=0):
        raise NotImplementedError


def capture_session(backend, out_dir, n_frames, tolerance_ms=DEFAULT_SYNC_TOLERANCE_MS,
                    session="turntable", turntable=None, settle_s=0.4):
    """Capture ``n_frames`` frame sets and write them in the dataset layout.

    Frame sets whose exposures are not simultaneous enough are **dropped**, and
    the count of dropped sets goes in the metadata.  Keeping them would be the
    easiest way to produce a capture that looks fine and reconstructs wrongly.
    """
    import cv2

    out_dir = Path(out_dir)
    (out_dir / session).mkdir(parents=True, exist_ok=True)

    kept, dropped, records = 0, 0, []
    backend.open()
    try:
        for index in range(int(n_frames)):
            if turntable is not None:
                turntable.move_to(index)
                time.sleep(settle_s)

            frames = backend.trigger(index=index)
            ok, spread = check_sync(frames.timestamps, tolerance_ms)
            if not ok:
                dropped += 1
                print("frame %d dropped: exposures spread over %.3f ms (limit %.3f)"
                      % (index, spread, tolerance_ms), file=sys.stderr)
                continue

            d = out_dir / session / ("%03d" % kept)
            d.mkdir(parents=True, exist_ok=True)
            for camera, image in frames.images.items():
                cv2.imwrite(str(d / ("cam%d.png" % camera)), image)
            records.append({
                "frame": kept,
                "trigger_index": index,
                "spread_ms": spread,
                "timestamps": {str(k): float(v) for k, v in frames.timestamps.items()},
                "turntable_deg": None if turntable is None else turntable.angle_deg(index),
            })
            kept += 1
    finally:
        backend.close()

    (out_dir / session / "capture_log.json").write_text(
        json.dumps({
            "session": session,
            "cameras": backend.n_cameras,
            "frames_kept": kept,
            "frames_dropped": dropped,
            "sync_tolerance_ms": tolerance_ms,
            "frames": records,
        }, indent=2),
        encoding="utf-8",
    )
    return kept, dropped


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


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=("opencv", "harvesters"), default="opencv")
    p.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    p.add_argument("--cti", default=None, help="GenTL producer .cti file")
    p.add_argument("--out", default=str(ROOT / "data" / "hardware"))
    p.add_argument("--session", default="turntable")
    p.add_argument("--frames", type=int, default=18)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--exposure-us", type=float, default=4000.0)
    p.add_argument("--sync-tolerance-ms", type=float, default=DEFAULT_SYNC_TOLERANCE_MS)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(__doc__.splitlines()[2])
    if args.backend == "opencv":
        backend = OpenCVBackend(args.devices, args.width, args.height,
                                exposure=args.exposure_us)
    else:
        backend = HarvestersBackend(args.cti, args.devices, args.exposure_us)
    kept, dropped = capture_session(
        backend, args.out, args.frames, args.sync_tolerance_ms, args.session
    )
    print("kept %d frame sets, dropped %d" % (kept, dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
