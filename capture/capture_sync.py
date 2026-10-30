"""Synchronised multi-camera capture, from the command line.

**What has been run and what has not.**  The GenICam backend (``--backend
aravis``) has been run in CI against three of Aravis's *simulated* GigE Vision
cameras on the loopback interface, through the real GenICam / GigE Vision
path: feature access over GVCP, ``AcquisitionStart``, a software trigger per
frame set, image streaming over GVSP and device timestamps.  Frames captured
that way are fed through the calibration, reconstruction and pose pipeline by
``scripts/genicam_demo.py``.  **It has never been run against a real camera.**
There is no motion control for a real turntable either.  STATUS.md lists what a
real rig would need checking first.

The logic lives in :mod:`multicam.capture`; this file is the command line.

Backends
--------
``--backend aravis`` (GenICam, through Aravis) is what a real machine-vision
rig would use.  It arms every camera on ``--trigger-source``: ``Line1`` by
default, a hardware trigger input wired to one pulse for all cameras, in which
case nothing here fires the trigger; or ``Software``, which fires
``TriggerSoftware`` on each camera in turn and is not synchronised on real
hardware.  Every frame set goes through the timestamp check in
:func:`multicam.capture.check_sync` and is dropped (and re-triggered at the
same station, up to ``--max-attempts``) if its exposures are spread by more
than ``--sync-tolerance-ms``.

``--backend opencv`` uses ``cv2.VideoCapture``, which is what a USB webcam rig
would use and which **cannot** be hardware triggered; it is there for a bench
test and it warns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multicam.capture import DEFAULT_SYNC_TOLERANCE_MS, capture_session, check_sync  # noqa: F401


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=("aravis", "opencv"), default="aravis")
    p.add_argument("--devices", nargs="+", default=None,
                   help="aravis: device ids or IP addresses; opencv: device indices")
    p.add_argument("--trigger-source", default="Line1",
                   help="aravis: GenICam TriggerSource, e.g. Line1 or Software")
    p.add_argument("--out", default=str(ROOT / "data" / "hardware"))
    p.add_argument("--session", default="turntable")
    p.add_argument("--frames", type=int, default=18)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--exposure-us", type=float, default=4000.0)
    p.add_argument("--gain", type=float, default=0.0)
    p.add_argument("--sync-tolerance-ms", type=float, default=DEFAULT_SYNC_TOLERANCE_MS)
    p.add_argument("--max-attempts", type=int, default=3)
    return p


def make_backend(args):
    if args.backend == "opencv":
        from multicam.capture.opencv_backend import OpenCVBackend

        return OpenCVBackend(args.devices or ["0", "1", "2", "3"], args.width,
                             args.height, exposure=args.exposure_us)
    from multicam.capture.genicam import AravisBackend

    if not args.devices:
        raise SystemExit("--devices is required for the aravis backend")
    return AravisBackend(args.devices, width=args.width, height=args.height,
                         exposure_us=args.exposure_us, gain=args.gain,
                         trigger_source=args.trigger_source)


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = capture_session(
        make_backend(args), args.out, args.frames, args.sync_tolerance_ms,
        args.session, settle_s=0.0, max_attempts=args.max_attempts,
    )
    print("kept %d frame sets, dropped %d, missed %d stations"
          % (log["frames_kept"], log["frame_sets_dropped"], log["stations_missed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
