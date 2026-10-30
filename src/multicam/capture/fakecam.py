"""Aravis's simulated GigE Vision cameras, as local processes.  SIMULATED.

Two programs, both built by ``scripts/genicam/setup_aravis.sh`` from the
pinned Aravis release:

``arv-fake-gv-camera-0.8``
    Aravis's own GigE Vision camera simulator, unmodified.  Real GVCP and GVSP
    over UDP, a GenICam description, software and free-running triggers plus a
    diagonal ramp for an image.  The protocol tests run against this.
``arv-scene-gv-camera``
    The same Aravis fake-camera object with its fill-pattern callback replaced,
    so that it serves frames rendered by :mod:`multicam.sim` instead of the
    ramp (``capture/fakecam/arv-scene-gv-camera.c``).  The pipeline check in
    ``scripts/genicam_demo.py`` runs against this, because a ramp has nothing
    in it to reconstruct.

Each camera needs its own address, because each answers GVCP on port 3956;
camera ``i`` listens on ``127.0.0.(i+1)``, which
``scripts/genicam/loopback_addresses.sh`` adds to ``lo``.  Nothing leaves the
loopback interface.

What the simulation does and does not show
------------------------------------------

The fake cameras all take their timestamps from the one host clock, which is
the simulated stand-in for PTP-synchronised cameras.  In software-trigger mode
a fake camera does not expose the moment the trigger arrives: it exposes on
the next tick of a 25 Hz grid aligned to that clock.  Three cameras triggered
within the same 40 ms window therefore expose on the same tick, give or take
how promptly each of the three processes is scheduled, which on a desktop
Docker VM put the spread at a fraction of a millisecond typically and a few
milliseconds at worst.  Now and then one camera's frame lands a tick or more
late instead (a spread of 40 to about 100 ms); the synchronisation check
has to catch it.  The first frame after ``AcquisitionStart`` is the most
likely to: until a camera is streaming it waits on a 100 ms window of its own
rather than on the shared grid.

Both are properties of the simulator.  A real software trigger is not
quantised like this and is not synchronised to anything like a millisecond;
the spreads measured here say nothing about a real rig.  That is also why the
simulated runs use :data:`SIMULATED_SYNC_TOLERANCE_MS` rather than the 0.2 ms
meant for a hardware trigger: it sits above the scheduling jitter and far
below the 40 ms tick, so it still rejects every set the simulator gets wrong.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .sync import Turntable

FAKE_CAMERA = "arv-fake-gv-camera-0.8"
GVCP_PORT = 3956
SCENE_CAMERA = "arv-scene-gv-camera"

#: The fake camera's GenICam register the scene camera reads the frame index
#: from.  A spare debug register in Aravis's fake-camera description.
SCENE_INDEX_FEATURE = "TestRegister"

#: The fake camera's software-trigger grid, from Aravis's source: in trigger
#: mode it exposes on the next multiple of 1/25 s of the host clock.
FAKE_TRIGGER_PERIOD_S = 1.0 / 25.0

#: Sync tolerance for the simulated cameras, in milliseconds.  See the module
#: docstring; the hardware-trigger default is
#: :data:`multicam.capture.DEFAULT_SYNC_TOLERANCE_MS`.
SIMULATED_SYNC_TOLERANCE_MS = 5.0


def find_tool(name):
    """Path to an Aravis tool on PATH or under ``$ARAVIS_PREFIX/bin``."""
    found = shutil.which(name)
    if found:
        return found
    prefix = os.environ.get("ARAVIS_PREFIX")
    if prefix:
        candidate = Path(prefix) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None


def camera_addresses(n):
    return ["127.0.0.%d" % (i + 1) for i in range(int(n))]


def loopback_has(addresses):
    """Are these addresses all configured on ``lo``?"""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", "lo"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    have = {tok.split("/")[0] for tok in out.split() if tok.count(".") == 3}
    return all(a in have for a in addresses)


def gvcp_port_in_use(address):
    """Is UDP port 3956 (GVCP) on ``address`` already bound?"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((address, GVCP_PORT))
    except OSError:
        return True
    finally:
        sock.close()
    return False


def diagonal_ramp(width, height, frame_id, exposure_us=10000.0, gain=0):
    """The Mono8 image Aravis's fake camera sends for ``frame_id``.

    A transcription of ``arv_fake_camera_diagonal_ramp`` in Aravis 0.8: the
    value at ``(x, y)`` is ``(x + y + frame_id) mod 255``, scaled by
    ``1 + gain + log10(exposure_us / 10000)`` and clamped.  At the default
    exposure and zero gain the scale is exactly one.
    """
    y, x = np.mgrid[0:int(height), 0:int(width)]
    value = ((x + y + int(frame_id)) % 255).astype(np.float64)
    value *= 1.0 + float(gain) + np.log10(float(exposure_us) / 10000.0)
    return np.clip(value, 0, 255).astype(np.uint8)


def write_pgm(path, image):
    """A binary 8-bit PGM, the format the scene camera reads."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"P5\n%d %d\n255\n" % (w, h))
        fh.write(image.tobytes())


class FakeCameraFleet:
    """Start ``n`` simulated GigE Vision cameras and stop them afterwards.

    ``kind="fake"`` runs Aravis's ``arv-fake-gv-camera``; ``kind="scene"``
    runs ``arv-scene-gv-camera`` with ``frame_dirs[i]`` as camera ``i``'s
    frames.  Use as a context manager.
    """

    def __init__(self, n=3, kind="fake", frame_dirs=None, serial_prefix="SIM",
                 startup_timeout_s=10.0):
        if kind not in ("fake", "scene"):
            raise ValueError("kind must be 'fake' or 'scene'")
        if kind == "scene" and (frame_dirs is None or len(frame_dirs) != n):
            raise ValueError("a scene camera needs one frame directory per camera")
        self.n = int(n)
        self.kind = kind
        self.frame_dirs = [str(d) for d in frame_dirs] if frame_dirs else None
        self.serials = ["%s%d" % (serial_prefix, i) for i in range(self.n)]
        self.addresses = camera_addresses(self.n)
        self.startup_timeout_s = float(startup_timeout_s)
        self.processes = []
        self._logs = None

    @property
    def tool(self):
        return FAKE_CAMERA if self.kind == "fake" else SCENE_CAMERA

    def start(self):
        exe = find_tool(self.tool)
        if exe is None:
            raise FileNotFoundError(
                "%s not found on PATH or under $ARAVIS_PREFIX/bin; run "
                "scripts/genicam/setup_aravis.sh build" % self.tool)
        if not loopback_has(self.addresses):
            raise RuntimeError(
                "the loopback interface does not have %s; run "
                "scripts/genicam/loopback_addresses.sh %d"
                % (", ".join(self.addresses), self.n))
        busy = [a for a in self.addresses if gvcp_port_in_use(a)]
        if busy:
            raise RuntimeError("something already answers GVCP on %s; another camera "
                               "simulator is still running" % ", ".join(busy))
        self._logs = tempfile.TemporaryDirectory(prefix="fakecam-")
        for i, (addr, serial) in enumerate(zip(self.addresses, self.serials)):
            cmd = [exe, "-i", addr, "-s", serial]
            if self.kind == "scene":
                cmd += ["-d", self.frame_dirs[i]]
            log = open(Path(self._logs.name) / ("%s.log" % serial), "wb")
            self.processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
            log.close()
        try:
            self._wait_until_reachable()
        except Exception:
            self.stop()
            raise
        return self

    def _wait_until_reachable(self):
        from .genicam import import_aravis, release_control

        Aravis = import_aravis()
        deadline = time.monotonic() + self.startup_timeout_s
        for addr, proc in zip(self.addresses, self.processes):
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("%s on %s exited with %s:\n%s"
                                       % (self.tool, addr, proc.returncode, self.log_text()))
                try:
                    camera = Aravis.Camera.new(addr)
                    release_control(camera)
                    del camera
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("%s on %s did not answer within %.0f s:\n%s"
                                           % (self.tool, addr, self.startup_timeout_s,
                                              self.log_text()))
                    time.sleep(0.1)

    def log_text(self):
        if self._logs is None:
            return ""
        parts = []
        for path in sorted(Path(self._logs.name).glob("*.log")):
            parts.append("--- %s\n%s" % (path.name, path.read_text(errors="replace")[-2000:]))
        return "\n".join(parts)

    def stop(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.processes = []
        if self._logs is not None:
            self._logs.cleanup()
            self._logs = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


class SceneStage(Turntable):
    """The simulated turntable for the scene cameras.  SIMULATED.

    Moving to station ``k`` writes ``k`` into every scene camera's
    ``TestRegister`` over GVCP, after which each camera serves its ``k``-th
    rendered frame.  It stands in for a table that has moved and settled, so
    there is nothing to wait for.
    """

    def __init__(self, backend, n_stations):
        super().__init__(n_stations)
        self.backend = backend

    def move_to(self, index):
        self.backend.set_integer_all(SCENE_INDEX_FEATURE, int(index))

    def angle_deg(self, index):
        return None
