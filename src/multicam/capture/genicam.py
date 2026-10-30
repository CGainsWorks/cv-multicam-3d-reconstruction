"""GenICam / GigE Vision capture through Aravis.

Aravis is an open-source GenICam library: it parses the GenICam XML a camera
describes itself with, reads and writes features by their Standard Features
Naming Convention (SFNC) names over GVCP and receives the image stream over
GVSP.  It talks to any compliant GigE Vision (and USB3 Vision) camera, which is
why it is the backend here rather than a vendor SDK.

**What has been run.**  This module has been exercised in CI against Aravis's
own simulated GigE Vision cameras (``arv-fake-gv-camera``, see
:mod:`multicam.capture.fakecam`): three of them on the loopback interface, with
real GVCP feature access, real GVSP streaming, software triggers and device
timestamps.  **It has never been run against a real camera.**  The things a
real rig would need checking first are listed in STATUS.md.

The configuration applied to every camera:

* ``AcquisitionMode = Continuous`` and ``TriggerSelector = FrameStart``,
  ``TriggerMode = On``: every frame waits for a trigger.
* ``TriggerSource`` is ``Line1`` by default, the hardware trigger input a real
  rig wires to a common pulse, in which case the software only arms the
  cameras.  ``Software`` fires ``TriggerSoftware`` on each camera in turn,
  which is what the simulated cameras support and what CI uses.
* Exposure, gain and white balance fixed, with their ``*Auto`` features off
  where the camera has them.
* PTP (``GevIEEE1588`` or ``PtpEnable``) switched on where the camera has it,
  so that device timestamps from different cameras share one clock.  Aravis's
  simulated cameras do not have it; their timestamps come from the host clock
  they all run on, which is the simulated equivalent and is recorded as such.

Features a camera does not offer are skipped and listed in :meth:`describe`,
never silently assumed to have been set.
"""

from __future__ import annotations

import time

import numpy as np

from .sync import CaptureBackend, FrameLost, FrameSet

#: Required version of the introspection namespace.
ARAVIS_API = "0.8"


def import_aravis():
    """``gi.repository.Aravis``, or ImportError saying what is missing."""
    try:
        import gi

        gi.require_version("Aravis", ARAVIS_API)
        from gi.repository import Aravis
    except (ImportError, ValueError) as exc:
        raise ImportError(
            "The GenICam backend needs Aravis %s with introspection data and "
            "PyGObject.  On Ubuntu 24.04: scripts/genicam/setup_aravis.sh apt "
            "&& scripts/genicam/setup_aravis.sh build && pip install -r "
            "requirements-genicam.txt, then export the variables printed by "
            "scripts/genicam/setup_aravis.sh env.  (%s)" % (ARAVIS_API, exc)
        ) from exc
    return Aravis


def aravis_version():
    """The Aravis release in use.

    The 0.8 introspection data has no version call, so this reads the
    pkg-config file of the install under ``$ARAVIS_PREFIX`` that
    ``scripts/genicam/setup_aravis.sh`` made.
    """
    import os
    from pathlib import Path

    import_aravis()
    prefix = os.environ.get("ARAVIS_PREFIX")
    if prefix:
        pc = Path(prefix) / "lib" / "pkgconfig" / ("aravis-%s.pc" % ARAVIS_API)
        if pc.exists():
            for line in pc.read_text().splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    return "%s (release unknown)" % ARAVIS_API


def take_control(camera, timeout_s=5.0):
    """Make sure this process is the GigE Vision controller of ``camera``.

    A GigE Vision device accepts writes from one controller at a time.  Aravis
    asks for control when it opens a device and, if another controller still
    holds it, carries on without it: every feature write and every trigger is
    then ignored by the camera, which shows up much later as a frame that
    never arrives.  A controller that went away without letting go (a crashed
    capture, or a device object Python has not yet freed) is released by the
    camera once its heartbeat times out, so this keeps asking until then.
    """
    device = camera.get_device()
    if not hasattr(device, "is_controller"):
        return True                      # USB3 Vision has no control channel
    deadline = time.monotonic() + float(timeout_s)
    while not device.is_controller():
        if device.take_control():
            break
        if time.monotonic() > deadline:
            raise RuntimeError("could not take control of %s within %.0f s: another "
                               "application is controlling it"
                               % (camera.get_device_id(), timeout_s))
        time.sleep(0.2)
    return True


def release_control(camera):
    """Give up control explicitly rather than whenever Python frees the device."""
    device = camera.get_device()
    if hasattr(device, "leave_control"):
        try:
            device.leave_control()
        except Exception:
            pass


class AravisBackend(CaptureBackend):
    """Several GenICam cameras, triggered together, through Aravis.

    ``devices`` are anything Aravis can open: a device id from discovery
    (``"<vendor>-<serial>"``) or, for GigE Vision, an IP address.
    """

    def __init__(self, devices, width=None, height=None, pixel_format="Mono8",
                 exposure_us=4000.0, gain=0.0, trigger_source="Line1",
                 timeout_s=2.0, n_buffers=8, control_timeout_s=5.0, packet_size=None):
        self.devices = [str(d) for d in devices]
        self.width = None if width is None else int(width)
        self.height = None if height is None else int(height)
        self.pixel_format = pixel_format
        self.exposure_us = None if exposure_us is None else float(exposure_us)
        self.gain = None if gain is None else float(gain)
        self.trigger_source = trigger_source
        self.timeout_s = float(timeout_s)
        self.n_buffers = int(n_buffers)
        self.control_timeout_s = float(control_timeout_s)
        self.packet_size = packet_size
        self.cameras, self.streams = [], []
        self.stale_buffers = 0
        self._description = {}
        self._epoch_ns = None

    @property
    def n_cameras(self):
        return len(self.devices)

    @property
    def software_triggered(self):
        return self.trigger_source == "Software"

    # -- set-up ------------------------------------------------------------

    @staticmethod
    def _feature(camera, name, value, skipped):
        """Set a feature if the camera has it; record it as skipped if not."""
        if not camera.is_feature_available(name):
            skipped.append(name)
            return False
        if isinstance(value, bool):
            camera.set_boolean(name, value)
        elif isinstance(value, str):
            camera.set_string(name, value)
        elif isinstance(value, int):
            camera.set_integer(name, value)
        else:
            camera.set_float(name, float(value))
        return True

    def _configure(self, camera):
        skipped = []
        f = self._feature
        f(camera, "AcquisitionMode", "Continuous", skipped)
        if self.width is not None or self.height is not None:
            f(camera, "OffsetX", 0, skipped)
            f(camera, "OffsetY", 0, skipped)
        if self.width is not None:
            f(camera, "Width", self.width, skipped)
        if self.height is not None:
            f(camera, "Height", self.height, skipped)
        f(camera, "PixelFormat", self.pixel_format, skipped)

        # Frame-start trigger, armed, from one source.
        f(camera, "TriggerSelector", "FrameStart", skipped)
        f(camera, "TriggerMode", "On", skipped)
        if not f(camera, "TriggerSource", self.trigger_source, skipped):
            raise RuntimeError("camera has no TriggerSource feature; cannot arm it")
        if not self.software_triggered:
            f(camera, "TriggerActivation", "RisingEdge", skipped)

        # Nothing automatic.
        f(camera, "ExposureAuto", "Off", skipped)
        f(camera, "GainAuto", "Off", skipped)
        f(camera, "BalanceWhiteAuto", "Off", skipped)
        if self.exposure_us is not None:
            if camera.is_exposure_time_available():
                camera.set_exposure_time(self.exposure_us)
            else:
                skipped.append("ExposureTime")
        if self.gain is not None:
            if camera.is_gain_available():
                camera.set_gain(self.gain)
            else:
                skipped.append("Gain")

        # One clock for every camera's timestamps.
        ptp = (f(camera, "GevIEEE1588", True, skipped)
               or f(camera, "PtpEnable", True, skipped))

        x, y, w, h = camera.get_region()
        return {
            "vendor": camera.get_vendor_name(),
            "model": camera.get_model_name(),
            "serial": camera.get_device_serial_number(),
            "device_id": camera.get_device_id(),
            "region": [int(x), int(y), int(w), int(h)],
            "pixel_format": camera.get_pixel_format_as_string(),
            "trigger_source": camera.get_string("TriggerSource"),
            "trigger_mode": camera.get_string("TriggerMode"),
            "exposure_us": (camera.get_exposure_time()
                            if camera.is_exposure_time_available() else None),
            "gain": camera.get_gain() if camera.is_gain_available() else None,
            "ptp": bool(ptp),
            "features_not_offered": skipped,
        }

    def open(self):
        Aravis = import_aravis()
        self.cameras, self.streams, cameras = [], [], []
        self._epoch_ns = None
        try:
            for device in self.devices:
                camera = Aravis.Camera.new(device)
                self.cameras.append(camera)
                take_control(camera, self.control_timeout_s)
                cameras.append(self._configure(camera))
            for camera in self.cameras:
                stream = camera.create_stream(None, None)
                self._tune_stream(Aravis, camera, stream)
                payload = camera.get_payload()
                for _ in range(self.n_buffers):
                    stream.push_buffer(Aravis.Buffer.new_allocate(payload))
                self.streams.append(stream)
            for camera in self.cameras:
                camera.start_acquisition()
        except Exception:
            self.close()
            raise
        self._description = {
            "backend": "aravis",
            "aravis": aravis_version(),
            "trigger_source": self.trigger_source,
            "devices": self.devices,
            "cameras": cameras,
        }
        return self

    def _tune_stream(self, Aravis, camera, stream):
        """GigE Vision receive settings.

        The kernel's default UDP receive buffer is a few hundred kilobytes
        while a frame arrives as a burst of packets.  If the buffer cannot hold
        one whole frame, packets are dropped whenever the stream thread is not
        scheduled; a camera without packet resend (the simulator has none)
        then delivers the frame as a timeout.  Aravis's ``AUTO`` setting sizes
        the buffer to the payload, up to ``net.core.rmem_max``, which a GigE
        Vision host normally raises; CI raises it too.  The packet size is optional
        because it has to match the network's MTU.
        """
        if hasattr(Aravis, "GvStream") and isinstance(stream, Aravis.GvStream):
            stream.set_property("socket-buffer", Aravis.GvStreamSocketBuffer.AUTO)
            if self.packet_size:
                camera.gv_set_packet_size(int(self.packet_size))

    def describe(self):
        return dict(self._description)

    def close(self):
        for camera in self.cameras:
            try:
                camera.stop_acquisition()
            except Exception:
                pass
        self.streams = []
        for camera in self.cameras:
            release_control(camera)
        self.cameras = []

    # -- features at run time ----------------------------------------------

    def set_integer_all(self, name, value):
        """Write one integer feature on every camera, over GVCP."""
        for camera in self.cameras:
            camera.set_integer(name, int(value))

    # -- triggering ----------------------------------------------------------

    def _drain(self):
        """Return any buffer that arrived outside a trigger to its stream.

        A late frame from a previous trigger left in a queue would be paired
        with the wrong frame set by every pop that followed.
        """
        for stream in self.streams:
            while True:
                buffer = stream.try_pop_buffer()
                if buffer is None:
                    break
                self.stale_buffers += 1
                stream.push_buffer(buffer)

    def fire(self, delays_s=None):
        """Issue ``TriggerSoftware`` on every camera, in order.

        ``delays_s`` maps camera index to a pause before that camera's
        trigger; it exists so the tests can provoke a frame set the
        synchronisation check must reject.  Returns host times of issue.
        """
        issued = {}
        for i, camera in enumerate(self.cameras):
            if delays_s and delays_s.get(i):
                time.sleep(float(delays_s[i]))
            camera.software_trigger()
            issued[i] = time.perf_counter()
        return issued

    def trigger(self, index=0, delays_s=None):
        Aravis = import_aravis()
        self._drain()
        issued = self.fire(delays_s) if self.software_triggered else {}

        images, timestamps, meta = {}, {}, {"frame_id": {}, "timestamp_ns": {},
                                            "system_timestamp_ns": {}}
        lost = []
        # Every camera's buffer is collected before anything is raised, so a
        # lost frame on one camera does not leave the others' frames queued
        # to be paired with the next trigger.
        for i, stream in enumerate(self.streams):
            buffer = stream.timeout_pop_buffer(int(self.timeout_s * 1e6))
            if buffer is None:
                lost.append("camera %d (%s): no frame within %.1f s"
                            % (i, self.devices[i], self.timeout_s))
                continue
            try:
                status = buffer.get_status()
                if status != Aravis.BufferStatus.SUCCESS:
                    lost.append("camera %d (%s): buffer status %s"
                                % (i, self.devices[i], status.value_nick))
                    continue
                if buffer.get_image_pixel_format() != Aravis.PIXEL_FORMAT_MONO_8:
                    raise RuntimeError("camera %d: only Mono8 is handled" % i)
                w, h = buffer.get_image_width(), buffer.get_image_height()
                data = buffer.get_image_data()
                images[i] = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w).copy()
                ts = int(buffer.get_timestamp())
                # Relative to the first frame of the session: nanoseconds since
                # 1970 do not survive a float64, nanoseconds since the first
                # trigger do.
                if self._epoch_ns is None:
                    self._epoch_ns = ts
                timestamps[i] = (ts - self._epoch_ns) * 1e-9
                meta["frame_id"][str(i)] = int(buffer.get_frame_id())
                meta["timestamp_ns"][str(i)] = ts
                meta["system_timestamp_ns"][str(i)] = int(buffer.get_system_timestamp())
            finally:
                stream.push_buffer(buffer)
        if lost:
            raise FrameLost("; ".join(lost))

        if issued:
            meta["trigger_issue_spread_ms"] = 1000.0 * (max(issued.values())
                                                        - min(issued.values()))
        meta["software_trigger"] = self.software_triggered
        return FrameSet(index=index, images=images, timestamps=timestamps, metadata=meta)
