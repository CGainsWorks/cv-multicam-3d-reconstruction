"""Synchronised capture: the sync check, the capture loop and the GenICam path.

The first half needs no camera library.  The second half, marked ``genicam``,
runs against Aravis's simulated GigE Vision cameras (``arv-fake-gv-camera``)
on the loopback interface: real GVCP feature access, real GVSP streaming,
software triggers and device timestamps.  **Simulated**; no camera hardware
is involved anywhere.  Those tests skip when Aravis is not installed, unless
``MULTICAM_REQUIRE_GENICAM=1`` is set, which CI does so that a broken install
fails rather than quietly skipping.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from multicam.capture import CaptureBackend, FrameSet, capture_session, check_sync
from multicam.capture.fakecam import (
    FAKE_TRIGGER_PERIOD_S,
    SIMULATED_SYNC_TOLERANCE_MS,
    FakeCameraFleet,
    camera_addresses,
    diagonal_ramp,
    find_tool,
    loopback_has,
    write_pgm,
)

# -- the sync check and the capture loop, no camera ----------------------------


def test_simultaneous_frames_pass_the_sync_check():
    ok, spread = check_sync({0: 10.0, 1: 10.00005, 2: 10.00001})
    assert ok
    assert spread == pytest.approx(0.05)


def test_a_spread_frame_set_is_rejected():
    ok, spread = check_sync({0: 10.0, 1: 10.0, 2: 10.04})
    assert not ok
    assert spread == pytest.approx(40.0)


def test_the_tolerance_is_inclusive_and_configurable():
    assert check_sync({0: 0.0, 1: 0.001}, tolerance_ms=1.0)[0]
    assert not check_sync({0: 0.0, 1: 0.001}, tolerance_ms=0.999)[0]


def test_one_camera_is_trivially_synchronised():
    assert check_sync({0: 123.0}) == (True, 0.0)


def test_frame_set_reports_its_spread():
    frames = FrameSet(index=0, images={}, timestamps={0: 1.0, 1: 1.0003})
    assert frames.spread_ms == pytest.approx(0.3)


class _ScriptedBackend(CaptureBackend):
    """Returns frame sets whose timestamp spreads follow a script."""

    def __init__(self, spreads_ms, n=3, shape=(6, 8)):
        self.spreads = list(spreads_ms)
        self.n, self.shape = n, shape
        self.opened = self.closed = 0
        self.triggers = 0

    @property
    def n_cameras(self):
        return self.n

    def open(self):
        self.opened += 1

    def close(self):
        self.closed += 1

    def describe(self):
        return {"backend": "scripted"}

    def trigger(self, index=0):
        spread = self.spreads[self.triggers] * 1e-3
        self.triggers += 1
        images = {c: np.full(self.shape, 10 * index + c, np.uint8) for c in range(self.n)}
        stamps = {c: 5.0 + (spread if c == self.n - 1 else 0.0) for c in range(self.n)}
        return FrameSet(index=index, images=images, timestamps=stamps)


def test_capture_session_drops_unsynchronised_sets(tmp_path):
    backend = _ScriptedBackend([0.0, 40.0, 0.01])
    log = capture_session(backend, tmp_path, 3, settle_s=0)
    assert (log["frames_kept"], log["frame_sets_dropped"], log["stations_missed"]) == (2, 1, 1)
    assert backend.opened == backend.closed == 1
    assert [f["station"] for f in log["frames"]] == [0, 2]
    # Kept frames are numbered contiguously, so a dropped set leaves no hole.
    assert (tmp_path / "turntable" / "001" / "cam2.png").exists()
    assert not (tmp_path / "turntable" / "002").exists()
    on_disk = json.loads((tmp_path / "turntable" / "capture_log.json").read_text())
    assert on_disk["frames_kept"] == 2


def test_capture_session_retriggers_at_the_same_station(tmp_path):
    backend = _ScriptedBackend([40.0, 40.0, 0.0, 0.0])
    log = capture_session(backend, tmp_path, 2, settle_s=0, max_attempts=3)
    assert log["frames_kept"] == 2
    assert log["frame_sets_dropped"] == 2
    assert [f["attempts"] for f in log["frames"]] == [3, 1]


def test_capture_session_retriggers_a_lost_frame(tmp_path):
    from multicam.capture import FrameLost

    backend = _ScriptedBackend([0.0, 0.0])
    real = backend.trigger
    calls = []

    def flaky(index=0):
        calls.append(index)
        if len(calls) == 1:
            raise FrameLost("camera 1: buffer status timeout")
        return real(index)

    backend.trigger = flaky
    log = capture_session(backend, tmp_path, 2, settle_s=0, max_attempts=2)
    assert calls == [0, 0, 1]
    assert (log["frames_kept"], log["frame_sets_lost"]) == (2, 1)


def test_capture_session_writes_where_it_is_told(tmp_path):
    import cv2

    backend = _ScriptedBackend([0.0, 0.0])

    def frame_path(station, kept, camera):
        return None if camera == 1 else "s%d/c%d.png" % (station, camera)

    capture_session(backend, tmp_path, 2, settle_s=0, frame_path=frame_path,
                    log_name="log.json")
    assert (tmp_path / "log.json").exists()
    assert not (tmp_path / "s0" / "c1.png").exists()
    image = cv2.imread(str(tmp_path / "s1" / "c2.png"), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (6, 8) and int(image[0, 0]) == 12


def test_diagonal_ramp_matches_the_aravis_formula():
    ramp = diagonal_ramp(300, 4, frame_id=7)
    assert ramp[0, 0] == 7 and ramp[3, 10] == 20
    assert ramp[0, 248] == 0            # (248 + 7) mod 255 wraps
    # Half the default exposure scales by 1 + log10(0.5).
    dim = diagonal_ramp(4, 1, frame_id=100, exposure_us=5000.0)
    assert dim[0, 0] == int(100 * (1 + np.log10(0.5)))


def test_pgm_is_what_the_scene_camera_parses(tmp_path):
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    write_pgm(tmp_path / "f.pgm", image)
    raw = (tmp_path / "f.pgm").read_bytes()
    assert raw.startswith(b"P5\n4 3\n255\n")
    assert raw[len(b"P5\n4 3\n255\n"):] == image.tobytes()


# -- the GenICam path, against Aravis's simulated cameras ----------------------

N_CAMERAS = 3


def _genicam_unavailable():
    if not sys.platform.startswith("linux"):
        return "Aravis's fake GigE Vision cameras are run on Linux only"
    try:
        from multicam.capture.genicam import import_aravis

        import_aravis()
    except ImportError as exc:
        return str(exc).split("(")[0].strip()
    if find_tool("arv-fake-gv-camera-0.8") is None:
        return "arv-fake-gv-camera-0.8 not found; run scripts/genicam/setup_aravis.sh build"
    if not loopback_has(camera_addresses(N_CAMERAS)):
        return "loopback addresses missing; run scripts/genicam/loopback_addresses.sh 3"
    return None


_REASON = _genicam_unavailable()
if _REASON and os.environ.get("MULTICAM_REQUIRE_GENICAM") == "1":
    raise RuntimeError("MULTICAM_REQUIRE_GENICAM=1 but: %s" % _REASON)

genicam = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")

WIDTH, HEIGHT = 320, 240


def _trigger(backend, attempts=3, **kwargs):
    """One frame set, re-triggered if a frame was lost in transit.

    The simulator has no packet resend, so a frame whose packets are dropped
    by the host (most often the very first after AcquisitionStart) arrives as
    a timeout; the capture loop retries those and so do the tests.
    """
    from multicam.capture import FrameLost

    for attempt in range(attempts):
        try:
            return backend.trigger(**kwargs)
        except FrameLost:
            if attempt == attempts - 1:
                raise


@pytest.fixture
def fleet():
    with FakeCameraFleet(N_CAMERAS, kind="fake", serial_prefix="FAKE") as f:
        yield f
        # Shown by pytest only when the test failed.
        print(f.log_text())


@pytest.fixture
def backend(fleet):
    from multicam.capture.genicam import AravisBackend

    b = AravisBackend(fleet.addresses, width=WIDTH, height=HEIGHT, exposure_us=10000.0,
                      gain=0.0, trigger_source="Software", timeout_s=2.0)
    b.open()
    yield b
    b.close()


@pytest.mark.genicam
@genicam
def test_genicam_features_are_set_and_read_back(backend, fleet):
    info = backend.describe()
    assert info["backend"] == "aravis" and info["aravis"].startswith("0.8.")
    assert len(info["cameras"]) == N_CAMERAS
    for cam, serial in zip(info["cameras"], fleet.serials):
        assert cam["vendor"] == "Aravis"
        assert cam["serial"] == serial
        assert cam["region"] == [0, 0, WIDTH, HEIGHT]
        assert cam["pixel_format"] == "Mono8"
        assert (cam["trigger_mode"], cam["trigger_source"]) == ("On", "Software")
        assert cam["exposure_us"] == pytest.approx(10000.0)
        # The simulator has no PTP and no white balance; that is recorded,
        # not assumed away.
        assert cam["ptp"] is False
        assert "BalanceWhiteAuto" in cam["features_not_offered"]


@pytest.mark.genicam
@genicam
def test_an_armed_camera_waits_for_its_trigger(backend):
    for stream in backend.streams:
        assert stream.timeout_pop_buffer(int(3 * FAKE_TRIGGER_PERIOD_S * 1e6)) is None


@pytest.mark.genicam
@genicam
def test_software_triggered_frame_sets_arrive_intact(backend):
    _trigger(backend)                   # off the simulator's grid; see fakecam
    passed = 0
    for k in range(8):
        frames = _trigger(backend, index=k)
        assert sorted(frames.images) == list(range(N_CAMERAS))
        ids = {frames.metadata["frame_id"][str(c)] for c in range(N_CAMERAS)}
        # One trigger, one frame per camera: the cameras' frame counters move
        # together, so a frame set never mixes frames of different triggers.
        assert len(ids) == 1, "a frame set mixes frames of different triggers"
        for c, image in frames.images.items():
            # Every pixel crossed GVSP; the fake camera's pattern is known
            # exactly, so the image is checked exactly.
            frame_id = frames.metadata["frame_id"][str(c)]
            assert np.array_equal(image, diagonal_ramp(WIDTH, HEIGHT, frame_id))
        if check_sync(frames.timestamps, SIMULATED_SYNC_TOLERANCE_MS)[0]:
            passed += 1
    # Most sets pass; the ones that do not are the simulator's late ticks,
    # which is what the check is there to catch.
    assert passed >= 6
    assert backend.stale_buffers == 0


@pytest.mark.genicam
@genicam
def test_the_sync_check_rejects_a_late_trigger(backend):
    # The simulator only settles onto its trigger grid once it is streaming,
    # so let it deliver a couple of frame sets first.
    for k in range(2):
        _trigger(backend, index=k)
    # Hold the last camera's trigger back by one and a half grid periods.
    frames = _trigger(backend, delays_s={N_CAMERAS - 1: 1.5 * FAKE_TRIGGER_PERIOD_S})
    ok, spread = check_sync(frames.timestamps, SIMULATED_SYNC_TOLERANCE_MS)
    assert not ok
    assert spread > 1000 * FAKE_TRIGGER_PERIOD_S * 0.9


@pytest.mark.genicam
@genicam
def test_capture_session_through_genicam(fleet, tmp_path):
    from multicam.capture.genicam import AravisBackend

    backend = AravisBackend(fleet.addresses, width=WIDTH, height=HEIGHT,
                            exposure_us=10000.0, trigger_source="Software")
    log = capture_session(backend, tmp_path, 4, tolerance_ms=SIMULATED_SYNC_TOLERANCE_MS,
                          settle_s=0, max_attempts=4, session="turntable")
    assert log["frames_kept"] == 4 and log["stations_missed"] == 0
    assert log["backend"]["trigger_source"] == "Software"
    assert log["spread_ms"]["max"] <= log["sync_tolerance_ms"]
    assert (tmp_path / "turntable" / "003" / "cam2.png").exists()


@pytest.mark.genicam
@genicam
@pytest.mark.skipif(find_tool("arv-scene-gv-camera") is None,
                    reason="arv-scene-gv-camera not built")
def test_scene_camera_serves_the_frame_it_is_told_to(tmp_path):
    from multicam.capture.fakecam import SceneStage
    from multicam.capture.genicam import AravisBackend

    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, size=(2, N_CAMERAS, 48, 64), dtype=np.uint8)
    dirs = []
    for c in range(N_CAMERAS):
        d = tmp_path / ("cam%d" % c)
        for k in range(2):
            write_pgm(d / ("%06d.pgm" % k), frames[k, c])
        dirs.append(d)

    with FakeCameraFleet(N_CAMERAS, kind="scene", frame_dirs=dirs) as scene:
        backend = AravisBackend(scene.addresses, width=64, height=48,
                                trigger_source="Software")
        backend.open()
        try:
            stage = SceneStage(backend, 2)
            for k in (1, 0, 1):
                stage.move_to(k)
                got = _trigger(backend, index=k)
                for c in range(N_CAMERAS):
                    assert np.array_equal(got.images[c], frames[k, c])
        finally:
            backend.close()
