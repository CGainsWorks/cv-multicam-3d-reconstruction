"""``cv2.VideoCapture``: a bench test of the plumbing, never a synchronised rig."""

from __future__ import annotations

import sys
import time

from .sync import CaptureBackend, FrameSet


class OpenCVBackend(CaptureBackend):
    """``cv2.VideoCapture``.  Software triggered and therefore not synchronised.

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
