"""Pipelined camera capture and frame processing threads.

Architecture
------------
CaptureThread  --frame_queue-->  ProcessThread  --DisplaySlot-->  MainThread

CaptureThread only captures; ProcessThread only dithers.  Because Numba
``@njit`` and numpy C extensions release the GIL, these two threads genuinely
run on separate CPU cores, overlapping capture latency with processing.
"""

from __future__ import annotations

import logging
import queue
import time

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

from settings import SettingsModel
from pipeline import FrameBuffers, DisplaySlot, process_frame

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


class CaptureThread(QThread):
    """Captures frames from picamera2 and pushes them into *frame_queue*.

    No image processing happens here -- that is the job of ProcessThread.
    """

    camera_ready = pyqtSignal()
    camera_error = pyqtSignal(str)

    def __init__(
        self,
        frame_queue: queue.Queue,
        width: int = 320,
        height: int = 240,
        parent=None,
    ):
        super().__init__(parent)
        self._queue = frame_queue
        self._width = width
        self._height = height
        self._running = False

    def run(self):
        self._running = True
        camera = None
        try:
            log.info("Creating Picamera2 instance")
            camera = Picamera2()

            # Cap at 30fps and let AGC/AWB converge before emitting ready.
            controls = {"FrameDurationLimits": (33333, 33333)}
            config = camera.create_preview_configuration(
                main={"size": (self._width, self._height), "format": "RGB888"},
                controls=controls,
                buffer_count=2,
            )
            camera.configure(config)
            log.info("Starting camera  size=%dx%d", self._width, self._height)
            camera.start()

            # libcamera needs ~1-2s for AGC/AWB to settle.
            # We emit camera_ready here so the UI clears "Starting camera..."
            # immediately; the first few frames may look dark/washed but that
            # is normal and corrects itself within a second.
            time.sleep(2.0)
            log.info("Camera ready, entering capture loop")
            self.camera_ready.emit()

            while self._running:
                try:
                    frame = camera.capture_array()
                except Exception as exc:
                    log.error("capture_array failed: %s", exc, exc_info=True)
                    time.sleep(0.05)
                    continue

                if frame is None or frame.size == 0:
                    log.debug("Empty frame skipped")
                    continue

                # Drop alpha channel if camera returns XRGB/RGBA
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                # picamera2 RGB888 is already RGB -- no conversion needed
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass

        except Exception as exc:
            log.critical("CaptureThread crashed: %s", exc, exc_info=True)
            self.camera_error.emit(str(exc))
        finally:
            if camera is not None:
                try:
                    camera.stop()
                    camera.close()
                except Exception:
                    pass
            log.info("CaptureThread exited")

    def stop(self):
        self._running = False
        self.wait(3000)


class ProcessThread(QThread):
    """Pulls raw frames from *frame_queue*, applies the dithering pipeline,
    and writes results into a :class:`DisplaySlot`.
    """

    frame_ready = pyqtSignal()

    def __init__(
        self,
        frame_queue: queue.Queue,
        settings: SettingsModel,
        display: DisplaySlot,
        parent=None,
    ):
        super().__init__(parent)
        self._queue = frame_queue
        self._settings = settings
        self._display = display
        self._running = False

    def run(self):
        self._running = True
        buffers: FrameBuffers | None = None

        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            if buffers is None or buffers.height != h or buffers.width != w:
                buffers = FrameBuffers(h, w)

            snap = self._settings.snapshot()
            result = process_frame(frame, snap, buffers)

            self._display.put(result)
            self.frame_ready.emit()

    def stop(self):
        self._running = False
        self.wait(3000)
