"""Pipelined camera capture and frame processing threads.

Architecture
------------
CaptureSource (abstract)
    ├── PiCameraSource    -- picamera2 / libcamera (CSI ribbon cable)
    └── WebcamSource      -- any OpenCV-compatible USB/UVC camera

CaptureThread  --frame_queue-->  ProcessThread  --DisplaySlot-->  MainThread

Adding a new input source only requires subclassing CaptureSource and
implementing open(), read_frame(), and close().  CaptureThread and
everything above it stays untouched.
"""

from __future__ import annotations

import logging
import queue
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from settings import SettingsModel
from pipeline import FrameBuffers, DisplaySlot, process_frame

log = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Abstract source interface
# ---------------------------------------------------------------------------

class CaptureSource(ABC):
    """Minimal interface every capture backend must implement.

    ``open`` / ``read_frame`` / ``close`` are called only from CaptureThread.
    """

    @abstractmethod
    def open(self, width: int, height: int) -> None:
        """Initialise hardware and start streaming. Raises on failure."""

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        """Return the next RGB uint8 frame, or None if not yet available."""

    @abstractmethod
    def close(self) -> None:
        """Stop streaming and release hardware resources."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# PiCamera2 source (CSI ribbon cable)
# ---------------------------------------------------------------------------

class PiCameraSource(CaptureSource):
    """libcamera-backed CSI camera via picamera2."""

    def __init__(self):
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("picamera2 is not installed")
        self._camera: Picamera2 | None = None

    def open(self, width: int, height: int) -> None:
        log.info("PiCameraSource: creating Picamera2 instance")
        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)},
            buffer_count=2,
        )
        self._camera.configure(config)
        log.info("PiCameraSource: starting  size=%dx%d", width, height)
        self._camera.start()
        # libcamera needs ~2s for AGC/AWB to settle
        time.sleep(2.0)
        log.info("PiCameraSource: ready")

    def read_frame(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        frame = self._camera.capture_array()
        if frame is None or frame.size == 0:
            return None
        # Drop alpha channel if camera returns XRGB/RGBA
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        # picamera2 RGB888 is already RGB
        return frame

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception as exc:
                log.warning("PiCameraSource close error: %s", exc)
            finally:
                self._camera = None
        log.info("PiCameraSource: closed")


# ---------------------------------------------------------------------------
# OpenCV / USB webcam source
# ---------------------------------------------------------------------------

class WebcamSource(CaptureSource):
    """Any OpenCV-compatible camera: USB webcam, virtual cam, etc.

    Pass ``device_index=0`` for the first USB camera,
    or a ``/dev/video*`` path string for a specific device.
    """

    def __init__(self, device_index: int | str = 0):
        self._device = device_index
        self._cap: cv2.VideoCapture | None = None

    def open(self, width: int, height: int) -> None:
        log.info("WebcamSource: opening device %s", self._device)
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam device: {self._device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        log.info("WebcamSource: ready  size=%dx%d", width, height)

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        # OpenCV returns BGR -- convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                log.warning("WebcamSource close error: %s", exc)
            finally:
                self._cap = None
        log.info("WebcamSource: closed")


# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------

def make_source(prefer: str = "picamera") -> CaptureSource:
    """Return the best available source.

    prefer: "picamera"  -- try PiCameraSource, fall back to WebcamSource
            "webcam"    -- use WebcamSource directly
            "auto"      -- same as "picamera"
    """
    if prefer == "webcam":
        return WebcamSource()
    if PICAMERA_AVAILABLE:
        return PiCameraSource()
    log.warning("picamera2 not available, falling back to WebcamSource")
    return WebcamSource()


# ---------------------------------------------------------------------------
# CaptureThread -- source-agnostic
# ---------------------------------------------------------------------------

class CaptureThread(QThread):
    """Captures frames from any CaptureSource and pushes them into *frame_queue*.

    The source backend is injected at construction time; this thread never
    imports picamera2 or cv2 directly.
    """

    camera_ready = pyqtSignal()
    camera_error = pyqtSignal(str)

    def __init__(
        self,
        frame_queue: queue.Queue,
        source: CaptureSource | None = None,
        width: int = 320,
        height: int = 240,
        parent=None,
    ):
        super().__init__(parent)
        self._queue = frame_queue
        self._source = source or make_source()
        self._width = width
        self._height = height
        self._running = False

    def run(self):
        self._running = True
        log.info("CaptureThread: using source %s", self._source.name)
        try:
            self._source.open(self._width, self._height)
            self.camera_ready.emit()

            while self._running:
                try:
                    frame = self._source.read_frame()
                except Exception as exc:
                    log.error("read_frame failed: %s", exc, exc_info=True)
                    time.sleep(0.05)
                    continue

                if frame is None:
                    continue

                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass

        except IndexError:
            msg = "No CSI camera detected. Check cable or switch to webcam."
            log.critical(msg, exc_info=True)
            self.camera_error.emit(msg)
        except Exception as exc:
            log.critical("CaptureThread crashed: %s", exc, exc_info=True)
            self.camera_error.emit(str(exc))
        finally:
            self._source.close()
            log.info("CaptureThread exited")

    def stop(self):
        self._running = False
        self.wait(3000)


# ---------------------------------------------------------------------------
# ProcessThread -- unchanged
# ---------------------------------------------------------------------------

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
