"""Pipelined camera capture and frame processing threads.

Architecture
------------
CaptureThread  --frame_queue-->  ProcessThread  --DisplaySlot-->  MainThread

CaptureThread only captures; ProcessThread only dithers.  Because Numba
``@njit`` and numpy C extensions release the GIL, these two threads genuinely
run on separate CPU cores, overlapping capture latency with processing.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

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
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"size": (self._width, self._height), "format": "RGB888"},
            )
            camera.configure(config)
            camera.start()
            time.sleep(0.5)

            test = camera.capture_array()
            if test is None:
                self.camera_error.emit("Camera test capture returned None")
                return

            self.camera_ready.emit()

            while self._running:
                frame = camera.capture_array()
                if frame is None:
                    continue

                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                # picamera2 with RGB888 already outputs RGB -- no conversion
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
            self.camera_error.emit(str(exc))
        finally:
            if camera is not None:
                try:
                    camera.stop()
                    camera.close()
                except Exception:
                    pass

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
