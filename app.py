"""Main application window: fullscreen camera view with OSD overlay.

Wires together the camera pipeline, settings model, GPIO controller,
and OSD overlay into a single cohesive application.
"""

from __future__ import annotations

import logging
import os
import queue
import time
from datetime import datetime

log = logging.getLogger(__name__)

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPainter, QKeyEvent
from PyQt6.QtWidgets import QMainWindow, QWidget

import config
from settings import SettingsModel
from pipeline import DisplaySlot
from camera import CaptureThread, ProcessThread, CaptureSource, make_source
from overlay import OSDOverlay
from controls import GPIOController


# ---------------------------------------------------------------------------
# Camera view widget
# ---------------------------------------------------------------------------

class CameraView(QWidget):
    """Fills the window, drawing the latest camera frame and the OSD on top."""

    def __init__(self, osd: OSDOverlay, parent=None):
        super().__init__(parent)
        self._osd = osd
        self._frame: np.ndarray | None = None
        self._qimage: QImage | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def update_frame(self, frame: np.ndarray):
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        self._frame = frame
        h, w = frame.shape[:2]
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            self._qimage = QImage(
                frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888,
            )
        else:
            self._qimage = QImage(
                frame.data, w, h, frame.strides[0], QImage.Format.Format_Grayscale8,
            )
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        if self._qimage:
            painter.drawImage(self.rect(), self._qimage)
        else:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)

        self._osd.paint(painter, self.rect())
        painter.end()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class DitherApp(QMainWindow):
    def __init__(
        self,
        fullscreen: bool = False,
        width: int = 480,
        height: int = 320,
        source: CaptureSource | None = None,
    ):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(width, height)

        # --- Core state ---
        self.settings = SettingsModel(self)

        # --- OSD ---
        self._osd = OSDOverlay(self.settings)

        # --- View ---
        self._view = CameraView(self._osd, self)
        self.setCentralWidget(self._view)

        # --- Display slot (ProcessThread -> MainThread) ---
        self._display = DisplaySlot()

        # --- Camera threads ---
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._capture_thread: CaptureThread | None = None
        self._process_thread: ProcessThread | None = None
        self._source = source  # injected CaptureSource, or None = auto-detect

        # --- Stats tracking ---
        self._frame_times: list[float] = []
        self._frames_received: int = 0
        self._frames_dropped: int = 0
        self._last_queue_size: int = 0
        self._stats_log_interval: int = 10   # log to file every N seconds

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        self._stats_log_counter: int = 0

        # --- GPIO ---
        self._gpio = GPIOController(self)
        self._wire_gpio()

        # --- Re-process on settings change while camera is not running ---
        self.settings.changed.connect(self._on_settings_changed)

        # --- Fullscreen ---
        if fullscreen:
            self.showFullScreen()

        # --- Auto-start camera ---
        # Always start -- the source was chosen by the caller (picamera / webcam / auto).
        # WebcamSource works without picamera2 being installed.
        QTimer.singleShot(100, self._start_camera)

    # -- GPIO wiring -----------------------------------------------------

    def _wire_gpio(self):
        g = self._gpio
        g.navigate_up.connect(self._handle_up)
        g.navigate_down.connect(self._handle_down)
        g.adjust_left.connect(self._handle_left)
        g.adjust_right.connect(self._handle_right)
        g.capture.connect(self._save_capture)

    def _handle_up(self):
        self.settings.move_up()
        self._osd.touch()
        self._view.update()

    def _handle_down(self):
        self.settings.move_down()
        self._osd.touch()
        self._view.update()

    def _handle_left(self):
        self.settings.adjust_left()
        self._osd.touch()
        self._view.update()

    def _handle_right(self):
        self.settings.adjust_right()
        self._osd.touch()
        self._view.update()

    # -- Camera lifecycle ------------------------------------------------

    def _start_camera(self):
        self._osd.set_status("Starting camera...")
        self._view.update()

        self._capture_thread = CaptureThread(
            self._frame_queue,
            source=self._source,
            parent=self,
        )
        self._capture_thread.camera_ready.connect(self._on_camera_ready)
        self._capture_thread.camera_error.connect(self._on_camera_error)

        self._process_thread = ProcessThread(
            self._frame_queue, self.settings, self._display, parent=self,
        )
        self._process_thread.frame_ready.connect(self._on_frame_ready)

        self._capture_thread.start()
        self._process_thread.start()

    def _on_camera_ready(self):
        self._osd.set_status(None)
        self._osd.touch()
        self._view.update()

    def _on_camera_error(self, msg: str):
        log.error("Camera error: %s", msg)
        self._osd.set_status(f"Camera error: {msg}")
        self._view.update()

    # -- Frame display ---------------------------------------------------

    def _on_frame_ready(self):
        frame = self._display.get()
        if frame is not None:
            self._view.update_frame(frame)
            self._frame_times.append(time.monotonic())
            self._frames_received += 1
        else:
            self._frames_dropped += 1
        self._last_queue_size = self._frame_queue.qsize()

    def _update_fps(self):
        now = time.monotonic()
        self._frame_times = [t for t in self._frame_times if now - t < 2.0]

        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
        else:
            fps = 0.0

        self._osd.set_fps(fps)

        # Log detailed stats every N seconds
        self._stats_log_counter += 1
        if self._stats_log_counter >= self._stats_log_interval:
            self._stats_log_counter = 0
            self._log_stats(fps)

    def _log_stats(self, fps: float):
        snap = self.settings.snapshot()

        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            cpu = psutil.cpu_percent(interval=None)
            sys_info = f"  cpu={cpu:.1f}%  mem={mem_mb:.1f}MB"
        except Exception:
            sys_info = ""

        log.info(
            "STATS  fps=%.1f  frames=%d  dropped=%d  queue=%d/%d"
            "  alg=%s  scale=%d  mode=%s%s",
            fps,
            self._frames_received,
            self._frames_dropped,
            self._last_queue_size,
            self._frame_queue.maxsize,
            snap.algorithm,
            snap.scale,
            snap.color_mode,
            sys_info,
        )

    # -- Settings change (for static images / OSD refresh) ---------------

    def _on_settings_changed(self):
        self._osd.touch()
        self._view.update()

    # -- Capture / save --------------------------------------------------

    def _save_capture(self):
        frame = self._display.get()
        if frame is None:
            print("No frame to save")
            return

        public = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
        os.makedirs(public, exist_ok=True)

        ext = config.DEFAULT_CAPTURE_FORMAT.lstrip(".")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(public, f"capture_{ts}.{ext}")

        try:
            import cv2
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            log.info("Captured frame saved: %s", path)
        except Exception as exc:
            log.error("Save failed: %s", exc, exc_info=True)

    # -- Keyboard fallback -----------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Up:
            self._handle_up()
        elif key == Qt.Key.Key_Down:
            self._handle_down()
        elif key == Qt.Key.Key_Left:
            self._handle_left()
        elif key == Qt.Key.Key_Right:
            self._handle_right()
        elif key in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self._save_capture()
        elif key == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        elif key == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    # -- Cleanup ---------------------------------------------------------

    def closeEvent(self, event):
        if self._capture_thread:
            self._capture_thread.stop()
        if self._process_thread:
            self._process_thread.stop()
        self._gpio.cleanup()
        event.accept()
