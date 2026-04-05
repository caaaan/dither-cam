"""On-screen display overlay rendered via QPainter.

The OSD appears on any button/key interaction and auto-fades after a timeout.
It draws a compact settings panel at the bottom of the screen and an FPS
counter in the top-right corner.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QFont

from settings import SettingsModel


class OSDOverlay:
    FADE_DELAY = 3.0       # seconds of inactivity before fade begins
    FADE_DURATION = 0.5    # seconds for the fade-out animation

    def __init__(self, settings: SettingsModel):
        self._settings = settings
        self._last_interact = time.monotonic()
        self._status: str | None = "Starting camera..."
        self._fps: float = 0.0

        self._font = QFont()
        self._font.setPointSize(10)

        self._small_font = QFont()
        self._small_font.setPointSize(8)

    # -- Public API ------------------------------------------------------

    def touch(self):
        """Reset the fade timer (call on any user interaction)."""
        self._last_interact = time.monotonic()

    def set_status(self, status: str | None):
        self._status = status

    def set_fps(self, fps: float):
        self._fps = fps

    # -- Painting --------------------------------------------------------

    def paint(self, painter: QPainter, rect: QRectF):
        painter.save()

        if self._status:
            self._paint_status(painter, QRectF(rect))
            painter.restore()
            return

        self._paint_fps(painter, QRectF(rect))

        opacity = self._compute_opacity()
        if opacity > 0.01:
            painter.setOpacity(opacity)
            self._paint_settings_panel(painter, QRectF(rect))

        painter.restore()

    # -- Internals -------------------------------------------------------

    def _compute_opacity(self) -> float:
        elapsed = time.monotonic() - self._last_interact
        if elapsed < self.FADE_DELAY:
            return 1.0
        progress = (elapsed - self.FADE_DELAY) / self.FADE_DURATION
        if progress >= 1.0:
            return 0.0
        return 1.0 - progress

    def _paint_status(self, painter: QPainter, rect: QRectF):
        """Centered status message (e.g. 'Starting camera...')."""
        painter.setFont(self._font)
        painter.setPen(QColor(255, 255, 255))
        painter.fillRect(rect, QColor(0, 0, 0))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._status)

    def _paint_fps(self, painter: QPainter, rect: QRectF):
        painter.setFont(self._small_font)
        painter.setPen(QColor(200, 200, 200, 180))
        fps_rect = QRectF(rect.right() - 72, rect.top() + 4, 68, 16)
        painter.drawText(
            fps_rect,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self._fps:.0f} FPS",
        )

    def _paint_settings_panel(self, painter: QPainter, rect: QRectF):
        sm = self._settings
        row_h = 18
        pad = 6
        count = sm.param_count
        panel_h = count * row_h + pad * 2
        panel = QRectF(rect.left(), rect.bottom() - panel_h, rect.width(), panel_h)

        painter.fillRect(panel, QColor(0, 0, 0, 170))
        painter.setFont(self._font)

        y = panel.top() + pad
        label_w = panel.width() * 0.45
        value_x = panel.left() + pad + label_w + 4
        value_w = panel.width() - label_w - pad * 2 - 4

        for i in range(count):
            selected = i == sm.selected_index
            label = sm.param_label(i)
            value = sm.param_value_display(i)

            row = QRectF(panel.left(), y, panel.width(), row_h)

            if selected:
                painter.fillRect(row, QColor(255, 255, 255, 40))
                painter.setPen(QColor(255, 255, 255))
                prefix = "\u25b6 "
            else:
                painter.setPen(QColor(170, 170, 170))
                prefix = "  "

            label_rect = QRectF(panel.left() + pad, y, label_w, row_h)
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{prefix}{label}",
            )

            val_rect = QRectF(value_x, y, value_w, row_h)
            painter.drawText(
                val_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                value,
            )

            y += row_h
