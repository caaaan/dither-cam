"""Thread-safe settings model with navigable parameters for GPIO/keyboard control."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from PyQt6.QtCore import QObject, pyqtSignal

ALGORITHMS = ("Floyd-Steinberg", "Bayer", "Simple Threshold")


@dataclass(frozen=True)
class SettingsSnapshot:
    """Immutable snapshot for thread-safe reads by ProcessThread."""
    algorithm: str
    threshold: int
    contrast: float
    scale: int
    color_mode: str      # "RGB" or "L"
    pass_through: bool


class _Param:
    __slots__ = ("key", "label")

    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label


PARAMS = (
    _Param("algorithm", "Algorithm"),
    _Param("threshold", "Threshold"),
    _Param("contrast", "Contrast"),
    _Param("scale", "Pixel Scale"),
    _Param("color_mode", "Color"),
    _Param("pass_through", "Pass-through"),
)


class SettingsModel(QObject):
    """Owns all adjustable parameters.

    Modified from the MainThread (via queued GPIO signals or keyboard events).
    Read from any thread via ``snapshot()``.
    """

    changed = pyqtSignal()
    nav_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()

        self._alg_idx: int = 0
        self._threshold: int = 128
        self._contrast: int = 100   # divide by 100.0 for the float value
        self._scale: int = 1
        self._color_rgb: bool = True
        self._pass_through: bool = False

        self._sel: int = 0

    # -- Thread-safe snapshot for ProcessThread --------------------------

    def snapshot(self) -> SettingsSnapshot:
        with self._lock:
            return SettingsSnapshot(
                algorithm=ALGORITHMS[self._alg_idx],
                threshold=self._threshold,
                contrast=self._contrast / 100.0,
                scale=self._scale,
                color_mode="RGB" if self._color_rgb else "L",
                pass_through=self._pass_through,
            )

    # -- OSD helpers (MainThread only) -----------------------------------

    @property
    def selected_index(self) -> int:
        return self._sel

    @property
    def param_count(self) -> int:
        return len(PARAMS)

    def param_label(self, idx: int) -> str:
        return PARAMS[idx].label

    def param_value_display(self, idx: int) -> str:
        key = PARAMS[idx].key
        with self._lock:
            if key == "algorithm":
                return ALGORITHMS[self._alg_idx]
            if key == "threshold":
                return str(self._threshold)
            if key == "contrast":
                return f"{self._contrast / 100.0:.1f}"
            if key == "scale":
                return str(self._scale)
            if key == "color_mode":
                return "RGB" if self._color_rgb else "Grayscale"
            if key == "pass_through":
                return "ON" if self._pass_through else "OFF"
        return ""

    # -- Navigation (MainThread) -----------------------------------------

    def move_up(self):
        self._sel = (self._sel - 1) % len(PARAMS)
        self.nav_changed.emit()

    def move_down(self):
        self._sel = (self._sel + 1) % len(PARAMS)
        self.nav_changed.emit()

    def adjust_left(self):
        self._do_adjust(-1)

    def adjust_right(self):
        self._do_adjust(1)

    def _do_adjust(self, d: int):
        key = PARAMS[self._sel].key
        with self._lock:
            if key == "algorithm":
                self._alg_idx = (self._alg_idx + d) % len(ALGORITHMS)
            elif key == "threshold":
                self._threshold = max(1, min(254, self._threshold + d * 5))
            elif key == "contrast":
                self._contrast = max(10, min(500, self._contrast + d * 10))
            elif key == "scale":
                self._scale = max(1, min(8, self._scale + d))
            elif key == "color_mode":
                self._color_rgb = not self._color_rgb
            elif key == "pass_through":
                self._pass_through = not self._pass_through
        self.changed.emit()
        self.nav_changed.emit()
