"""GPIO button controller with graceful fallback on non-Pi platforms.

Emits Qt signals that are connected to SettingsModel navigation methods
in app.py.  Because gpiozero callbacks fire on a background thread, all
signal emissions are automatically queued by Qt, ensuring SettingsModel
mutations happen on the MainThread.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

import config

try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class GPIOController(QObject):
    """Maps physical GPIO buttons to Qt signals."""

    navigate_up = pyqtSignal()
    navigate_down = pyqtSignal()
    adjust_left = pyqtSignal()
    adjust_right = pyqtSignal()
    capture = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons: dict[int, Button] = {}

        if GPIO_AVAILABLE:
            self._init_gpio()
        else:
            print("GPIO not available -- using keyboard controls only")

    def _init_gpio(self):
        pin_signal_map = {
            config.GPIO_UP: self.navigate_up,
            config.GPIO_DOWN: self.navigate_down,
            config.GPIO_LEFT: self.adjust_left,
            config.GPIO_RIGHT: self.adjust_right,
            config.GPIO_CAPTURE: self.capture,
        }
        for pin, sig in pin_signal_map.items():
            try:
                btn = Button(pin, pull_up=True, bounce_time=0.05)
                btn.when_pressed = lambda s=sig: s.emit()
                self._buttons[pin] = btn
            except Exception as exc:
                print(f"GPIO pin {pin} init failed: {exc}")

    def cleanup(self):
        for btn in self._buttons.values():
            try:
                btn.close()
            except Exception:
                pass
        self._buttons.clear()
