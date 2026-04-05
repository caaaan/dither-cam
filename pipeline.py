"""Unified frame processing pipeline and pre-allocated buffer management.

Replaces the duplicated ``process_frame_array`` / ``apply_dither`` code paths
from the original monolithic main.py with a single ``process_frame`` function.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from helper import (
    fs_dither,
    bayer_dither,
    simple_threshold_rgb_ps1,
    block_average_rgb,
    block_average_gray,
    nearest_upscale_rgb,
    nearest_upscale_gray,
    downscale_dither_upscale,
)
from settings import SettingsSnapshot


# ---------------------------------------------------------------------------
# Pre-allocated buffer pool
# ---------------------------------------------------------------------------

@dataclass
class FrameBuffers:
    """Pre-allocated numpy arrays sized for one resolution.

    Created once when the camera starts (or resolution changes).
    Reused every frame to avoid per-frame ``np.empty`` / GC pressure.
    """

    height: int
    width: int

    work_rgb: np.ndarray = field(init=False, repr=False)
    work_gray: np.ndarray = field(init=False, repr=False)
    upscale_rgb: np.ndarray = field(init=False, repr=False)
    upscale_gray: np.ndarray = field(init=False, repr=False)

    _ds_cache: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        h, w = self.height, self.width
        self.work_rgb = np.empty((h, w, 3), dtype=np.uint8)
        self.work_gray = np.empty((h, w), dtype=np.uint8)
        self.upscale_rgb = np.empty((h, w, 3), dtype=np.uint8)
        self.upscale_gray = np.empty((h, w), dtype=np.uint8)

    def downscaled(self, scale: int, rgb: bool) -> np.ndarray:
        """Return a cached downscaled buffer, allocating only on shape change."""
        sh = max(1, self.height // scale)
        sw = max(1, self.width // scale)
        key = (scale, rgb)
        shape = (sh, sw, 3) if rgb else (sh, sw)
        buf = self._ds_cache.get(key)
        if buf is None or buf.shape != shape:
            self._ds_cache[key] = np.empty(shape, dtype=np.float32)
        return self._ds_cache[key]


# ---------------------------------------------------------------------------
# Thread-safe display slot (written by ProcessThread, read by MainThread)
# ---------------------------------------------------------------------------

class DisplaySlot:
    """Holds the latest processed frame (always RGB uint8).

    ``put`` is called by ProcessThread; ``get`` by MainThread.
    The internal copy ensures the ProcessThread can safely reuse its buffers.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray):
        copy = frame.copy()
        with self._lock:
            self._frame = copy

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_frame(
    frame: np.ndarray,
    settings: SettingsSnapshot,
    buffers: FrameBuffers,
) -> np.ndarray:
    """Apply the full dithering pipeline to a single RGB frame.

    Always returns an RGB uint8 ndarray suitable for display.
    """
    if settings.pass_through:
        return frame

    h, w = frame.shape[:2]
    is_rgb = settings.color_mode == "RGB"
    alg = settings.algorithm
    thr = settings.threshold
    scale = settings.scale

    if is_rgb:
        np.copyto(buffers.work_rgb[:h, :w], frame)
        work = buffers.work_rgb[:h, :w]
    else:
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, dst=buffers.work_gray[:h, :w])
        work = buffers.work_gray[:h, :w]

    if abs(settings.contrast - 1.0) > 0.01:
        f = work.astype(np.float32)
        np.multiply(f, settings.contrast, out=f)
        np.add(f, 128.0 * (1.0 - settings.contrast), out=f)
        np.clip(f, 0, 255, out=f)
        work = f.astype(np.uint8)

    mode = "RGB" if is_rgb else "L"
    result = (
        _dither_direct(work, alg, mode, thr, is_rgb)
        if scale <= 1
        else _dither_scaled(work, alg, mode, thr, scale, h, w, is_rgb, buffers)
    )

    if not is_rgb:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dither_direct(work, alg, mode, thr, is_rgb):
    if alg == "Floyd-Steinberg":
        return fs_dither(work.astype(np.float32), mode, thr)
    if alg == "Bayer":
        return bayer_dither(work, mode, thr)
    if is_rgb:
        return simple_threshold_rgb_ps1(work, thr)
    return np.where(work < thr, np.uint8(0), np.uint8(255)).astype(np.uint8)


def _dither_scaled(work, alg, mode, thr, scale, h, w, is_rgb, buf):
    if alg == "Floyd-Steinberg":
        return downscale_dither_upscale(work, thr, scale, mode)

    sh = max(1, h // scale)
    sw = max(1, w // scale)

    if alg == "Bayer":
        ds = buf.downscaled(scale, is_rgb)
        if is_rgb:
            small = block_average_rgb(work, ds, sh, sw, scale)
            dithered = bayer_dither(small, "RGB", thr)
            return nearest_upscale_rgb(
                dithered, buf.upscale_rgb[:h, :w], h, w, sh, sw, scale,
            )
        small = block_average_gray(work, ds, sh, sw, scale)
        dithered = bayer_dither(small, "L", thr)
        return nearest_upscale_gray(
            dithered, buf.upscale_gray[:h, :w], h, w, sh, sw, scale,
        )

    if is_rgb:
        return simple_threshold_rgb_ps1(work, thr)
    return np.where(work < thr, np.uint8(0), np.uint8(255)).astype(np.uint8)
