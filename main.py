"""DitherCam -- entry point.

Starts background Numba JIT warmup, then launches the Qt application.
The warmup runs on a daemon thread so the UI and camera appear instantly;
the first few frames may be slightly slower while compilation finishes.
"""

from __future__ import annotations

import sys
import argparse
import threading


def _warmup_numba():
    """Trigger Numba JIT compilation for all dithering kernels."""
    try:
        import os
        import numpy as np
        import cv2
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

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ditherer.jpeg")
        if not os.path.exists(path):
            print("Warmup image not found, skipping Numba warmup")
            return

        img = cv2.imread(path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape[:2]

        fs_dither(img.astype(np.float32), "RGB", 128)
        bayer_dither(img.copy(), "RGB", 128)
        simple_threshold_rgb_ps1(img, 128)

        fs_dither(gray.astype(np.float32), "L", 128)
        bayer_dither(gray.copy(), "L", 128)

        for ps in (2, 4):
            downscale_dither_upscale(img, 128, ps, "RGB")
            downscale_dither_upscale(gray, 128, ps, "L")

            sh, sw = max(1, h // ps), max(1, w // ps)

            ds_rgb = np.empty((sh, sw, 3), dtype=np.float32)
            block_average_rgb(img, ds_rgb, sh, sw, ps)
            up_rgb = np.empty((h, w, 3), dtype=np.uint8)
            nearest_upscale_rgb(ds_rgb, up_rgb, h, w, sh, sw, ps)

            ds_g = np.empty((sh, sw), dtype=np.float32)
            block_average_gray(gray, ds_g, sh, sw, ps)
            up_g = np.empty((h, w), dtype=np.uint8)
            nearest_upscale_gray(ds_g, up_g, h, w, sh, sw, ps)

        print("Numba warmup complete")
    except Exception as exc:
        print(f"Numba warmup error (non-critical): {exc}")


def main():
    parser = argparse.ArgumentParser(description="DitherCam")
    parser.add_argument("--fullscreen", "-f", action="store_true",
                        help="Launch in fullscreen mode")
    parser.add_argument("--resolution", "-r", default="480x320",
                        help="Window resolution WxH (default: 480x320)")
    args = parser.parse_args()

    threading.Thread(target=_warmup_numba, daemon=True).start()

    from PyQt6.QtWidgets import QApplication
    from app import DitherApp

    qapp = QApplication(sys.argv)

    try:
        w, h = map(int, args.resolution.split("x"))
    except ValueError:
        print(f"Invalid resolution '{args.resolution}', using 480x320")
        w, h = 480, 320

    window = DitherApp(fullscreen=args.fullscreen, width=w, height=h)
    window.show()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
