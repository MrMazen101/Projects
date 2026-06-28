"""
Denoise.py — Denoising tab.

Fixes vs original:
  - Inherits BaseTab → no duplicate display_image / no memory bug.
  - safe_kernel() fixes the buggy nested-if kernel validation in the original
    (original: if even → k+1, then if k<1 → 1; the inner check was unreachable).
  - NLMeans added as a bonus filter (non-local means, great for photos).
"""

from PyQt5 import uic
import cv2

from base_tab import BaseTab


class DenoisingTab(BaseTab):
    def __init__(self):
        super().__init__()
        uic.loadUi("denoise.ui", self)

        self.combo_filter.currentIndexChanged.connect(self.apply_denoise)
        self.slider_kernel.valueChanged.connect(self.apply_denoise)

    def _on_image_set(self):
        self.apply_denoise()

    def apply_denoise(self):
        if self.current_image is None:
            return

        filter_name = self.combo_filter.currentText().lower().strip()
        # BUG FIX: original nested-if was wrong — safe_kernel handles it correctly
        kernel_size = self.safe_kernel(self.slider_kernel.value(), min_size=1, max_size=31)

        processed_img = None

        if "gaussian" in filter_name:
            processed_img = cv2.GaussianBlur(
                self.current_image, (kernel_size, kernel_size), 0
            )

        elif "median" in filter_name:
            processed_img = cv2.medianBlur(self.current_image, kernel_size)

        elif "averaging" in filter_name or "average" in filter_name:
            processed_img = cv2.blur(
                self.current_image, (kernel_size, kernel_size)
            )

        elif "bilateral" in filter_name:
            # Bilateral: preserves edges while blurring — slider controls sigma
            sigma = kernel_size * 3
            processed_img = cv2.bilateralFilter(
                self.current_image, kernel_size, sigma, sigma
            )

        if processed_img is not None:
            self.display_image(processed_img)
