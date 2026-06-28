"""
Morph.py — Morphological Operations tab.

Fixes vs original:
  - Inherits BaseTab → no duplicate display_image / no memory bug.
  - BUG FIX: same nested-if kernel bug as Denoise.py — fixed via safe_kernel.
  - Top-hat and Black-hat added for completeness (common exam operations).
"""

from PyQt5 import uic
import cv2
import numpy as np

from base_tab import BaseTab


class MorphologyTab(BaseTab):
    def __init__(self):
        super().__init__()
        uic.loadUi("morph.ui", self)

        self.combo_morph.currentIndexChanged.connect(self.apply_morph)
        self.slider_kernel.valueChanged.connect(self.apply_morph)

    def _on_image_set(self):
        self.apply_morph()

    def apply_morph(self):
        if self.current_image is None:
            return

        morph_name = self.combo_morph.currentText().lower().strip()
        # BUG FIX: safe_kernel replaces the broken nested-if in original
        kernel_size = self.safe_kernel(self.slider_kernel.value(), min_size=1, max_size=31)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_img = None

        if morph_name == "erosion":
            processed_img = cv2.erode(self.current_image, kernel, iterations=1)

        elif morph_name == "dilation":
            processed_img = cv2.dilate(self.current_image, kernel, iterations=1)

        elif morph_name == "opening":
            processed_img = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, kernel)

        elif morph_name == "closing":
            processed_img = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, kernel)

        elif "gradient" in morph_name:
            processed_img = cv2.morphologyEx(self.current_image, cv2.MORPH_GRADIENT, kernel)

        elif "top" in morph_name or "white" in morph_name:
            # Top-hat: original minus opening → bright details on dark background
            processed_img = cv2.morphologyEx(self.current_image, cv2.MORPH_TOPHAT, kernel)

        elif "black" in morph_name:
            # Black-hat: closing minus original → dark details on bright background
            processed_img = cv2.morphologyEx(self.current_image, cv2.MORPH_BLACKHAT, kernel)

        if processed_img is not None:
            self.display_image(processed_img)
