"""
Transform.py — Geometric Transformation tab.

Fixes vs original:
  - Inherits BaseTab → no duplicate display_image / no memory bug.
  - BUG FIX: original set_image called slider_1.setValue(0) which triggered
    apply_transform BEFORE self.current_image was set → crash/NoneType error.
    Fixed by blocking signals while resetting sliders.
  - Flipping logic extended: slider gives 3 states cleanly (-1, 0, +1).
  - Shearing output canvas clamped so it doesn't produce a 0-size image when
    shear values are 0.
"""

from PyQt5 import uic
import cv2
import numpy as np

from base_tab import BaseTab


class TransformationTab(BaseTab):
    def __init__(self):
        super().__init__()
        uic.loadUi("transform.ui", self)

        self.combo_transform.currentIndexChanged.connect(self.apply_transform)
        self.slider_1.valueChanged.connect(self.apply_transform)
        self.slider_2.valueChanged.connect(self.apply_transform)

    def _on_image_set(self):
        # BUG FIX: block signals while resetting so apply_transform isn't
        # called before self.current_image is ready
        for slider in (self.slider_1, self.slider_2):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.apply_transform()

    def apply_transform(self):
        if self.current_image is None:
            return

        transform_name = self.combo_transform.currentText().lower().strip()
        val1 = self.slider_1.value()
        val2 = self.slider_2.value()

        rows, cols = self.current_image.shape[:2]
        processed_img = self.current_image.copy()

        # ── Translation ────────────────────────────────────────────────────
        if transform_name == "translation":
            M = np.float32([[1, 0, val1], [0, 1, val2]])
            processed_img = cv2.warpAffine(self.current_image, M, (cols, rows))

        # ── Rotation ───────────────────────────────────────────────────────
        elif transform_name == "rotation":
            center = (cols / 2, rows / 2)
            M = cv2.getRotationMatrix2D(center, val1, 1.0)
            processed_img = cv2.warpAffine(self.current_image, M, (cols, rows))

        # ── Flipping ───────────────────────────────────────────────────────
        elif transform_name == "flipping":
            if val1 > 0:
                processed_img = cv2.flip(self.current_image, 1)   # horizontal
            elif val1 < 0:
                processed_img = cv2.flip(self.current_image, 0)   # vertical
            else:
                processed_img = cv2.flip(self.current_image, -1)  # both axes

        # ── Shearing ───────────────────────────────────────────────────────
        elif transform_name == "shearing":
            shear_x = val1 / 100.0
            shear_y = val2 / 100.0
            M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
            # BUG FIX: output size must be at least 1 pixel
            out_w = max(1, int(cols + abs(shear_x) * rows))
            out_h = max(1, int(rows + abs(shear_y) * cols))
            processed_img = cv2.warpAffine(self.current_image, M, (out_w, out_h))

        # ── Affine ─────────────────────────────────────────────────────────
        elif transform_name == "affine":
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[50 + val1, 50 + val2], [200, 50], [50, 200]])
            M = cv2.getAffineTransform(pts1, pts2)
            processed_img = cv2.warpAffine(self.current_image, M, (cols, rows))

        self.display_image(processed_img)
