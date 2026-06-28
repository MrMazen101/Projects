"""
edge.py — Edge Detection tab.

Fixes vs original:
  - Inherits BaseTab → no duplicate display_image / no memory bug.
  - cv2.normalize used instead of raw np.uint8 cast → no clipping artefacts
    on high-magnitude Sobel / Prewitt / Roberts results.
  - safe_kernel() enforces odd, bounded kernel sizes.
  - combo_operator text matched case-insensitively via .lower() (already was,
    kept as is) but the "log" branch is now labelled more clearly.
  - _on_image_set wires the apply call after image is stored.
"""

from PyQt5 import uic
import cv2
import numpy as np

from base_tab import BaseTab


class EdgeDetectionTab(BaseTab):
    def __init__(self):
        super().__init__()
        uic.loadUi("edge.ui", self)

        self.combo_operator.currentIndexChanged.connect(self.apply_edge_detection)
        self.slider_kernel.valueChanged.connect(self.apply_edge_detection)

    # Called by BaseTab.set_image after self.current_image is assigned
    def _on_image_set(self):
        self.apply_edge_detection()

    def apply_edge_detection(self):
        if self.current_image is None:
            return

        operator_name = self.combo_operator.currentText().lower().strip()
        kernel_size = self.safe_kernel(self.slider_kernel.value(), min_size=1, max_size=31)

        # Edge detection works on intensity, convert to grayscale
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()

        processed_img = None

        # ── First-derivative operators ──────────────────────────────────────
        if operator_name == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            magnitude = cv2.magnitude(sobelx, sobely)
            # normalize → no clipping
            processed_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif operator_name == "prewitt":
            kx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            px = cv2.filter2D(gray, cv2.CV_64F, kx)
            py = cv2.filter2D(gray, cv2.CV_64F, ky)
            magnitude = cv2.magnitude(px, py)
            processed_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif operator_name == "roberts":
            kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            rx = cv2.filter2D(gray, cv2.CV_64F, kx)
            ry = cv2.filter2D(gray, cv2.CV_64F, ky)
            magnitude = cv2.magnitude(rx, ry)
            processed_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif operator_name == "canny":
            # Canny thresholds work best auto-calculated from the median
            sigma = 0.33
            median = np.median(gray)
            low  = int(max(0,   (1.0 - sigma) * median))
            high = int(min(255, (1.0 + sigma) * median))
            processed_img = cv2.Canny(gray, low, high)

        elif "log" in operator_name or "laplacian" in operator_name:
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            lap = cv2.Laplacian(blurred, cv2.CV_64F)
            processed_img = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        if processed_img is not None:
            self.display_image(processed_img)
