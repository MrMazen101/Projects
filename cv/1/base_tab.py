"""
base_tab.py — Shared base class for all processing tabs.
Fixes:
  - QImage memory bug: keeps a reference to the numpy array so it isn't
    garbage-collected while QImage still points at its data buffer.
  - display_image centralised: one implementation instead of 5 identical copies.
  - AspectRatioMode uses the correct Qt enum instead of the magic number 1.
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2


class BaseTab(QWidget):
    """All processing tabs inherit from this class."""

    def __init__(self):
        super().__init__()
        self.current_image = None
        # Keep a reference so the numpy buffer isn't freed before QPixmap
        # finishes using it (classic PyQt5 memory bug).
        self._display_buffer = None

    def set_image(self, image):
        """Receive a BGR numpy array from the main window."""
        self.current_image = image
        self._on_image_set()

    def _on_image_set(self):
        """Override in subclasses to run logic after a new image arrives."""
        pass

    # ------------------------------------------------------------------
    # Shared display helper
    # ------------------------------------------------------------------
    def display_image(self, img):
        """
        Convert an OpenCV numpy array (BGR or grayscale) to a QPixmap and
        show it on self.label_image, scaled to fit while keeping aspect ratio.
        """
        if img is None:
            return

        if len(img.shape) == 2:
            # Grayscale
            h, w = img.shape
            # Keep reference alive
            self._display_buffer = img.copy()
            q_img = QImage(
                self._display_buffer.data, w, h, w,
                QImage.Format_Grayscale8
            )
        else:
            # Colour: convert BGR → RGB for Qt
            h, w, ch = img.shape
            self._display_buffer = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(
                self._display_buffer.data, w, h, ch * w,
                QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(q_img)
        self.label_image.setPixmap(
            pixmap.scaled(
                self.label_image.width(),
                self.label_image.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # ------------------------------------------------------------------
    # Shared kernel-size helper
    # ------------------------------------------------------------------
    @staticmethod
    def safe_kernel(value, min_size=1, max_size=31):
        """Return an odd kernel size in [min_size, max_size]."""
        k = max(min_size, min(value, max_size))
        if k % 2 == 0:
            k += 1
        return k
