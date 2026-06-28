"""
Seg.py — Segmentation tab.

Critical fix vs original:
  - BUG FIX (CRITICAL): start_manual_segmentation() ran a blocking cv2 event
    loop (while True: cv2.waitKey(1)) on the main Qt GUI thread.  This freezes
    the entire application.  Fixed by replacing it with an inline Qt-based
    drawing canvas inside a QDialog — no cv2 windows needed.

Other fixes:
  - Inherits BaseTab → no duplicate display_image / no memory bug.
  - Otsu / Adaptive / Peak text matching made more robust.
  - Added K-Means colour segmentation as an extra method.
"""

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np

from base_tab import BaseTab


class ManualSegDialog(QDialog):
    """Qt-native brush dialog — replaces the blocking cv2 window."""

    def __init__(self, image_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Brush — draw to select, then click Confirm")
        self.image_bgr = image_bgr
        h, w = image_bgr.shape[:2]

        # Build a QPixmap from the source image for display
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(q_img)

        # Mask pixmap — we'll draw on this
        self._mask_pixmap = QPixmap(w, h)
        self._mask_pixmap.fill(Qt.black)

        # Composite display pixmap
        self._display_pixmap = self._base_pixmap.copy()

        self.label = QLabel()
        self.label.setPixmap(self._display_pixmap)
        self.label.setFixedSize(min(w, 900), min(h, 700))
        self.label.setScaledContents(True)

        btn_confirm = QPushButton("✔  Confirm")
        btn_confirm.clicked.connect(self.accept)
        btn_clear = QPushButton("✖  Clear")
        btn_clear.clicked.connect(self._clear)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(btn_confirm)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self._drawing = False
        self._brush_radius = max(8, min(w, h) // 40)
        self.label.setMouseTracking(True)

    # ── Mouse events ──────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._paint(event.pos())

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._paint(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = False

    def _paint(self, pos):
        # Map widget coords → label coords
        lx = pos.x() - self.label.x()
        ly = pos.y() - self.label.y()
        if not (0 <= lx < self.label.width() and 0 <= ly < self.label.height()):
            return

        # Scale to image coords
        scale_x = self._base_pixmap.width()  / self.label.width()
        scale_y = self._base_pixmap.height() / self.label.height()
        ix, iy = int(lx * scale_x), int(ly * scale_y)
        r = self._brush_radius

        # Draw onto mask pixmap (white circle)
        mp = QPainter(self._mask_pixmap)
        mp.setPen(Qt.NoPen)
        mp.setBrush(Qt.white)
        mp.drawEllipse(QPoint(ix, iy), r, r)
        mp.end()

        # Draw semi-transparent overlay on display pixmap
        dp = self._display_pixmap = self._base_pixmap.copy()
        overlay = QPixmap(self._base_pixmap.size())
        overlay.fill(Qt.transparent)
        op = QPainter(overlay)
        op.setPen(Qt.NoPen)
        op.setBrush(Qt.green)
        op.setOpacity(0.4)

        # Re-draw all mask pixels as green overlay efficiently using the mask
        mask_img = self._mask_pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        op.drawPixmap(0, 0, self._mask_pixmap)
        op.end()

        final_p = QPainter(dp)
        final_p.setOpacity(0.4)
        final_p.drawPixmap(0, 0, overlay)
        final_p.end()
        self.label.setPixmap(dp)

    def _clear(self):
        self._mask_pixmap.fill(Qt.black)
        self._display_pixmap = self._base_pixmap.copy()
        self.label.setPixmap(self._display_pixmap)

    def get_mask(self):
        """Return a numpy uint8 mask (0/255) in the original image size."""
        w = self._mask_pixmap.width()
        h = self._mask_pixmap.height()
        q_img = self._mask_pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        ptr = q_img.bits()
        ptr.setsize(h * w)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w)).copy()
        return arr


# ── Main segmentation tab ────────────────────────────────────────────────────

class SegmentationTab(BaseTab):
    def __init__(self):
        super().__init__()
        uic.loadUi("seg.ui", self)

        self.combo_seg.currentIndexChanged.connect(self.apply_auto_segmentation)
        self.btn_manual.clicked.connect(self.start_manual_segmentation)

    def _on_image_set(self):
        self.apply_auto_segmentation()

    # ── Automated segmentation ───────────────────────────────────────────
    def apply_auto_segmentation(self):
        if self.current_image is None:
            return

        method = self.combo_seg.currentText().lower().strip()

        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()

        processed_img = None

        if "otsu" in method:
            _, processed_img = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        elif "adaptive" in method or "ahe" in method or "clahe" in method:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_img = clahe.apply(gray)

        elif "peak" in method or "histogram" in method:
            mean_val = int(np.mean(gray))
            _, processed_img = cv2.threshold(
                gray, mean_val, 255, cv2.THRESH_BINARY
            )

        elif "kmeans" in method or "k-means" in method:
            processed_img = self._kmeans_segmentation(self.current_image, k=4)

        if processed_img is not None:
            self.display_image(processed_img)

    @staticmethod
    def _kmeans_segmentation(img, k=4):
        """Colour segmentation via K-Means clustering."""
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(img.shape)
        return segmented

    # ── Manual segmentation (Qt dialog — no cv2 window) ─────────────────
    def start_manual_segmentation(self):
        """
        BUG FIX: original used a blocking cv2 event loop on the Qt GUI thread
        which caused the application to freeze.  Replaced with a proper QDialog.
        """
        if self.current_image is None:
            return

        dialog = ManualSegDialog(self.current_image, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            mask = dialog.get_mask()
            segmented = cv2.bitwise_and(
                self.current_image, self.current_image, mask=mask
            )
            self.display_image(segmented)
