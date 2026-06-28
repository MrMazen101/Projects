"""
main.py — Image Processing Vision Tool — entry point.

Fixes vs original:
  - Imports now reference the fixed modules.
  - open_image: added a status bar message so the user knows an image was loaded.
  - open_image: handles non-ASCII paths on Windows via imdecode + np.fromfile
    (avoids cv2.imread silently returning None for Arabic/Chinese paths).
  - Stylesheet extended: cleaner modern look, status bar styled, label hint.
  - Window title and minimum size set.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic

from Denoise import DenoisingTab
from Transform import TransformationTab
from Morph import MorphologyTab
from Seg import SegmentationTab

try:
    from edge import EdgeDetectionTab
except ImportError as e:
    print(f"Module Import Error: {e}")
    sys.exit(1)


class ImageProcessingVisionTool(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)

        self.setWindowTitle("Image Processing Vision Tool")
        self.setMinimumSize(900, 650)

        # Clear placeholder tabs, add real modules
        self.tabWidget.clear()
        self.edge_section      = EdgeDetectionTab()
        self.denoise_section   = DenoisingTab()
        self.transform_section = TransformationTab()
        self.morph_section     = MorphologyTab()
        self.seg_section       = SegmentationTab()

        self.tabWidget.addTab(self.edge_section,      "1. Edge Detection")
        self.tabWidget.addTab(self.denoise_section,   "2. Denoising")
        self.tabWidget.addTab(self.transform_section, "3. Transformation")
        self.tabWidget.addTab(self.morph_section,     "4. Morphology")
        self.tabWidget.addTab(self.seg_section,       "5. Segmentation")

        self.btn_open_image.clicked.connect(self.open_image)
        self.statusBar().showMessage("Ready — open an image to begin.")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not file_path:
            return

        # FIX: use imdecode so paths with non-ASCII characters work correctly
        raw = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if image is None:
            self.statusBar().showMessage(f"⚠  Could not read: {file_path}")
            return

        self.edge_section.set_image(image)
        self.denoise_section.set_image(image)
        self.transform_section.set_image(image)
        self.morph_section.set_image(image)
        self.seg_section.set_image(image)

        # Show filename in title and status bar
        import os
        fname = os.path.basename(file_path)
        self.setWindowTitle(f"Image Processing Vision Tool — {fname}")
        self.statusBar().showMessage(f"Loaded: {file_path}  |  Size: {image.shape[1]}×{image.shape[0]}")


STYLESHEET = """
/* ── Window ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1E1E2E;
    color: #CDD6F4;
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}

/* ── Tab bar ─────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 6px;
    background: #181825;
}
QTabBar::tab {
    background: #313244;
    color: #BAC2DE;
    padding: 8px 18px;
    margin: 2px 2px 0 2px;
    border-radius: 6px 6px 0 0;
    min-width: 120px;
}
QTabBar::tab:selected {
    background: #89B4FA;
    color: #1E1E2E;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #45475A;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #89B4FA;
    color: #1E1E2E;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #B4D0F8; }
QPushButton:pressed{ background-color: #5A8FD6; }

/* ── ComboBox ────────────────────────────────────────────────────────── */
QComboBox {
    background-color: #313244;
    color: #CDD6F4;
    border: 1px solid #45475A;
    border-radius: 5px;
    padding: 4px 10px;
    min-height: 26px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #313244;
    color: #CDD6F4;
    selection-background-color: #89B4FA;
    selection-color: #1E1E2E;
}

/* ── Slider ──────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #45475A;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #89B4FA;
    width: 16px; height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #89B4FA; border-radius: 2px; }

/* ── Labels ──────────────────────────────────────────────────────────── */
QLabel {
    color: #CDD6F4;
}
QLabel#label_image {
    background-color: #11111B;
    border: 1px solid #313244;
    border-radius: 6px;
}

/* ── Status bar ──────────────────────────────────────────────────────── */
QStatusBar {
    background: #181825;
    color: #6C7086;
    font-size: 11px;
}

/* ── GroupBox ────────────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    color: #89B4FA;
    font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; }
"""


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = ImageProcessingVisionTool()
    window.show()
    sys.exit(app.exec_())
