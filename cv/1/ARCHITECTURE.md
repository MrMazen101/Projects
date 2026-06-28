# Architecture — Image Processing Vision Tool

## Class Hierarchy

```
QMainWindow
└── ImageProcessingVisionTool  (main.py)
    └── QTabWidget
        ├── EdgeDetectionTab   (edge.py)    ──┐
        ├── DenoisingTab       (Denoise.py) ──┤
        ├── TransformationTab  (Transform.py)─┤── all inherit BaseTab (base_tab.py)
        ├── MorphologyTab      (Morph.py)   ──┤
        └── SegmentationTab    (Seg.py)     ──┘
```

---

## Module Responsibilities

### `main.py` — Entry point & orchestrator

- Loads `main.ui` and clears placeholder tabs
- Instantiates one instance of each module class
- Adds each module as a tab in the `QTabWidget`
- Handles the **Open Image** button → reads the image via `cv2.imdecode` (supports non-ASCII paths) → broadcasts it to all five modules via `set_image()`
- Applies the global dark-mode stylesheet
- Shows image metadata in the status bar

```
User clicks "Open Image"
        │
        ▼
  QFileDialog → file_path
        │
        ▼
  np.fromfile + cv2.imdecode → BGR numpy array
        │
        ├──► edge_section.set_image(image)
        ├──► denoise_section.set_image(image)
        ├──► transform_section.set_image(image)
        ├──► morph_section.set_image(image)
        └──► seg_section.set_image(image)
```

---

### `base_tab.py` — Shared base class

All five modules inherit from `BaseTab(QWidget)`. It provides:

| Method / attr | Purpose |
|---|---|
| `self.current_image` | Stores the current BGR numpy array |
| `self._display_buffer` | Keeps a reference to the display copy to prevent PyQt5 memory-release bug |
| `set_image(image)` | Stores the image, then calls `_on_image_set()` |
| `_on_image_set()` | Hook overridden by each subclass to trigger its processing |
| `display_image(img)` | Converts BGR/grayscale numpy → `QPixmap` → scales to `label_image` with `KeepAspectRatio` |
| `safe_kernel(v, min, max)` | Returns the nearest odd integer within [min, max] — avoids OpenCV kernel-size errors |

**Memory safety**: The original code had a PyQt5 bug where the numpy array was garbage-collected before `QImage` finished reading from its data buffer. `BaseTab` keeps `self._display_buffer = img.copy()` alive for the lifetime of the widget.

---

### `edge.py` — Edge Detection

**UI controls**: `combo_operator` (dropdown), `slider_kernel` (kernel size)

**Processing flow**:
```
BGR image
    │
    ▼
cv2.cvtColor → Grayscale
    │
    ▼
operator_name from combo ──► Sobel / Prewitt / Roberts / Canny / LoG
    │
    ▼
cv2.normalize (NORM_MINMAX) → uint8   ← fixes clipping bug from np.uint8 cast
    │
    ▼
display_image()
```

**Canny improvement**: Thresholds auto-calculated from image median (sigma=0.33) instead of hardcoded 100/200, making it adaptive to image brightness.

---

### `Denoise.py` — Denoising Filters

**UI controls**: `combo_filter` (dropdown), `slider_kernel` (kernel size)

**Supported filters**: Gaussian Blur, Median, Averaging, Bilateral

**Key fix**: The original had a broken nested-if for kernel validation (inner `if k<1` was unreachable). Replaced with `BaseTab.safe_kernel()`.

---

### `Transform.py` — Geometric Transformations

**UI controls**: `combo_transform` (dropdown), `slider_1`, `slider_2` (transform parameters)

**Supported transforms**: Translation (dx, dy), Rotation (angle), Flipping (H/V/Both), Shearing (shear_x, shear_y), Affine (3-point warp)

**Key fix**: The original called `slider.setValue(0)` in `set_image()` which triggered `apply_transform` via signal before `self.current_image` was updated → crash. Fixed with `slider.blockSignals(True/False)` around the reset.

---

### `Morph.py` — Morphological Operations

**UI controls**: `combo_morph` (dropdown), `slider_kernel` (kernel size)

**Supported ops**: Erosion, Dilation, Opening, Closing, Gradient, Top-Hat, Black-Hat

**Kernel**: `np.ones((k, k), np.uint8)` — same kernel-validation fix as Denoise.

---

### `Seg.py` — Segmentation

**UI controls**: `combo_seg` (dropdown), `btn_manual` (brush button)

**Automated methods**:

| Method | OpenCV call |
|---|---|
| Otsu's Thresholding | `cv2.threshold(..., THRESH_BINARY + THRESH_OTSU)` |
| CLAHE (AHE) | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` |
| Histogram Peak | Mean-value thresholding |
| K-Means | `cv2.kmeans(k=4)` on flattened pixel array |

**Manual Segmentation** (critical bug fix):

Original: opened `cv2.namedWindow` + `while True: cv2.waitKey(1)` on the Qt main thread → **GUI freeze**.

Fixed: replaced entirely with `ManualSegDialog(QDialog)` — a native Qt dialog containing:
- A `QLabel` canvas that shows the image
- `mousePressEvent / mouseMoveEvent / mouseReleaseEvent` for brush drawing
- A `QPainter` overlay rendering green strokes and building the mask simultaneously
- Confirm / Clear buttons
- `get_mask()` returning a `numpy uint8` mask for `cv2.bitwise_and`

---

## Data Flow (full round-trip)

```
┌─────────────────────────────────────────────────┐
│                  main.py                        │
│                                                 │
│  QFileDialog → file_path                        │
│       │                                         │
│       ▼                                         │
│  np.fromfile(path) + cv2.imdecode               │
│  → image: np.ndarray (H, W, 3) BGR uint8        │
│       │                                         │
│       ├──► EdgeDetectionTab.set_image(image)    │
│       ├──► DenoisingTab.set_image(image)        │
│       ├──► TransformationTab.set_image(image)   │
│       ├──► MorphologyTab.set_image(image)       │
│       └──► SegmentationTab.set_image(image)     │
└─────────────────────────────────────────────────┘

Inside each Tab (via BaseTab):
┌──────────────────────────────────────────────┐
│  set_image(image)                            │
│       │ stores image + calls _on_image_set() │
│       ▼                                      │
│  apply_*(...)  ← triggered by signal or call │
│       │ reads combo + sliders                │
│       │ runs OpenCV processing               │
│       ▼                                      │
│  display_image(processed_img)                │
│       │ copy to _display_buffer              │
│       │ QImage → QPixmap → scaled to label   │
│       ▼                                      │
│  label_image.setPixmap(...)                  │
└──────────────────────────────────────────────┘
```


