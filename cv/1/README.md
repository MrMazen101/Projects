# Image Processing Vision Tool

A desktop application built with **PyQt5** and **OpenCV** for interactive image processing. It covers five core computer vision modules: Edge Detection, Denoising, Geometric Transformations, Morphological Operations, and Image Segmentation — each as a self-contained Python class with its own UI.

---

## Features

| Module | Techniques |
|---|---|
| **Edge Detection** | Sobel, Prewitt, Roberts, Canny, Laplacian of Gaussian (LoG) |
| **Denoising** | Gaussian Blur, Median Filter, Averaging Filter, Bilateral Filter |
| **Transformation** | Translation, Rotation, Flipping, Shearing, Affine |
| **Morphology** | Erosion, Dilation, Opening, Closing, Gradient, Top-Hat, Black-Hat |
| **Segmentation** | Otsu's Thresholding, AHE (CLAHE), Histogram Peak, K-Means, Manual Brush |

---

## Project Structure

```
project/
│
├── main.py           ← Entry point — assembles all tabs into the main window
├── base_tab.py       ← Shared base class (display logic, memory safety, kernel utils)
│
├── edge.py           ← Edge Detection module
├── Denoise.py        ← Denoising module
├── Transform.py      ← Geometric Transformation module
├── Morph.py          ← Morphology module
├── Seg.py            ← Segmentation module
│
├── main.ui           ← Main window Qt Designer layout
├── edge.ui           ← Edge Detection tab layout
├── denoise.ui        ← Denoising tab layout
├── transform.ui      ← Transformation tab layout
├── morph.ui          ← Morphology tab layout
└── seg.ui            ← Segmentation tab layout
```

---

## Setup

### 1. Clone / download the project

```bash
git clone https://github.com/MrMazen101/Projects/cv/1.git
cd Projects/cv/1
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

---

## Usage

1. Launch the app — you'll see five tabs across the top.
2. Click **Open Image** to load any PNG, JPG, BMP, or TIFF file.
3. Navigate to any tab and use the dropdown to select a technique.
4. Move the sliders to adjust parameters — the result updates in real-time.
5. For **Manual Segmentation**, click the brush button, draw on the pop-up canvas, then click Confirm.

---

## Task Experimental — Minnie Mouse & Portrait

The two provided noisy images were processed as follows:

### Image 1 — Minnie Mouse (cartoon, salt-and-pepper noise)
<img width="388" height="263" alt="Screenshot 2026-06-28 055030" src="https://github.com/user-attachments/assets/19b7176d-7414-435d-a8a9-88c0aafb03fa" />


**Denoising → Median Filter (kernel 3×3)**

The noise pattern consists of isolated bright and dark pixels scattered randomly — this is classic salt-and-pepper noise. The Median Filter is the optimal choice because it replaces each pixel with the median of its neighbourhood, which completely removes isolated outliers without blurring sharp cartoon edges (unlike Gaussian which would smear them).

**Edge Detection → Canny**

After denoising, Canny was applied with auto-computed thresholds (based on image median). Canny is ideal here because the cartoon has clean, high-contrast outlines and the dual-threshold hysteresis ensures only strong, connected edges are kept — producing a clean wireframe of the character.

---

### Image 2 — Portrait (grayscale photo, Gaussian noise)
<img width="284" height="227" alt="image" src="https://github.com/user-attachments/assets/1fb75bbe-091b-43ac-86d4-8208904bc75d" />


**Denoising → Gaussian Blur (kernel 5×5, σ=1)**

The noise has a smooth, random distribution across the entire image — characteristic of Gaussian/sensor noise from a low-light grayscale photo. A Gaussian Blur with a moderate kernel (5×5) effectively averages out this type of continuous noise. Median Filter would be overkill here and would lose fine detail; Bilateral Filter is an alternative if edge preservation is the priority.

**Edge Detection → Sobel**

Sobel was chosen because the portrait has gradual intensity transitions (skin tones, shadows) rather than sharp edges. Sobel computes first-order gradients in both X and Y directions and is robust to residual noise post-filtering. Canny would over-detect on smooth skin regions; Sobel gives a more natural result that highlights meaningful structural edges (jaw, eyes, hair boundary).

---

## Architecture

See `ARCHITECTURE.md` for the full module diagram and class hierarchy.

---

## Requirements

- Python 3.9+
- PyQt5 5.15+
- OpenCV 4.9+
- NumPy 1.26+

---

## Notes

- All `.ui` files must reside in the **same directory** as the `.py` files when running (PyQt5 loads them at runtime via relative path).
- The application uses a dark theme (Catppuccin Mocha palette) defined entirely in `main.py`'s stylesheet — no external CSS files needed.
- The manual segmentation brush opens as a native Qt dialog (not a separate OpenCV window) to avoid GUI thread conflicts.
