<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=8B0000,D4AF37&height=200&section=header&text=Real-Time%20Face%20Detection%20System&fontSize=32&fontColor=D4AF37&fontAlignY=38&desc=Computer%20Vision%20%C2%B7%20OpenCV%20%C2%B7%20Python&descAlignY=58&descColor=00E5FF" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-D4AF37?style=for-the-badge&logo=python&logoColor=0A0A0A&labelColor=8B0000"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-00E5FF?style=for-the-badge&logo=opencv&logoColor=0A0A0A&labelColor=111111"/>
  <img src="https://img.shields.io/badge/NumPy-Enabled-D4AF37?style=for-the-badge&logo=numpy&logoColor=0A0A0A&labelColor=111111"/>
  <img src="https://img.shields.io/badge/License-MIT-00E5FF?style=for-the-badge&labelColor=111111"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-D4AF37?style=for-the-badge&labelColor=8B0000"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Haar%20Cascade-Face%20%7C%20Eyes%20%7C%20Smile-D4AF37?style=flat-square&labelColor=111111"/>
  <img src="https://img.shields.io/badge/Modes-Image%20%7C%20Webcam%20%7C%20Batch%20%7C%20Test-00E5FF?style=flat-square&labelColor=111111"/>
  <img src="https://img.shields.io/badge/Output-Annotated%20PNG%20%7C%20Structured%20Dict-D4AF37?style=flat-square&labelColor=111111"/>
</p>

<br/>

> **A production-ready CV pipeline — not a notebook demo.**  
> Built with deployment, extensibility, and client handoff in mind.

<br/>

</div>

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install opencv-python numpy matplotlib pillow

# 2. Run with default test image
python main.py

# 3. Or detect faces in your own image
python main.py
# → Enter choice: 1
# → Enter image path: photo.jpg
```

**Drop-in integration — 4 lines:**

```python
from main import FaceDetector
import cv2

detector = FaceDetector()
image    = cv2.imread('photo.jpg')
faces    = detector.detect_faces(image, detect_eyes=True, detect_smile=True)
result   = detector.draw_detections(image, faces, draw_eyes=True)
```

---

## 🖼️ Output

<div align="center">

| Input | Detection Result |
|:---:|:---:|
| Original image | Face detected with bounding box |
| No annotations | `FACE (Smiling)` label + eye markers |

```
══════════════════════════════════════════════════════
DETECTING FACES IN IMAGE: photo.jpg
══════════════════════════════════════════════════════

Detecting faces...
Found 1 face(s)!

Detection Details:
──────────────────────────────────────────────────────
Face 1:
  Position : (x=111, y=113)
  Size     : 178 × 178 px
  Eyes     : 2 detected
  Smile    : Yes

✓ Face detection complete!
══════════════════════════════════════════════════════
```

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACE DETECTION PIPELINE                      │
├─────────────┐                                                   │
│   INPUT     │  Image File  ──┐                                  │
│             │  Webcam Feed ──┼──► Grayscale ──► Cascade ──►    │
│             │  Folder Batch──┘    Conversion     Detector       │
└─────────────┘                                                   │
                                         │                        │
                              ┌──────────▼──────────┐            │
                              │   ROI ANALYSIS      │            │
                              │  ┌────────────────┐ │            │
                              │  │ Upper half ROI │ │            │
                              │  │  → Eye cascade │ │            │
                              │  └────────────────┘ │            │
                              │  ┌────────────────┐ │            │
                              │  │ Lower half ROI │ │            │
                              │  │ → Smile cascade│ │            │
                              │  └────────────────┘ │            │
                              └──────────┬──────────┘            │
                                         │                        │
                              ┌──────────▼──────────┐            │
                              │      OUTPUT         │            │
                              │  Annotated Image    │            │
                              │  Structured Dict    │            │
                              │  {bbox, eyes[],     │            │
                              │   smile: bool}      │            │
                              └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔲 Face Detection
Multi-scale Haar Cascade scanning with configurable:
- `scaleFactor` — image reduction per scale step
- `minNeighbors` — candidate validation threshold
- `minSize` — minimum detection size in pixels

</td>
<td width="50%">

### 👁️ Eye Detection
Region-of-interest detection scoped to each face's upper half bounding box. Returns absolute image coordinates for downstream use.

</td>
</tr>
<tr>
<td width="50%">

### 😊 Smile Detection
Lower-half ROI analysis prevents false positives from eyebrows and forehead features. Returns `True/None` per face.

</td>
<td width="50%">

### 📹 Real-Time Webcam
Live frame-by-frame detection with:
- On-screen face count overlay
- `S` key → save screenshot
- `Q` key → clean quit

</td>
</tr>
<tr>
<td width="50%">

### 🗂️ Batch Processing
Scan entire directories. Supports `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`. Returns per-file detection report with totals.

</td>
<td width="50%">

### 📊 Visualization Export
Side-by-side 150 DPI matplotlib output. Original vs annotated, labeled bounding boxes, saved as PNG.

</td>
</tr>
</table>

---

## 🚀 Operation Modes

```
python main.py
```

```
Choose a mode:

  [1]  Detect faces in an image file
  [2]  Real-time webcam detection
  [3]  Create test image + detect
  [4]  Batch scan a folder of images
  [ ]  Press Enter → auto test mode
```

---

## 📦 Return Structure

```python
# detect_faces() returns a list of dicts:
[
  {
    "bbox":  (x, y, w, h),     # bounding box in pixels
    "eyes":  [(x,y,w,h), ...], # absolute coords, may be []
    "smile": True | None        # None if detect_smile=False
  },
  # ... one dict per face detected
]
```

---

## 🔌 Upgrade Path

This system is built to scale. Swap the detector engine without changing the calling interface:

```python
# Current: Haar Cascade (fast, CPU-only, zero dependencies)
detector = FaceDetector()

# Future: MediaPipe (higher accuracy, still CPU-friendly)
# detector = FaceDetectorMediaPipe()

# Future: YOLOv8 (GPU, real-time, state-of-the-art)
# detector = FaceDetectorYOLO()

# ─── Same interface. Zero refactoring. ────────────────
faces  = detector.detect_faces(image)
result = detector.draw_detections(image, faces)
```

---

## 🌐 Real-World Applications

| Domain | Use Case |
|--------|----------|
| 🔐 Access Control | First-stage detection for attendance & verification systems |
| 📸 Photo Management | Auto-tag, sort, and export portraits from large libraries |
| 📊 Retail Analytics | Count faces in footage for footfall & engagement metrics |
| 🛡️ Security | Flag unrecognized faces in restricted zones |
| 🤖 Robotics / IoT | Lightweight perception module for embedded systems |
| 🎓 EdTech | Student presence monitoring for online assessments |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|:------|:-----------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV 4.x |
| Numerical | NumPy |
| Visualization | Matplotlib |
| Image I/O | Pillow |
| Classifiers | Haar Cascade (Face · Eye · Smile) |

</div>

---

## 📁 Project Structure

```
face-detection/
│
├── main.py                  # Core pipeline
│   ├── class FaceDetector   # Detection engine
│   ├── detect_from_image()  # Image mode
│   ├── detect_from_webcam() # Webcam mode
│   ├── batch_detect_faces() # Batch mode
│   └── create_test_image()  # Test generator
│
├── requirements.txt         # pip dependencies
├── README.md                # This file
└── sample_output.png        # Example detection result
```

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/omarmomtaz/face-detection-system.git
cd face-detection-system

# Install
pip install -r requirements.txt

# Run
python main.py
```

**requirements.txt:**
```
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
pillow>=8.0.0
```

---

<div align="center">

## 👤 Built By

### Omar Momtaz
**AI/ML Engineer · Computer Vision · Python**

*Building systems that deploy — not demos that impress once.*

<br/>

<a href="[https://www.upwork.com](https://www.upwork.com/freelancers/~01044b5d38a3457dc1?mp_source=share)">
  <img src="https://img.shields.io/badge/Hire%20on-Upwork-D4AF37?style=for-the-badge&labelColor=8B0000"/>
</a>
<a href="[https://www.linkedin.com](https://www.linkedin.com/in/omar-momtaz-/)">
  <img src="https://img.shields.io/badge/Connect-LinkedIn-00E5FF?style=for-the-badge&labelColor=111111"/>
</a>
<a href="[https://twitter.com](https://x.com/omarmomtaz_main)">
  <img src="https://img.shields.io/badge/Follow-X-D4AF37?style=for-the-badge&labelColor=111111"/>
</a>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=8B0000,D4AF37&height=100&section=footer" width="100%"/>

</div>
