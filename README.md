# SENTINEL — Face Recognition System

A real-time face recognition system designed for deployment in low-resource environments. Built with a **YOLOv8 + MTCNN + FaceNet** pipeline, a **Flask REST API**, and a **Streamlit** surveillance interface. Includes a data augmentation module optimised for equatorial African lighting conditions (strong sunlight, varied skin tones, mixed indoor/outdoor lighting).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Streamlit Interface](#streamlit-interface)
- [Data Augmentation](#data-augmentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

SENTINEL solves a specific problem: identifying whether the **same person** appears across two different inputs — a reference photo (Place A) and a live camera, video, or image (Place B) — without requiring the person to be pre-registered in any database.

**Key capabilities:**

- Direct face-to-face comparison using embedding similarity (no database required)
- Named recognition against a stored database of known people
- One-photo data augmentation to bootstrap training data from a single image
- Non-blocking video/webcam scanning (frame batching via `st.rerun()`)
- Real-time similarity scoring displayed on each frame

---

## Project Structure

```
facial_rec_project/
│
├── api/
│   ├── main.py                  # Flask REST API (all endpoints)
│   └── augment.py               # Face detection + data augmentation module
│
├── detection/
│   └── weights/
│       └── best.pt              # YOLOv8 face detection weights (not in git)
│
├── streamlit/
│   ├── sentinel_app.py          # Main Place A / Place B comparison page
│   ├── pages/
│   │   ├── 1_Face_Manager.py    # Add / remove people from the database
│   │   └── 2_Recognition.py    # Named recognition page
│   ├── videos/                  # Sample surveillance footage
│   └── Dockerfile
│
├── known_faces/                 # Face images per person (not in git)
│   └── <PersonName>/
│       └── *.jpg
│
├── known_embeddings.pkl         # Stored face embeddings (not in git)
├── .gitignore
└── README.md
```

---

## How It Works

### Face Recognition Pipeline

```
Input image
    │
    ▼
[YOLOv8] ──── detects face bounding boxes
    │
    ▼
Largest face crop + 25% padding
    │
    ▼
[MTCNN] ──── aligns face using facial landmarks (eyes, nose, mouth)
    │         standardises position across all images
    ▼
160×160 aligned face tensor
    │
    ▼
[InceptionResnetV1 / FaceNet] ──── produces 512-dimensional embedding
    │
    ▼
L2-normalised embedding vector
    │
    ├── /compare_faces  → cosine similarity against a second embedding
    └── /recognize_names → Euclidean distance against known_embeddings.pkl
```

### Similarity Metric

Embeddings are L2-normalised before comparison:

```
distance   = ||emb_A - emb_B||₂        # 0 = identical, 2 = opposite
cosine     = emb_A · emb_B              # 1 = identical, -1 = opposite
similarity = (cosine + 1) / 2           # mapped to [0, 1] for display
match      = distance < 1.0             # threshold (≈ 75% similarity)
```

The same threshold (`0.9` Euclidean distance on raw embeddings) is used for named recognition.

### YOLO Fallback Strategy

If YOLO finds no face in an image, the system falls back gracefully:

1. YOLO on full image → crop largest box
2. MTCNN detect on full image → crop largest box
3. Embed the full image directly (last resort)

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slower)
- Anaconda or virtualenv

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/facial_rec_project.git
cd facial_rec_project
```

### 2. Create and activate environment

```bash
conda create -n facial_rec python=3.10
conda activate facial_rec
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install facenet-pytorch ultralytics flask streamlit opencv-python pillow numpy requests
```

Optional (richer data augmentation):

```bash
pip install albumentations
```

### 4. Add your YOLO weights

Place your trained `best.pt` file at:

```
detection/weights/best.pt
```

> The weights file is not included in the repository (too large for git). Keep a local copy or store it on a shared drive.

---

## Running the System

The API and Streamlit interface are **two separate processes** — run them in two terminals.

### Terminal 1 — Start the Flask API

```bash
cd api
flask --app main run
# or for production:
python main.py
```

API runs at `http://localhost:5000`

### Terminal 2 — Start the Streamlit interface

```bash
cd streamlit
streamlit run sentinel_app.py
```

Interface opens at `http://localhost:8501`

> **Important:** Do not place `main.py` inside the `streamlit/pages/` folder. Streamlit scans that folder and will try to run the Flask server as a page, causing a crash.

---

## API Reference

Base URL: `http://localhost:5000`

### `POST /compare_faces` — Direct face comparison (no database needed)

Compare two faces directly. The person does not need to be registered.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `reference` | file | Place A photo (JPEG/PNG) |
| `scene` | file | Place B photo or frame (JPEG/PNG) |

**Response:**

```json
{
  "match": true,
  "similarity": 0.9124,
  "distance": 0.3871,
  "error": null
}
```

---

### `POST /recognize_names` — Identify from database

Identify all faces in an image against stored known embeddings.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | Image to scan (JPEG/PNG) |

**Response:**

```json
{
  "names": ["Jean Dupont", "Marie Claire"]
}
```

---

### `POST /recognize` — Annotated image response

Returns the input image with face bounding boxes and name labels drawn on it.

**Request:** `multipart/form-data` — `image` field

**Response:** `image/jpeg`

---

### `POST /add` — Add a person to the database

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Person's full name |
| `images` | file(s) | One or more photos |

**Response:**

```json
{
  "message": "Jean Dupont added with 3 embedding(s)."
}
```

---

### `DELETE /remove` — Remove a person from the database

**Request:** `application/json`

```json
{ "name": "Jean Dupont" }
```

---

### `GET /people` — List all registered people

**Response:**

```json
{
  "Jean Dupont": 5,
  "Marie Claire": 3
}
```

---

### `POST /augment_and_add` — One photo → augment → encode → store

Upload a single photo, generate augmented variants, and register the person in one call.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Person's full name |
| `image` | file | One reference photo |
| `count` | int (optional) | Number of augmented images (default: 100) |

**Response:**

```json
{
  "message": "Jean Dupont added successfully.",
  "images_saved": 100,
  "embeddings_saved": 87
}
```

---

## Streamlit Interface

### sentinel_app.py — Place A vs Place B

The main comparison interface.

| Zone | Purpose |
|------|---------|
| **Place A** | Upload a reference photo of the person to find |
| **Place B** | Upload an image, video, or use the webcam to search |

The system calls `/compare_faces` on every scanned frame and displays:
- Live similarity percentage on each frame
- A match banner when similarity exceeds the threshold
- An event log of all match detections with timestamps

### Face Manager page

Add and remove people from the recognition database. Accepts multiple photos per person to improve accuracy.

### Recognition page

Scan any image or video for all known faces simultaneously.

---

## Data Augmentation

The `augment.py` module generates training data from a single photo, specifically tuned for deployment in Gabon and Central Africa.

### Usage

```bash
# Command line
python augment.py --image photo.jpg --name "Jean Dupont" --count 100

# As a module
from augment import augment_face
augment_face("photo.jpg", "Jean Dupont", out_dir="known_faces", target_count=100)
```

### What it does

1. **Detects the face** using YOLO + MTCNN (background is discarded)
2. **Aligns the face** using MTCNN landmark alignment (eyes always at same position)
3. **Augments the clean crop** across three pipelines:

| Pipeline | Augmentations | Why |
|----------|---------------|-----|
| OpenCV | Rotation, brightness, gamma, CLAHE, noise, perspective warp, white balance | Core geometric and photometric variation |
| PIL | Sharpness, colour saturation, contrast, brightness | Phone camera processing variation |
| Albumentations | Shadow, sun flare, fog, rain, JPEG compression, ISO noise, occlusion | Equatorial environment simulation |

### Augmentations chosen for Central Africa

| Augmentation | Reason |
|---|---|
| Gamma 0.5–2.2 | Deep melanin levels — preserve shadow detail in dark skin tones |
| CLAHE | Equatorial noon sun blows out highlights — adaptive equalisation recovers face structure |
| Random shadow | Trees, buildings, car windows create partial face shadows |
| Sun flare | Direct equatorial sunlight at low angles (morning/evening) |
| White balance shifts | Fluorescent office lighting vs outdoor daylight |
| Fog/haze | Libreville rainy season (June–August) atmospheric haze |
| JPEG compression | Budget smartphones with heavy compression |
| Coarse dropout | Partial occlusion: sunglasses, masks, hats |

---

## Configuration

Key constants at the top of each file:

**`api/main.py`**

```python
KNOWN_FACES_DIR = "path/to/known_faces"   # where face images are stored
EMBEDDINGS_FILE = "known_embeddings.pkl"  # embedding database
```

**`streamlit/sentinel_app.py`**

```python
API_URL          = "http://localhost:5000"
FRAME_INTERVAL   = 5      # scan every Nth frame (higher = faster but less accurate)
```

**`api/augment.py`**

```python
# CLI defaults
--count   100     # number of images to generate
--size    256     # output image size in pixels
--padding 0.25    # face box expansion (0.25 = 25% padding around the face)
```

---

## Troubleshooting

**`No face detected by YOLO in reference`**

The reference photo does not contain a detectable face. The system falls back to MTCNN on the full image, then embeds the full image as a last resort. Use a clear, well-lit, front-facing photo.

**`signal only works in main thread`**

`main.py` was placed inside the `streamlit/pages/` folder. Move it to `api/main.py` and run it separately.

**`use_container_width` deprecation warnings**

Update `st.image()` calls to use `width="stretch"` instead of `use_container_width=True`.

**`UnboundLocalError: local variable 'e' referenced before assignment`**

An error handler referenced `e` outside its `except` block. The fallback return must not use `str(e)` — use a pre-defined error string variable instead.

**Video freezes / UI hangs during scanning**

Do not use a `while` loop for video playback in Streamlit. Use the batch + `st.rerun()` pattern: read 10 frames per rerun, display the last frame, then call `st.rerun()` to trigger the next batch.

**Same image gives 0% similarity**

Check that the BGR→RGB conversion happens before passing crops to MTCNN. MTCNN expects RGB; OpenCV reads BGR. Missing this conversion causes MTCNN to return `None` silently.

---

## .gitignore

```
known_faces/
known_embeddings.pkl
*.pt
*.pkl
__pycache__/
*.pyc
.env
videos/
```

> Face images, embeddings, and model weights should never be committed to git.
