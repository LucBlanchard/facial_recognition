# SENTINEL — Face Recognition System

A real-time face recognition system designed for deployment in low-resource environments. Built with a **YOLOv8 + MTCNN + FaceNet** pipeline, a **Fast API**, and a **Streamlit** surveillance interface. 

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
│   ├── main.py                  # FastAPI REST API (all endpoints)
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