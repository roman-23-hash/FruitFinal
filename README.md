# 🍈 Fruit Ripeness Predictor — Full Stack API

A full-stack web application for classifying guava fruit ripeness from images using a custom Keras/TensorFlow model with thermal imaging output and HSV-based guava color gate filtering.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [REST API Reference](#rest-api-reference)
6. [Guava Color Gate](#guava-color-gate)
7. [Model Details](#model-details)
8. [Environment Variables](#environment-variables)
9. [Frontend](#frontend)
10. [VS Code Tasks](#vs-code-tasks)
11. [Troubleshooting](#troubleshooting)
12. [Local Testing Checklist](#local-testing-checklist)

---

## Project Overview

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Backend | Python 3.10 + FastAPI + Uvicorn |
| ML Model | TensorFlow 2.10.1 / Keras 2.10.0 |
| Image Processing | OpenCV 4.8.0.76 + NumPy 1.23.5 |
| Guava Gate | HSV color spectrum filtering (pure OpenCV) |

**Key features:**
- Upload a fruit image via drag-and-drop or file picker
- HSV color gate filters out non-guava images before model inference
- Multi-output Keras model returns ripeness score + thermal heatmap
- Thermal heatmap rendered in INFERNO colormap and returned as base64 PNG
- Fully local — no cloud services, no data leaves your machine

---

## Architecture

```
Browser (http://localhost:5173)
        │
        │  POST /predict  (multipart/form-data)
        ▼
┌─────────────────────────────────────────┐
│           FastAPI Backend               │
│         (http://localhost:8000)         │
│                                         │
│  1. Decode image (OpenCV → Pillow)      │
│  2. HSV Color Gate                      │
│     ├─ < 20% green/yellow → REJECT      │
│     └─ ≥ 20% green/yellow → CONTINUE   │
│  3. Preprocess (resize, grayscale,      │
│     normalise to [0,1])                 │
│  4. Model Inference (CPU)               │
│     ├─ thermal_out  (512×512×1)  ──┐   │
│     ├─ ripeness_out (1,)         ──┤   │
│     └─ guava_guard  (1,)  unused  │   │
│  5. Render thermal → INFERNO PNG  │   │
│  6. Build prediction labels       │   │
│  7. Return JSON response ─────────┘   │
└─────────────────────────────────────────┘
        │
        │  JSON response
        ▼
Browser displays:
  - Ripeness label + confidence bars
  - Thermal heatmap (expandable panel)
  - Gate rejection message (if not guava)
```

---

## Repository Structure

```
fruit-ripeness/
├── README.md
├── .gitignore
├── .vscode/
│   └── tasks.json                  ← VS Code one-click start (Ctrl+Shift+B)
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.cjs
│   ├── postcss.config.cjs
│   ├── .env.example
│   ├── favicon.svg
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       └── components/
│           └── ImageUploader.jsx
│
├── backend/
│   ├── README.md
│   ├── requirements.txt
│   ├── runtime.txt                 ← python-3.10
│   ├── Dockerfile
│   ├── .env                        ← gate + model config (create this)
│   ├── model/
│   │   ├── README.md
│   │   ├── model.h5                ← YOUR MODEL (place here)
│   │   └── labels.json             ← optional class labels
│   └── app/
│       ├── __init__.py
│       ├── main.py                 ← FastAPI app + lifespan
│       ├── models.py               ← Pydantic response schemas
│       ├── utils.py                ← image processing + inference
│       ├── gate.py                 ← HSV color gate
│       └── routes/
│           ├── __init__.py
│           └── predict.py          ← POST /predict endpoint
│
└── scripts/
    └── test_predict.py             ← CLI test script
```

---

## Quick Start

### Prerequisites

- Python 3.10 (required — TF 2.10.1 is incompatible with 3.11+)
- Node.js 18+

### Backend Setup

```bash
cd backend

# Create virtual environment
python3.10 -m venv venv

# Activate (Windows cmd)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Place your model
# Copy your .h5 file → backend/model/model.h5

# Create .env file
# (copy the example below into backend/.env)

# Start server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### backend/.env (create this file)

```ini
GATE_ENABLED=true
GATE_THRESHOLD=20.0
MODEL_PATH=./model/model.h5
LABELS_PATH=./model/labels.json
CORS_ORIGINS=http://localhost:5173
```

---

## REST API Reference

### Base URL
```
http://localhost:8000
```

---

### `GET /`

Returns basic API info.

**Response `200 OK`**
```json
{
  "name": "Fruit Ripeness API",
  "version": "2.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### `GET /health`

Returns server health status and model load state.

**Response `200 OK` — model loaded**
```json
{
  "status": "ok",
  "model_loaded": true,
  "gate_available": true,
  "gate_enabled": true,
  "gate_threshold": 20.0,
  "model_input_shape": [null, 512, 512, 1],
  "labels_loaded": false,
  "message": null
}
```

**Response `200 OK` — model NOT loaded**
```json
{
  "status": "ok",
  "model_loaded": false,
  "gate_available": true,
  "gate_enabled": true,
  "gate_threshold": 20.0,
  "model_input_shape": null,
  "labels_loaded": false,
  "message": "Place model.h5 at './model/model.h5' and restart."
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` if server is running |
| `model_loaded` | bool | Whether model.h5 was loaded successfully |
| `gate_available` | bool | Whether guava gate is initialised |
| `gate_enabled` | bool | Whether gate is actively filtering |
| `gate_threshold` | float | Minimum % green/yellow pixels required |
| `model_input_shape` | array\|null | Model's expected input shape |
| `labels_loaded` | bool | Whether labels.json was found and loaded |
| `message` | string\|null | Helpful message when model is not loaded |

---

### `POST /predict`

Upload an image and receive ripeness predictions with optional thermal heatmap.

**Request**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Image file — JPEG, PNG, WEBP, BMP. Max 20 MB. |

Content-Type: `multipart/form-data`

**cURL Example**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/guava.jpg"
```

**Python Example**
```python
import requests

with open("guava.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("guava.jpg", f, "image/jpeg")}
    )
print(response.json())
```

---

#### Response — Guava detected ✅

**`200 OK`**
```json
{
  "success": true,
  "is_guava": true,
  "predictions": [
    { "label": "ripe",   "confidence": 0.87 },
    { "label": "unripe", "confidence": 0.13 }
  ],
  "thermal_image": "data:image/png;base64,iVBORw0KGgo...",
  "meta": {
    "model_input_shape": [null, 512, 512, 1],
    "processing_time_ms": 1245.6,
    "gate_confidence": 0.63,
    "gate_message": "Guava color confirmed (63.2% green/yellow pixels)"
  }
}
```

#### Response — Not a guava 🚫

**`200 OK`** (not an error — a valid classification result)
```json
{
  "success": true,
  "is_guava": false,
  "message": "Not a guava — only 4.1% green/yellow pixels (need ≥20%)",
  "gate_confidence": 0.041,
  "predictions": [],
  "thermal_image": null,
  "meta": {
    "model_input_shape": [null, 512, 512, 1],
    "processing_time_ms": 0,
    "gate_confidence": 0.041,
    "gate_message": "Not a guava — only 4.1% green/yellow pixels (need ≥20%)"
  }
}
```

#### Response — Error cases

**`400 Bad Request` — invalid image**
```json
{
  "success": false,
  "error": "Cannot decode image",
  "detail": "Failed to decode image with both OpenCV and Pillow"
}
```

**`400 Bad Request` — file too large**
```json
{
  "success": false,
  "error": "File too large (max 20 MB)"
}
```

**`503 Service Unavailable` — model not loaded**
```json
{
  "success": false,
  "error": "Model not loaded",
  "detail": "Place model.h5 in backend/model/ and restart the server."
}
```

**`500 Internal Server Error` — inference failed**
```json
{
  "success": false,
  "error": "Inference failed",
  "detail": "<exception message>"
}
```

---

#### Response Field Reference

| Field | Type | Description |
|---|---|---|
| `success` | bool | `true` if request was processed (even if not guava) |
| `is_guava` | bool | `true` if image passed the color gate |
| `predictions` | array | Sorted list of `{label, confidence}` — empty if not guava |
| `predictions[].label` | string | Class label from labels.json or `"class_N"` |
| `predictions[].confidence` | float | Probability 0.0–1.0 |
| `thermal_image` | string\|null | Base64 PNG data URI of thermal heatmap |
| `meta.model_input_shape` | array | Shape the model expects e.g. `[null,512,512,1]` |
| `meta.processing_time_ms` | float | Model inference time in milliseconds |
| `meta.gate_confidence` | float | Fraction of pixels matching guava color (0.0–1.0) |
| `meta.gate_message` | string | Human-readable gate result |

---

## Guava Color Gate

The gate uses HSV (Hue-Saturation-Value) color analysis — no ML model required.

### How it works

```
Image (RGB)
    ↓
Convert to HSV color space
    ↓
Apply color mask: Hue 25–85 (green to yellow spectrum)
    ↓
Count matching pixels as % of total image
    ↓
match_pct < GATE_THRESHOLD  →  Reject (Not a Guava)
match_pct ≥ GATE_THRESHOLD  →  Accept (proceed to model)
```

### HSV Hue reference

```
Hue  0– 10  →  Red    (apples, tomatoes)   ✗ rejected
Hue 10– 25  →  Orange (oranges, papayas)   ✗ rejected
Hue 25– 45  →  Yellow (ripe guava)         ✓ passes
Hue 45– 85  →  Green  (unripe guava)       ✓ passes
Hue 85–130  →  Cyan/Blue                   ✗ rejected
Hue 130–180 →  Purple/Red                  ✗ rejected
```

### Tuning the gate

Edit `backend/.env`:

```ini
# Default — 20% of pixels must be green/yellow
GATE_THRESHOLD=20.0

# More permissive — useful for partially obscured guavas
GATE_THRESHOLD=10.0

# More strict — only strongly green/yellow images pass
GATE_THRESHOLD=35.0

# Disable completely — all images go to model
GATE_ENABLED=false
```

Restart uvicorn after any `.env` change.

---

## Model Details

### Expected file location
```
backend/model/model.h5
```

### Expected architecture (your model)

Your model is a multi-output U-Net style network with 3 output heads:

| Output name | Shape | Used for |
|---|---|---|
| `thermal_out` | `(None, 512, 512, 1)` | Thermal heatmap visualisation |
| `ripeness_out` | `(None, 1)` | Ripeness sigmoid score |
| `guava_guard` | `(None, 1)` | Not used (superseded by color gate) |

### Input requirements

| Property | Value |
|---|---|
| Input shape | `(None, 512, 512, 1)` |
| Color mode | Grayscale (1 channel) |
| Normalisation | Divide by 255.0 → range [0, 1] |
| Dtype | `float32` |

The backend reads `model.input_shape` automatically at startup and adapts preprocessing accordingly.

### labels.json (optional)

Place at `backend/model/labels.json`. JSON array of strings indexed by class:

```json
["unripe", "ripe", "overripe"]
```

If missing, predictions use `"class_0"`, `"class_1"`, etc.

For a binary sigmoid model (single output), only 2 labels are needed:
```json
["unripe", "ripe"]
```

### Thermal image

The `thermal_out` output is a `512×512` grayscale heatmap. The backend:

1. Squeezes to `(512, 512)`
2. Normalises to `uint8` (0–255)
3. Applies OpenCV `COLORMAP_INFERNO` (dark purple = cool, orange = warm, yellow = hot)
4. Encodes as PNG
5. Returns as `data:image/png;base64,...` string

The frontend displays this in a collapsible panel with a download button.

---

## Environment Variables

All variables go in `backend/.env`.

| Variable | Default | Description |
|---|---|---|
| `GATE_ENABLED` | `true` | `true`/`false` — enable/disable color gate |
| `GATE_THRESHOLD` | `20.0` | Min % of green/yellow pixels to pass gate |
| `MODEL_PATH` | `./model/model.h5` | Path to Keras model file |
| `LABELS_PATH` | `./model/labels.json` | Path to class labels JSON |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |

Frontend variable (in `frontend/.env`):

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend base URL |

---

## Frontend

Built with Vite + React 18 + Tailwind CSS.

### Features

- Drag-and-drop or click-to-upload image input
- Client-side image preview before submission
- XHR upload with real progress bar
- Loading spinner during inference
- Gate rejection panel with color confidence score
- Ripeness prediction with confidence bars (color-coded)
- Collapsible thermal heatmap panel with download button
- Accessible — ARIA labels, keyboard navigation, role attributes
- Responsive layout

### Run

```bash
cd frontend
npm install
npm run dev       # development → http://localhost:5173
npm run build     # production build → dist/
npm run preview   # preview production build
```

### Configure backend URL

```bash
cp .env.example .env
# Edit VITE_API_BASE_URL if backend is not on localhost:8000
```

---

## VS Code Tasks

Press **`Ctrl + Shift + B`** to start both servers simultaneously in separate VS Code terminals.

Config is at `.vscode/tasks.json`. Each server gets its own dedicated terminal panel labeled `Backend :8000` and `Frontend :5173`.

---

## TensorFlow & GPU Notes

- **TF 2.10.1** is the last version with native Windows GPU support
- CPU-only works out of the box — no CUDA needed
- For GPU: install **CUDA 11.2** + **cuDNN 8.1** (see [TF install guide](https://www.tensorflow.org/install/pip))
- To force CPU: set `CUDA_VISIBLE_DEVICES=-1` before starting server
- To allow gradual GPU memory growth: set `TF_FORCE_GPU_ALLOW_GROWTH=true`
- All inference runs under `tf.device("/CPU:0")` by default for compatibility

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `tensorflow==2.10.1` not found | You are on Python 3.11+. Install Python 3.10 and use `py -3.10 -m venv venv` |
| `libGL.so.1` missing (Linux) | `sudo apt-get install libgl1-mesa-glx libglib2.0-0` |
| `&&` not valid in PowerShell | Use `cmd.exe` or VS Code tasks (`Ctrl+Shift+B`) |
| `ModuleNotFoundError: No module named 'app'` | Run uvicorn from inside the `backend/` directory |
| Gate rejects valid guavas | Lower `GATE_THRESHOLD` in `.env` (try `10.0`) |
| Gate passes non-guavas | Raise `GATE_THRESHOLD` (try `30.0`) |
| 500 Inference error | Check uvicorn terminal for full traceback |
| Thermal image not showing | Check uvicorn logs for "Could not render thermal image" |
| CORS error in browser | Add frontend URL to `CORS_ORIGINS` in `.env` |
| Model loads but wrong input shape | Check `model.input_shape` in startup logs |
| OOM on GPU | Set `CUDA_VISIBLE_DEVICES=-1` to force CPU |

---

## Local Testing Checklist

Steps to verify everything works on a fresh machine:

- [ ] Install Python 3.10 from python.org and Node.js 18+ from nodejs.org
- [ ] `cd backend && py -3.10 -m venv venv && venv\Scripts\activate`
- [ ] `pip install --upgrade pip && pip install -r requirements.txt` — no errors
- [ ] Copy `model.h5` to `backend/model/model.h5`
- [ ] Create `backend/.env` with `GATE_ENABLED=true` and `GATE_THRESHOLD=20.0`
- [ ] `uvicorn app.main:app --reload --port 8000` — server starts, model summary printed
- [ ] Visit `http://localhost:8000/health` — `model_loaded: true`
- [ ] Visit `http://localhost:8000/docs` — Swagger UI loads
- [ ] `cd frontend && npm install && npm run dev` — Vite starts
- [ ] Open `http://localhost:5173` — UI renders, no console errors
- [ ] Upload a guava image → prediction + thermal heatmap displayed
- [ ] Upload a red apple → "Not a guava" rejection message shown
- [ ] Upload a non-image file → friendly error message shown
- [ ] Test via cURL: `curl -X POST http://localhost:8000/predict -F "file=@guava.jpg"`
- [ ] Test via script: `python scripts/test_predict.py path/to/guava.jpg`
- [ ] Press `Ctrl+Shift+B` in VS Code → both terminals start cleanly