# 🍎 Fruit Ripeness Predictor

A full-stack web app for classifying fruit ripeness from images using a Keras/TensorFlow model.

## Repository Tree

```
fruit-ripeness/
├── README.md                        ← you are here
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.cjs
│   ├── postcss.config.cjs
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       └── components/
│           └── ImageUploader.jsx
├── backend/
│   ├── README.md
│   ├── requirements.txt
│   ├── runtime.txt
│   ├── Dockerfile
│   ├── model/
│   │   ├── README.md
│   │   └── labels.json         ← optional: create this for named labels
│   └── app/
│       ├── main.py
│       ├── models.py
│       ├── utils.py
│       └── routes/
│           └── predict.py
└── scripts/
    └── test_predict.py
```

---

## Quick Start

### Prerequisites
- Python 3.10 (strongly recommended; TF 2.10.1 is incompatible with 3.12+)
- Node.js 18+

### 1. Backend

```bash
cd backend

# Create a virtualenv
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Place your model file
# Copy model.h5 → backend/model/model.h5
# (Optional) Copy labels.json → backend/model/labels.json

# Start the server
uvicorn app.main:app --reload --port 8000
```

Server runs at **http://localhost:8000**  
Health check: http://localhost:8000/health

### 2. Frontend

```bash
cd frontend

npm install

# (Optional) configure backend URL
cp .env.example .env
# Edit .env if your backend is not at http://localhost:8000

npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Model Setup

1. Train or obtain a Keras `.h5` model that classifies fruit ripeness.
2. Place the file at `backend/model/model.h5`.
3. **Optional:** Create `backend/model/labels.json` — a JSON array of class label strings indexed by class index:
   ```json
   ["ripe", "overripe", "unripe"]
   ```
   If this file is absent, predictions will use numeric class indices (`"class_0"`, `"class_1"`, …).

### Checking your model's input shape

```python
import keras
m = keras.models.load_model("backend/model/model.h5", compile=False)
print(m.input_shape)   # e.g. (None, 224, 224, 3)
m.summary()
```

The backend reads `model.input_shape` automatically and resizes images to match. If the shape cannot be determined, it falls back to `224×224` and logs a warning.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend base URL (frontend) |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins (backend, comma-separated) |
| `MODEL_PATH` | `./model/model.h5` | Path to model (relative to `backend/`) |

---

## TensorFlow & GPU Notes

- **TF 2.10.1** is the last release with native Windows GPU support.
- CPU-only works out of the box — no CUDA needed.
- For GPU inference you need **CUDA 11.2** + **cuDNN 8.1** (see [TF install guide](https://www.tensorflow.org/install/pip)).
- To force CPU inference set: `CUDA_VISIBLE_DEVICES=-1` before starting the server.
- To limit GPU memory: add `TF_FORCE_GPU_ALLOW_GROWTH=true` to your environment.

---

## cURL Example

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/your/fruit.jpg" | python3 -m json.tool
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: Could not find a version that satisfies the requirement tensorflow==2.10.1` | Ensure Python 3.10 and pip 22+. Run `pip install --upgrade pip`. |
| OpenCV import error `libGL.so.1` missing | `sudo apt-get install -y libgl1-mesa-glx libglib2.0-0` |
| `model.input_shape` returns `None` or dynamic | Backend falls back to 224×224. Check logs; override by editing `DEFAULT_IMG_SIZE` in `app/utils.py`. |
| TF OOM on GPU | Set `CUDA_VISIBLE_DEVICES=-1` to force CPU. |
| CORS errors in browser | Add your frontend origin to `CORS_ORIGINS` env var. |
| `wheel not found` for opencv on ARM | Use `opencv-python-headless==4.8.0.76` instead in `requirements.txt`. |
| PIL fallback activated | OpenCV failed to decode the image (corrupted or unsupported format). PIL will attempt recovery. |

---

## How I Tested It Locally (Checklist)

1. [ ] Install Python 3.10 via pyenv or python.org installer.
2. [ ] `cd backend && python3.10 -m venv venv && source venv/bin/activate`
3. [ ] `pip install --upgrade pip && pip install -r requirements.txt` — no errors.
4. [ ] Copy a valid Keras `.h5` model to `backend/model/model.h5`.
5. [ ] `uvicorn app.main:app --reload --port 8000` — server starts, model summary printed.
6. [ ] Visit http://localhost:8000/health — returns `{"status":"ok","model_loaded":true}`.
7. [ ] `cd ../frontend && npm install && npm run dev` — Vite dev server starts.
8. [ ] Open http://localhost:5173 — UI renders without console errors.
9. [ ] Drag a fruit image onto the uploader — preview appears.
10. [ ] Click **Predict** — spinner shows, then prediction label + confidence appears.
11. [ ] `cd ../scripts && python test_predict.py path/to/fruit.jpg` — JSON result printed.
12. [ ] Test with a non-image file — friendly error message shown in UI.
