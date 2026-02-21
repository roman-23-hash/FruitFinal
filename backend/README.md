# Backend — Fruit Ripeness API

FastAPI server that loads a Keras `.h5` model and exposes a `/predict` endpoint.

## Setup

```bash
# From repo root
cd backend

python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Place your model

```
backend/
└── model/
    ├── model.h5        ← your Keras model (REQUIRED)
    └── labels.json     ← optional class labels array
```

`labels.json` example:
```json
["unripe", "ripe", "overripe"]
```

## Run

```bash
# from backend/ directory, with venv active
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
# from backend/ directory
docker build -t fruit-ripeness-backend .
docker run -p 8000:8000 -v $(pwd)/model:/app/model fruit-ripeness-backend
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root info |
| GET | `/health` | Health check + model status |
| POST | `/predict` | Upload image, get predictions |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./model/model.h5` | Path to `.h5` model file |
| `LABELS_PATH` | `./model/labels.json` | Path to labels JSON |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins (comma-separated) |
