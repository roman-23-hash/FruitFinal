"""
Fruit Ripeness API — FastAPI application entry point.

Run:
  uvicorn app.main:app --reload --port 8000
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.gate import GuavaGate
from app.models import HealthResponse
from app.routes.predict import router as predict_router
from app.utils import get_model_channels, get_model_input_size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH       = Path(os.getenv("MODEL_PATH",  "./model/model.h5"))
LABELS_PATH      = Path(os.getenv("LABELS_PATH", "./model/labels.json"))
GATE_THRESHOLD   = float(os.getenv("GATE_THRESHOLD", "0.5"))
GATE_ENABLED     = os.getenv("GATE_ENABLED", "true").lower() == "true"

_cors_raw    = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Fruit Ripeness API  —  starting up")
    logger.info("=" * 60)

    model      = None
    labels     = None
    input_size = (224, 224)
    channels   = 3

    # ── Load ripeness model ───────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        logger.warning("Model not found at '%s'. /predict will return 503.", MODEL_PATH)
    else:
        logger.info("Loading model from '%s' …", MODEL_PATH)
        try:
            import keras
            model      = keras.models.load_model(str(MODEL_PATH), compile=False)
            input_size = get_model_input_size(model)
            channels   = get_model_channels(model)
            logger.info("Model loaded. Input: %s  Channels: %d", input_size, channels)
            model.summary(print_fn=logger.info)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)

    # ── Load labels ───────────────────────────────────────────────────────────
    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as fh:
                labels = json.load(fh)
            logger.info("Loaded %d labels: %s", len(labels), labels)
        except Exception as exc:
            logger.warning("Could not load labels.json: %s", exc)
    else:
        logger.info("No labels.json found — using class indices.")

    # ── Set up guava gate (model-based, no CLIP needed) ───────────────────────
    gate = GuavaGate(threshold=GATE_THRESHOLD, enabled=GATE_ENABLED)

    # ── Store on app state ────────────────────────────────────────────────────
    app.state.model      = model
    app.state.labels     = labels
    app.state.input_size = input_size
    app.state.channels   = channels
    app.state.gate       = gate

    logger.info("Gate enabled: %s | Threshold: %.2f | CORS: %s",
                GATE_ENABLED, GATE_THRESHOLD, CORS_ORIGINS)
    logger.info("=" * 60)

    yield

    logger.info("Shutting down …")


app = FastAPI(
    title="Fruit Ripeness API",
    version="2.0.0",
    description="Guava ripeness classifier with model-based guava guard and thermal imaging.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, tags=["Inference"])


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Fruit Ripeness API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    model  = app.state.model
    loaded = model is not None
    shape  = list(model.input_shape) if loaded else None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        gate_available=app.state.gate.available,
        gate_enabled=app.state.gate.enabled,
        gate_threshold=app.state.gate.threshold,
        model_input_shape=shape,
        labels_loaded=app.state.labels is not None,
        message=None if loaded else f"Place model.h5 at '{MODEL_PATH}' and restart.",
    )