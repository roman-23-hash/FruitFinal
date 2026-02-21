"""
Fruit Ripeness API — FastAPI application entry point.

Startup:
  - Loads Keras model from MODEL_PATH (default: ./model/model.h5)
  - Loads optional class labels from LABELS_PATH (default: ./model/labels.json)
  - Stores model, input_size, and labels in app.state for use by routes

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

from app.models import HealthResponse
from app.routes.predict import router as predict_router
from app.utils import get_model_input_size, get_model_channels

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "./model/model.h5"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "./model/labels.json"))
_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",")]


# ---------------------------------------------------------------------------
# Lifespan: load model + labels on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and labels at startup; release on shutdown."""

    # --- Keras / TF import deferred to avoid slow startup messages polluting
    #     the log before our banner is printed. ---
    logger.info("=" * 60)
    logger.info("  Fruit Ripeness API  —  starting up")
    logger.info("=" * 60)

    model: Optional[object] = None
    labels: Optional[list] = None
    input_size = (224, 224)  # default
    channels = 3  # default

    # Load model
    if not MODEL_PATH.exists():
        logger.warning(
            "Model file not found at '%s'. "
            "Server will start, but /predict will return 503 until a model is placed there.",
            MODEL_PATH,
        )
    else:
        logger.info("Loading model from '%s' …", MODEL_PATH)
        try:
            import keras

            model = keras.models.load_model(str(MODEL_PATH), compile=False)
            input_size = get_model_input_size(model)
            channels = get_model_channels(model)
            logger.info("Model loaded successfully. Input size: %s, Channels: %d", input_size, channels)
            model.summary(print_fn=logger.info)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            logger.error(
                "Fix: ensure the file is a valid Keras HDF5 model saved with "
                "model.save('model.h5') using TF 2.x."
            )

    # Load labels
    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as fh:
                labels = json.load(fh)
            if not isinstance(labels, list):
                raise ValueError("labels.json must be a JSON array of strings")
            logger.info("Loaded %d class labels: %s", len(labels), labels)
        except Exception as exc:
            logger.warning("Could not load labels.json: %s. Using class indices.", exc)
    else:
        logger.info(
            "No labels.json found at '%s'. Predictions will use class indices.",
            LABELS_PATH,
        )

    # Expose on app state
    app.state.model = model
    app.state.labels = labels
    app.state.input_size = input_size
    app.state.channels = channels

    logger.info("Startup complete. CORS origins: %s", CORS_ORIGINS)
    logger.info("=" * 60)

    yield  # ← server is running

    logger.info("Shutting down Fruit Ripeness API …")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fruit Ripeness API",
    version="1.0.0",
    description="Upload a fruit image and get ripeness predictions from a Keras model.",
    lifespan=lifespan,
)

# CORS — allow Vite dev server by default; expand via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, tags=["Inference"])


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Fruit Ripeness API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Return server health and model status."""
    model = app.state.model
    loaded = model is not None

    shape = None
    num_classes = None
    if loaded:
        shape = list(model.input_shape)
        try:
            # last-layer output gives number of classes
            num_classes = model.output_shape[-1]
        except Exception:
            pass

    message = None
    if not loaded:
        message = (
            f"Model not loaded. Place your Keras .h5 model at '{MODEL_PATH}' "
            "and restart the server."
        )

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_input_shape=shape,
        labels_loaded=app.state.labels is not None,
        num_classes=num_classes,
        message=message,
    )