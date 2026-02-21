"""POST /predict — image upload and inference endpoint."""

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models import ErrorResponse, PredictionResponse
from app.utils import (
    build_predictions,
    decode_image,
    preprocess_image,
    run_inference,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum upload size: 20 MB
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",  # some browsers send this for images
}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request / invalid image"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Inference error"},
    },
    summary="Predict fruit ripeness from an uploaded image",
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)"),
) -> Any:
    """
    Accept a multipart image upload, run inference, and return predictions.

    Returns JSON:
    ```json
    {
      "success": true,
      "predictions": [{"label": "ripe", "confidence": 0.92}],
      "meta": {"model_input_shape": [1, 224, 224, 3], "processing_time_ms": 45.2}
    }
    ```
    """
    state = request.app.state

    # --- Guard: model must be loaded ---
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Model not loaded",
                "detail": (
                    "Place model.h5 in backend/model/ and restart the server. "
                    "See backend/model/README.md for instructions."
                ),
            },
        )

    # --- Validate content type (soft check) ---
    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": f"Unsupported content type: {content_type}",
                "detail": "Please upload a JPEG, PNG, WEBP, or BMP image.",
            },
        )

    # --- Read file bytes ---
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "Empty file", "detail": "No bytes received."},
        )
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "File too large",
                "detail": f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
            },
        )

    # --- Decode image ---
    try:
        img_rgb = decode_image(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "Cannot decode image", "detail": str(exc)},
        )

    # --- Preprocess ---
    input_array = preprocess_image(img_rgb, state.input_size, channels=state.channels)

    # --- Inference ---
    try:
        preds, elapsed_ms = run_inference(state.model, input_array)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "Inference failed", "detail": str(exc)},
        )

    # --- Build response ---
    predictions = build_predictions(preds, state.labels)

    meta_shape = list(state.model.input_shape)

    return PredictionResponse(
        success=True,
        predictions=predictions,
        meta={
            "model_input_shape": meta_shape,
            "processing_time_ms": round(elapsed_ms, 2),
        },
    )