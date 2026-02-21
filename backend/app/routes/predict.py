"""POST /predict — color gate check, inference, thermal output."""

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.utils import (
    build_predictions,
    decode_image,
    preprocess_image,
    render_thermal_image,
    run_inference,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff",
    "application/octet-stream",
}


@router.post("/predict", summary="Predict guava ripeness from an uploaded image")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)"),
) -> Any:
    state = request.app.state

    # ── Guard: model must be loaded ───────────────────────────────────────────
    if state.model is None:
        raise HTTPException(status_code=503, detail={
            "success": False,
            "error": "Model not loaded",
            "detail": "Place model.h5 in backend/model/ and restart the server.",
        })

    # ── Validate content type ─────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": f"Unsupported content type: {content_type}",
            "detail": "Please upload a JPEG, PNG, WEBP, or BMP image.",
        })

    # ── Read bytes ────────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail={
            "success": False, "error": "Empty file",
        })
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        })

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img_rgb = decode_image(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={
            "success": False, "error": "Cannot decode image", "detail": str(exc),
        })

    # ── Color-based guava gate (HSV spectrum check) ───────────────────────────
    is_guava, match_pct, gate_message = state.gate.check_color(img_rgb)

    if not is_guava:
        logger.info("Color gate rejected image: %s", gate_message)
        return JSONResponse(content={
            "success": True,
            "is_guava": False,
            "message": gate_message,
            "gate_confidence": round(match_pct / 100.0, 4),  # normalise to 0-1 for UI
            "predictions": [],
            "thermal_image": None,
            "meta": {
                "model_input_shape": list(state.model.input_shape),
                "processing_time_ms": 0,
                "gate_confidence": round(match_pct / 100.0, 4),
                "gate_message": gate_message,
            },
        })

    # ── Preprocess ────────────────────────────────────────────────────────────
    input_array = preprocess_image(img_rgb, state.input_size, channels=state.channels)

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        outputs, elapsed_ms = run_inference(state.model, input_array)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail={
            "success": False, "error": "Inference failed", "detail": str(exc),
        })

    # ── Build ripeness predictions ────────────────────────────────────────────
    predictions = build_predictions(outputs["ripeness"], state.labels)

    # ── Render thermal image ──────────────────────────────────────────────────
    thermal_b64 = None
    if outputs["thermal"] is not None:
        try:
            thermal_b64 = render_thermal_image(outputs["thermal"])
            logger.info("Thermal image rendered successfully")
        except Exception as exc:
            logger.warning("Could not render thermal image: %s", exc)

    return JSONResponse(content={
        "success": True,
        "is_guava": True,
        "predictions": predictions,
        "thermal_image": thermal_b64,
        "meta": {
            "model_input_shape": list(state.model.input_shape),
            "processing_time_ms": round(elapsed_ms, 2),
            "gate_confidence": round(match_pct / 100.0, 4),
            "gate_message": gate_message,
        },
    })