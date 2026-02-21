"""Image preprocessing utilities for the fruit ripeness API."""

import base64
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_IMG_SIZE: Tuple[int, int] = (224, 224)


# ── Model shape helpers ───────────────────────────────────────────────────────

def get_model_input_size(model) -> Tuple[int, int]:
    try:
        shape = model.input_shape
        if shape is None:
            raise ValueError("model.input_shape is None")
        if len(shape) == 4:
            _, h, w, _ = shape
            if isinstance(h, int) and isinstance(w, int):
                logger.info("Model input size detected: %dx%d", h, w)
                return (h, w)
        raise ValueError(f"Cannot parse input shape: {shape}")
    except Exception as exc:
        logger.warning(
            "Could not determine model input shape (%s). Falling back to %dx%d.",
            exc, DEFAULT_IMG_SIZE[0], DEFAULT_IMG_SIZE[1],
        )
        return DEFAULT_IMG_SIZE


def get_model_channels(model) -> int:
    try:
        channels = model.input_shape[-1]
        if isinstance(channels, int):
            return channels
    except Exception:
        pass
    return 3


# ── Image decode ──────────────────────────────────────────────────────────────

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to RGB uint8 numpy array. OpenCV first, Pillow fallback."""
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info("Image decoded via OpenCV, shape=%s", img_rgb.shape)
        return img_rgb
    except Exception as cv_exc:
        logger.warning("OpenCV decode failed (%s); trying Pillow fallback.", cv_exc)

    try:
        from io import BytesIO
        from PIL import Image
        pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img_rgb = np.array(pil_img, dtype=np.uint8)
        logger.info("Image decoded via Pillow, shape=%s", img_rgb.shape)
        return img_rgb
    except Exception as pil_exc:
        raise ValueError(f"Failed to decode image: {pil_exc}")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int],
    channels: int = 3,
) -> np.ndarray:
    """Resize + normalise to (1, H, W, C) float32 in [0, 1]."""
    h, w = target_size
    if channels == 1:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (w, h), interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_float, axis=0)
        img_batch = np.expand_dims(img_batch, axis=-1)
    else:
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_float, axis=0)

    logger.info("Preprocessed to shape=%s (channels=%d)", img_batch.shape, channels)
    return img_batch


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, input_array: np.ndarray) -> Tuple[dict, float]:
    """
    Run model inference and return parsed outputs dict + elapsed ms.

    Your model has 3 outputs:
        thermal_out   (None, 512, 512, 1)  — spatial heatmap
        ripeness_out  (None, 1)            — sigmoid ripeness probability
        guava_guard   (None, 1)            — sigmoid guava detection probability

    Returns dict with keys: ripeness, thermal, guava_guard
    """
    import tensorflow as tf

    start = time.perf_counter()
    with tf.device("/CPU:0"):
        raw = model.predict(input_array, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    outputs = _parse_model_outputs(raw)
    logger.info(
        "Inference: %.1f ms | ripeness=%s | thermal=%s | guava_guard=%s",
        elapsed_ms,
        outputs["ripeness"].shape  if outputs["ripeness"]    is not None else None,
        outputs["thermal"].shape   if outputs["thermal"]     is not None else None,
        outputs["guava_guard"]     if outputs["guava_guard"] is not None else None,
    )
    return outputs, elapsed_ms


def _parse_model_outputs(raw) -> dict:
    """
    Split multi-output model results into named outputs.

    Model output order (from model.summary()):
        index 0 — thermal_out   shape (1, 512, 512, 1)  → spatial / 4-D
        index 1 — ripeness_out  shape (1, 1)            → classification
        index 2 — guava_guard   shape (1, 1)            → classification

    Strategy:
        - 4-D outputs  → thermal candidate (pick largest)
        - 2-D outputs  → classification candidates, sorted by original index
          first  classification output = ripeness
          second classification output = guava_guard
    """
    result = {"ripeness": None, "thermal": None, "guava_guard": None, "raw": raw}

    # Single output model
    if not isinstance(raw, list):
        arr = raw if isinstance(raw, np.ndarray) else np.array(raw, dtype=np.float32)
        result["ripeness"] = arr
        return result

    spatial_candidates        = []
    classification_candidates = []

    for i, arr in enumerate(raw):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 4:
            spatial_candidates.append((i, arr))
        elif arr.ndim == 2:
            classification_candidates.append((i, arr))
        elif arr.ndim == 1:
            classification_candidates.append((i, arr.reshape(1, -1)))

    # Thermal — largest spatial output
    if spatial_candidates:
        spatial_candidates.sort(key=lambda x: x[1].size, reverse=True)
        result["thermal"] = spatial_candidates[0][1]
        logger.info("Thermal output: index=%d shape=%s", spatial_candidates[0][0], spatial_candidates[0][1].shape)

    # Classification outputs — preserve original model order
    classification_candidates.sort(key=lambda x: x[0])

    if len(classification_candidates) >= 1:
        result["ripeness"]    = classification_candidates[0][1]
        logger.info("Ripeness output: index=%d shape=%s", classification_candidates[0][0], classification_candidates[0][1].shape)
    if len(classification_candidates) >= 2:
        result["guava_guard"] = classification_candidates[1][1]
        logger.info("Guava guard output: index=%d shape=%s value=%.4f",
                    classification_candidates[1][0],
                    classification_candidates[1][1].shape,
                    float(classification_candidates[1][1].flatten()[0]))

    return result


# ── Thermal image rendering ───────────────────────────────────────────────────

def render_thermal_image(thermal_array: np.ndarray) -> str:
    """
    Convert raw thermal model output to a colourised PNG base64 data URI.

    thermal_array: (1, H, W, 1) values in [0, 1]
    Returns: "data:image/png;base64,..."
    """
    arr = thermal_array.squeeze()

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-6:
        arr_norm = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    else:
        arr_norm = np.zeros_like(arr, dtype=np.uint8)

    # COLORMAP_INFERNO: dark purple=cool, orange=warm, yellow=hot
    coloured = cv2.applyColorMap(arr_norm, cv2.COLORMAP_INFERNO)

    success, buffer = cv2.imencode(".png", coloured)
    if not success:
        raise RuntimeError("Failed to encode thermal image to PNG")

    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ── Prediction labels ─────────────────────────────────────────────────────────

def build_predictions(
    ripeness_array: np.ndarray,
    labels: Optional[list],
    top_k: int = 5,
) -> list:
    """Convert ripeness sigmoid output to sorted [{label, confidence}] list."""
    if not isinstance(ripeness_array, np.ndarray):
        ripeness_array = np.array(ripeness_array, dtype=np.float32)

    probs = ripeness_array.flatten()

    # Single sigmoid → binary [not ripe, ripe]
    if len(probs) == 1:
        p = float(probs[0])
        probs = np.array([1.0 - p, p], dtype=np.float32)

    num_classes = len(probs)
    if labels is None:
        label_list = [f"class_{i}" for i in range(num_classes)]
    else:
        label_list = list(labels)
        if len(label_list) < num_classes:
            label_list += [f"class_{i}" for i in range(len(label_list), num_classes)]

    results = [
        {"label": label_list[i], "confidence": float(probs[i])}
        for i in range(num_classes)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_k]