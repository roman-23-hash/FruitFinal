"""Image preprocessing utilities for the fruit ripeness API."""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_IMG_SIZE: Tuple[int, int] = (224, 224)


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
        logger.warning("Could not determine model input shape (%s). Falling back to %dx%d.",
                       exc, DEFAULT_IMG_SIZE[0], DEFAULT_IMG_SIZE[1])
        return DEFAULT_IMG_SIZE


def get_model_channels(model) -> int:
    try:
        channels = model.input_shape[-1]
        if isinstance(channels, int):
            return channels
    except Exception:
        pass
    return 3


def decode_image(file_bytes: bytes) -> np.ndarray:
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


def preprocess_image(img_rgb: np.ndarray, target_size: Tuple[int, int], channels: int = 3) -> np.ndarray:
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


def _find_classification_output(output_list: list) -> np.ndarray:
    """
    Given a list of numpy arrays from a multi-output model, find the one
    most likely to be a classification/regression scalar output.

    Strategy: prefer outputs whose shape is (batch, N) where N is small
    (i.e. not a spatial map). Among those, prefer the one named
    'ripeness_out' if available, else the smallest non-spatial output.
    """
    candidates = []
    for i, arr in enumerate(output_list):
        if arr.ndim == 2:          # (batch, N) — classification head
            candidates.append((i, arr))
        elif arr.ndim == 1:        # (batch,) — also fine
            candidates.append((i, arr.reshape(arr.shape[0], 1)))

    if not candidates:
        # Fallback: take the last output whatever it is
        logger.warning("No 2-D output found; using last model output as classification head.")
        arr = output_list[-1]
        return arr.reshape(1, -1)

    # Among candidates prefer the smallest (fewest classes) that isn't 1×spatial
    candidates.sort(key=lambda x: x[1].shape[-1])
    chosen_idx, chosen_arr = candidates[0]
    logger.info("Multi-output model: selected output index %d, shape=%s for classification",
                chosen_idx, chosen_arr.shape)
    return chosen_arr


def run_inference(model, input_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run model inference. Handles single-output and multi-output models."""
    import tensorflow as tf

    start = time.perf_counter()
    with tf.device("/CPU:0"):
        raw = model.predict(input_array, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Multi-output model returns a list of arrays
    if isinstance(raw, list):
        logger.info("Multi-output model detected: %d outputs with shapes %s",
                    len(raw), [getattr(o, 'shape', type(o)) for o in raw])
        preds = _find_classification_output(raw)
    else:
        preds = raw

    if not isinstance(preds, np.ndarray):
        preds = np.array(preds, dtype=np.float32)

    logger.info("Inference: %.1f ms, final preds shape=%s dtype=%s",
                elapsed_ms, preds.shape, preds.dtype)
    return preds, elapsed_ms


def build_predictions(preds: np.ndarray, labels: Optional[list], top_k: int = 5) -> list:
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds, dtype=np.float32)

    # Flatten to 1D probabilities
    probs = preds.flatten()

    # Single sigmoid value → binary [not ripe, ripe]
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

    results = [{"label": label_list[i], "confidence": float(probs[i])} for i in range(num_classes)]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_k]