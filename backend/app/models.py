"""Pydantic response models for the Fruit Ripeness API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictionItem(BaseModel):
    label: str
    confidence: float  # 0.0 – 1.0


class PredictionMeta(BaseModel):
    model_input_shape: List[Optional[int]]
    processing_time_ms: float


class PredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionItem]
    meta: PredictionMeta


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_input_shape: Optional[List[Optional[int]]] = None
    labels_loaded: bool = False
    num_classes: Optional[int] = None
    message: Optional[str] = None
