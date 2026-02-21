"""Pydantic response models for the Fruit Ripeness API."""

from typing import List, Optional
from pydantic import BaseModel


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictionMeta(BaseModel):
    model_input_shape: List[Optional[int]]
    processing_time_ms: float
    gate_confidence: Optional[float] = None
    gate_message: Optional[str] = None


class PredictionResponse(BaseModel):
    success: bool
    is_guava: bool = True
    predictions: List[PredictionItem]
    thermal_image: Optional[str] = None
    meta: PredictionMeta


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gate_available: bool = False
    gate_enabled: bool = True
    gate_threshold: float = 0.5
    model_input_shape: Optional[List[Optional[int]]] = None
    labels_loaded: bool = False
    message: Optional[str] = None