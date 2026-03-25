"""Pydantic request/response models for the prediction API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models: List[ModelStatus]


class PredictionResponse(BaseModel):
    status: str
    model: str
    settlement_point: str
    generated_at: str
    predictions: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    status: str
    info: Dict[str, Any]


class AlertConfigRequest(BaseModel):
    chat_ids: Optional[List[str]] = None
    spike_alert_threshold: Optional[float] = None
    spike_cooldown_minutes: Optional[int] = None
    bot_token: Optional[str] = None


class CreateKeyRequest(BaseModel):
    name: str
    tier: str = "free"


class KeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str
    tier: str
    active: bool
    created_at: str
    last_used: Optional[str]
    request_count: int
