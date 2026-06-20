from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    user_id: str
    host_id: str
    process_name: str
    event_type: str
    source_ip: str
    destination_ip: str
    bytes_in: int = Field(ge=0)
    bytes_out: int = Field(ge=0)
    event_hour: int = Field(ge=0, le=23)
    is_business_hour: bool

class PredictionResponse(BaseModel):
    prediction: int = Field(ge=0, le=1)
    classification: Literal["normal", "suspicious"]
    suspicious_probability: float = Field(ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_loaded: bool
    model_path: str