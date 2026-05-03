from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, Dict, Optional
from datetime import datetime


class PatientInput(BaseModel):
    """Input payload containing features for a single patient/cita.

    The client should send a JSON object with a `features` dictionary
    whose keys match the feature column names used during training.
    """

    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    label: int
    final_label: Optional[int] = None
    probability: Optional[float]
    verification: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None


class AppointmentCreate(BaseModel):
    medic_id: str = Field(..., min_length=1, max_length=64)
    patient_id: str = Field(..., min_length=1, max_length=64)
    hour: int = Field(..., ge=0, le=23)
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)

    @field_validator("medic_id", "patient_id")
    @classmethod
    def strip_and_validate_ids(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must not be empty")
        return v


class AppointmentUpdate(BaseModel):
    medic_id: str = Field(..., min_length=1, max_length=64)
    patient_id: str = Field(..., min_length=1, max_length=64)
    hour: int = Field(..., ge=0, le=23)
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    appointment_type: int = Field(..., ge=0, le=2)

    @field_validator("medic_id", "patient_id")
    @classmethod
    def strip_and_validate_ids(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must not be empty")
        return v


class AppointmentOut(BaseModel):
    id: int
    medic_id: str
    patient_id: str
    hour: int
    day: int
    month: int
    appointment_type: int
    created_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)
