from pydantic import BaseModel
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
    probability: Optional[float]
    model_version: Optional[str] = None


class AppointmentCreate(BaseModel):
    patient_id: str
    hour: int
    day: int
    month: int
    # appointment_type is set by server when creating a new appointment
    pass


class AppointmentUpdate(BaseModel):
    patient_id: str
    hour: int
    day: int
    month: int
    appointment_type: int


class AppointmentOut(BaseModel):
    id: int
    patient_id: str
    hour: int
    day: int
    month: int
    appointment_type: int
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
