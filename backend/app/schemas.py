from pydantic import BaseModel
from typing import Any, Dict, Optional


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
