from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainResult:
    model: Any
    threshold: float
    metrics: Dict[str, float]


class BaseTrainer:
    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X, y) -> Any:
        self.model.fit(X, y)
        return self.model
