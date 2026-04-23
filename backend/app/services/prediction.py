"""Prediction service: loads the serialized pipeline and exposes a helper
to run predictions from a feature dict.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
)


BASE_BACKEND = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_BACKEND / "models" / "model.pkl"
METRICS_PATH = BASE_BACKEND / "models" / "metrics.json"


def _load_artifacts() -> Dict[str, Any]:
    model = None
    metrics = {}
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH!s}")

    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as fp:
            metrics = json.load(fp)

    return {"model": model, "metrics": metrics}


ARTIFACTS = _load_artifacts()
MODEL = ARTIFACTS["model"]
METRICS = ARTIFACTS["metrics"]


def predict_from_dict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single prediction given a dict of feature values.

    - Builds a one-row DataFrame using the `feature_columns` saved in metrics.json.
    - Calls the pipeline's `predict` and `predict_proba` when available.
    - Returns a dict with `label`, `probability` and optional `model_version`.
    """

    feature_columns = METRICS.get("feature_columns", [])
    if not feature_columns:
        # fallback: use keys from features
        row = features
    else:
        # ensure all expected columns are present (fill missing with None)
        row = {c: features.get(c, None) for c in feature_columns}

    df = pd.DataFrame([row])

    proba: Optional[float] = None
    if hasattr(MODEL, "predict_proba"):
        try:
            p = MODEL.predict_proba(df)
            # binary classification assumed: prob of class 1
            if p.shape[1] == 2:
                proba = float(p[0, 1])
            else:
                proba = float(p[0].max())
        except Exception:
            proba = None

    # Predict label
    threshold = METRICS.get("threshold")
    if proba is not None and isinstance(threshold, (int, float)):
        label = int(proba >= float(threshold))
    else:
        y_pred = MODEL.predict(df)
        label = int(y_pred[0])

    return {
        "label": label,
        "probability": proba,
        "model_version": METRICS.get("model_version"),
    }


def info() -> Dict[str, Any]:
    return {"model_path": str(MODEL_PATH), "metrics": METRICS}
