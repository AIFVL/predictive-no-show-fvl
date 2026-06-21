"""Prediction service for no-show inference.

Supports two artifact formats currently present in the repository:

1. `outputs/catboost/catboost_final.joblib` (Primary)
   A dictionary artifact created by `src/train.py` containing the trained
   CatBoost model, feature columns and threshold.
2. `backend/models/model.pkl` (Legacy fallback)
   A serialized sklearn-compatible pipeline saved directly.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from .double_verification import DEFAULT_DOUBLE_VERIFICATION_CONFIG, double_verify


warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_BACKEND = REPO_ROOT / "backend"

CATBOOST_MODEL_PATH = REPO_ROOT / "outputs" / "catboost" / "catboost_final.joblib"
CATBOOST_METRICS_PATH = REPO_ROOT / "outputs" / "catboost" / "catboost_final_metrics.json"
STACKING_MODEL_PATH = REPO_ROOT / "outputs" / "stacking" / "stacking_final.joblib"
STACKING_METRICS_PATH = REPO_ROOT / "outputs" / "stacking" / "stacking_final_metrics.json"
LEGACY_MODEL_PATH = BASE_BACKEND / "models" / "model.pkl"
LEGACY_METRICS_PATH = BASE_BACKEND / "models" / "metrics.json"


def _resolve_artifact_paths() -> tuple[Path, Path]:
    """Pick the preferred model artifact, allowing overrides by env vars.
    
    Priority order:
    1. Environment variable override (PREDICTIVE_MODEL_PATH)
    2. CatBoost model (primary)
    3. Stacking model (fallback)
    4. Legacy model (legacy fallback)
    """

    model_override = os.environ.get("PREDICTIVE_MODEL_PATH")
    metrics_override = os.environ.get("PREDICTIVE_METRICS_PATH")
    if model_override:
        model_path = Path(model_override)
        metrics_path = Path(metrics_override) if metrics_override else CATBOOST_METRICS_PATH
        return model_path, metrics_path

    # Prefer CatBoost model
    if CATBOOST_MODEL_PATH.exists():
        return CATBOOST_MODEL_PATH, CATBOOST_METRICS_PATH

    # Fallback to Stacking
    if STACKING_MODEL_PATH.exists():
        return STACKING_MODEL_PATH, STACKING_METRICS_PATH

    return LEGACY_MODEL_PATH, LEGACY_METRICS_PATH


MODEL_PATH, METRICS_PATH = _resolve_artifact_paths()


def _load_artifacts() -> Dict[str, Any]:
    model = None
    scaler = None
    metrics: Dict[str, Any] = {}
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
            scaler = artifact.get("scaler")
            metrics.setdefault("feature_columns", artifact.get("feature_cols") or [])
            metrics.setdefault("threshold", artifact.get("threshold"))
            metrics.setdefault("artifact_type", "training_bundle")
        else:
            model = artifact
            metrics.setdefault("artifact_type", "pipeline")
    else:
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH!s}")

    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as fp:
            loaded_metrics = json.load(fp)
            if isinstance(loaded_metrics, dict):
                metrics.update(loaded_metrics)

    metrics.setdefault("model_path", str(MODEL_PATH))
    metrics.setdefault("metrics_path", str(METRICS_PATH))
    metrics.setdefault("model_version", _infer_model_version())

    return {"model": model, "scaler": scaler, "metrics": metrics}


def _infer_model_version() -> str:
    model_name = MODEL_PATH.name.lower()
    if "catboost" in model_name:
        return "catboost_final"
    if "stack" in model_name:
        return "stacking_final"
    if "lightgbm" in model_name:
        return "lightgbm_smote"
    return MODEL_PATH.stem

_MODEL: Any | None = None
_SCALER: Any | None = None
METRICS: Dict[str, Any] = {}


def _ensure_loaded() -> Any:
    global _MODEL, _SCALER, METRICS
    if _MODEL is not None:
        return _MODEL
    artifacts = _load_artifacts()
    _MODEL = artifacts["model"]
    _SCALER = artifacts.get("scaler")
    # IMPORTANT: update in-place so any modules that imported `METRICS`
    # (e.g., route modules) see the loaded values.
    METRICS.clear()
    METRICS.update(artifacts.get("metrics", {}) or {})
    return _MODEL


def _prepare_bundle_input(features: Dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Rebuild the transformed feature frame expected by training bundles.

    `src/train.py` trains stacking artifacts over one-hot encoded and scaled
    tabular inputs. At inference time we reconstruct those one-hot columns
    deterministically from the raw feature dict before aligning to the
    persisted `feature_cols`.
    """

    raw = {k: features.get(k) for k in features}
    row: Dict[str, Any] = {}

    numeric_candidates = {
        "Age",
        "Number of Diseases",
        "Recent Hospitalization",
        "Number of Medications",
        "Hour",
        "Creation to Assignment Interval",
        "Number of Previous Attendance",
        "Number of Previous Non-Attendance",
    }
    categorical_candidates = {
        "Sex",
        "Insurance Type",
        "Day",
        "Month",
    }

    for col in feature_columns:
        if col in numeric_candidates:
            row[col] = raw.get(col, 0)
            continue

        if col in categorical_candidates:
            row[col] = str(raw.get(col, "0"))
            continue

        matched = False
        for base_name in categorical_candidates:
            prefix = f"{base_name}_"
            if col.startswith(prefix):
                suffix = col[len(prefix):]
                value = raw.get(base_name)
                row[col] = 1 if str(value) == suffix else 0
                matched = True
                break

        if not matched:
            row[col] = raw.get(col, 0)

    return pd.DataFrame([row], columns=feature_columns).fillna(0)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _clamp_probability(value: Optional[float]) -> Optional[float]:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return max(0.0, min(1.0, parsed))


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _unwrap_predictor(model: Any) -> Any:
    """Return the estimator that owns the predictive weights when possible."""

    if hasattr(model, "named_steps"):
        steps = getattr(model, "named_steps", {}) or {}
        if "clf" in steps:
            return steps["clf"]
        if steps:
            return list(steps.values())[-1]
    return model


def _vector_to_floats(values: Any, expected_len: int) -> Optional[list[float]]:
    try:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        vector = [abs(float(v)) for v in values]
        if len(vector) != expected_len:
            return None
        total = sum(vector)
        if total <= 0:
            return None
        return [v / total for v in vector]
    except Exception:
        return None


def _direct_feature_weights(estimator: Any, expected_len: int) -> Optional[list[float]]:
    if hasattr(estimator, "feature_importances_"):
        weights = _vector_to_floats(getattr(estimator, "feature_importances_"), expected_len)
        if weights:
            return weights

    if hasattr(estimator, "coef_"):
        weights = _vector_to_floats(getattr(estimator, "coef_"), expected_len)
        if weights:
            return weights

    return None


def _model_feature_weights(model: Any, feature_columns: list[str]) -> Optional[list[float]]:
    """Best-effort weights for local explanation.

    The active stacking artifact does not persist SHAP values. For the stack,
    combine each base estimator's feature importances using the meta-model
    coefficients as weights. This keeps the explanation tied to model weights
    while leaving the actual probability untouched.
    """

    if not feature_columns:
        return None

    predictor = _unwrap_predictor(model)
    expected_len = len(feature_columns)

    direct = _direct_feature_weights(predictor, expected_len)
    if direct:
        return direct

    estimators = getattr(predictor, "estimators_", None)
    if not estimators:
        return None

    meta_weights = [1.0 for _ in estimators]
    final_estimator = getattr(predictor, "final_estimator_", None)
    if final_estimator is not None and hasattr(final_estimator, "coef_"):
        parsed = _vector_to_floats(getattr(final_estimator, "coef_"), len(estimators))
        if parsed:
            meta_weights = parsed

    combined = [0.0 for _ in feature_columns]
    for idx, estimator in enumerate(estimators):
        base_weights = _direct_feature_weights(_unwrap_predictor(estimator), expected_len)
        if not base_weights:
            continue
        meta_weight = meta_weights[idx] if idx < len(meta_weights) else 1.0
        for col_idx, value in enumerate(base_weights):
            combined[col_idx] += meta_weight * value

    total = sum(combined)
    if total <= 0:
        return None
    return [v / total for v in combined]


def _feature_label(feature_name: str) -> str:
    categorical_prefixes = ("Sex_", "Insurance Type_", "Day_", "Month_")
    for prefix in categorical_prefixes:
        if feature_name.startswith(prefix):
            return feature_name.replace("_", ": ", 1)
    return feature_name


def _base_feature_name(feature_name: str) -> str:
    categorical_prefixes = ("Sex_", "Insurance Type_", "Day_", "Month_")
    for prefix in categorical_prefixes:
        if feature_name.startswith(prefix):
            return prefix[:-1]
    return feature_name


def _build_explanation_weights(
    *,
    model: Any,
    df: pd.DataFrame,
    raw_features: Dict[str, Any],
    feature_columns: list[str],
) -> Dict[str, Dict[str, float]]:
    weights = _model_feature_weights(model, feature_columns)
    if not weights or not feature_columns:
        return {}

    row = df.iloc[0].to_dict() if not df.empty else {}
    explanation: Dict[str, Dict[str, float]] = {}

    for idx, column in enumerate(feature_columns):
        weight = float(weights[idx] if idx < len(weights) else 0.0)
        if weight <= 0:
            continue

        value = _as_float(row.get(column))
        raw_value = _json_safe_value(raw_features.get(column))
        local_strength = float(weight * abs(value if value is not None else 1.0))
        feature_names = {column, _base_feature_name(column)}

        for feature_name in feature_names:
            current = explanation.setdefault(
                feature_name,
                {
                    "weight": 0.0,
                    "local_strength": 0.0,
                    "value": raw_value if raw_value is not None else _json_safe_value(row.get(column)),
                },
            )
            current["weight"] += weight
            current["local_strength"] += local_strength

    return explanation


def _build_model_analysis(
    *,
    model: Any,
    df: pd.DataFrame,
    raw_features: Dict[str, Any],
    feature_columns: list[str],
    probability_no_show: Optional[float],
    model_label: Optional[int],
    final_label: Optional[int],
    explanation_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Return probabilities and explanatory weights for the frontend.

    There is no persisted SHAP explainer in this repository. The response keeps
    that explicit and falls back to model-derived feature weights so the UI can
    show the automatic probability separately from double verification.
    """

    prob_no_show = _clamp_probability(probability_no_show)
    prob_attend = None if prob_no_show is None else 1.0 - prob_no_show
    decision_label = final_label if final_label is not None else model_label
    predicted_outcome = "no_show" if decision_label == 1 else "attend" if decision_label == 0 else None
    display_probability = prob_no_show if predicted_outcome == "no_show" else prob_attend

    factors: list[Dict[str, Any]] = []
    if explanation_weights:
        scored = [
            (
                float(info.get("local_strength", 0.0) or 0.0),
                float(info.get("weight", 0.0) or 0.0),
                feature,
                info.get("value"),
            )
            for feature, info in explanation_weights.items()
        ]

        for strength, weight, feature, value in sorted(scored, key=lambda item: item[0], reverse=True)[:6]:
            if weight <= 0:
                continue
            factors.append(
                {
                    "feature": feature,
                    "label": _feature_label(feature),
                    "value": value,
                    "weight": float(weight),
                    "local_strength": float(strength),
                }
            )

    return {
        "method": "model_feature_weights",
        "method_label": "Pesos del modelo",
        "shap_available": False,
        "note": "No hay un explainer SHAP persistido; se usan pesos/importancias del modelo activo.",
        "predicted_outcome": predicted_outcome,
        "display_probability": display_probability,
        "probability_attend": prob_attend,
        "probability_no_show": prob_no_show,
        "top_factors": factors,
    }


def predict_from_dict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single prediction given a dict of feature values.

    - Builds a one-row DataFrame using the `feature_columns` saved in metrics.json.
    - Calls the pipeline's `predict` and `predict_proba` when available.
    - Returns a dict with `label`, `probability` and optional `model_version`.
    """

    model = _ensure_loaded()

    feature_columns = METRICS.get("feature_columns", [])
    artifact_type = METRICS.get("artifact_type")

    if not feature_columns:
        # fallback: use keys from features
        row = features
        df = pd.DataFrame([row])
    elif artifact_type == "training_bundle":
        df = _prepare_bundle_input(features, feature_columns)
        if _SCALER is not None:
            transformed = _SCALER.transform(df)
            df = pd.DataFrame(transformed, columns=feature_columns)
    else:
        # ensure all expected columns are present (fill missing with None)
        row = {c: features.get(c, None) for c in feature_columns}
        df = pd.DataFrame([row])

    proba: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(df)
            # binary classification assumed: prob of class 1
            if p.shape[1] == 2:
                proba = float(p[0, 1])
            else:
                proba = float(p[0].max())
        except Exception:
            proba = None

    # Predict label (model threshold)
    threshold = METRICS.get("threshold")
    if proba is not None and isinstance(threshold, (int, float)):
        label = int(proba >= float(threshold))
    else:
        y_pred = model.predict(df)
        label = int(y_pred[0])

    explanation_weights = _build_explanation_weights(
        model=model,
        df=df,
        raw_features=features,
        feature_columns=feature_columns,
    )

    verification = None
    final_label = None
    dv_cfg = METRICS.get("double_verification")
    if not isinstance(dv_cfg, dict):
        dv_cfg = DEFAULT_DOUBLE_VERIFICATION_CONFIG
    if isinstance(dv_cfg, dict) and proba is not None and isinstance(threshold, (int, float)):
        try:
            verification = double_verify(
                features=features,
                probability_no_show=float(proba),
                model_threshold=float(threshold),
                config=dv_cfg,
                explanation_weights=explanation_weights,
            )
            final_label = verification.get("decision", {}).get("final_label")
        except Exception:
            verification = None
            final_label = None

    prob_no_show = _clamp_probability(proba)
    prob_attend = None if prob_no_show is None else 1.0 - prob_no_show
    model_analysis = _build_model_analysis(
        model=model,
        df=df,
        raw_features=features,
        feature_columns=feature_columns,
        probability_no_show=prob_no_show,
        model_label=label,
        final_label=final_label,
        explanation_weights=explanation_weights,
    )

    return {
        "label": label,
        "final_label": final_label,
        "probability": proba,
        "prob_no_show": prob_no_show,
        "prob_attend": prob_attend,
        "model_analysis": model_analysis,
        "shap_analysis": model_analysis,
        "verification": verification,
        "model_version": METRICS.get("model_version"),
    }


def info() -> Dict[str, Any]:
    # Do not force loading model.pkl for info.
    return {
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "metrics": METRICS,
    }


def reload_model() -> Dict[str, Any]:
    """
    Recarga el modelo desde disco. Se usa después del reentrenamiento
    automático para que el API use el nuevo modelo sin reiniciar.
    
    Retorna información del modelo recargado.
    """
    global _MODEL, _SCALER, METRICS, MODEL_PATH, METRICS_PATH
    
    # Reset variables globales
    _MODEL = None
    _SCALER = None
    METRICS.clear()
    MODEL_PATH, METRICS_PATH = _resolve_artifact_paths()
    
    # Fuerza recarga
    _ensure_loaded()
    
    return {
        "status": "model_reloaded",
        "model_version": METRICS.get("model_version"),
        "model_path": str(MODEL_PATH),
        "metrics": METRICS,
    }
