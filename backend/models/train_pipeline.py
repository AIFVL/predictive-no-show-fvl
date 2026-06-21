"""Reusable training script for the appointment attendance model.

This module mirrors the preprocessing and modeling steps developed in
notebooks/02_Limpieza.ipynb. It loads the raw dataset, applies the
normalization and cleaning rules, trains a CatBoost sklearn-compatible
pipeline, reports metrics, and serializes the fitted pipeline for inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import difflib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostClassifier


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATHS = [
    REPO_ROOT / "data" / "raw" / "database_non-shows.csv",
    REPO_ROOT / "data" / "raw" / "database_non-shows.xlsx",
]
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACT_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
CATBOOST_OUTPUT_DIR = REPO_ROOT / "outputs" / "catboost"
CATBOOST_MODEL_ARTIFACT_PATH = CATBOOST_OUTPUT_DIR / "catboost_final.joblib"
CATBOOST_METRICS_PATH = CATBOOST_OUTPUT_DIR / "catboost_final_metrics.json"

TARGET_COLUMN = "Appointment Type"
MAX_CREATION_ASSIGNMENT_INTERVAL = 365

# CatBoost is the production training model. Keep these params aligned with
# the tuned CatBoost config used by src/train.py.
CATBOOST_BEST_PARAMS: Dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": 42,
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 600,
    "l2_leaf_reg": 3.0,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "allow_writing_files": False,
}

SMOTE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "random_state": 42,
    "k_neighbors": 5,
}

RENAME_MAP = {
    "0ppointment Type": "Appointment Type",
    "0ppointment_Type": "Appointment Type",
    "0ppointmentType": "Appointment Type",
    "Number of diseases": "Number of Diseases",
    "Number of diseases ": "Number of Diseases",
}

NUMERIC_FEATURE_CANDIDATES = [
    "Age",
    "Number of Diseases",
    "Recent Hospitalization",
    "Number of Medications",
    "Hour",
    "Creation to Assignment Interval",
    "Number of Previous Attendance",
    "Number of Previous Non-Attendance",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "Sex",
    "Insurance Type",
    "Day",
    "Month",
]


# ---------------------------------------------------------------------------
# Double verification (rule-based) configuration
# ---------------------------------------------------------------------------

# These rules are used as a second pass to validate model predictions.
# They are serialized into metrics.json so inference can expose explanations.
DOUBLE_VERIFICATION_CONFIG: Dict[str, Any] = {
    "version": 1,
    # Conservative defaults: require a minimum score to confirm model outputs.
    # You can adjust these values and retrain to update the persisted config.
    "non_attendance": {
        "min_score_confirm": 4,
        "rules": [
            {
                "id": "NA1",
                "name": "Historial de inasistencias alto",
                "condition": "Number of Previous Non-Attendance >= 2",
                "weight": 2,
                "enabled": True,
            },
            {
                "id": "NA2",
                "name": "Alta tasa de no-show",
                "condition": "Prev_NoShow_Rate > 0.5",
                "weight": 2,
                "enabled": True,
            },
            {
                "id": "NA3",
                "name": "Baja experiencia con citas",
                "condition": "Prev_Total <= 2",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "NA4",
                "name": "Última cita fue no-show",
                "condition": "Last_Attendance == No-Show",
                "weight": 2,
                "enabled": True,
            },
            {
                "id": "NA5",
                "name": "Intervalo largo de asignación",
                "condition": "Creation to Assignment Interval > 7",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "NA6",
                "name": "Cita con poca anticipación",
                "condition": "Creation to Assignment Interval <= 2",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "NA7",
                "name": "Paciente joven + bajo compromiso",
                "condition": "Age < 30 AND Number of Previous Attendance <= 1",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "NA8",
                "name": "Baja carga clínica + bajo compromiso",
                "condition": "Number of Diseases <= 1 AND Number of Previous Attendance <= 1",
                "weight": 1,
                "enabled": True,
            },
        ],
    },
    "attendance": {
        "min_score_confirm": 4,
        "rules": [
            {
                "id": "A1",
                "name": "Historial alto de asistencia",
                "condition": "Number of Previous Attendance >= 3",
                "weight": 2,
                "enabled": True,
            },
            {
                "id": "A2",
                "name": "Baja tasa de no-show",
                "condition": "Prev_NoShow_Rate <= 0.2",
                "weight": 2,
                "enabled": True,
            },
            {
                "id": "A3",
                "name": "Última cita fue asistida",
                "condition": "Last_Attendance == Show",
                "weight": 2,
                "enabled": True,
            },
            {
                "id": "A4",
                "name": "Alta experiencia con citas",
                "condition": "Prev_Total >= 4",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "A5",
                "name": "Intervalo moderado",
                "condition": "3 <= Creation to Assignment Interval <= 7",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "A6",
                "name": "Paciente con mayor carga clínica",
                "condition": "Number of Diseases >= 2",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "A7",
                "name": "Mayor consumo de medicamentos",
                "condition": "Number of Medications >= 2",
                "weight": 1,
                "enabled": True,
            },
            {
                "id": "A8",
                "name": "(Ambiguo) Baja carga clínica + bajo compromiso",
                "condition": "Number of Diseases <= 1 AND Number of Previous Attendance <= 1",
                "weight": 1,
                "enabled": False,
                "notes": "En la tabla compartida este criterio aparece también en inasistencia con la misma condición; se deja deshabilitado para evitar contradicciones hasta confirmar la regla correcta.",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def _default_raw_data_path() -> Path:
    for candidate in DEFAULT_RAW_DATA_PATHS:
        if candidate.exists():
            return candidate
    return DEFAULT_RAW_DATA_PATHS[0]


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw appointment dataset from CSV or Excel.

    The path can be provided explicitly (CLI), via the RAW_DATA_PATH env var,
    or will fall back to the default repo location.
    """

    if path is not None:
        candidate = Path(path)
    elif RAW_DATA_PATH:
        candidate = Path(RAW_DATA_PATH)
    else:
        candidate = _default_raw_data_path()

    if not candidate.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Expected a CSV or Excel file at: "
            f"{candidate!s}. "
            "\n\nSoluciones:\n"
            "- Copia el CSV a: data/raw/database_non-shows.csv\n"
            "- O copia el Excel a: data/raw/database_non-shows.xlsx\n"
            "- O ejecuta: python backend/models/train_pipeline.py --raw-data \"C:/ruta/tu_archivo.csv\"\n"
            "- O define env var RAW_DATA_PATH con la ruta del dataset."
        )

    suffix = candidate.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(candidate)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(candidate)
    raise ValueError(f"Formato no soportado para dataset raw: {candidate.suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column renames, string trimming, and numeric coercion."""

    df_norm = df.copy()

    # Apply exact renames if present
    renames = {old: new for old, new in RENAME_MAP.items() if old in df_norm.columns}
    if renames:
        df_norm = df_norm.rename(columns=renames)
        print(f"Applied exact renames: {renames}")

    # Fuzzy-match common header typos (e.g. '0ppointment Type') against
    # important target names so the pipeline finds the target column.
    existing_cols = list(df_norm.columns)
    existing_lower = [c.lower() for c in existing_cols]

    # Map any existing column names that are similar to canonical names
    canonical_names = set(list(RENAME_MAP.values()) + [TARGET_COLUMN])
    canonical_lower = {c.lower(): c for c in canonical_names}
    fuzzy_map = {}
    for orig_col in existing_cols:
        if orig_col in canonical_names:
            continue
        match = difflib.get_close_matches(orig_col.lower(), list(canonical_lower.keys()), n=1, cutoff=0.7)
        if match:
            matched_canon = canonical_lower[match[0]]
            fuzzy_map[orig_col] = matched_canon

    if fuzzy_map:
        df_norm = df_norm.rename(columns=fuzzy_map)
        print(f"Applied fuzzy renames: {fuzzy_map}")

    object_cols = df_norm.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df_norm[col] = df_norm[col].astype(str).str.strip()

    for col in NUMERIC_FEATURE_CANDIDATES:
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce")

    return df_norm


def filter_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that violate domain rules or are duplicated."""

    df_filtered = df.copy()
    invalid_mask = pd.Series(False, index=df_filtered.index)

    def _rule(column: str, condition) -> None:
        nonlocal invalid_mask
        if column in df_filtered.columns:
            mask = condition(df_filtered[column])
            invalid_mask = invalid_mask | mask

    _rule(TARGET_COLUMN, lambda s: ~s.isin([0, 1]))
    _rule("Age", lambda s: (s < 18) | (s > 120))
    _rule("Sex", lambda s: ~s.isin([0, 1, 2]))
    _rule("Insurance Type", lambda s: ~s.isin(range(0, 9)))
    _rule("Number of Diseases", lambda s: s < 0)
    _rule("Recent Hospitalization", lambda s: s < 0)
    _rule("Number of Medications", lambda s: s < 0)
    _rule("Hour", lambda s: (s < 0) | (s > 23))
    _rule("Day", lambda s: ~s.between(0, 6))
    _rule("Month", lambda s: ~s.between(1, 12))
    _rule("Creation to Assignment Interval", lambda s: s < 0)
    _rule("Number of Previous Attendance", lambda s: s < 0)
    _rule("Number of Previous Non-Attendance", lambda s: s < 0)

    removed = int(invalid_mask.sum())
    if removed:
        print(f"Removing {removed} invalid records based on domain rules.")

    df_filtered = df_filtered[~invalid_mask].copy()

    duplicate_count = int(df_filtered.duplicated().sum())
    if duplicate_count:
        print(f"Dropping {duplicate_count} duplicate rows.")
        df_filtered = df_filtered.drop_duplicates()

    return df_filtered


def remove_outliers(df: pd.DataFrame, upper_bound: int = MAX_CREATION_ASSIGNMENT_INTERVAL) -> pd.DataFrame:
    """Remove rows where the creation-to-assignment interval exceeds a bound."""

    if "Creation to Assignment Interval" not in df.columns:
        return df.copy()

    df_filtered = df.copy()
    before = len(df_filtered)
    df_filtered = df_filtered[df_filtered["Creation to Assignment Interval"] <= upper_bound]
    after = len(df_filtered)

    removed = before - after
    if removed:
        print(
            "Removed",
            removed,
            "records with Creation to Assignment Interval greater than",
            upper_bound,
        )

    return df_filtered


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    numeric: List[str]
    categorical: List[str]


def resolve_feature_sets(df: pd.DataFrame) -> FeatureConfig:
    """Pick the subset of candidate features that exist in the dataframe."""

    numeric = [col for col in NUMERIC_FEATURE_CANDIDATES if col in df.columns]
    categorical = [col for col in CATEGORICAL_FEATURE_CANDIDATES if col in df.columns]

    if not numeric and not categorical:
        raise ValueError("No candidate features found in dataframe after cleaning.")

    missing = set(NUMERIC_FEATURE_CANDIDATES + CATEGORICAL_FEATURE_CANDIDATES) - set(
        numeric + categorical
    )
    if missing:
        print(f"Skipping missing feature columns: {sorted(missing)}")

    return FeatureConfig(numeric=numeric, categorical=categorical)


def build_pipeline(config: FeatureConfig, model: Any | None = None) -> Pipeline:
    """Construct the sklearn pipeline with preprocessing and estimator."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, config.numeric),
            ("categorical", categorical_transformer, config.categorical),
        ],
        remainder="drop",
    )

    estimator = model or CatBoostClassifier(**CATBOOST_BEST_PARAMS)

    steps = [("preprocess", preprocessor)]
    if SMOTE_CONFIG.get("enabled", False):
        smote_params = {k: v for k, v in SMOTE_CONFIG.items() if k != "enabled"}
        steps.append(("smote", SMOTE(**smote_params)))
    steps.append(("model", estimator))

    return ImbPipeline(steps=steps)


def tune_threshold_f1_macro(y_true: pd.Series, proba: pd.Series) -> float:
    """Find threshold that maximizes macro F1 on a clean validation set."""

    thresholds = [i / 1000 for i in range(200, 801)]  # 0.200 .. 0.800
    best_th = 0.5
    best_score = -1.0

    y_true_arr = y_true.to_numpy()
    proba_arr = proba.to_numpy()
    for th in thresholds:
        preds = (proba_arr >= th).astype(int)
        score = f1_score(y_true_arr, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_th = float(th)

    return best_th


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train_and_serialize(model: Any | None = None, *, raw_data_path: Path | None = None) -> Dict[str, object]:
    """Train the pipeline and persist artifacts.

    Returns a dictionary with evaluation metrics.
    """

    df_raw = load_raw_data(raw_data_path)
    print(f"Loaded raw dataset with shape {df_raw.shape}.")

    df_norm = normalize_columns(df_raw)
    df_valid = filter_invalid_records(df_norm)
    df_ready = remove_outliers(df_valid)

    if TARGET_COLUMN not in df_ready.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found after cleaning.")

    config = resolve_feature_sets(df_ready)

    feature_columns = config.numeric + config.categorical
    X = df_ready[feature_columns].copy()
    y = df_ready[TARGET_COLUMN].astype(int)

    # 70 / 15 / 15 split (train / val / test) so we can tune a threshold
    stratify = y if y.nunique() > 1 else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        random_state=42,
        stratify=(y_tmp if y_tmp.nunique() > 1 else None),
    )

    pipeline = build_pipeline(config, model=model)
    pipeline.fit(X_train, y_train)

    # Threshold tuning on validation
    val_proba = pd.Series(pipeline.predict_proba(X_val)[:, 1])
    best_threshold = tune_threshold_f1_macro(y_val.reset_index(drop=True), val_proba)

    # Evaluate on test using tuned threshold
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_threshold).astype(int)

    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(y_test, test_proba))
    except Exception:
        roc_auc = None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_ARTIFACT_PATH)
    print(f"Serialized trained pipeline to {MODEL_ARTIFACT_PATH!s}.")

    CATBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, CATBOOST_MODEL_ARTIFACT_PATH)
    print(f"Serialized CatBoost production artifact to {CATBOOST_MODEL_ARTIFACT_PATH!s}.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "df_limpio.csv"
    df_ready.to_csv(processed_path, index=False)
    print(f"Saved cleaned dataset snapshot to {processed_path!s}.")

    metrics: Dict[str, Any] = {
        "model_version": "catboost",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "threshold": float(best_threshold),
        "feature_columns": feature_columns,
        "double_verification": DOUBLE_VERIFICATION_CONFIG,
        "target_distribution": y.value_counts(normalize=True).to_dict(),
        "classification_report": report,
        "model_params": {
            "catboost": dict(CATBOOST_BEST_PARAMS),
            "smote": dict(SMOTE_CONFIG),
        },
    }

    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, default=float)
    print(f"Stored metrics report at {METRICS_PATH!s}.")

    with CATBOOST_METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, default=float)
    print(f"Stored CatBoost metrics report at {CATBOOST_METRICS_PATH!s}.")

    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and serialize the no-show pipeline.")
    parser.add_argument(
        "--raw-data",
        dest="raw_data",
        default=None,
        help="Path to the raw CSV/Excel dataset (overrides data/raw/database_non-shows.csv)",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_data) if args.raw_data else None
    metrics = train_and_serialize(raw_data_path=raw_path)
    print("Training finished. Accuracy:", metrics["accuracy"])


if __name__ == "__main__":
    main()
