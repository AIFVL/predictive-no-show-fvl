"""Reusable training script for the appointment attendance model.

This module mirrors the preprocessing and modeling steps developed in
notebooks/02_Limpieza.ipynb. It loads the raw dataset, applies the
normalization and cleaning rules, trains a sklearn pipeline, reports
metrics, and serializes the fitted pipeline for inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

import lightgbm as lgb


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "database_non-shows.xlsx"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACT_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "Appointment Type"
MAX_CREATION_ASSIGNMENT_INTERVAL = 365

# Best-performing config taken from notebooks/05_Stacking_SMOTE_Correcto.ipynb
LGBM_BEST_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "subsample": 0.8,
    "reg_lambda": 0.5,
    "reg_alpha": 0.3,
    "num_leaves": 63,
    "n_estimators": 600,
    "min_child_samples": 10,
    "max_depth": 5,
    "learning_rate": 0.08,
    "colsample_bytree": 0.7,
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
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw appointment dataset from Excel."""

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path!s}")
    return pd.read_excel(path)


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

    estimator = model or lgb.LGBMClassifier(**LGBM_BEST_PARAMS)

    # Important: SMOTE must only run during fitting on the training split.
    # Using imblearn's Pipeline ensures the sampler is applied during fit,
    # and is effectively a no-op during inference.
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("model", estimator),
        ]
    )


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

def train_and_serialize(model: Any | None = None) -> Dict[str, object]:
    """Train the pipeline and persist artifacts.

    Returns a dictionary with evaluation metrics.
    """

    df_raw = load_raw_data()
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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "df_limpio.csv"
    df_ready.to_csv(processed_path, index=False)
    print(f"Saved cleaned dataset snapshot to {processed_path!s}.")

    metrics: Dict[str, Any] = {
        "model_version": "lightgbm_smote",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "threshold": float(best_threshold),
        "feature_columns": feature_columns,
        "target_distribution": y.value_counts(normalize=True).to_dict(),
        "classification_report": report,
        "model_params": {
            "lightgbm": dict(LGBM_BEST_PARAMS),
            "smote": {"random_state": 42, "k_neighbors": 5},
        },
    }

    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, default=float)
    print(f"Stored metrics report at {METRICS_PATH!s}.")

    return metrics


def main() -> None:
    metrics = train_and_serialize()
    print("Training finished. Accuracy:", metrics["accuracy"])


if __name__ == "__main__":
    main()
