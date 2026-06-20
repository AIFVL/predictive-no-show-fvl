from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Original features
DEFAULT_FEATURES_NUM = [
    "Age",
    "Number of Diseases",
    "Recent Hospitalization",
    "Number of Medications",
    "Hour",
    "Creation to Assignment Interval",
    "Number of Previous Attendance",
    "Number of Previous Non-Attendance",
]

DEFAULT_FEATURES_CAT = [
    "Sex",
    "Insurance Type",
    "Day",
    "Month",
]



def find_repo_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    return start


def load_processed_df(processed_csv: str | Path | None = None) -> pd.DataFrame:
    repo_root = find_repo_root()
    if processed_csv is None:
        processed_csv = repo_root / "data" / "processed" / "df_limpio.csv"
    else:
        processed_csv = Path(processed_csv)
        if not processed_csv.is_absolute():
            processed_csv = repo_root / processed_csv

    if not processed_csv.exists():
        raise FileNotFoundError(f"No se encontro dataset en: {processed_csv}")

    return pd.read_csv(processed_csv)


def infer_target(df: pd.DataFrame, target: str | None = None) -> str:
    if target and target in df.columns:
        return target

    for cand in ["Appointment Type", "appointment_type", "AppointmentType"]:
        if cand in df.columns:
            return cand

    for col in df.columns:
        values = set(df[col].dropna().unique())
        if values <= {0, 1} and 0.15 <= df[col].mean() <= 0.55:
            return col

    raise ValueError("No se pudo inferir la columna objetivo.")


def build_feature_frame(
    df: pd.DataFrame,
    features_num: List[str] | None,
    features_cat: List[str] | None,
    target: str | None,
    use_one_hot: bool = True,
    return_cat_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, str] | Tuple[pd.DataFrame, pd.Series, str, List[int]]:
    
    target_col = infer_target(df, target)

    num = [c for c in (features_num or DEFAULT_FEATURES_NUM) if c in df.columns]
    cat = [c for c in (features_cat or DEFAULT_FEATURES_CAT) if c in df.columns]

    X = df[num + cat].copy()
    y = df[target_col].copy().reset_index(drop=True)

    if cat:
        if use_one_hot:
            X = pd.get_dummies(X, columns=cat, drop_first=True)
        else:
            for c in cat:
                X[c] = X[c].astype(str)

    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)

    if return_cat_features:
        cat_features = [X.columns.get_loc(c) for c in cat if c in X.columns]
        return X, y, target_col, cat_features

    return X, y, target_col


def align_feature_columns(X: pd.DataFrame, feature_cols: List[str] | None) -> pd.DataFrame:
    if feature_cols is None:
        return X
    return X.reindex(columns=feature_cols, fill_value=0)


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, StandardScaler]:
    X_tr_raw, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    X_val_raw, X_te_raw, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state, stratify=y_tmp
    )

    scaler = StandardScaler()
    feature_cols = X.columns.tolist()

    X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=feature_cols).reset_index(drop=True)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=feature_cols).reset_index(drop=True)
    X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=feature_cols).reset_index(drop=True)

    y_tr = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)

    return X_tr, X_val, X_te, y_tr, y_val, y_te, scaler
