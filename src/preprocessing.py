from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Find repo root by looking for `requirements.txt` + `data/`.

    This keeps notebooks/scripts runnable from any cwd.
    """

    start = start or Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    return start


PROJECT_ROOT: Final[Path] = find_project_root()
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_PATH: Final[Path] = DATA_DIR / "raw" / "database_non-shows.xlsx"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
PROCESSED_PATH: Final[Path] = PROCESSED_DIR / "dataset_desbalanceado.csv"

TARGET_COL: Final[str] = "Appointment Type"

NUM_COLS: Final[list[str]] = [
    "Age",
    "Number of Diseases",
    "Recent Hospitalization",
    "Number of Medications",
    "Hour",
    "Creation to Assignment Interval",
    "Number of Previous Attendance",
    "Number of Previous Non-Attendance",
]

CAT_COLS: Final[list[str]] = ["Sex", "Insurance Type", "Day", "Month"]

REQUIRED_COLS: Final[list[str]] = [TARGET_COL, *NUM_COLS, *CAT_COLS]


def build_processed_dataset(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Excel and apply basic cleaning rules."""

    if not raw_path.exists():
        raise FileNotFoundError(f"No encuentro el Excel fuente en: {raw_path}")

    df_raw = pd.read_excel(raw_path)
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas en el Excel: " + ", ".join(missing))

    df_clean = df_raw[REQUIRED_COLS].copy().dropna()

    condiciones = {
        TARGET_COL: ~df_clean[TARGET_COL].isin([0, 1]),
        "Age": (df_clean["Age"] < 18) | (df_clean["Age"] > 120),
        "Sex": ~df_clean["Sex"].isin([0, 1, 2]),
        "Insurance Type": ~df_clean["Insurance Type"].isin(range(0, 9)),
        "Number of Diseases": (df_clean["Number of Diseases"] < 0),
        "Recent Hospitalization": (df_clean["Recent Hospitalization"] < 0),
        "Number of Medications": (df_clean["Number of Medications"] < 0),
        "Hour": (df_clean["Hour"] < 0) | (df_clean["Hour"] > 23),
        "Day": ~df_clean["Day"].between(0, 6),
        "Month": ~df_clean["Month"].between(1, 12),
        "Creation to Assignment Interval": (df_clean["Creation to Assignment Interval"] < 0),
        "Number of Previous Attendance": (df_clean["Number of Previous Attendance"] < 0),
        "Number of Previous Non-Attendance": (df_clean["Number of Previous Non-Attendance"] < 0),
    }

    mask_invalid = pd.Series(False, index=df_clean.index)
    for cond in condiciones.values():
        mask_invalid = mask_invalid | cond

    return df_clean.loc[~mask_invalid].copy()


def load_processed_dataset(
    processed_path: Path = PROCESSED_PATH,
    raw_path: Path = RAW_PATH,
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Read the processed CSV if present; otherwise build it from the raw Excel."""

    if processed_path.exists() and not force_rebuild:
        return pd.read_csv(processed_path)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean = build_processed_dataset(raw_path)
    df_clean.to_csv(processed_path, index=False)
    return df_clean


def split_features_target(df: pd.DataFrame):
    """Return X, y using the canonical column split."""

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL]
    return X, y
