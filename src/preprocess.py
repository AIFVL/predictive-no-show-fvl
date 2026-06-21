from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from utils.data_loader import find_repo_root


NUM_COLS = [
    "Age",
    "Number of Diseases",
    "Recent Hospitalization",
    "Number of Medications",
    "Hour",
    "Creation to Assignment Interval",
    "Number of Previous Attendance",
    "Number of Previous Non-Attendance",
]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Appointment Type": "Appointment Type",
        "Appointment_Type": "Appointment Type",
        "AppointmentType": "Appointment Type",
        "Number of diseases": "Number of Diseases",
        "Number of diseases ": "Number of Diseases",
    }

    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def load_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato no soportado para dataset raw: {path.suffix}")


def main(input_path: str | None = None, output_path: str | None = None) -> None:
    repo_root = find_repo_root()
    raw_path = resolve_path(
        input_path or "data/raw/database_non-shows.csv",
        repo_root,
    )
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset en: {raw_path}")

    df = load_raw(raw_path)
    df = clean_raw(df)

    output = resolve_path(output_path or "data/processed/df_limpio.csv", repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, index=False)
    print(f"Dataset limpio guardado en: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar dataset raw a data/processed/df_limpio.csv")
    parser.add_argument(
        "--input",
        default="data/raw/database_non-shows.csv",
        help="Ruta del dataset raw (.csv, .xlsx o .xls)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/df_limpio.csv",
        help="Ruta del CSV procesado",
    )
    args = parser.parse_args()
    main(args.input, args.output)
