from __future__ import annotations

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


def main() -> None:
    repo_root = find_repo_root()
    raw_path = repo_root / "data" / "raw" / "database_non-shows.xlsx"
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset en: {raw_path}")

    df = pd.read_excel(raw_path)
    df = clean_raw(df)

    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "df_limpio.csv"

    df.to_csv(output_path, index=False)
    print(f"Dataset limpio guardado en: {output_path}")


if __name__ == "__main__":
    main()
