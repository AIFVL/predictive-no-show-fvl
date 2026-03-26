from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from . import preprocessing
from .models import make_logreg_baseline, make_preprocessor, MODEL_NAME_BASELINE


def train_baseline_model(*, out_dir: Path) -> Path:
    df = preprocessing.load_processed_dataset()
    X, y = preprocessing.split_features_target(df)

    pre = make_preprocessor(preprocessing.NUM_COLS, preprocessing.CAT_COLS)
    model = make_logreg_baseline(pre)
    model.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL_NAME_BASELINE}.joblib"
    joblib.dump(model, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model and save artifact")
    parser.add_argument(
        "--out-dir",
        default=str(preprocessing.PROJECT_ROOT / "outputs" / "saved_models"),
        help="Output directory for trained model artifact",
    )
    args = parser.parse_args()

    out_path = train_baseline_model(out_dir=Path(args.out_dir))
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
