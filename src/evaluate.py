from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import classification_report

from . import preprocessing


def evaluate_model(*, model_path: Path) -> str:
    df = preprocessing.load_processed_dataset()
    X, y = preprocessing.split_features_target(df)

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    return classification_report(y, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "model_path",
        help="Path to a saved model .joblib (e.g., outputs/saved_models/logreg_baseline.joblib)",
    )
    args = parser.parse_args()

    report = evaluate_model(model_path=Path(args.model_path))
    print(report)


if __name__ == "__main__":
    main()
