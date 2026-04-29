from __future__ import annotations

import argparse
from pathlib import Path
import sys

import json
import yaml
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.data_loader import load_processed_df, build_feature_frame, align_feature_columns


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    pc, rc, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rc, pc)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_no_show": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision_no_show": precision_score(y_true, y_pred, zero_division=0),
        "recall_no_show": recall_score(y_true, y_pred, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": pr_auc,
    }


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def main(model_path: str, config_path: str, output_path: str | None) -> None:
    config_path = resolve_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_artifact = load(resolve_path(model_path))
    pipeline = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_cols = model_artifact["feature_cols"]
    threshold = model_artifact["threshold"]

    df = load_processed_df(cfg["data"]["processed_csv"])
    use_one_hot = cfg["data"].get("use_one_hot", True)
    if not use_one_hot:
        X, y, _target, _cat_features = build_feature_frame(
            df,
            cfg["data"].get("features_num"),
            cfg["data"].get("features_cat"),
            cfg["data"].get("target"),
            use_one_hot=use_one_hot,
            return_cat_features=True,
        )
    else:
        X, y, _target = build_feature_frame(
            df,
            cfg["data"].get("features_num"),
            cfg["data"].get("features_cat"),
            cfg["data"].get("target"),
            use_one_hot=use_one_hot,
        )
    X = align_feature_columns(X, feature_cols)

    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    random_state = cfg["split"]["random_state"]

    X_tr_raw, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    X_val_raw, X_te_raw, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state, stratify=y_tmp
    )

    if scaler is not None:
        X_te_eval = scaler.transform(X_te_raw)
    else:
        X_te_eval = X_te_raw

    probs_test = pipeline.predict_proba(X_te_eval)[:, 1]
    preds_test = (probs_test >= threshold).astype(int)

    metrics_test = compute_metrics(y_te, preds_test, probs_test)

    if output_path:
        output_path = resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_test, f, indent=2)

    print("Metricas de validacion (test):")
    for k, v in metrics_test.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validar modelo LightGBM")
    parser.add_argument("--model", default="outputs/lightgbm_smote.joblib")
    parser.add_argument("--config", default="configs/training.yml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    main(args.model, args.config, args.output)
