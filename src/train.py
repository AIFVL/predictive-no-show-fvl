from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import importlib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import yaml
from joblib import dump
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer

from utils.data_loader import (
    load_processed_df,
    build_feature_frame,
    align_feature_columns,
    split_and_scale,
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: str | Path) -> dict:
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tune_threshold(
    probs,
    y_true,
    start,
    end,
    points,
    objective,
    min_accuracy,
    min_precision,
    min_recall,
):
    grid = [start + i * (end - start) / (points - 1) for i in range(points)]
    best_th = grid[0]
    best_score = -1.0
    fallback = (grid[0], -1.0)

    for th in grid:
        preds = (probs >= th).astype(int)
        acc = accuracy_score(y_true, preds)
        f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
        f1_no_show = f1_score(y_true, preds, average="binary", zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)

        if objective == "f1_no_show":
            score = f1_no_show
        elif objective == "f1_macro":
            score = f1_macro
        elif objective == "combo":
            score = 0.7 * f1_no_show + 0.3 * acc
        elif objective == "f1_recall":
            score = 0.6 * f1_no_show + 0.4 * recall
        else:
            score = f1_macro

        if (
            acc >= min_accuracy
            and precision >= min_precision
            and recall >= min_recall
            and score > best_score
        ):
            best_score = score
            best_th = th

        if score > fallback[1]:
            fallback = (th, score)

    if best_score < 0:
        return fallback[0], fallback[1]

    return best_th, best_score


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


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    model_cfg = load_yaml(cfg["model"]["config_path"])
    module_path = cfg["model"].get("module", "models.lightgbm.model")
    model_module = importlib.import_module(module_path)
    build_pipeline = getattr(model_module, "build_pipeline")

    df = load_processed_df(cfg["data"]["processed_csv"])
    use_one_hot = cfg["data"].get("use_one_hot", True)
    return_cat_features = not use_one_hot
    if return_cat_features:
        X, y, target_col, cat_features = build_feature_frame(
            df,
            cfg["data"].get("features_num"),
            cfg["data"].get("features_cat"),
            cfg["data"].get("target"),
            use_one_hot=use_one_hot,
            return_cat_features=True,
        )
    else:
        X, y, target_col = build_feature_frame(
            df,
            cfg["data"].get("features_num"),
            cfg["data"].get("features_cat"),
            cfg["data"].get("target"),
            use_one_hot=use_one_hot,
        )
        cat_features = None

    X = align_feature_columns(X, None)

    scale_features = cfg["data"].get("scale_features", True)
    if scale_features:
        X_tr, X_val, X_te, y_tr, y_val, y_te, scaler = split_and_scale(
            X,
            y,
            cfg["split"]["test_size"],
            cfg["split"]["val_size"],
            cfg["split"]["random_state"],
        )
    else:
        X_tr_raw, X_tmp, y_tr, y_tmp = train_test_split(
            X,
            y,
            test_size=cfg["split"]["test_size"] + cfg["split"]["val_size"],
            random_state=cfg["split"]["random_state"],
            stratify=y,
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp,
            y_tmp,
            test_size=0.5,
            random_state=cfg["split"]["random_state"],
            stratify=y_tmp,
        )
        X_tr = X_tr_raw.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_te = X_te.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)
        scaler = None

    smote_cfg = model_cfg.get("smote", {})
    if smote_cfg.get("enabled", False):
        print(
            f"\n SMOTE habilitado (via pipeline): "
            f"{len(X_tr)} registros train, {(y_tr == 1).sum()} No-Show\n"
        )

    pipeline = build_pipeline(model_cfg)

    search_cfg = cfg.get("search", {})
    fit_params = {}
    if cat_features is not None:
        fit_params = {"clf__cat_features": cat_features}

    if search_cfg.get("enabled", False):
        scorer = make_scorer(f1_score, average="binary", zero_division=0)
        search = RandomizedSearchCV(
            pipeline,
            search_cfg.get("params", {}),
            n_iter=search_cfg.get("n_iter", 30),
            cv=search_cfg.get("cv", 5),
            scoring=scorer,
            n_jobs=-1,
            random_state=cfg["split"]["random_state"],
            verbose=1,
        )
        search.fit(X_tr, y_tr, **fit_params)
        pipeline = search.best_estimator_
        print("Mejores params (f1_no_show CV):", search.best_params_)
        print("Mejor score CV (f1_no_show):", round(search.best_score_, 4))
    else:
        pipeline.fit(X_tr, y_tr, **fit_params)

    probs_val = pipeline.predict_proba(X_val)[:, 1]
    th, score_val = tune_threshold(
        probs_val,
        y_val,
        cfg["threshold"]["grid_start"],
        cfg["threshold"]["grid_end"],
        cfg["threshold"]["grid_points"],
        cfg["threshold"].get("objective", "f1_macro"),
        cfg["threshold"].get("min_accuracy", 0.0),
        cfg["threshold"].get("min_precision", 0.0),
        cfg["threshold"].get("min_recall", 0.0),
    )

    preds_val = (probs_val >= th).astype(int)
    metrics_val = compute_metrics(y_val, preds_val, probs_val)

    probs_test = pipeline.predict_proba(X_te)[:, 1]
    preds_test = (probs_test >= th).astype(int)
    metrics_test = compute_metrics(y_te, preds_test, probs_test)

    output_model = resolve_path(cfg["output"]["model_path"])
    output_model.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": pipeline,
            "scaler": scaler,
            "feature_cols": X.columns.tolist(),
            "threshold": th,
            "target": target_col,
        },
        output_model,
    )

    metrics_path = resolve_path(cfg["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": th,
                "score_val": score_val,
                "val": metrics_val,
                "test": metrics_test,
            },
            f,
            indent=2,
        )

    print("Modelo entrenado y guardado en:", output_model)
    print("Metricas guardadas en:", metrics_path)
    print("Threshold:", round(th, 4))
    print("Score (val):", round(score_val, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar LightGBM + SMOTE")
    parser.add_argument(
        "--config",
        default="configs/training.yml",
        help="Ruta al archivo training.yml",
    )
    args = parser.parse_args()
    main(args.config)
