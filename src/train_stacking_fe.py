import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "imblearn no esta instalado. Instala con: pip install imbalanced-learn"
    ) from exc

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("xgboost no esta instalado. Instala con: pip install xgboost") from exc

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("lightgbm no esta instalado. Instala con: pip install lightgbm") from exc

from feature_engineering import add_feature_engineering

warnings.filterwarnings("ignore")
np.random.seed(42)


def load_dataset() -> pd.DataFrame:
    possible_paths = [
        Path("data/processed/df_limpio.csv"),
        Path("notebooks/data/processed/df_limpio.csv"),
        Path("../data/processed/df_limpio.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            df = pd.read_csv(p)
            print(f"OK dataset cargado: {p.resolve()} -> {df.shape}")
            return df
    raise FileNotFoundError("No se encontro df_limpio.csv")


def find_target(df: pd.DataFrame) -> str:
    for cand in ["Appointment Type", "appointment_type", "AppointmentType"]:
        if cand in df.columns:
            return cand
    # heuristica: binaria con prevalencia razonable
    for col in df.columns:
        vals = set(df[col].dropna().unique())
        if vals.issubset({0, 1}):
            rate = df[col].mean()
            if 0.15 <= rate <= 0.55:
                return col
    raise ValueError("No se detecto el target")


def prepare_features(df: pd.DataFrame, target: str):
    df_fe = add_feature_engineering(df)
    df_fe = df_fe.drop(columns=[c for c in df_fe.columns if "Unnamed" in c], errors="ignore")
    # LightGBM warning: remove whitespace in feature names
    df_fe.columns = [str(c).replace(" ", "_") for c in df_fe.columns]

    target_norm = target.replace(" ", "_")
    if target not in df_fe.columns and target_norm in df_fe.columns:
        target = target_norm

    y = df_fe[target].copy()
    X = df_fe.drop(columns=[target])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.fillna(X.median(numeric_only=True))
    return X, y, num_cols, cat_cols


def split_data(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def evaluate_with_threshold(name, model, X_val, y_val, X_te, y_te):
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_te = model.predict_proba(X_te)[:, 1]

    grid = np.linspace(0.20, 0.80, 300)
    f1g = [f1_score(y_val, (proba_val >= th).astype(int), average="macro") for th in grid]
    best_th = float(grid[int(np.argmax(f1g))])

    y_pred = (proba_te >= best_th).astype(int)

    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    f1w = f1_score(y_te, y_pred, average="weighted")
    f1c1 = f1_score(y_te, y_pred, average="binary")
    kappa = cohen_kappa_score(y_te, y_pred)
    bal = balanced_accuracy_score(y_te, y_pred)
    roc = roc_auc_score(y_te, proba_te)

    pc, rc, _ = precision_recall_curve(y_te, proba_te)
    prauc = np.trapz(pc, rc)

    print("\n" + "=" * 72)
    print(f"Modelo: {name}")
    print("=" * 72)
    print(f"Threshold (val F1 macro): {best_th:.4f}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal:.4f}")
    print(f"F1 Macro          : {f1m:.4f}")
    print(f"F1 Weighted       : {f1w:.4f}")
    print(f"F1 No-Show        : {f1c1:.4f}")
    print(f"Cohen Kappa       : {kappa:.4f}")
    print(f"ROC-AUC           : {roc:.4f}")
    print(f"PR-AUC            : {prauc:.4f}")
    print(classification_report(y_te, y_pred, digits=4))

    return {
        "Modelo": name,
        "Threshold": best_th,
        "Accuracy": acc,
        "Balanced Acc": bal,
        "F1 Macro": f1m,
        "F1 Weighted": f1w,
        "F1 No-Show": f1c1,
        "ROC-AUC": roc,
        "PR-AUC": prauc,
        "Cohen Kappa": kappa,
    }


def evaluate_split(name, model, X, y, threshold):
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Balanced Acc": balanced_accuracy_score(y, y_pred),
        "F1 Macro": f1_score(y, y_pred, average="macro"),
        "F1 Weighted": f1_score(y, y_pred, average="weighted"),
        "F1 No-Show": f1_score(y, y_pred, average="binary"),
        "Cohen Kappa": cohen_kappa_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def overfitting_report(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te, threshold):
    tr = evaluate_split(name, model, X_tr, y_tr, threshold)
    va = evaluate_split(name, model, X_val, y_val, threshold)
    te = evaluate_split(name, model, X_te, y_te, threshold)
    print("\n" + "-" * 72)
    print(f"Overfitting check: {name}")
    print("-" * 72)
    print(f"Train F1 Macro: {tr['F1 Macro']:.4f}")
    print(f"Val   F1 Macro: {va['F1 Macro']:.4f}")
    print(f"Test  F1 Macro: {te['F1 Macro']:.4f}")
    print(f"Gap Train-Test: {(tr['F1 Macro'] - te['F1 Macro']):.4f}")


def main():
    df = load_dataset()
    target = find_target(df)
    print(f"Target: {target}")

    X, y, _, _ = prepare_features(df, target)
    print(f"Features finales: {X.shape}")

    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
    print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- XGBoost ----
    xgb_pipe = ImbPipeline(
        [
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb_params = {
        "clf__n_estimators": [600, 800, 1000],
        "clf__max_depth": [4, 5, 6],
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__subsample": [0.8, 0.9],
        "clf__colsample_bytree": [0.7, 0.8, 0.9],
        "clf__gamma": [0, 0.1, 0.2],
        "clf__min_child_weight": [1, 3],
        "clf__reg_lambda": [1, 2, 3],
        "clf__reg_alpha": [0.0, 0.1, 0.3],
    }

    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        xgb_params,
        n_iter=35,
        cv=cv5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    xgb_search.fit(X_tr, y_tr)
    xgb_best = xgb_search.best_estimator_
    print(f"XGB CV F1 macro: {xgb_search.best_score_:.4f}")

    # ---- LightGBM ----
    lgb_pipe = ImbPipeline(
        [
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            (
                "clf",
                lgb.LGBMClassifier(
                    objective="binary",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    min_gain_to_split=0.0,
                    force_col_wise=True,
                ),
            ),
        ]
    )

    lgb_params = {
        "clf__n_estimators": [500, 800, 1000],
        "clf__max_depth": [-1, 5, 7, 9],
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__num_leaves": [31, 50, 63, 80],
        "clf__subsample": [0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9],
        "clf__min_child_samples": [20, 40, 60],
        "clf__min_gain_to_split": [0.0, 0.01, 0.05],
        "clf__reg_lambda": [0, 0.5, 1.0, 2.0],
        "clf__reg_alpha": [0, 0.1, 0.3],
    }

    lgb_search = RandomizedSearchCV(
        lgb_pipe,
        lgb_params,
        n_iter=35,
        cv=cv5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    lgb_search.fit(X_tr, y_tr)
    lgb_best = lgb_search.best_estimator_
    print(f"LGBM CV F1 macro: {lgb_search.best_score_:.4f}")

    # ---- HistGradientBoosting ----
    hgb_pipe = ImbPipeline(
        [
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    hgb_params = {
        "clf__max_iter": [200, 400, 600],
        "clf__max_depth": [None, 5, 7, 9],
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__max_leaf_nodes": [31, 50, 63, 80],
        "clf__min_samples_leaf": [10, 20, 30, 50],
        "clf__l2_regularization": [0, 0.1, 0.5, 1.0],
        "clf__max_features": [0.7, 0.8, 1.0],
    }

    hgb_search = RandomizedSearchCV(
        hgb_pipe,
        hgb_params,
        n_iter=30,
        cv=cv5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    hgb_search.fit(X_tr, y_tr)
    hgb_best = hgb_search.best_estimator_
    print(f"HGB CV F1 macro: {hgb_search.best_score_:.4f}")

    # ---- Evaluacion individual ----
    results = []
    res_xgb = evaluate_with_threshold("XGBoost + FE", xgb_best, X_val, y_val, X_te, y_te)
    overfitting_report(
        "XGBoost + FE",
        xgb_best,
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        res_xgb["Threshold"],
    )
    results.append(res_xgb)

    res_lgb = evaluate_with_threshold("LightGBM + FE", lgb_best, X_val, y_val, X_te, y_te)
    overfitting_report(
        "LightGBM + FE",
        lgb_best,
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        res_lgb["Threshold"],
    )
    results.append(res_lgb)

    res_hgb = evaluate_with_threshold("HistGB + FE", hgb_best, X_val, y_val, X_te, y_te)
    overfitting_report(
        "HistGB + FE",
        hgb_best,
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        res_hgb["Threshold"],
    )
    results.append(res_hgb)

    # ---- Stacking ----
    print("\nConstruyendo Stacking...")
    xgb_clf = xgb_best.named_steps["clf"]
    lgb_clf = lgb_best.named_steps["clf"]
    hgb_clf = hgb_best.named_steps["clf"]

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    final_est = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    stack = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("hgb", hgb_clf)],
        final_estimator=final_est,
        cv=5,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )

    stack.fit(X_tr_sm, y_tr_sm)
    res_stack = evaluate_with_threshold(
        "Stacking (XGB+LGB+HGB) + FE", stack, X_val, y_val, X_te, y_te
    )
    overfitting_report(
        "Stacking (XGB+LGB+HGB) + FE",
        stack,
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        res_stack["Threshold"],
    )
    results.append(res_stack)

    # ---- Resumen ----
    df_res = pd.DataFrame(results).sort_values("F1 Macro", ascending=False).reset_index(drop=True)
    print("\nRanking final (top 5):")
    print(df_res.head(5))

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out / "resultados_stacking_fe.csv", index=False)
    print(f"Resultados guardados en: {(out / 'resultados_stacking_fe.csv').resolve()}")


if __name__ == "__main__":
    main()
