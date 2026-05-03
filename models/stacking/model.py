from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def build_pipeline(config: dict) -> Pipeline:
    smote_cfg = config.get("smote", {})
    stack_cfg = config.get("stack", {})

    lgb_cfg = config.get("lightgbm", {})
    xgb_cfg = config.get("xgboost", {})
    cat_cfg = config.get("catboost", {})
    meta_cfg = config.get("meta", {})

    use_smote = smote_cfg.get("enabled", False)
    steps = []
    if use_smote:
        smote_params = {k: v for k, v in smote_cfg.items() if k != "enabled"}
        steps.append(("smote", SMOTE(**smote_params)))

    estimators = [
        ("xgb", XGBClassifier(**xgb_cfg)),
        ("lgb", lgb.LGBMClassifier(**lgb_cfg)),
        ("cat", CatBoostClassifier(**cat_cfg)),
    ]

    final_estimator = LogisticRegression(**meta_cfg)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=stack_cfg.get("cv", 5),
        stack_method=stack_cfg.get("stack_method", "predict_proba"),
        passthrough=stack_cfg.get("passthrough", False),
        n_jobs=stack_cfg.get("n_jobs", -1),
    )

    steps.append(("clf", stack))
    return Pipeline(steps)
