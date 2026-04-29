"""CatBoost model for no-show prediction."""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


def build_pipeline(config: dict) -> Pipeline:
    smote_cfg = config.get("smote", {})
    cat_cfg = config.get("catboost", {})

    steps = []
    if smote_cfg.get("enabled", False):
        smote_params = {k: v for k, v in smote_cfg.items() if k != "enabled"}
        steps.append(("smote", SMOTE(**smote_params)))

    steps.append(("clf", CatBoostClassifier(**cat_cfg)))
    return Pipeline(steps)
