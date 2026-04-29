"""LightGBM model for no-show prediction."""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb


def build_pipeline(config: dict) -> Pipeline:
    smote_cfg = config.get("smote", {})
    lgb_cfg = config.get("lightgbm", {})

    steps = []
    if smote_cfg.get("enabled", False):
        smote_params = {k: v for k, v in smote_cfg.items() if k != "enabled"}
        steps.append(("smote", SMOTE(**smote_params)))

    steps.append(("clf", lgb.LGBMClassifier(**lgb_cfg)))
    return Pipeline(steps)
