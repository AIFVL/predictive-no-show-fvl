from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def build_pipeline(config: dict) -> Pipeline:
    smote_cfg = config.get("smote", {})
    xgb_cfg = config.get("xgboost", {})
    use_smote = smote_cfg.get("enabled", False)

    steps = []
    if use_smote:
        smote_params = {k: v for k, v in smote_cfg.items() if k != "enabled"}
        steps.append(("smote", SMOTE(**smote_params)))

    steps.append(("clf", XGBClassifier(**xgb_cfg)))

    return Pipeline(steps)
