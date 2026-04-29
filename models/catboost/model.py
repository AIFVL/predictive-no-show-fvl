from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier


def build_pipeline(config: dict) -> Pipeline:
    cat_cfg = config.get("catboost", {})
    clf = CatBoostClassifier(**cat_cfg)

    return Pipeline([
        ("clf", clf),
    ])
