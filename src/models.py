from __future__ import annotations

from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", "passthrough", cat_cols),
        ]
    )


def make_logreg_baseline(preprocessor: ColumnTransformer) -> Pipeline:
    """Simple baseline pipeline.

    Note: categorical columns are assumed to be already numeric-coded.
    """

    clf = LogisticRegression(max_iter=2000)
    return Pipeline([("pre", preprocessor), ("clf", clf)])


MODEL_NAME_BASELINE: Final[str] = "logreg_baseline"
