"""
Baselines and models (paper, Sections 5.3, 7 and 8.3).

Every model must beat the baselines on unseen posts to earn a place in
planning. All learned models predict log(1 + engagement), which keeps a
few viral posts from dominating, and convert back to engagement counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# ---------------------------------------------------------------------
# Baselines: the "plain cup" every model has to beat
# ---------------------------------------------------------------------


class HistoricalMean:
    """Predict the training-period mean for every post."""

    name = "Baseline: historical mean"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMean":
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_)


class HistoricalMedian(HistoricalMean):
    """Predict the training-period median: robust to viral outliers."""

    name = "Baseline: historical median"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMedian":
        self.value_ = float(np.median(y))
        return self


class RecentAccountMedian:
    """
    Predict each account's median engagement over its recent completed
    posts ("same as our usual"). Falls back to the overall training
    median for accounts with no history yet.
    """

    name = "Baseline: account's recent median"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecentAccountMedian":
        self.fallback_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        recent = np.expm1(X["prior_median_log_engagement"].to_numpy(dtype=float))
        return np.where(X["has_history"].to_numpy() == 1, recent, self.fallback_)


# ---------------------------------------------------------------------
# Learned models
# ---------------------------------------------------------------------


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", numeric, NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _log_target(regressor: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )


def linear_regression() -> TransformedTargetRegressor:
    """Ridge regression: transparent, stable on small datasets."""
    model = _log_target(
        Pipeline([("prep", _preprocessor(scale_numeric=True)), ("model", Ridge(alpha=1.0))])
    )
    model.name = "Linear regression (ridge)"
    return model


def random_forest() -> TransformedTargetRegressor:
    model = _log_target(
        Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        max_features=0.5,
                        n_jobs=-1,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        )
    )
    model.name = "Random forest"
    return model


def gradient_boosting() -> TransformedTargetRegressor:
    model = _log_target(
        Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        )
    )
    model.name = "Gradient boosting"
    return model


def all_models() -> list:
    """Baselines first, then models in increasing complexity."""
    return [
        HistoricalMean(),
        HistoricalMedian(),
        RecentAccountMedian(),
        linear_regression(),
        random_forest(),
        gradient_boosting(),
    ]


LEARNED_MODEL_FACTORIES = {
    "linear": linear_regression,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
}


def model_name(model) -> str:
    return getattr(model, "name", type(model).__name__)


def is_baseline(model) -> bool:
    return model_name(model).startswith("Baseline")
