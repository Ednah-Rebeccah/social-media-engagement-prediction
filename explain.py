"""
Explaining predictions (paper, Section 7.4 to 7.6).

Two levels of explanation:

* GLOBAL: which inputs matter most across all test posts. Uses
  permutation importance from scikit-learn: shuffle one input and see
  how much worse the forecasts get. Works for any model, no extra
  installs needed.

* LOCAL: why one particular planned post received its forecast.
    - For the linear model, contributions are computed exactly from the
      coefficients (they add up to the prediction on the log scale).
    - For tree models, SHAP values are used if the optional `shap`
      package is installed (pip install shap).

Remember: explanations describe what the MODEL learned from history.
They are not proof that changing a feature will CAUSE engagement to
change. Test that deliberately (paper, Section 10.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Plain-language names for reports.
FRIENDLY = {
    "platform": "Platform",
    "account": "Account",
    "content_format": "Format",
    "topic": "Topic",
    "paid": "Paid or organic",
    "day_of_week": "Day of week",
    "log_followers": "Audience size",
    "hour": "Posting hour",
    "is_weekend": "Weekend",
    "caption_length": "Caption length",
    "hashtag_count": "Hashtags",
    "mention_count": "Mentions",
    "has_question": "Asks a question",
    "has_call_to_action": "Call to action",
    "has_link": "Contains a link",
    "prior_median_log_engagement": "Account's recent typical engagement",
    "has_history": "Account has history",
}


def global_importance(model, X_test: pd.DataFrame, y_test: pd.Series,
                      n_repeats: int = 10) -> pd.DataFrame:
    """Increase in MAE when each input is shuffled. Bigger = more important."""
    result = permutation_importance(
        model, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=n_repeats, random_state=config.RANDOM_STATE,
    )
    out = pd.DataFrame({
        "feature": X_test.columns,
        "label": [FRIENDLY.get(c, c) for c in X_test.columns],
        "MAE_increase_when_shuffled": result.importances_mean,
        "std": result.importances_std,
    })
    return out.sort_values("MAE_increase_when_shuffled", ascending=False).reset_index(drop=True)


def _origin_map(prep) -> list[str]:
    """Map every transformed column back to the original input it came from."""
    cats = prep.named_transformers_["cat"].categories_
    origin = []
    for col, values in zip(CATEGORICAL_FEATURES, cats):
        origin.extend([col] * len(values))
    origin.extend(NUMERIC_FEATURES)
    if len(origin) != len(prep.get_feature_names_out()):
        raise RuntimeError("Could not map transformed features back to inputs.")
    return origin


def _aggregate(contrib: np.ndarray, origin: list[str], index) -> pd.DataFrame:
    """Sum one-hot columns back into one contribution per original input."""
    frame = pd.DataFrame(contrib, columns=origin, index=index)
    return frame.T.groupby(level=0, sort=False).sum().T


def linear_contributions(model, X_train: pd.DataFrame, X: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Exact additive contributions for the ridge model on the log scale.

    prediction_log = base + sum(contributions), where base is the
    prediction for an 'average' training post.
    """
    pipe = model.regressor_
    prep, ridge = pipe.named_steps["prep"], pipe.named_steps["model"]
    z_train = prep.transform(X_train)
    z_mean = z_train.mean(axis=0)
    z = prep.transform(X)
    contrib = (z - z_mean) * ridge.coef_
    base = float(ridge.intercept_ + z_mean @ ridge.coef_)
    return base, _aggregate(contrib, _origin_map(prep), X.index)


def shap_contributions(model, X_train: pd.DataFrame, X: pd.DataFrame):
    """SHAP values for tree models, if `shap` is installed. Returns None otherwise."""
    try:
        import shap  # optional dependency
    except ImportError:
        return None
    try:
        pipe = model.regressor_
        prep, tree = pipe.named_steps["prep"], pipe.named_steps["model"]
        explainer = shap.TreeExplainer(tree)
        values = explainer.shap_values(prep.transform(X))
        base = float(np.ravel(explainer.expected_value)[0])
        return base, _aggregate(np.asarray(values), _origin_map(prep), X.index)
    except Exception as exc:  # unsupported model or shap version
        print(f"SHAP explanation unavailable ({exc}); using linear explanation instead.")
        return None


def describe_value(feature: str, row: pd.Series) -> str:
    value = row[feature]
    if feature == "log_followers":
        return f"{10 ** value:,.0f} followers"
    if feature == "prior_median_log_engagement":
        return f"about {np.expm1(value):,.0f} engagements" if row["has_history"] else "no history yet"
    if feature in ("is_weekend", "has_question", "has_call_to_action", "has_link", "has_history"):
        return "yes" if value else "no"
    return str(value)


def explain_posts(contrib: pd.DataFrame, X: pd.DataFrame, top_k: int = 4) -> list[str]:
    """
    Turn contributions into one plain-language line per post, e.g.
    'Topic = grant_call (+21%); Format = carousel (+12%); Posting hour = 21 (-8%)'.
    Percentages are the approximate multiplicative effect on (1 + engagement).
    """
    lines = []
    for idx in X.index:
        c = contrib.loc[idx]
        top = c.reindex(c.abs().sort_values(ascending=False).index)[:top_k]
        parts = [
            f"{FRIENDLY.get(f, f)} = {describe_value(f, X.loc[idx])} "
            f"({np.expm1(v) * 100:+.0f}%)"
            for f, v in top.items()
        ]
        lines.append("; ".join(parts))
    return lines
