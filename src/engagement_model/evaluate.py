"""
Evaluation (paper, Section 8).

Every score here is computed on posts the model did NOT see during
training. Error analysis by group shows where the model is reliable
and where it is not, which often matters more than the headline score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import all_models, is_baseline, model_name
from .split import rolling_origin_splits


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R², MAPE and median absolute error, on the engagement scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "MAPE_%": mape,  # excludes posts with zero actual engagement
        "MedianAE": float(np.median(np.abs(err))),
        "n": int(len(y_true)),
    }


def evaluate_holdout(
    X: pd.DataFrame, y: pd.Series, train_idx, test_idx
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Fit every baseline and model on the training posts, score all of them
    on the held-out test posts.

    Returns:
      results  - one row of metrics per model, best MAE first
      fitted   - {model name: fitted model}
      preds    - test-set actuals and each model's predictions
    """
    rows, fitted = [], {}
    preds = pd.DataFrame({"actual": y.loc[test_idx]}, index=test_idx)
    for model in all_models():
        model.fit(X.loc[train_idx], y.loc[train_idx])
        p = np.clip(model.predict(X.loc[test_idx]), 0, None)
        name = model_name(model)
        fitted[name] = model
        preds[name] = p
        rows.append({"model": name, "baseline": is_baseline(model),
                     **regression_metrics(y.loc[test_idx], p)})

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_baseline_mae = results.loc[results["baseline"], "MAE"].min()
    results["MAE_vs_best_baseline_%"] = (
        (best_baseline_mae - results["MAE"]) / best_baseline_mae * 100
    ).round(1)
    return results, fitted, preds


def evaluate_rolling(X: pd.DataFrame, y: pd.Series, posts: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling-origin evaluation: one round per recent month (paper, 8.4).
    The spread of scores across rounds shows whether accuracy is stable.
    """
    rows = []
    for month, train_idx, test_idx in rolling_origin_splits(posts):
        for model in all_models():
            model.fit(X.loc[train_idx], y.loc[train_idx])
            p = np.clip(model.predict(X.loc[test_idx]), 0, None)
            m = regression_metrics(y.loc[test_idx], p)
            rows.append({"test_month": month, "model": model_name(model),
                         "MAE": m["MAE"], "n_test": m["n"]})
    return pd.DataFrame(rows)


def error_by_group(
    preds: pd.DataFrame, posts: pd.DataFrame, model: str,
    groups=("platform", "content_format", "topic"),
) -> pd.DataFrame:
    """Where does the model fail? MAE and bias for each group (paper, 9.4 step 9)."""
    frame = preds[["actual", model]].join(posts[list(groups)])
    frame["abs_error"] = (frame["actual"] - frame[model]).abs()
    frame["error"] = frame[model] - frame["actual"]  # + means over-predicted
    out = []
    for g in groups:
        s = frame.groupby(g).agg(
            n=("actual", "size"),
            MAE=("abs_error", "mean"),
            mean_bias=("error", "mean"),
            actual_median=("actual", "median"),
        ).reset_index().rename(columns={g: "value"})
        s.insert(0, "group", g)
        out.append(s)
    return pd.concat(out, ignore_index=True).round(1)


def residual_band(preds: pd.DataFrame, model: str, coverage: float = 0.8) -> tuple[float, float]:
    """
    Empirical range on the log scale from test-set errors. Used to give
    planned posts a "likely range" instead of a falsely precise number.
    """
    log_err = np.log1p(preds["actual"]) - np.log1p(preds[model])
    lo, hi = np.quantile(log_err, [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
    return float(lo), float(hi)
