"""
The full study, end to end (paper, Section 9.4).

  1. Load and clean the data, recording every change.
  2. Build leakage-checked, pre-publication features.
  3. Chronological 80/20 holdout: baselines vs models.
  4. Rolling-origin evaluation, one round per recent month.
  5. Error analysis by platform, format and topic.
  6. Global explanation (permutation importance).
  7. Decision gate: does any model beat the best baseline?
  8. Refit the chosen approach on all data and save it for planning.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import load_posts
from .evaluate import error_by_group, evaluate_holdout, evaluate_rolling, residual_band
from .explain import global_importance
from .features import build_features
from .models import LEARNED_MODEL_FACTORIES, linear_regression
from .report import plot_predictions, plot_rolling, write_report
from .split import chronological_split

# A model must reduce MAE by at least this much versus the best
# baseline to be adopted. Small gains are rarely worth the upkeep.
MIN_IMPROVEMENT_PERCENT = 5.0


def run_study(data_path: str, out_dir: str = "outputs") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data
    posts, log = load_posts(data_path)
    print(f"Loaded {len(posts)} clean posts.")

    # 2. Features (history uses only earlier, completed posts)
    X = build_features(posts)
    y = posts[config.TARGET].astype(float)

    # 3. Holdout
    train_idx, test_idx = chronological_split(posts)
    results, fitted, preds = evaluate_holdout(X, y, train_idx, test_idx)
    results.to_csv(out / "holdout_results.csv", index=False)
    preds.join(posts[[config.POST_ID, config.PUBLISHED_AT]]).to_csv(
        out / "test_predictions.csv"
    )
    print("\nHoldout results (test set = most recent posts):")
    print(results.round(2).to_string(index=False))

    # 4. Rolling origin
    rolling = evaluate_rolling(X, y, posts)
    rolling.to_csv(out / "rolling_origin_results.csv", index=False)

    # 7. Decision gate
    learned = results[~results["baseline"]]
    best_learned = learned.iloc[0]
    best_baseline = results[results["baseline"]].iloc[0]
    gate_passed = best_learned["MAE_vs_best_baseline_%"] >= MIN_IMPROVEMENT_PERCENT
    chosen = best_learned["model"] if gate_passed else best_baseline["model"]

    # 5. Error analysis for the chosen approach
    errors = error_by_group(preds, posts, chosen)
    errors.to_csv(out / "error_by_group.csv", index=False)

    # 6. Global importance for the best learned model
    importance = global_importance(
        fitted[best_learned["model"]], X.loc[test_idx], y.loc[test_idx]
    )
    importance.to_csv(out / "feature_importance.csv", index=False)

    band = residual_band(preds, chosen)

    # 8. Refit on all data and save
    factory = {m().name: m for m in LEARNED_MODEL_FACTORIES.values()}
    if gate_passed:
        final = factory[chosen]().fit(X, y)
    else:
        final = fitted[chosen].fit(X, y)
    explainer_model = linear_regression().fit(X, y)
    bundle = {
        "model": final,
        "model_name": chosen,
        "gate_passed": bool(gate_passed),
        "explainer_model": explainer_model,
        "X_reference": X,
        "log_error_band_80": band,
        "trained_on_posts": int(len(posts)),
        "trained_until": str(posts[config.PUBLISHED_AT].max()),
    }
    joblib.dump(bundle, out / "model_bundle.joblib")

    # Report and charts
    plot_predictions(preds, chosen, out / "predicted_vs_actual.png")
    plot_rolling(rolling, out / "rolling_origin_mae.png")
    summary = {
        "posts": int(len(posts)),
        "train_posts": int(len(train_idx)),
        "test_posts": int(len(test_idx)),
        "test_period": [
            str(posts.loc[test_idx, config.PUBLISHED_AT].min().date()),
            str(posts.loc[test_idx, config.PUBLISHED_AT].max().date()),
        ],
        "best_baseline": best_baseline["model"],
        "best_baseline_MAE": round(float(best_baseline["MAE"]), 2),
        "best_model": best_learned["model"],
        "best_model_MAE": round(float(best_learned["MAE"]), 2),
        "improvement_%": float(best_learned["MAE_vs_best_baseline_%"]),
        "gate_passed": bool(gate_passed),
        "chosen_for_planning": chosen,
        "likely_range_80%_multipliers": [round(float(np.exp(b)), 2) for b in band],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out / "REPORT.md", summary, log, results, rolling, errors, importance,
                 outliers=int(posts["is_extreme_outlier"].sum()))

    print("\n" + ("GATE PASSED: " if gate_passed else "GATE NOT PASSED: ")
          + f"using '{chosen}' for planning. Full report: {out / 'REPORT.md'}")
    return summary


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def score_planned(bundle: dict, planned: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Forecast planned posts, with a likely range and a plain-language explanation."""
    from .explain import explain_posts, linear_contributions

    X_new = build_features(planned, history=history)
    point = np.clip(bundle["model"].predict(X_new), 0, None)
    lo, hi = bundle["log_error_band_80"]
    base_log = np.log1p(point)

    _, contrib = linear_contributions(
        bundle["explainer_model"], bundle["X_reference"], X_new
    )
    out = planned[[config.POST_ID]].copy()
    out["predicted_engagement_48h"] = point.round(0).astype(int)
    out["likely_low"] = np.clip(np.expm1(base_log + lo), 0, None).round(0).astype(int)
    out["likely_high"] = np.expm1(base_log + hi).round(0).astype(int)
    out["main_reasons"] = explain_posts(contrib, X_new)
    out["model"] = bundle["model_name"]
    return out
