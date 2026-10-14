"""Charts and a plain-language Markdown report of the study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, ACCENT, GRID = "#1f2328", "#6e7781", "#2f6fdb", "#e5e7eb"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_predictions(preds: pd.DataFrame, model: str, path: Path) -> None:
    """Predicted vs actual on the test posts (log axes: engagement is skewed)."""
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)
    a, p = preds["actual"] + 1, preds[model] + 1
    ax.scatter(a, p, s=14, alpha=0.6, color=ACCENT, edgecolor="none")
    lims = [min(a.min(), p.min()) * 0.8, max(a.max(), p.max()) * 1.2]
    ax.plot(lims, lims, color=MUTED, linewidth=1, linestyle="--", label="Perfect forecast")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual engagements within 48 hours (+1, log scale)", color=INK, fontsize=9)
    ax.set_ylabel("Predicted engagements (+1, log scale)", color=INK, fontsize=9)
    ax.set_title(f"Test posts: {model}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_rolling(rolling: pd.DataFrame, path: Path) -> None:
    """MAE per rolling-origin round, one line per approach."""
    if rolling.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    table = rolling.pivot(index="test_month", columns="model", values="MAE")
    dashes = iter(["--", ":", "-."])
    for name in table.columns:
        base = name.startswith("Baseline")
        ax.plot(table.index, table[name], marker="o", markersize=4,
                linewidth=1.4 if base else 2,
                linestyle=next(dashes, "--") if base else "-",
                color=MUTED if base else None, alpha=0.8 if base else 1,
                label=name)
    ax.set_ylabel("Mean absolute error (engagements)", color=INK, fontsize=9)
    ax.set_xlabel("Test month (model trained on all earlier months)", color=INK, fontsize=9)
    ax.set_title("Is accuracy stable over time?", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _table(df: pd.DataFrame, floatfmt: int = 1) -> str:
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].round(floatfmt)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_report(path: Path, s: dict, log, results, rolling, errors, importance, outliers: int) -> None:
    gate = (
        f"**Gate passed.** {s['best_model']} reduced MAE by {s['improvement_%']}% "
        f"versus the best baseline ({s['best_baseline']}). It is used for planning."
        if s["gate_passed"]
        else f"**Gate not passed.** The best model ({s['best_model']}) improved MAE by only "
        f"{s['improvement_%']}% versus {s['best_baseline']}. Keep using the baseline "
        "and improve the data before trying again."
    )
    stability = (
        rolling.groupby("model")["MAE"].agg(["mean", "min", "max"]).reset_index()
        .rename(columns={"mean": "mean_MAE", "min": "best_month", "max": "worst_month"})
        .sort_values("mean_MAE")
    )
    lo, hi = s["likely_range_80%_multipliers"]
    text = f"""# Engagement prediction study: results

## The answer

{gate}

- Posts analysed: {s['posts']} ({s['train_posts']} for training, {s['test_posts']} held back for testing)
- Test period (never seen by the model): {s['test_period'][0]} to {s['test_period'][1]}
- Best baseline MAE: {s['best_baseline_MAE']} engagements
- Best model MAE: {s['best_model_MAE']} engagements
- About 80% of test posts landed between {lo}x and {hi}x of their forecast (on the 1 + engagement scale). Treat every forecast as a range, not a promise.

## How the data were cleaned

{log}

{outliers} extreme posts were kept but flagged. Viral breakouts are the least predictable posts (paper, Section 4.5); check whether results change without them.

## Holdout results

MAE and RMSE are in engagements; R2 is the share of variation explained; MAPE excludes posts with zero engagement.

{_table(results[['model', 'MAE', 'RMSE', 'R2', 'MAPE_%', 'MedianAE', 'MAE_vs_best_baseline_%']], 2)}

![Predicted vs actual](predicted_vs_actual.png)

## Stability over time (rolling origin)

{_table(stability, 1)}

![Rolling origin](rolling_origin_mae.png)

## Where the forecasts are weakest

Positive `mean_bias` means the model over-predicts for that group.

{_table(errors, 1)}

## What the best model relies on

Increase in MAE when each input is shuffled. These describe what the model learned from history, not what causes engagement.

{_table(importance[['label', 'MAE_increase_when_shuffled', 'std']], 2)}

## Before you use these results

- Do the important inputs make sense to someone who knows the audience?
- Which groups above have large errors, and are they posts you care about most?
- Has anything changed (platform rules, strategy, audience) since {s['test_period'][1]}?
- Record the decision and its owner in MODEL_CARD.md.
"""
    Path(path).write_text(text)
