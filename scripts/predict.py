"""
Forecast engagement for posts you are PLANNING to publish.

Usage:
    python scripts/predict.py \
        --model outputs/model_bundle.joblib \
        --history data/sample_posts.csv \
        --planned data/sample_planned_posts.csv \
        --out outputs/planned_forecasts.csv

The planned-posts CSV needs the same pre-publication columns as the
history file (no engagement column, obviously). Each forecast comes
with a likely range and the main reasons behind it.

A forecast is a prompt for a conversation, not a decision. Publish
strategically important content regardless of its forecast.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement_model.data import load_posts  # noqa: E402
from engagement_model.study import load_bundle, score_planned  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--history", required=True, help="CSV of published posts with outcomes")
    parser.add_argument("--planned", required=True, help="CSV of planned posts")
    parser.add_argument("--out", default="outputs/planned_forecasts.csv")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    if not bundle["gate_passed"]:
        print("Note: no model beat the baseline by enough, so these forecasts come "
              f"from '{bundle['model_name']}'.")
    history, _ = load_posts(args.history)
    planned = pd.read_csv(args.planned)
    planned["published_at"] = pd.to_datetime(planned["published_at"])

    latest = history["published_at"].max()
    if (planned["published_at"] - latest).dt.days.max() > 120:
        print("Warning: some planned posts are months after the latest data. "
              "Refresh the history and retrain before relying on these forecasts.")

    forecasts = score_planned(bundle, planned, history)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.out, index=False)
    pd.set_option("display.max_colwidth", 120)
    print(forecasts.to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
