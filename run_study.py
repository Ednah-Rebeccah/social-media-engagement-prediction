"""
Run the full engagement prediction study on a CSV of past posts.

Usage:
    python scripts/run_study.py --data data/sample_posts.csv --out outputs

Produces in the output folder: REPORT.md (start here), holdout and
rolling-origin results, error analysis, feature importance, charts,
and model_bundle.joblib for forecasting planned posts.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement_model.study import run_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV of published posts (see data/DATA_DICTIONARY.md)")
    parser.add_argument("--out", default="outputs", help="Folder for results")
    args = parser.parse_args()
    run_study(args.data, args.out)


if __name__ == "__main__":
    main()
