"""
Tests for the safeguards the paper insists on.

Run with either:
    python -m unittest discover -s tests -v
    pytest
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engagement_model import config  # noqa: E402
from engagement_model.data import clean_posts  # noqa: E402
from engagement_model.evaluate import regression_metrics  # noqa: E402
from engagement_model.features import (  # noqa: E402
    FEATURE_COLUMNS, LeakageError, assert_no_leakage, build_features, prior_history,
)
from engagement_model.split import chronological_split  # noqa: E402


def _posts(times, ys, account="main", platform="facebook"):
    n = len(times)
    return pd.DataFrame({
        "post_id": [f"P{i}" for i in range(n)],
        "published_at": pd.to_datetime(times),
        "platform": platform, "account": account,
        "content_format": "image", "topic": "event", "paid": "organic",
        "followers_at_post": 10000, "caption": "Hello #world",
        "engagement_48h": ys,
    })


class TestMetrics(unittest.TestCase):
    """The worked example in Section 8.2 of the paper."""

    actual = [80, 120, 100, 160]

    def test_baseline(self):
        m = regression_metrics(self.actual, [100, 100, 100, 100])
        self.assertAlmostEqual(m["MAE"], 25.0)
        self.assertAlmostEqual(m["RMSE"], 33.17, places=2)
        self.assertAlmostEqual(m["R2"], -0.257, places=3)

    def test_model(self):
        m = regression_metrics(self.actual, [90, 110, 105, 145])
        self.assertAlmostEqual(m["MAE"], 10.0)
        self.assertAlmostEqual(m["RMSE"], 10.61, places=2)
        self.assertAlmostEqual(m["R2"], 0.871, places=3)
        self.assertAlmostEqual(m["MAPE_%"], 8.80, places=2)

    def test_mape_skips_zero_actuals(self):
        m = regression_metrics([0, 10], [5, 5])
        self.assertAlmostEqual(m["MAPE_%"], 50.0)


class TestLeakage(unittest.TestCase):
    def test_guard_rejects_post_publication_columns(self):
        with self.assertRaises(LeakageError):
            assert_no_leakage(["content_format", "lifetime_reach"])
        with self.assertRaises(LeakageError):
            assert_no_leakage([config.TARGET])

    def test_features_contain_no_forbidden_columns(self):
        df = _posts(pd.date_range("2025-01-01", periods=5, freq="7D"), [1, 2, 3, 4, 5])
        df["lifetime_reach"] = 999
        X = build_features(df)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertFalse(set(X.columns) & set(config.FORBIDDEN_PREDICTORS))

    def test_history_ignores_unfinished_and_future_posts(self):
        t0 = pd.Timestamp("2025-01-01 09:00")
        times = [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=24),
                 t0 + pd.Timedelta(hours=48, minutes=30)]
        df = _posts(times, [100, 10000, 5, 7])
        h = prior_history(df)
        # First post: nothing earlier.
        self.assertEqual(h.loc[0, "has_history"], 0)
        # 24h later: earlier posts have not finished their 48h window yet.
        self.assertEqual(h.loc[2, "has_history"], 0)
        # 48.5h later: only the first post's window has closed.
        self.assertAlmostEqual(h.loc[3, "prior_median_log_engagement"], np.log1p(100))

    def test_history_unchanged_by_future_outcomes(self):
        times = pd.date_range("2025-01-01", periods=6, freq="5D")
        a = prior_history(_posts(times, [10, 20, 30, 40, 50, 60]))
        b = prior_history(_posts(times, [10, 20, 30, 40, 50, 99999]))
        pd.testing.assert_frame_equal(a.iloc[:5], b.iloc[:5])

    def test_history_is_per_account_and_platform(self):
        times = pd.date_range("2025-01-01", periods=4, freq="5D")
        df = pd.concat([_posts(times[:2], [1000, 1000], account="big"),
                        _posts(times[2:], [5, 5], account="small")], ignore_index=True)
        df["post_id"] = [f"Q{i}" for i in range(4)]
        h = prior_history(df)
        self.assertEqual(h.loc[2, "has_history"], 0)  # 'small' has no past of its own


class TestSplitAndCleaning(unittest.TestCase):
    def test_split_is_chronological(self):
        times = pd.date_range("2024-01-01", periods=300, freq="D")
        df = _posts(times.to_series().sample(frac=1, random_state=1).to_numpy(), range(300))
        train, test = chronological_split(df, 0.2)
        self.assertEqual(len(test), 60)
        self.assertLess(df.loc[train, "published_at"].max(), df.loc[test, "published_at"].min())

    def test_cleaning(self):
        df = _posts(pd.date_range("2025-01-01", periods=4, freq="D"), [5, -1, 7, 8])
        df.loc[0, "platform"] = "  Facebook "
        df = pd.concat([df, df.iloc[[2]]])  # duplicate post
        clean, log = clean_posts(df)
        self.assertEqual(len(clean), 3)  # duplicate and negative removed
        self.assertEqual(set(clean["platform"]), {"facebook"})
        self.assertIn("duplicate", str(log))


class TestEndToEnd(unittest.TestCase):
    def test_study_and_forecast(self):
        from generate_sample_data import make_planned, make_posts
        from engagement_model.study import run_study, score_planned, load_bundle
        from engagement_model.data import load_posts

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "posts.csv"
            make_posts(500, seed=3).to_csv(data, index=False)
            summary = run_study(str(data), str(Path(tmp) / "out"))
            for f in ["REPORT.md", "holdout_results.csv", "model_bundle.joblib",
                      "predicted_vs_actual.png", "rolling_origin_mae.png"]:
                self.assertTrue((Path(tmp) / "out" / f).exists(), f)
            self.assertIn("gate_passed", summary)

            bundle = load_bundle(str(Path(tmp) / "out" / "model_bundle.joblib"))
            history, _ = load_posts(str(data))
            planned = make_planned(0)
            planned["published_at"] = pd.to_datetime(planned["published_at"])
            out = score_planned(bundle, planned, history)
            self.assertEqual(len(out), len(planned))
            self.assertTrue((out["likely_low"] <= out["predicted_engagement_48h"]).all())
            self.assertTrue((out["predicted_engagement_48h"] <= out["likely_high"]).all())


if __name__ == "__main__":
    unittest.main()
