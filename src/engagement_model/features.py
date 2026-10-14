"""
Feature engineering (paper, Sections 5 and 6).

Turns each planned or published post into a row of numbers a model can
use, using ONLY information available before the post goes live.

Three kinds of feature are built:
  * structured metadata: platform, format, topic, timing, audience size;
  * simple text features from the caption (no AI needed);
  * the account's recent history, computed strictly from EARLIER posts.

The leakage guard at the bottom refuses to continue if any column
measured after publication finds its way into the features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

# Words that signal a call to action. Extend with your own languages,
# for example Luganda or Swahili phrases your team uses.
CALL_TO_ACTION_PATTERNS = [
    r"\bapply\b", r"\bregister\b", r"\bsign up\b", r"\bjoin\b", r"\bdonate\b",
    r"\bread more\b", r"\blearn more\b", r"\bwatch\b", r"\bdownload\b",
    r"\blink in bio\b", r"\bclick\b", r"\bshare\b", r"\bnominate\b", r"\brsvp\b",
]
_CTA_RE = re.compile("|".join(CALL_TO_ACTION_PATTERNS), flags=re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

CATEGORICAL_FEATURES = config.CATEGORICAL_INPUTS + ["day_of_week"]

NUMERIC_FEATURES = [
    "log_followers",
    "hour",
    "is_weekend",
    "caption_length",
    "hashtag_count",
    "mention_count",
    "has_question",
    "has_call_to_action",
    "has_link",
    "prior_median_log_engagement",
    "has_history",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class LeakageError(RuntimeError):
    """Raised when a post-publication column is about to be used as a predictor."""


def assert_no_leakage(feature_columns: list[str]) -> None:
    """The leakage test from the paper, applied automatically."""
    leaked = sorted(set(feature_columns) & set(config.FORBIDDEN_PREDICTORS))
    if leaked:
        raise LeakageError(
            "These columns are only known AFTER publication and cannot be "
            f"used to predict engagement before posting: {leaked}. "
            "Remove them from the feature list (see config.FORBIDDEN_PREDICTORS)."
        )


def text_features(captions: pd.Series) -> pd.DataFrame:
    """Simple, transparent caption features (paper, Section 6.1)."""
    captions = captions.fillna("").astype(str)
    return pd.DataFrame(
        {
            "caption_length": captions.str.len(),
            "hashtag_count": captions.str.count(r"#\w+"),
            "mention_count": captions.str.count(r"@\w+"),
            "has_question": captions.str.contains(r"\?", regex=True).astype(int),
            "has_call_to_action": captions.apply(lambda t: int(bool(_CTA_RE.search(t)))),
            "has_link": captions.apply(lambda t: int(bool(_URL_RE.search(t)))),
        },
        index=captions.index,
    )


def timing_features(published_at: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(published_at)
    return pd.DataFrame(
        {
            "day_of_week": ts.dt.day_name().str.lower(),
            "hour": ts.dt.hour,
            "is_weekend": (ts.dt.dayofweek >= 5).astype(int),
        },
        index=published_at.index,
    )


def prior_history(
    posts: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    For each post, the median (log) engagement of the same account on the
    same platform over its previous HISTORY_WINDOW *completed* posts.

    A subtle leak is blocked here. A post published one hour earlier has
    not finished its 48-hour window, so its final engagement was not
    known when the next post was planned. Only earlier posts whose
    outcome window had already closed are counted.

    `history` = published posts with known outcomes (used when scoring
    planned posts). If omitted, `posts` serves as its own history, which
    is what we want when building features for past posts.
    """
    hist = posts if history is None else history
    hist = hist[hist[config.TARGET].notna()] if config.TARGET in hist.columns else hist.iloc[0:0]
    window = config.HISTORY_WINDOW
    lag = pd.Timedelta(hours=config.OUTCOME_WINDOW_HOURS)

    # Pre-sort each account/platform history once.
    groups: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ("account", "platform"):
            df[c] = df[c].astype(str).str.strip().str.lower()
        return df

    hist = _norm(hist)
    posts = _norm(posts)

    if len(hist):
        for key, g in hist.groupby(["account", "platform"], sort=False):
            g = g.sort_values(config.PUBLISHED_AT, kind="mergesort")
            groups[key] = (
                pd.to_datetime(g[config.PUBLISHED_AT]).to_numpy(),
                np.log1p(g[config.TARGET].to_numpy(dtype=float)),
            )

    values = np.full(len(posts), np.nan)
    times = pd.to_datetime(posts[config.PUBLISHED_AT]).to_numpy()
    keys = list(zip(posts["account"], posts["platform"]))
    for i, (key, t) in enumerate(zip(keys, times)):
        if key not in groups:
            continue
        h_times, h_logy = groups[key]
        # Earlier posts whose outcome window closed by time t.
        cut = np.searchsorted(h_times, t - lag, side="right")
        if cut == 0:
            continue
        values[i] = np.median(h_logy[max(0, cut - window):cut])

    result = pd.DataFrame(
        {"prior_median_log_engagement": values}, index=posts.index
    )
    result["has_history"] = result["prior_median_log_engagement"].notna().astype(int)
    # No completed earlier posts: use 0 and let `has_history` tell the model.
    result["prior_median_log_engagement"] = result["prior_median_log_engagement"].fillna(0.0)
    return result


def build_features(
    posts: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the full, leakage-checked feature table for a set of posts."""
    X = pd.DataFrame(index=posts.index)
    for col in config.CATEGORICAL_INPUTS:
        X[col] = posts[col].astype(str).str.strip().str.lower()
    X["log_followers"] = np.log10(posts["followers_at_post"].astype(float))
    X = X.join(timing_features(posts[config.PUBLISHED_AT]))
    X = X.join(text_features(posts[config.TEXT_INPUT]))
    X = X.join(prior_history(posts, history))

    X = X[FEATURE_COLUMNS]
    assert_no_leakage(list(X.columns))
    return X
