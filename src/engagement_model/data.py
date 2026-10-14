"""
Loading and cleaning post-level data (paper, Sections 9.2 and 11, step 3).

One row = one published post. The functions here never invent data:
they standardise labels, remove duplicates and rows that cannot be
used, and record every change in a cleaning log so the analysis can be
repeated and audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningLog:
    """A plain-language record of every change made to the raw data."""

    steps: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.steps.append(message)

    def __str__(self) -> str:
        return "\n".join(f"- {s}" for s in self.steps) or "- No changes needed."


REQUIRED_COLUMNS = (
    [config.POST_ID, config.PUBLISHED_AT, config.TARGET, config.TEXT_INPUT]
    + config.CATEGORICAL_INPUTS
    + config.NUMERIC_INPUTS
)


def load_posts(path: str) -> tuple[pd.DataFrame, CleaningLog]:
    """Read a CSV of posts, validate it and return a clean, time-ordered table."""
    raw = pd.read_csv(path)
    return clean_posts(raw)


def clean_posts(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    df = raw.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The data is missing required columns: "
            + ", ".join(missing)
            + ". See data/DATA_DICTIONARY.md for the expected format."
        )

    start = len(df)
    log.add(f"Loaded {start} rows.")

    # Duplicates: the same post exported twice.
    df = df.drop_duplicates(subset=config.POST_ID, keep="first")
    if len(df) < start:
        log.add(f"Removed {start - len(df)} duplicate post IDs.")

    # Dates.
    df[config.PUBLISHED_AT] = pd.to_datetime(df[config.PUBLISHED_AT], errors="coerce")
    bad_dates = df[config.PUBLISHED_AT].isna().sum()
    if bad_dates:
        df = df[df[config.PUBLISHED_AT].notna()]
        log.add(f"Removed {bad_dates} rows with unreadable publication dates.")

    # Standardise category labels: ' Instagram ' and 'instagram' are one platform.
    for col in config.CATEGORICAL_INPUTS:
        df[col] = (
            df[col].astype("string").str.strip().str.lower().fillna("unknown")
        )
    log.add("Standardised category labels (trimmed spaces, lower case).")

    df[config.TEXT_INPUT] = df[config.TEXT_INPUT].fillna("").astype(str)

    # Outcome: must exist and cannot be negative.
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    n_missing_y = df[config.TARGET].isna().sum()
    if n_missing_y:
        df = df[df[config.TARGET].notna()]
        log.add(f"Removed {n_missing_y} rows with no recorded outcome.")
    n_negative = (df[config.TARGET] < 0).sum()
    if n_negative:
        df = df[df[config.TARGET] >= 0]
        log.add(f"Removed {n_negative} rows with impossible negative engagement.")

    # Audience size: must be positive.
    for col in config.NUMERIC_INPUTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col].isna() | (df[col] <= 0)).sum()
        if bad:
            df = df[df[col].notna() & (df[col] > 0)]
            log.add(f"Removed {bad} rows with missing or impossible '{col}'.")

    # Flag (do not delete) extreme outliers so they can be investigated.
    y = df[config.TARGET]
    q1, q3 = np.percentile(y, [25, 75])
    upper = q3 + 3 * (q3 - q1)
    n_outliers = int((y > upper).sum())
    df["is_extreme_outlier"] = y > upper
    log.add(
        f"Flagged {n_outliers} extreme high-engagement posts (above {upper:.0f}); "
        "kept in the data but reported separately."
    )

    df = df.sort_values(config.PUBLISHED_AT, kind="mergesort").reset_index(drop=True)
    log.add(f"Final dataset: {len(df)} posts, sorted oldest to newest.")
    return df, log
