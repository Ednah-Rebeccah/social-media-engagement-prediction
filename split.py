"""
Time-ordered train/test splits (paper, Sections 2.5 and 8.4).

Social media forecasting is about using the past to predict the future,
so posts are never split at random. Older posts train the model; newer
posts, which it has never seen, test it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from . import config


def chronological_split(
    df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Earliest (1 - test_fraction) of posts for training, the rest for testing."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    order = np.argsort(df[config.PUBLISHED_AT].to_numpy(), kind="mergesort")
    n_test = max(1, int(round(len(df) * test_fraction)))
    train_idx, test_idx = order[:-n_test], order[-n_test:]
    if len(train_idx) < config.MIN_TRAIN_POSTS:
        raise ValueError(
            f"Only {len(train_idx)} posts available for training; at least "
            f"{config.MIN_TRAIN_POSTS} are needed for a meaningful model."
        )
    return df.index[train_idx].to_numpy(), df.index[test_idx].to_numpy()


def rolling_origin_splits(
    df: pd.DataFrame, n_months: int = config.ROLLING_TEST_MONTHS
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """
    Rolling-origin evaluation: for each of the last `n_months` calendar
    months, train on everything published before that month and test on
    that month only. Yields (month label, train index, test index).
    """
    months = df[config.PUBLISHED_AT].dt.to_period("M")
    for month in sorted(months.unique())[-n_months:]:
        train_mask = months < month
        test_mask = months == month
        if train_mask.sum() < config.MIN_TRAIN_POSTS or test_mask.sum() == 0:
            continue
        yield str(month), df.index[train_mask].to_numpy(), df.index[test_mask].to_numpy()
