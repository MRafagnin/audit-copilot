"""Feature engineering for journal-entry anomaly detection.

Inputs are raw GL rows produced by ``scripts/gen_journal_entries.py``. The
output is a numeric feature matrix plus metadata needed downstream (the
original ``tx_id`` and ground-truth labels when present).

Features
--------

* ``amount``: signed posting amount (``debit - credit``).
* ``amount_abs``: absolute amount.
* ``amount_log``: ``log1p(amount_abs)``.
* ``is_round_amount``: 1.0 when ``amount_abs`` is a multiple of 1000.
* ``benford_first_digit``: leading digit of ``amount_abs`` (1-9; 0 if amount is 0).
* ``hour``: posting hour, 0-23.
* ``is_after_hours``: 1.0 when hour < 6 or hour >= 20.
* ``is_weekend``: 1.0 when posting day is Saturday or Sunday.
* ``user_account_freq``: relative frequency of the (user, account) pair.
* ``user_freq``: relative frequency of the user.
* ``account_freq``: relative frequency of the account.
* ``amount_zscore_per_account``: z-score of ``amount_abs`` within the account.
* ``desc_len``: character length of the description.
"""

# ruff: noqa: N806  # X is the standard sklearn name for a feature matrix.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


FEATURE_COLUMNS: tuple[str, ...] = (
    "amount",
    "amount_abs",
    "amount_log",
    "is_round_amount",
    "benford_first_digit",
    "hour",
    "is_after_hours",
    "is_weekend",
    "user_account_freq",
    "user_freq",
    "account_freq",
    "amount_zscore_per_account",
    "desc_len",
)


@dataclass(frozen=True)
class FeatureMatrix:
    """Container for engineered features and aligned metadata.

    Attributes:
        X: ``(n, d)`` float matrix in column order :data:`FEATURE_COLUMNS`.
        tx_ids: Transaction ids parallel to ``X`` rows.
        labels: Ground-truth ``is_anomaly`` labels (``None`` when unavailable).
    """

    X: NDArray[np.float64]
    tx_ids: NDArray[Any]
    labels: NDArray[np.bool_] | None


def _first_digit(value: float) -> int:
    """Return the leading non-zero digit of ``|value|`` (0 when value is 0)."""
    abs_val = abs(value)
    if abs_val == 0:
        return 0
    while abs_val < 1:
        abs_val *= 10
    while abs_val >= 10:
        abs_val //= 10
    return int(abs_val)


def build_features(df: pd.DataFrame) -> FeatureMatrix:
    """Compute engineered features for a GL DataFrame.

    Args:
        df: Raw GL frame as produced by ``gen_journal_entries.py``. Must have
            columns ``tx_id, account, debit, credit, user, posting_ts,
            description``. ``is_anomaly`` is optional.

    Returns:
        :class:`FeatureMatrix` with the same row order as ``df``.
    """
    if df.empty:
        return FeatureMatrix(
            X=np.zeros((0, len(FEATURE_COLUMNS)), dtype=float),
            tx_ids=np.array([], dtype=object),
            labels=None,
        )

    debit = df["debit"].astype(float).to_numpy()
    credit = df["credit"].astype(float).to_numpy()
    amount = debit - credit
    amount_abs = np.abs(amount)

    posting_ts = pd.to_datetime(df["posting_ts"])
    hour = posting_ts.dt.hour.to_numpy()
    weekday = posting_ts.dt.weekday.to_numpy()
    is_after_hours = ((hour < 6) | (hour >= 20)).astype(float)
    is_weekend = (weekday >= 5).astype(float)

    is_round = ((amount_abs > 0) & (np.mod(amount_abs, 1000.0) == 0)).astype(float)
    benford = np.array([_first_digit(v) for v in amount_abs], dtype=float)

    n = len(df)
    user_counts = df["user"].value_counts(normalize=True)
    account_counts = df["account"].value_counts(normalize=True)
    pair_key = df["user"].astype(str) + "|" + df["account"].astype(str)
    pair_counts = pair_key.value_counts(normalize=True)

    user_freq = df["user"].map(user_counts).to_numpy(dtype=float)
    account_freq = df["account"].map(account_counts).to_numpy(dtype=float)
    user_account_freq = pair_key.map(pair_counts).to_numpy(dtype=float)

    # z-score of amount_abs within each account
    amount_series = pd.Series(amount_abs, index=df.index)
    grouped = amount_series.groupby(df["account"])
    mean_per_account = grouped.transform("mean")
    std_per_account = grouped.transform("std").replace(0.0, 1.0).fillna(1.0)
    zscore = ((amount_series - mean_per_account) / std_per_account).to_numpy()

    desc_len = df["description"].astype(str).str.len().to_numpy(dtype=float)
    amount_log = np.log1p(amount_abs)

    X = np.column_stack(
        [
            amount,
            amount_abs,
            amount_log,
            is_round,
            benford,
            hour.astype(float),
            is_after_hours,
            is_weekend,
            user_account_freq,
            user_freq,
            account_freq,
            zscore,
            desc_len,
        ]
    )
    # Guard against NaN/inf from edge cases (single-row accounts, etc.)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert X.shape[1] == len(FEATURE_COLUMNS), "feature column count mismatch"

    labels = (
        df["is_anomaly"].astype(bool).to_numpy()
        if "is_anomaly" in df.columns
        else None
    )
    logger.info(
        "features built",
        extra={"rows": n, "cols": X.shape[1], "labels": labels is not None},
    )
    return FeatureMatrix(X=X, tx_ids=df["tx_id"].to_numpy(), labels=labels)
