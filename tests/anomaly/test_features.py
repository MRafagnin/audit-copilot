"""Tests for ``src.anomaly.features``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly.features import FEATURE_COLUMNS, _first_digit, build_features


def _row(
    *,
    tx_id: str = "t1",
    account: str = "1000-Cash",
    debit: float = 100.0,
    credit: float = 0.0,
    user: str = "user_001",
    posting_ts: str = "2024-06-03T10:30:00",  # Monday 10:30
    description: str = "ordinary posting",
    is_anomaly: bool = False,
) -> dict[str, object]:
    return {
        "tx_id": tx_id,
        "account": account,
        "debit": debit,
        "credit": credit,
        "user": user,
        "posting_ts": posting_ts,
        "description": description,
        "is_anomaly": is_anomaly,
    }


def test_first_digit_basic() -> None:
    assert _first_digit(0) == 0
    assert _first_digit(7) == 7
    assert _first_digit(123.45) == 1
    assert _first_digit(0.0042) == 4
    assert _first_digit(-987) == 9


def test_build_features_empty() -> None:
    df = pd.DataFrame(
        columns=[
            "tx_id",
            "account",
            "debit",
            "credit",
            "user",
            "posting_ts",
            "description",
        ]
    )
    fm = build_features(df)
    assert fm.X.shape == (0, len(FEATURE_COLUMNS))
    assert fm.tx_ids.shape == (0,)
    assert fm.labels is None


def test_build_features_flags_weekend_after_hours_round_amount() -> None:
    rows = [
        _row(tx_id="normal", debit=137.42),
        _row(
            tx_id="weekend",
            posting_ts="2024-06-08T11:00:00",  # Saturday
            debit=212.0,
        ),
        _row(
            tx_id="late",
            posting_ts="2024-06-03T23:30:00",  # Monday 23:30
            debit=212.0,
        ),
        _row(tx_id="round", debit=10_000.0),
    ]
    df = pd.DataFrame(rows)
    fm = build_features(df)
    cols = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

    is_weekend = fm.X[:, cols["is_weekend"]]
    is_after_hours = fm.X[:, cols["is_after_hours"]]
    is_round = fm.X[:, cols["is_round_amount"]]
    benford = fm.X[:, cols["benford_first_digit"]]

    assert is_weekend.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert is_after_hours.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert is_round.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert benford[3] == 1.0  # 10000 -> first digit 1


def test_build_features_preserves_labels_and_ids() -> None:
    df = pd.DataFrame(
        [
            _row(tx_id="a", is_anomaly=False),
            _row(tx_id="b", is_anomaly=True),
        ]
    )
    fm = build_features(df)
    assert fm.tx_ids.tolist() == ["a", "b"]
    assert fm.labels is not None
    assert fm.labels.tolist() == [False, True]
    assert np.isfinite(fm.X).all()
