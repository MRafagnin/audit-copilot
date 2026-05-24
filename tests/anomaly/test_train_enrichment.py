"""Tests for the cross-row flag enrichment in scripts/train_anomaly.py."""

from __future__ import annotations

import importlib

import pandas as pd

from src.anomaly.features import build_features

train_anomaly = importlib.import_module("scripts.train_anomaly")


def _build_frame() -> pd.DataFrame:
    """Synthetic GL frame exercising each cross-row signal."""
    base = []
    # Bulk of normal rows so quantile/std make sense.
    for i in range(40):
        base.append(
            {
                "tx_id": f"n{i}",
                "date": "2024-06-10",
                "account": "6000-OperatingExpense",
                "debit": 1_000.0 + i * 10,
                "credit": 0.0,
                "user": "user_001",
                "posting_ts": f"2024-06-10T1{i % 6}:00:00",
                "description": "Note 13 \u00b7 p.180 \u2014 Operating expenses",
                "is_anomaly": False,
                "anomaly_type": "",
            }
        )
    # Unusual user/account pair (only occurrence of this pair).
    base.append(
        {
            "tx_id": "unusual",
            "date": "2024-06-11",
            "account": "3000-Equity",
            "debit": 500.0,
            "credit": 0.0,
            "user": "user_099",
            "posting_ts": "2024-06-11T10:00:00",
            "description": "Note 20 \u00b7 p.215 \u2014 Equity",
            "is_anomaly": True,
            "anomaly_type": "unusual_user_account",
        }
    )
    # Amount outlier within account.
    base.append(
        {
            "tx_id": "outlier",
            "date": "2024-06-12",
            "account": "6000-OperatingExpense",
            "debit": 5_000_000.0,
            "credit": 0.0,
            "user": "user_001",
            "posting_ts": "2024-06-12T10:00:00",
            "description": "Note 13 \u00b7 p.182 \u2014 Operating expenses",
            "is_anomaly": True,
            "anomaly_type": "amount_outlier",
        }
    )
    # Near-duplicate pair.
    base.append(
        {
            "tx_id": "dup_a",
            "date": "2024-06-13",
            "account": "2000-AccountsPayable",
            "debit": 2_500.0,
            "credit": 0.0,
            "user": "user_002",
            "posting_ts": "2024-06-13T10:00:00",
            "description": "Note 11 \u00b7 p.175 \u2014 Trade payables",
            "is_anomaly": True,
            "anomaly_type": "near_duplicate",
        }
    )
    base.append(
        {
            "tx_id": "dup_b",
            "date": "2024-06-13",
            "account": "2000-AccountsPayable",
            "debit": 2_500.0,
            "credit": 0.0,
            "user": "user_002",
            "posting_ts": "2024-06-13T10:05:00",
            "description": "Note 11 \u00b7 p.175 \u2014 Trade payables",
            "is_anomaly": True,
            "anomaly_type": "near_duplicate",
        }
    )
    return pd.DataFrame(base).reset_index(drop=True)


def test_enrich_feature_flags_detects_cross_row_signals() -> None:
    df = _build_frame()
    fm = build_features(df)

    def pos(tx_id: str) -> int:
        return int(df.index[df["tx_id"] == tx_id][0])

    import numpy as np

    top_idx = np.array([pos("unusual"), pos("outlier"), pos("dup_a"), pos("dup_b")])
    flags = train_anomaly.enrich_feature_flags(df, fm.X, top_idx)

    assert "is_unusual_user_account" in flags[0]
    # Equity is a sensitive account too
    assert "is_sensitive_account" in flags[0]
    assert "is_amount_outlier_for_account" in flags[1]
    assert "is_large_amount" in flags[1]
    assert "is_near_duplicate" in flags[2]
    assert "is_near_duplicate" in flags[3]


def test_enrich_feature_flags_orders_by_severity() -> None:
    df = _build_frame()
    fm = build_features(df)

    import numpy as np

    pos = int(df.index[df["tx_id"] == "outlier"][0])
    flags = train_anomaly.enrich_feature_flags(df, fm.X, np.array([pos]))[0]
    parts = flags.split(";")
    # Amount outlier should appear before large_amount (severity ordering).
    assert parts.index("is_amount_outlier_for_account") < parts.index("is_large_amount")
