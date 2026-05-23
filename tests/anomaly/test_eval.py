"""Tests for ``src.anomaly.eval``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.anomaly.eval import evaluate, persist, precision_at_k


def test_precision_at_k_perfect() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    assert precision_at_k(scores, labels, k=2) == 1.0


def test_precision_at_k_partial() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 0])
    assert precision_at_k(scores, labels, k=2) == 0.5


def test_precision_at_k_zero_k() -> None:
    assert precision_at_k(np.array([1.0]), np.array([1]), k=0) == 0.0


def test_evaluate_perfect_ranking() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1])
    result = evaluate(scores, labels, k=2)
    assert result.precision_at_k == 1.0
    assert result.roc_auc == 1.0
    assert result.pr_auc == 1.0
    assert result.n == 5
    assert result.anomalies == 2
    assert result.k == 2


def test_evaluate_single_class_falls_back_to_zero() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0])
    result = evaluate(scores, labels, k=1)
    assert result.roc_auc == 0.0
    assert result.pr_auc == 0.0
    assert result.anomalies == 0


def test_persist_writes_json(tmp_path: Path) -> None:
    scores = np.array([0.1, 0.9])
    labels = np.array([0, 1])
    result = evaluate(scores, labels, k=1)
    out = tmp_path / "metrics.json"
    persist({"ensemble": result}, out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "ensemble" in payload
    assert payload["ensemble"]["precision_at_k"] == 1.0
    assert payload["ensemble"]["n"] == 2
