"""Tests for the FastAPI application."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import pandas as pd  # type: ignore[import-untyped]
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.state import get_explainer, get_flagged_df, get_pipeline
from src.fusion.explain import ExplainResult
from src.rag.pipeline import AnswerResult, Citation


def _sample_row(tx_id: str = "tx-1") -> dict[str, object]:
    return {
        "tx_id": tx_id,
        "date": "2024-06-22",
        "account": "1200-Inventory",
        "debit": 0.0,
        "credit": 235309.94,
        "user": "user_006",
        "posting_ts": "2024-06-22T17:25:34",
        "description": "Weekend inventory credit.",
        "is_anomaly": True,
        "anomaly_type": "weekend",
        "ensemble_score": 0.999,
        "isolation_forest_score": 0.68,
        "autoencoder_score": 6.6,
        "kmeans_score": 8.7,
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    """TestClient with cleared dependency overrides after the test."""
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    """/health responds with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_citations(client: TestClient) -> None:
    """/ask delegates to the pipeline and serialises the result."""
    pipeline = Mock()
    pipeline.answer.return_value = AnswerResult(
        answer="ASA 240 requires journal-entry testing [1].",
        citations=[
            Citation(tag=1, source="AUASB", section="p.31", page=31, chunk_id="c-1"),
        ],
        refused=False,
        reason="",
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    response = client.post("/ask", json={"question": "what does ASA 240 require?"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["answer"].startswith("ASA 240")
    assert body["citations"][0]["chunk_id"] == "c-1"
    pipeline.answer.assert_called_once_with("what does ASA 240 require?")


def test_ask_rejects_short_question(client: TestClient) -> None:
    """/ask returns 422 when the question is below the min length."""
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_scan_returns_top_n_flagged_rows(client: TestClient) -> None:
    """/scan returns at most N rows with derived feature flags."""
    df = pd.DataFrame([_sample_row("tx-a"), _sample_row("tx-b")])
    app.dependency_overrides[get_flagged_df] = lambda: df

    response = client.post("/scan", json={"n": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["tx_id"] == "tx-a"
    assert "is_weekend" in item["feature_flags"]
    assert item["anomaly_type"] == "weekend"


def test_scan_with_empty_data_returns_empty_list(client: TestClient) -> None:
    """/scan returns an empty items list when the flagged file is missing."""
    app.dependency_overrides[get_flagged_df] = lambda: pd.DataFrame()

    response = client.post("/scan", json={"n": 5})

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_explain_returns_narrative_for_known_tx(client: TestClient) -> None:
    """/explain looks up the tx and serialises the explainer result."""
    df = pd.DataFrame([_sample_row("tx-a")])
    explainer = Mock()
    explainer.explain.return_value = ExplainResult(
        tx_id="tx-a",
        narrative="Weekend posting is risky [1].",
        citations=[
            Citation(tag=1, source="AUASB", section="p.31", page=31, chunk_id="c-1"),
        ],
        refused=False,
        reason="",
    )
    app.dependency_overrides[get_flagged_df] = lambda: df
    app.dependency_overrides[get_explainer] = lambda: explainer

    response = client.get("/explain/tx-a")

    assert response.status_code == 200
    body = response.json()
    assert body["tx_id"] == "tx-a"
    assert body["refused"] is False
    assert body["citations"][0]["chunk_id"] == "c-1"
    called_tx = explainer.explain.call_args.args[0]
    assert called_tx.tx_id == "tx-a"
    assert "is_weekend" in called_tx.feature_flags


def test_explain_returns_404_for_unknown_tx(client: TestClient) -> None:
    """/explain returns 404 when the tx_id is not in the flagged set."""
    df = pd.DataFrame([_sample_row("tx-a")])
    app.dependency_overrides[get_flagged_df] = lambda: df

    response = client.get("/explain/missing")

    assert response.status_code == 404


def test_explain_returns_503_when_data_unavailable(client: TestClient) -> None:
    """/explain returns 503 when the flagged data is missing."""
    app.dependency_overrides[get_flagged_df] = lambda: pd.DataFrame()

    response = client.get("/explain/tx-a")

    assert response.status_code == 503
