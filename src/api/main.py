"""FastAPI entry point exposing the RAG, scan, and explain endpoints.

Endpoints:

* ``GET  /health``            — liveness probe.
* ``POST /ask``               — grounded Q&A against the AUASB ASA corpus.
* ``POST /scan``              — return top-N flagged GL rows.
* ``GET  /explain/{tx_id}``   — grounded narrative for a flagged transaction.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import requests
from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    CompaniesResponse,
    CompanyOut,
    ExplainResponse,
    FlaggedTxOut,
    IngestResponse,
    ScanRequest,
    ScanResponse,
)
from src.api.state import get_explainer, get_flagged_df, get_pipeline, reset_caches
from src.core.logging_config import configure_logging
from src.fusion.explain import AnomalyExplainer, flagged_transaction_from_row
from src.rag.ingest_company import ingest_company
from src.rag.pipeline import RagPipeline
from src.rag.registry import ASX_ANNUAL_REPORTS, list_companies

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AuditCopilot API",
    version="0.1.0",
    description="Local-first AI assistant for Audit and Assurance.",
)


def _row_to_out(row: pd.Series) -> FlaggedTxOut:
    """Convert a flagged-CSV row to the API output schema."""
    tx = flagged_transaction_from_row(row.to_dict())
    return FlaggedTxOut(
        tx_id=tx.tx_id,
        date=tx.date,
        account=tx.account,
        debit=tx.debit,
        credit=tx.credit,
        user=tx.user,
        posting_ts=tx.posting_ts,
        description=tx.description,
        ensemble_score=tx.ensemble_score,
        feature_flags=list(tx.feature_flags),
        is_anomaly=bool(row["is_anomaly"]) if "is_anomaly" in row.index else None,
        anomaly_type=(
            str(row["anomaly_type"])
            if "anomaly_type" in row.index and pd.notna(row["anomaly_type"])
            else None
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
FlaggedDfDep = Annotated[pd.DataFrame, Depends(get_flagged_df)]
ExplainerDep = Annotated[AnomalyExplainer, Depends(get_explainer)]


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    pipeline: PipelineDep,
) -> AskResponse:
    """Answer an audit-standards question with citations."""
    logger.info(
        "ask request",
        extra={"question_len": len(payload.question), "company": payload.company},
    )
    result = pipeline.answer(payload.question, company=payload.company)
    return AskResponse(
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        refused=result.refused,
        reason=result.reason,
    )


@app.get("/companies", response_model=CompaniesResponse)
def companies() -> CompaniesResponse:
    """List the curated ASX company allowlist with per-ticker indexed flag."""
    items = [CompanyOut(**c.__dict__) for c in list_companies()]
    return CompaniesResponse(items=items)


@app.post("/companies/{ticker}/ingest", response_model=IngestResponse)
def ingest(ticker: str) -> IngestResponse:
    """Fetch, chunk, and index one company's annual report on demand."""
    ticker_upper = ticker.upper()
    if ticker_upper not in ASX_ANNUAL_REPORTS:
        raise HTTPException(status_code=404, detail="ticker not in allowlist")
    logger.info("ingest request", extra={"ticker": ticker_upper})
    try:
        result = ingest_company(ticker_upper)
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "ingest download failed",
            extra={"ticker": ticker_upper, "error": str(exc)},
        )
        raise HTTPException(
            status_code=502,
            detail=f"failed to download {ticker_upper} annual report: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("ingest failed", extra={"ticker": ticker_upper})
        raise HTTPException(
            status_code=500,
            detail=f"ingest failed for {ticker_upper}: {exc}",
        ) from exc
    reset_caches()
    return IngestResponse(
        ticker=result.ticker,
        chunks_added=result.chunks_added,
        took_ms=result.took_ms,
        cached=result.cached,
    )


@app.post("/scan", response_model=ScanResponse)
def scan(
    payload: ScanRequest,
    df: FlaggedDfDep,
) -> ScanResponse:
    """Return the top-N flagged GL rows by ensemble score."""
    if df.empty:
        return ScanResponse(items=[])
    head = df.head(payload.n)
    items = [_row_to_out(row) for _, row in head.iterrows()]
    logger.info("scan response", extra={"n": payload.n, "returned": len(items)})
    return ScanResponse(items=items)


@app.get("/explain/{tx_id}", response_model=ExplainResponse)
def explain(
    tx_id: str,
    df: FlaggedDfDep,
    explainer: ExplainerDep,
    company: str | None = None,
) -> ExplainResponse:
    """Generate a grounded narrative for one flagged transaction."""
    if df.empty:
        raise HTTPException(status_code=503, detail="flagged data unavailable")
    matches = df.loc[df["tx_id"] == tx_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="tx_id not found")
    tx = flagged_transaction_from_row(matches.iloc[0].to_dict())
    logger.info("explain request", extra={"tx_id": tx_id, "company": company})
    result = explainer.explain(tx, company=company)
    return ExplainResponse(
        tx_id=result.tx_id,
        narrative=result.narrative,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        refused=result.refused,
        reason=result.reason,
    )
