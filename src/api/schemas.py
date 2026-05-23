"""Pydantic request and response models for the FastAPI surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Question payload for ``POST /ask``."""

    question: str = Field(min_length=3, max_length=2000)


class CitationOut(BaseModel):
    """Citation returned alongside generated answers and narratives."""

    tag: int
    source: str
    section: str
    page: int | None = None
    chunk_id: str


class AskResponse(BaseModel):
    """Response from ``POST /ask``."""

    answer: str
    citations: list[CitationOut]
    refused: bool
    reason: str


class ScanRequest(BaseModel):
    """Pagination payload for ``POST /scan``."""

    n: int = Field(default=10, ge=1, le=200)


class FlaggedTxOut(BaseModel):
    """A single flagged journal-entry row as returned by ``POST /scan``."""

    tx_id: str
    date: str
    account: str
    debit: float
    credit: float
    user: str
    posting_ts: str
    description: str
    ensemble_score: float
    feature_flags: list[str]
    is_anomaly: bool | None = None
    anomaly_type: str | None = None


class ScanResponse(BaseModel):
    """Response from ``POST /scan``."""

    items: list[FlaggedTxOut]


class ExplainResponse(BaseModel):
    """Response from ``GET /explain/{tx_id}``."""

    tx_id: str
    narrative: str
    citations: list[CitationOut]
    refused: bool
    reason: str
