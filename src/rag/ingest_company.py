"""On-demand ingest of a single company annual report.

Used by the API endpoint ``POST /companies/{ticker}/ingest`` and the
Streamlit sidebar to pull a ticker not yet present in the index.

The function is idempotent: if Chroma already contains chunks for the
ticker's source label, it is a no-op.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.rag._http import download_pdf
from src.rag.indexer import ChromaIndex
from src.rag.ingest import chunk_pdf
from src.rag.registry import get_entry, pdf_path_for, source_label

logger = logging.getLogger(__name__)

CHUNKS_JSONL_PATH = Path("data/chunks/chunks.jsonl")


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingest call.

    Attributes:
        ticker: Uppercase ticker that was ingested.
        chunks_added: Number of new chunks written to the index. Zero when
            the ticker was already indexed.
        took_ms: Wall-clock duration in milliseconds.
        cached: True when the call was a no-op because chunks already existed.
    """

    ticker: str
    chunks_added: int
    took_ms: int
    cached: bool


def _existing_chunk_count(index: ChromaIndex, source: str) -> int:
    """Return the number of chunks already indexed under ``source``."""
    collection: Any = index.collection
    try:
        result = collection.get(where={"source": source}, limit=1, include=[])
    except TypeError:
        result = collection.get(where={"source": source}, limit=1)
    ids = result.get("ids", []) if isinstance(result, dict) else []
    return len(ids)


def _append_chunks_jsonl(chunks: list[Any], out_path: Path) -> None:
    """Append serialised chunks to the JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def ingest_company(
    ticker: str,
    *,
    index: ChromaIndex | None = None,
    chunks_path: Path = CHUNKS_JSONL_PATH,
) -> IngestResult:
    """Fetch, chunk, embed, and upsert one company's annual report.

    Args:
        ticker: Allowlist ticker (case-insensitive).
        index: Optional pre-built Chroma index (used by tests). When omitted,
            constructs the default persistent index.
        chunks_path: JSONL file to append new chunk records to.

    Returns:
        :class:`IngestResult` describing what happened.

    Raises:
        KeyError: When ``ticker`` is not in the allowlist.
    """
    ticker_upper = ticker.upper()
    entry = get_entry(ticker_upper)
    source = source_label(ticker_upper)
    start = time.perf_counter()

    if index is None:
        index = ChromaIndex()

    existing = _existing_chunk_count(index, source)
    if existing > 0:
        took_ms = int((time.perf_counter() - start) * 1000)
        logger.info("ingest cached", extra={"ticker": ticker_upper, "source": source})
        return IngestResult(ticker=ticker_upper, chunks_added=0, took_ms=took_ms, cached=True)

    pdf_path = pdf_path_for(ticker_upper)
    download_pdf(
        entry.url,
        pdf_path,
        user_agent=settings.http_user_agent,
        label=f"{ticker_upper}-annual-report",
    )

    chunks = chunk_pdf(pdf_path, source=source, doc_id=f"{ticker_upper}-annual-report")
    if not chunks:
        took_ms = int((time.perf_counter() - start) * 1000)
        logger.warning("ingest produced no chunks", extra={"ticker": ticker_upper})
        return IngestResult(ticker=ticker_upper, chunks_added=0, took_ms=took_ms, cached=False)

    index.add(chunks)
    _append_chunks_jsonl(chunks, chunks_path)

    took_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "ingest completed",
        extra={"ticker": ticker_upper, "chunks_added": len(chunks), "took_ms": took_ms},
    )
    return IngestResult(
        ticker=ticker_upper, chunks_added=len(chunks), took_ms=took_ms, cached=False
    )
