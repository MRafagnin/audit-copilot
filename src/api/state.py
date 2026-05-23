"""Lazy singleton factories for API dependencies.

The factories are cached so the heavy components (embedder, Chroma client,
HTTP client) are constructed once per process. Each factory is exposed as a
FastAPI dependency via ``Depends(...)`` in :mod:`src.api.main`, which lets
tests override them with fakes through ``app.dependency_overrides``.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from src.core.config import settings
from src.fusion.explain import AnomalyExplainer
from src.llm.client import OllamaClient
from src.rag.indexer import ChromaIndex
from src.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)

FLAGGED_CSV_PATH = Path("data/flagged/top_k.csv")
CHUNKS_JSONL_PATH = Path("data/chunks/chunks.jsonl")


@lru_cache(maxsize=1)
def get_bm25_documents() -> list[tuple[str, str, str]]:
    """Load ``(chunk_id, source, text)`` triples from the persisted chunks file."""
    if not CHUNKS_JSONL_PATH.exists():
        logger.warning("chunks jsonl missing", extra={"path": str(CHUNKS_JSONL_PATH)})
        return []
    docs: list[tuple[str, str, str]] = []
    with CHUNKS_JSONL_PATH.open("r", encoding="utf-8") as fp:
        for line in fp:
            obj = json.loads(line)
            docs.append((obj["chunk_id"], obj.get("source", ""), obj["text"]))
    logger.info("bm25 corpus loaded", extra={"docs": len(docs)})
    return docs


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Build the shared RAG pipeline (Chroma + Ollama + BM25)."""
    index = ChromaIndex()
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    return RagPipeline(index=index, llm=llm, bm25_documents=get_bm25_documents())


@lru_cache(maxsize=1)
def get_explainer() -> AnomalyExplainer:
    """Build the shared anomaly explainer over the cached pipeline."""
    return AnomalyExplainer(pipeline=get_pipeline())


@lru_cache(maxsize=1)
def get_flagged_df() -> pd.DataFrame:
    """Load the flagged-transactions table; empty when the file is missing."""
    if not FLAGGED_CSV_PATH.exists():
        logger.warning("flagged csv missing", extra={"path": str(FLAGGED_CSV_PATH)})
        return pd.DataFrame()
    df = pd.read_csv(FLAGGED_CSV_PATH)
    logger.info("flagged rows loaded", extra={"rows": len(df)})
    return df


def reset_caches() -> None:
    """Drop cached singletons. Used by tests."""
    get_bm25_documents.cache_clear()
    get_pipeline.cache_clear()
    get_explainer.cache_clear()
    get_flagged_df.cache_clear()
