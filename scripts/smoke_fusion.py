"""Smoke test: generate grounded narratives for top-flagged journal entries.

Loads the top-k flagged rows persisted by ``scripts/train_anomaly.py``, builds
``FlaggedTransaction`` records from them, and asks the live LLM (via the same
``RagPipeline`` used for question-answering) to explain each one with
citations to the AUASB ASA corpus.

Run::

    uv run python scripts/smoke_fusion.py --rows 3
"""

# ruff: noqa: T201  # this is a CLI smoke test; print is the interface.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.core.config import settings
from src.core.logging_config import configure_logging
from src.fusion.explain import AnomalyExplainer, flagged_transaction_from_row
from src.llm.client import OllamaClient
from src.rag.indexer import ChromaIndex
from src.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Run the fusion explainer smoke test.")
    parser.add_argument("--flagged", default="data/flagged/top_k.csv")
    parser.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args(argv)

    configure_logging()
    flagged_path = Path(args.flagged)
    if not flagged_path.exists():
        logger.error("flagged csv missing", extra={"path": str(flagged_path)})
        return 1

    df = pd.read_csv(flagged_path).head(args.rows)
    logger.info("flagged rows loaded", extra={"rows": len(df), "path": str(flagged_path)})

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        logger.error("chunks jsonl missing", extra={"path": str(chunks_path)})
        return 1
    bm25_docs: list[tuple[str, str, str]] = []
    with chunks_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            obj = json.loads(line)
            bm25_docs.append((obj["chunk_id"], obj.get("source", ""), obj["text"]))
    logger.info("bm25 corpus loaded", extra={"docs": len(bm25_docs)})

    index = ChromaIndex()
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    pipeline = RagPipeline(index=index, llm=llm, bm25_documents=bm25_docs)
    explainer = AnomalyExplainer(pipeline=pipeline)

    for _, row in df.iterrows():
        tx = flagged_transaction_from_row(row.to_dict())
        print("=" * 72)
        print(f"tx_id           : {tx.tx_id}")
        print(f"account         : {tx.account}")
        print(f"amount          : {max(tx.debit, tx.credit):.2f}")
        print(f"posting_ts      : {tx.posting_ts}")
        print(f"ensemble_score  : {tx.ensemble_score:.3f}")
        print(f"feature_flags   : {', '.join(tx.feature_flags) or '(none)'}")
        result = explainer.explain(tx)
        if result.refused:
            print(f"REFUSED: {result.reason}")
            print(result.narrative)
            continue
        print("\nNARRATIVE:")
        print(result.narrative)
        print("\nCITATIONS:")
        for c in result.citations:
            print(f"  [{c.tag}] {c.source} | {c.section} | {c.chunk_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
