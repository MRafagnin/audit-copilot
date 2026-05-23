"""Live RAG smoke test against Ollama using the built Chroma index.

Run after ``scripts/build_index.py``::

    uv run python scripts/smoke_rag.py "What does ASA 240 say about journal-entry testing?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.config import settings
from src.core.logging_config import configure_logging
from src.llm.client import OllamaClient
from src.rag.indexer import ChromaIndex
from src.rag.pipeline import RagPipeline


def _load_bm25_documents(jsonl_path: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            docs.append((row["chunk_id"], row["text"]))
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask one question against the live RAG pipeline.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What does ASA 240 say about journal-entry testing?",
    )
    parser.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    args = parser.parse_args(argv)

    configure_logging()
    index = ChromaIndex()
    bm25_docs = _load_bm25_documents(Path(args.chunks))
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    pipeline = RagPipeline(index=index, llm=llm, bm25_documents=bm25_docs)

    print(f"\nQ: {args.question}\n")
    result = pipeline.answer(args.question)

    print("=" * 80)
    print("ANSWER:")
    print(result.answer)
    print()
    if result.refused:
        print(f"[REFUSED] reason: {result.reason}")
        return 0
    print("CITATIONS:")
    for c in result.citations:
        print(f"  [{c.tag}] {c.source} | {c.section} | p.{c.page} | {c.chunk_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
