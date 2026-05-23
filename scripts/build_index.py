"""Chunk every PDF under ``data/corpus/`` and upsert into ChromaDB.

Run after ``scripts/fetch_corpus.py``::

    uv run python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core.logging_config import configure_logging
from src.rag.indexer import ChromaIndex
from src.rag.ingest import Chunk, chunk_pdf, write_chunks_jsonl

logger = logging.getLogger(__name__)


def _source_label_for(pdf_path: Path, corpus_root: Path) -> str:
    """Derive a source label from a PDF's location inside the corpus tree.

    Args:
        pdf_path: Path to the PDF file.
        corpus_root: Root of ``data/corpus``.

    Returns:
        Label like ``AUASB``, ``IAASB``, or ``ASX-WOW``.
    """
    rel = pdf_path.relative_to(corpus_root)
    category = rel.parts[0].lower()
    if category == "asx" and len(rel.parts) >= 3:
        ticker = rel.parts[1].upper()
        return f"ASX-{ticker}"
    return category.upper()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Chunk corpus PDFs and index in ChromaDB.")
    parser.add_argument("--corpus", default="data/corpus", help="Corpus root directory.")
    parser.add_argument(
        "--jsonl",
        default="data/chunks/chunks.jsonl",
        help="Where to write the combined chunk JSONL for inspection.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    corpus_root = Path(args.corpus)
    pdfs = sorted(corpus_root.rglob("*.pdf"))
    if not pdfs:
        logger.error("no corpus pdfs found", extra={"corpus": str(corpus_root)})
        return 2

    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        source = _source_label_for(pdf, corpus_root)
        logger.info("chunking pdf", extra={"path": str(pdf), "source": source})
        chunks = chunk_pdf(pdf, source=source)
        all_chunks.extend(chunks)
        logger.info("chunked pdf", extra={"path": str(pdf), "count": len(chunks)})

    write_chunks_jsonl(all_chunks, Path(args.jsonl))

    logger.info("loading embedder and opening chroma collection")
    index = ChromaIndex()
    index.add(all_chunks)
    logger.info("index build complete", extra={"chunks": len(all_chunks), "count": index.count()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
