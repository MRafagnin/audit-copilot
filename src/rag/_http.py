"""Shared PDF downloader used by corpus fetch and on-demand ingest.

Extracted from ``scripts/fetch_corpus.py`` so ``src/rag/ingest_company.py``
can reuse it without importing the CLI script (which would create a
``scripts -> src -> scripts`` cycle).
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def download_pdf(url: str, out_path: Path, *, user_agent: str, label: str | None = None) -> bool:
    """Download a PDF to ``out_path`` unless it is already cached.

    Args:
        url: Direct HTTPS URL to the PDF.
        out_path: Destination file path.
        user_agent: Value for the ``User-Agent`` header.
        label: Optional short identifier used in log lines.

    Returns:
        True when a new download occurred, False when the cached file was kept.
    """
    log_label = label or out_path.stem
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("corpus cached", extra={"label": log_label, "path": str(out_path)})
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("corpus downloading", extra={"label": log_label, "url": url})
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=60, stream=True)
    response.raise_for_status()
    with out_path.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                fp.write(chunk)
    logger.info("corpus downloaded", extra={"label": log_label, "bytes": out_path.stat().st_size})
    return True
