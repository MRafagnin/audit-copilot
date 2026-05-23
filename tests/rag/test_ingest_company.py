"""Tests for the on-demand company ingest path."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.rag.ingest import Chunk
from src.rag.ingest_company import ingest_company


def _stub_chunks(source: str = "ASX-WOW") -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{source}:annual:1:0", source=source, section="p.1", page=1, text="alpha"),
        Chunk(chunk_id=f"{source}:annual:1:1", source=source, section="p.1", page=1, text="beta"),
    ]


@pytest.fixture
def chunks_path(tmp_path: Path) -> Path:
    return tmp_path / "chunks.jsonl"


def test_ingest_company_downloads_chunks_and_upserts(chunks_path: Path) -> None:
    index = Mock()
    index.collection.get.return_value = {"ids": []}

    with (
        patch("src.rag.ingest_company.download_pdf", return_value=True) as mock_dl,
        patch("src.rag.ingest_company.chunk_pdf", return_value=_stub_chunks()),
    ):
        result = ingest_company("WOW", index=index, chunks_path=chunks_path)

    mock_dl.assert_called_once()
    index.add.assert_called_once()
    assert result.chunks_added == 2
    assert result.cached is False
    assert result.ticker == "WOW"

    lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["source"] == "ASX-WOW"


def test_ingest_company_idempotent_when_already_indexed(chunks_path: Path) -> None:
    """When Chroma already has the source, ingest is a no-op."""
    index = Mock()
    index.collection.get.return_value = {"ids": ["existing-id"]}

    with (
        patch("src.rag.ingest_company.download_pdf") as mock_dl,
        patch("src.rag.ingest_company.chunk_pdf") as mock_chunk,
    ):
        result = ingest_company("WOW", index=index, chunks_path=chunks_path)

    mock_dl.assert_not_called()
    mock_chunk.assert_not_called()
    index.add.assert_not_called()
    assert result.cached is True
    assert result.chunks_added == 0


def test_ingest_company_rejects_unknown_ticker(chunks_path: Path) -> None:
    with pytest.raises(KeyError):
        ingest_company("ZZZ", index=Mock(), chunks_path=chunks_path)
