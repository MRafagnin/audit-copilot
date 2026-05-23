"""Tests for PDF parsing and chunking helpers."""

from __future__ import annotations

from pathlib import Path

from src.rag.ingest import (
    Chunk,
    _detect_section,
    _normalize,
    _split_text,
    chunk_pdf,
    write_chunks_jsonl,
)


def test_normalize_collapses_whitespace_and_nulls() -> None:
    out = _normalize("hello\x00  world\n\tfoo")
    assert out == "hello world foo"


def test_detect_section_returns_fallback_when_no_match() -> None:
    assert _detect_section("plain text with no heading", fallback="p.7") == "p.7"


def test_detect_section_returns_numbered_heading() -> None:
    assert _detect_section("12.3 Some heading goes here", fallback="p.1") == "Section 12.3"


def test_split_text_returns_overlapping_windows() -> None:
    text = "a" * 5000
    pieces = _split_text(text)

    assert len(pieces) > 1
    assert all(len(p) <= 512 * 4 for p in pieces)


def test_split_text_empty_input() -> None:
    assert _split_text("") == []


def _build_pdf(path: Path, pages: list[str]) -> None:
    """Write a tiny multi-page PDF using pypdf."""
    from pypdf import PdfWriter

    # Use reportlab-free approach: pypdf can't easily author a text PDF without
    # a backend, so we instead create blank pages and monkeypatch extract_text
    # via a fixture. For this happy-path test we rely on chunk_pdf via fake.
    writer = PdfWriter()
    for _ in pages:
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as fp:
        writer.write(fp)


def test_chunk_pdf_with_monkeypatched_reader(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "tiny.pdf"
    _build_pdf(pdf_path, ["page one text", "page two text"])

    fake_pages = [
        "1. Introduction lorem ipsum " * 30,
        "2.1 Risk discussion text " * 30,
    ]

    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [_FakePage(t) for t in fake_pages]

    monkeypatch.setattr("src.rag.ingest.PdfReader", _FakeReader)

    chunks = chunk_pdf(pdf_path, source="TEST")

    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "TEST" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert any("Section 1" in c.section or "Section 2" in c.section for c in chunks)


def test_write_chunks_jsonl_roundtrip(tmp_path) -> None:
    chunks = [
        Chunk(chunk_id="a:1:0", source="A", section="Section 1", page=1, text="hello"),
        Chunk(chunk_id="a:1:1", source="A", section="Section 2", page=1, text="world"),
    ]
    out_path = tmp_path / "out" / "chunks.jsonl"

    write_chunks_jsonl(chunks, out_path)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "hello" in lines[0]
    assert "world" in lines[1]
