"""PDF parsing and chunking for the standards + annual-report corpus.

Inputs are PDFs already downloaded by ``scripts/fetch_corpus.py``. Each PDF is
parsed with ``pypdf``, normalised, and split into overlapping chunks using a
character-based recursive splitter sized to roughly ``CHUNK_SIZE_TOKENS``.

A simple 1-token ≈ 4-character heuristic is used because we want zero hard
dependencies on a tokenizer at ingest time. The downstream embedder handles
truncation if a chunk is slightly oversized.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from src.core.constants import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN: int = 4
_CHUNK_SIZE_CHARS: int = CHUNK_SIZE_TOKENS * _CHARS_PER_TOKEN
_CHUNK_OVERLAP_CHARS: int = CHUNK_OVERLAP_TOKENS * _CHARS_PER_TOKEN

_WHITESPACE_RE = re.compile(r"\s+")
_SECTION_RE = re.compile(r"^\s*((?:section\s+)?\d+(?:\.\d+)*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    """A single retrievable chunk with its source metadata.

    Attributes:
        chunk_id: Stable identifier; ``{source}:{doc_id}:{page}:{ordinal}``.
        source: Origin label (``AUASB``, ``IAASB``, ``ASX-<ticker>``).
        section: Best-effort section heading detected for the chunk.
        page: 1-based page number of the chunk's first character.
        text: Cleaned chunk text.
    """

    chunk_id: str
    source: str
    section: str
    page: int
    text: str


def _normalize(text: str) -> str:
    """Collapse whitespace and strip control characters from extracted text.

    Args:
        text: Raw text from a PDF page.

    Returns:
        Normalised single-spaced text.
    """
    cleaned = text.replace("\x00", " ")
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _detect_section(text: str, fallback: str) -> str:
    """Infer a section label from the first line of a chunk.

    Args:
        text: Chunk text.
        fallback: Section label to return when nothing matches.

    Returns:
        The first matching ``\\d+(\\.\\d+)*`` token, prefixed with ``Section``,
        or ``fallback`` when no heading-like pattern is found.
    """
    first_line = text.lstrip().split(" ", 4)[:4]
    candidate = " ".join(first_line)
    match = _SECTION_RE.match(candidate)
    if match is None:
        return fallback
    return f"Section {match.group(1)}"


def _split_text(text: str) -> list[str]:
    """Split a long string into overlapping character windows.

    Args:
        text: Cleaned input text.

    Returns:
        List of substrings of length ``_CHUNK_SIZE_CHARS`` (last may be
        shorter), each overlapping its predecessor by ``_CHUNK_OVERLAP_CHARS``.
    """
    if not text:
        return []
    step = max(1, _CHUNK_SIZE_CHARS - _CHUNK_OVERLAP_CHARS)
    out: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + _CHUNK_SIZE_CHARS]
        if piece:
            out.append(piece)
        if start + _CHUNK_SIZE_CHARS >= len(text):
            break
    return out


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    """Read a PDF and return ``(page_number, normalised_text)`` tuples.

    Args:
        path: Filesystem path to the PDF.

    Returns:
        One tuple per non-empty page, 1-indexed.
    """
    reader = PdfReader(str(path))
    out: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _normalize(raw)
        if text:
            out.append((idx, text))
    return out


def chunk_pdf(path: Path, *, source: str, doc_id: str | None = None) -> list[Chunk]:
    """Parse a PDF and produce chunked records for indexing.

    Args:
        path: Filesystem path to the PDF.
        source: Source label propagated into every chunk's metadata.
        doc_id: Per-document identifier used to make ``chunk_id`` unique when
            multiple PDFs share the same ``source`` (e.g. several AUASB
            standards). Defaults to the PDF's filename stem.

    Returns:
        List of :class:`Chunk` records covering every non-empty page.
    """
    pages = parse_pdf(path)
    doc = doc_id or path.stem
    out: list[Chunk] = []
    ordinal = 0
    for page_number, page_text in pages:
        for piece in _split_text(page_text):
            section = _detect_section(piece, fallback=f"p.{page_number}")
            out.append(
                Chunk(
                    chunk_id=f"{source}:{doc}:{page_number}:{ordinal}",
                    source=source,
                    section=section,
                    page=page_number,
                    text=piece,
                )
            )
            ordinal += 1
    return out


def write_chunks_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    """Persist chunks to a JSONL file.

    Args:
        chunks: Chunks to write.
        out_path: Output path. Parent directories are created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    logger.info("chunks written", extra={"count": len(chunks), "path": str(out_path)})
