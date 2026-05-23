"""Tests for prompt formatting."""

from __future__ import annotations

from src.rag.prompts import (
    RAG_SYSTEM_PROMPT,
    ContextChunk,
    format_context_block,
    format_user_prompt,
)


def _chunks() -> list[ContextChunk]:
    return [
        ContextChunk(tag=1, source="AUASB", section="Section 5", page=12, text="alpha"),
        ContextChunk(tag=2, source="ASX-WOW", section="Risk", page=None, text="beta"),
    ]


def test_format_context_block_includes_tags_and_metadata() -> None:
    block = format_context_block(_chunks())

    assert "[1] AUASB | Section 5 | p.12" in block
    assert "[2] ASX-WOW | Risk" in block
    assert "p.None" not in block
    assert "alpha" in block and "beta" in block


def test_format_user_prompt_embeds_question_and_context() -> None:
    out = format_user_prompt("what is ASA 240?", _chunks())

    assert "CONTEXT:" in out
    assert "QUESTION:" in out
    assert "what is ASA 240?" in out
    assert "[1]" in out
    assert "Cite chunks as [n]" in out


def test_system_prompt_locks_refusal_phrase() -> None:
    assert "INSUFFICIENT_GROUNDING" in RAG_SYSTEM_PROMPT
