"""Tests for `src.core.config.Settings`."""

from __future__ import annotations

import pytest


def test_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override defaults when a fresh Settings is built."""
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("RAG_TOP_K", "12")
    monkeypatch.setenv("DEMO_TICKER", "CBA")

    from src.core.config import Settings

    s = Settings(_env_file=None)  # type: ignore[call-arg]

    assert s.llm_model == "llama3.1:8b"
    assert s.rag_top_k == 12
    assert s.demo_ticker == "CBA"


def test_settings_singleton_importable() -> None:
    """The singleton `settings` import works."""
    from src.core.config import settings

    assert settings.llm_timeout_seconds > 0
