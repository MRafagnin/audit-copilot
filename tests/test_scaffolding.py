"""Smoke tests for project scaffolding."""

from __future__ import annotations

import src
from src.core.config import settings
from src.core.constants import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, EMBED_DIM
from src.llm.client import LLMClientError, OllamaClient


def test_package_version() -> None:
    """The package exposes a version string."""
    assert isinstance(src.__version__, str)
    assert src.__version__


def test_settings_defaults_loaded() -> None:
    """Settings load with sensible defaults even without a .env file."""
    assert settings.llm_model == "qwen2.5:7b-instruct"
    assert settings.demo_ticker == "WOW"
    assert 0.0 < settings.rag_min_score < 1.0
    assert settings.rag_top_k > 0


def test_constants_sane() -> None:
    """Chunking and embedding constants are internally consistent."""
    assert CHUNK_OVERLAP_TOKENS < CHUNK_SIZE_TOKENS
    assert EMBED_DIM == 384


def test_ollama_client_constructs() -> None:
    """The Ollama client stub accepts the documented constructor arguments."""
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:7b-instruct")
    assert client.base_url == "http://localhost:11434"
    assert client.model == "qwen2.5:7b-instruct"
    assert client.timeout_seconds == 120


def test_llm_error_is_runtime_error() -> None:
    """LLMClientError remains a RuntimeError subclass for broad except clauses."""
    assert issubclass(LLMClientError, RuntimeError)
