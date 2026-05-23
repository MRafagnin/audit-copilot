"""Tests for the Ollama LLM client."""

from __future__ import annotations

import httpx
import pytest

from src.llm.client import LLMClientError, OllamaClient


def _client_with_handler(handler):
    """Build an OllamaClient backed by an httpx MockTransport."""
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return OllamaClient(
        base_url="http://localhost:11434",
        model="qwen2.5:7b-instruct",
        client=http_client,
    )


def test_complete_returns_assistant_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert b"hello" in request.content
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    client = _client_with_handler(handler)

    result = client.complete(system="sys", user="hello")

    assert result == "hi"


def test_complete_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client_with_handler(handler)

    with pytest.raises(LLMClientError):
        client.complete(system="sys", user="hello")


def test_complete_raises_on_malformed_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client_with_handler(handler)

    with pytest.raises(LLMClientError):
        client.complete(system="sys", user="hello")
