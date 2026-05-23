"""LLMClient protocol and the Ollama HTTP implementation.

The protocol is the single seam between domain logic and any specific LLM
provider. The Azure OpenAI implementation will live next to ``OllamaClient`` in
this module when we wire deployment.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when an LLM call fails after the client's internal retries."""


class LLMClient(Protocol):
    """Provider-agnostic interface for chat-completion-style generation."""

    def complete(self, *, system: str, user: str, temperature: float = 0.1) -> str:
        """Run a single completion.

        Args:
            system: System prompt (locked by caller; never user-controlled).
            user: Templated user prompt.
            temperature: Sampling temperature; default low for deterministic output.

        Returns:
            The model's text response.

        Raises:
            LLMClientError: When the underlying call fails.
        """
        ...


class OllamaClient:
    """LLMClient implementation backed by the local Ollama HTTP server.

    Calls the ``/api/chat`` endpoint in non-streaming mode.

    Attributes:
        base_url: Ollama server base URL (e.g. ``http://localhost:11434``).
        model: Ollama model tag to invoke.
        timeout_seconds: Per-request HTTP timeout.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Ollama HTTP base URL.
            model: Model tag, e.g. ``qwen2.5:7b-instruct``.
            timeout_seconds: Per-request timeout.
            client: Optional pre-built ``httpx.Client`` (used by tests to inject
                a transport). When omitted, a new client is created.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def complete(self, *, system: str, user: str, temperature: float = 0.1) -> str:
        """Generate a completion via the Ollama ``/api/chat`` endpoint.

        Args:
            system: Locked system prompt.
            user: Templated user prompt.
            temperature: Sampling temperature.

        Returns:
            The assistant message content as a plain string.

        Raises:
            LLMClientError: On HTTP error, network failure, or malformed response.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        url = f"{self.base_url}/api/chat"
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("ollama call failed", extra={"model": self.model})
            raise LLMClientError(f"ollama request failed: {exc}") from exc

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("ollama response malformed", extra={"model": self.model})
            raise LLMClientError(f"malformed ollama response: {exc}") from exc

        if not isinstance(content, str):
            raise LLMClientError("ollama response content was not a string")
        return content
