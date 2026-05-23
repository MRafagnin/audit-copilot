"""Input-side safety layer for the RAG pipeline.

Two responsibilities:

* **Prompt-injection detection** — regex heuristics against common jailbreak
  phrases. The LLM-as-judge path is documented but not invoked here to keep
  the request cheap and deterministic.
* **PII scrubbing** — replace emails, Australian TFNs, credit-card numbers,
  and obvious phone numbers with placeholder tokens before the input reaches
  the LLM or any log line.

Both functions are pure and synchronous so callers can compose them freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?(system|previous)\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+", re.IGNORECASE),
    re.compile(r"(developer|admin|root)\s+mode", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(system|assistant)\s*>", re.IGNORECASE),
)

_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_TFN_PATTERN = re.compile(r"\b\d{3}\s?\d{3}\s?\d{3}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?61|0)[\s-]?[2-478](?:[\s-]?\d){8}\b")

_MAX_QUESTION_LEN: int = 2000


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of running guardrails on a user question.

    Attributes:
        ok: True when the question may proceed to retrieval and generation.
        reason: Human-readable refusal reason when ``ok`` is False, else empty.
        sanitized: PII-scrubbed question text. Equal to the original input
            when no PII was found.
    """

    ok: bool
    reason: str
    sanitized: str


def detect_prompt_injection(text: str) -> bool:
    """Return True when the input matches any known injection heuristic.

    Args:
        text: Raw user input.

    Returns:
        True if any pattern matched.
    """
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def scrub_pii(text: str) -> str:
    """Replace common PII fragments with placeholder tokens.

    The replacements are conservative — false positives are preferred over
    leaking real identifiers. Order matters: cards before phones so 16-digit
    card numbers aren't mistaken for phone runs.

    Args:
        text: Raw user input.

    Returns:
        Input string with PII fragments replaced by ``[REDACTED_*]`` tokens.
    """
    text = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = _CARD_PATTERN.sub("[REDACTED_CARD]", text)
    text = _TFN_PATTERN.sub("[REDACTED_TFN]", text)
    text = _PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    return text


def check_question(text: str) -> GuardrailResult:
    """Run all input-side guardrails against a user question.

    Args:
        text: Raw user input.

    Returns:
        GuardrailResult describing whether the question may proceed and, if
        so, the sanitized form to use downstream.
    """
    stripped = text.strip()
    if not stripped:
        return GuardrailResult(ok=False, reason="empty question", sanitized="")
    if len(stripped) > _MAX_QUESTION_LEN:
        return GuardrailResult(ok=False, reason="question too long", sanitized="")
    if detect_prompt_injection(stripped):
        return GuardrailResult(ok=False, reason="possible prompt injection", sanitized="")
    return GuardrailResult(ok=True, reason="", sanitized=scrub_pii(stripped))
