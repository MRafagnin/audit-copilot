"""Tests for `src.core.logging_config`."""

from __future__ import annotations

import json
import logging
from io import StringIO


def test_json_formatter_emits_valid_json() -> None:
    """The JSON formatter produces parseable single-line records."""
    from src.core.logging_config import _JsonFormatter

    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x",
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.resource_id = "abc-123"  # type: ignore[attr-defined]

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello"
    assert payload["resource_id"] == "abc-123"


def test_configure_logging_attaches_handler(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """configure_logging installs exactly one handler on the root logger."""
    from src.core.logging_config import configure_logging

    configure_logging()
    root = logging.getLogger()
    assert len(root.handlers) == 1

    buf = StringIO()
    root.handlers[0].stream = buf  # type: ignore[attr-defined]
    logging.getLogger("smoketest").info("rag pipeline started")
    assert "rag pipeline started" in buf.getvalue()
