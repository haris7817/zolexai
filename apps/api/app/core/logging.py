"""Structured logging.

Every log line is one JSON object carrying whatever correlation identifiers are
in scope — `request_id`, `job_id`, `user_id`, `worker_id` (directive §19). Those
travel in `contextvars`, so a handler sets them once and every downstream log
call inside that request or job inherits them without threading arguments
through the call stack.

No secrets are ever logged: `bind()` only receives identifiers, and the record
formatter emits a fixed set of fields plus `extra`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.core.config import settings

_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "zolexai_log_context", default={}
)

#: Never emitted, whatever a caller passes.
_REDACTED_KEYS = frozenset(
    {
        "password",
        "secret",
        "token",
        "authorization",
        "api_key",
        "access_key",
        "secret_key",
        "session",
        "cookie",
        "lease_token",
    }
)


def get_context() -> dict[str, Any]:
    return dict(_log_context.get())


@contextmanager
def bind(**values: Any) -> Iterator[None]:
    """Adds correlation identifiers for the duration of the block.

    Nested binds merge rather than replace, so a job handler inside a request
    keeps the request_id.
    """
    safe = {k: v for k, v in values.items() if v is not None and k not in _REDACTED_KEYS}
    token = _log_context.set({**_log_context.get(), **safe})
    try:
        yield
    finally:
        _log_context.reset(token)


def bind_context(**values: Any) -> None:
    """Adds identifiers for the rest of the current task, with no unwind.

    Use inside a request-scoped dependency, where there is no block to wrap and
    no need to restore: each request runs in its own asyncio task with its own
    copied context, so the values die with the request. `bind()` is the right
    tool anywhere a scope genuinely ends.
    """
    safe = {k: v for k, v in values.items() if v is not None and k not in _REDACTED_KEYS}
    _log_context.set({**_log_context.get(), **safe})


_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(get_context())

        for key, value in record.__dict__.items():
            if key not in _RESERVED and key.lower() not in _REDACTED_KEYS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable local alternative. Same data, easier to scan."""

    def format(self, record: logging.LogRecord) -> str:
        context = get_context()
        suffix = " ".join(f"{k}={v}" for k, v in context.items())
        base = f"{record.levelname:<7} {record.name:<28} {record.getMessage()}"
        line = f"{base}  {suffix}" if suffix else base
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging() -> None:
    """Installs the formatter on the root logger. Idempotent."""
    formatter: logging.Formatter = (
        JsonFormatter() if settings.log_format == "json" else ConsoleFormatter()
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # uvicorn ships its own handlers; drop them so everything goes through ours
    # and every line is structured.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # The access log is emitted by our own middleware with correlation ids
    # attached, so uvicorn's duplicate is silenced.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
