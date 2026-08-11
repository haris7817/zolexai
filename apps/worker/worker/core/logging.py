"""Structured logging for the worker.

Deliberately a copy of the API's approach rather than a shared import: the
worker does not depend on `apps/api`, and giving it one would couple two
independently deployable services over thirty lines of formatting. The output
shape is identical, so both services' logs aggregate together.

Correlation identifiers — `worker_id`, `job_id`, `attempt` — ride in
contextvars, so a log call inside an adapter is attributable without threading
arguments through it.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from worker.core.config import settings

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "zolexai_worker_context", default={}
)

_REDACTED = frozenset({"token", "secret", "password", "authorization", "lease_token", "url"})


@contextmanager
def bind(**values: Any) -> Iterator[None]:
    safe = {k: v for k, v in values.items() if v is not None and k not in _REDACTED}
    token = _context.set({**_context.get(), **safe})
    try:
        yield
    finally:
        _context.reset(token)


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
            "service": "worker",
        }
        payload.update(_context.get())
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key.lower() not in _REDACTED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        suffix = " ".join(f"{k}={v}" for k, v in _context.get().items())
        line = f"{record.levelname:<7} {record.name:<26} {record.getMessage()}"
        if suffix:
            line += f"  {suffix}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_format == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    # httpx logs every request at INFO, including presigned URLs — which carry
    # a signature and would be a credential in the log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
