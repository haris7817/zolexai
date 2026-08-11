"""HTTP middleware — correlation, access logging and security headers."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.core.logging import bind, get_logger

logger = get_logger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"

#: Health checks would otherwise dominate the log at container-scheduler cadence.
_QUIET_PATHS = frozenset({"/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, binds it for logging, and emits one access line.

    An inbound `X-Request-ID` is honoured so a trace survives a proxy or a call
    originating in the frontend; otherwise one is generated. It is always echoed
    back, which is what makes a user-reported failure findable in the logs.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        started = time.perf_counter()
        with bind(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                # Logged here as well as in the handler because a failure inside
                # streaming (SSE) never reaches an exception handler.
                logger.exception(
                    "request_errored",
                    extra={"method": request.method, "path": request.url.path},
                )
                raise

            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers[REQUEST_ID_HEADER] = request_id

            if request.url.path not in _QUIET_PATHS:
                logger.info(
                    "request_completed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": elapsed_ms,
                    },
                )
            return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative defaults for a JSON API.

    No CSP: this service returns JSON and SSE, never HTML, so a policy here
    would protect nothing. The frontend sets its own.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        if settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response
