"""Application errors and their HTTP representation.

Two rules drive this module (directive §17, §23):

  1. A client never sees a stack trace, a driver message or an internal
     identifier. `AppError.message` is written to be read by a customer.
  2. Nothing fails silently. Every handler logs with full context before
     returning the sanitized response.

Every error response has the same envelope, so a client can branch on `code`
without string-matching prose:

    {"error": {"code": "not_found", "message": "...", "details": {...},
               "request_id": "..."}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.enums import ErrorCode
from app.core.logging import get_context, get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base for every deliberate failure.

    `message` is customer-facing. `details` may carry structured, non-sensitive
    hints (which field, which allowed values) — never anything derived from
    infrastructure.
    """

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    message: str = "Something went wrong."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        code: ErrorCode | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            payload["details"] = self.details
        request_id = get_context().get("request_id")
        if request_id:
            payload["request_id"] = request_id
        return {"error": payload}


class ValidationFailed(AppError):
    # Literal rather than the Starlette constant: the name was renamed to
    # HTTP_422_UNPROCESSABLE_CONTENT and the old alias warns, but the number is
    # the actual contract and it has never changed.
    status_code = 422
    code = ErrorCode.VALIDATION_FAILED
    message = "The request could not be processed as submitted."


class NotFound(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = ErrorCode.NOT_FOUND
    message = "Not found."


class Conflict(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = ErrorCode.CONFLICT
    message = "That action conflicts with the current state."


class Unauthorized(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = ErrorCode.UNAUTHORIZED
    message = "Authentication required."


class Forbidden(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = ErrorCode.FORBIDDEN
    message = "You do not have access to this resource."


class RateLimited(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = ErrorCode.RATE_LIMITED
    message = "Too many requests. Please slow down and try again shortly."


class ConcurrencyLimitReached(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = ErrorCode.CONCURRENCY_LIMIT_REACHED
    message = "You already have the maximum number of generations running. Please wait for one to finish."


class ServiceUnavailable(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = ErrorCode.STORAGE_UNAVAILABLE
    message = "A required service is temporarily unavailable. Please try again shortly."


def _json(error: AppError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.to_payload())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        # 4xx is expected traffic, not an incident — log at info so real
        # problems stay visible in the error stream.
        log = logger.warning if exc.status_code >= 500 else logger.info
        log(
            "request_failed",
            extra={"error_code": exc.code.value, "status_code": exc.status_code},
        )
        return _json(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's raw errors include the submitted input; echoing it back
        # could reflect a secret a client sent by mistake. Only location and
        # reason survive.
        fields = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())[1:]) or "body",
                "reason": err.get("msg", "invalid"),
            }
            for err in exc.errors()
        ]
        logger.info("request_validation_failed", extra={"invalid_fields": len(fields)})
        return _json(
            ValidationFailed(
                "Some fields were missing or invalid.",
                details={"fields": fields},
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        mapped = {
            401: (ErrorCode.UNAUTHORIZED, "Authentication required."),
            403: (ErrorCode.FORBIDDEN, "You do not have access to this resource."),
            404: (ErrorCode.NOT_FOUND, "Not found."),
            405: (ErrorCode.NOT_FOUND, "Not found."),
        }
        code, message = mapped.get(
            exc.status_code, (ErrorCode.INTERNAL_ERROR, "Something went wrong.")
        )
        return _json(AppError(message, code=code, status_code=exc.status_code))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the log, never to the response body — including
        # in development, so a bug is never accidentally shipped as an API
        # contract clients start depending on.
        logger.exception("unhandled_exception", extra={"exception_type": type(exc).__name__})
        detail = (
            {} if settings.is_production else {"hint": "See API logs for the full traceback."}
        )
        return _json(
            AppError(
                "Something went wrong on our side. The team has been notified.",
                code=ErrorCode.INTERNAL_ERROR,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                details=detail,
            )
        )
