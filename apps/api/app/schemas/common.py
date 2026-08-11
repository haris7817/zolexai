"""Pagination and shared response envelopes.

Keyset ("cursor") pagination, not OFFSET. On a table the directive expects to
grow very large, `OFFSET 10000` makes PostgreSQL walk and discard ten thousand
rows on every page; a keyset seek jumps straight to the position using the
`(user_id, created_at DESC)` index. Cost stays flat no matter how deep the user
scrolls (directive §5).

The cursor is opaque on purpose — clients must treat it as a token, so its
encoding can change without a breaking API version.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import ValidationFailed

T = TypeVar("T")


class PageParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    limit: int = Field(default=24, ge=1, le=100)
    cursor: str | None = None

    def decoded(self) -> tuple[datetime, uuid.UUID] | None:
        """The (created_at, id) position to resume after.

        `id` breaks ties: two rows created in the same microsecond would
        otherwise make the sort unstable and a page could repeat or skip a row.
        """
        if not self.cursor:
            return None
        try:
            raw = base64.urlsafe_b64decode(self.cursor.encode()).decode()
            data = json.loads(raw)
            return datetime.fromisoformat(data["t"]), uuid.UUID(data["i"])
        except (
            binascii.Error,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            raise ValidationFailed("That page cursor is not valid.") from exc


def encode_cursor(created_at: datetime, item_id: uuid.UUID) -> str:
    payload = json.dumps({"t": created_at.isoformat(), "i": str(item_id)}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    """Absent when this is the last page. Clients must not construct one."""

    has_more: bool = False

    # Deliberately no `total`. Counting a large filtered history is a full scan
    # on every page request; the UI shows "load more" rather than "page 7 of
    # 214", so nothing pays for a number nobody reads.


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """The single error envelope every failure uses. Documented in OpenAPI."""

    error: ErrorDetail


class AcceptedResponse(BaseModel):
    """202 body for work that continues after the response."""

    job_id: uuid.UUID
    status: str
