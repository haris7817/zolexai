"""Public contracts for media upload, download and listing."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AssetKind, AssetSource, AssetStatus


class UploadUrlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=120)
    kind: AssetKind
    size_bytes: int = Field(ge=1)
    """
    Declared up front so an oversized upload is refused BEFORE a URL is issued.
    It is a claim, not proof — `POST /assets/{id}/confirm` re-reads the real
    size from storage and rejects a mismatch.
    """


class PresignedUploadResponse(BaseModel):
    url: str
    method: str
    headers: dict[str, str]
    """Must be sent verbatim: Content-Type is part of the signature."""
    expires_in: int


class UploadUrlResponse(BaseModel):
    asset_id: uuid.UUID
    upload: PresignedUploadResponse
    confirm_url: str
    """The client calls this after the PUT succeeds; until then the asset is
    `pending` and cannot be used as a generation input."""


class AssetConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class AssetPublic(BaseModel):
    id: uuid.UUID
    kind: AssetKind
    source: AssetSource
    status: AssetStatus

    name: str
    content_type: str
    size_bytes: int | None
    duration_seconds: float | None
    width: int | None
    height: int | None

    created_at: datetime

    url: str | None = None
    """Presigned GET, present on list and detail. Short-lived by design."""


class DownloadUrlResponse(BaseModel):
    url: str
    expires_in: int


class MediaCounts(BaseModel):
    """Tab counts for the media library, computed in one grouped query."""

    all: int = 0
    video: int = 0
    image: int = 0
    audio: int = 0
