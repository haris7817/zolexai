"""Media upload, confirmation, download and listing.

The upload architecture is the point of this module (directive §13):

    Browser ──presigned PUT──▶ Object storage

Bytes never pass through Next.js or FastAPI. The API issues a short-lived signed
URL and records a `pending` row; the browser uploads directly; a confirm call
verifies what landed and marks the asset usable.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.deps import CurrentUser, DbSession, Pagination, RedisClient
from app.core.config import settings
from app.core.enums import AssetKind, AssetSource, AssetStatus
from app.core.errors import ValidationFailed
from app.core.logging import get_logger
from app.schemas.asset import (
    AssetConfirmRequest,
    AssetPublic,
    DownloadUrlResponse,
    MediaCounts,
    PresignedUploadResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from app.schemas.common import Page
from app.services import rate_limit
from app.services.storage import AssetService

logger = get_logger(__name__)
router = APIRouter(tags=["media"])

_UPLOAD_RATE_LIMIT = 60
_UPLOAD_RATE_WINDOW = 60


@router.post(
    "/assets/upload-url",
    response_model=UploadUrlResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Get a direct-upload URL",
)
async def create_upload_url(
    payload: UploadUrlRequest,
    session: DbSession,
    redis: RedisClient,
    user: CurrentUser,
    response: Response,
) -> UploadUrlResponse:
    """Validates the declared type and size, then issues a presigned PUT.

    Rate-limited because issuing URLs is cheap for us and expensive to abuse:
    each one is a writable handle into the bucket for its lifetime.
    """
    await rate_limit.check_request_rate(
        redis,
        user.id,
        action="upload_url",
        limit=_UPLOAD_RATE_LIMIT,
        window_seconds=_UPLOAD_RATE_WINDOW,
    )

    service = AssetService(session)
    asset, presigned = await service.create_upload(
        user_id=user.id,
        filename=payload.filename,
        content_type=payload.content_type,
        kind=payload.kind,
        size_bytes=payload.size_bytes,
    )
    await session.commit()

    response.headers["Cache-Control"] = "no-store"
    return UploadUrlResponse(
        asset_id=asset.id,
        upload=PresignedUploadResponse(
            url=presigned.url,
            method=presigned.method,
            headers=presigned.headers,
            expires_in=presigned.expires_in,
        ),
        confirm_url=f"{settings.api_v1_prefix}/assets/{asset.id}/confirm",
    )


@router.post(
    "/assets/{asset_id}/confirm", response_model=AssetPublic, summary="Confirm an upload"
)
async def confirm_upload(
    asset_id: uuid.UUID,
    payload: AssetConfirmRequest,
    session: DbSession,
    user: CurrentUser,
    response: Response,
) -> AssetPublic:
    """Verifies the object landed and matches its declared size, then marks it
    ready. Until this succeeds the asset cannot be used as a generation input."""
    service = AssetService(session)
    asset = await service.require_for_user(asset_id, user.id)
    confirmed = await service.confirm_upload(
        asset=asset,
        duration_seconds=payload.duration_seconds,
        width=payload.width,
        height=payload.height,
    )
    await session.commit()
    response.headers["Cache-Control"] = "no-store"
    return service.to_public(confirmed)


@router.post(
    "/assets/{asset_id}/download-url",
    response_model=DownloadUrlResponse,
    summary="Get a download URL",
)
async def create_download_url(
    asset_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
    response: Response,
) -> DownloadUrlResponse:
    """Issues a short-lived signed GET.

    A signed URL, not a public object: the bucket stays private, so a leaked
    link expires instead of exposing a user's media indefinitely.
    """
    service = AssetService(session)
    asset = await service.require_for_user(asset_id, user.id)

    if asset.status != AssetStatus.READY:
        raise ValidationFailed("That media item is not ready yet.")

    response.headers["Cache-Control"] = "no-store"
    return DownloadUrlResponse(
        url=service.download_url(asset, as_attachment=True),
        expires_in=settings.storage_presign_expiry_seconds,
    )


@router.get("/media", response_model=Page[AssetPublic], summary="Media library")
async def list_media(
    session: DbSession,
    user: CurrentUser,
    page: Pagination,
    response: Response,
    kind: Annotated[AssetKind | None, Query()] = None,
    source: Annotated[AssetSource | None, Query()] = None,
) -> Page[AssetPublic]:
    service = AssetService(session)
    assets, next_cursor, has_more = await service.repo.list_for_user(
        user.id, page=page, kind=kind, source=source
    )
    response.headers["Cache-Control"] = "no-store"
    return Page[AssetPublic](
        items=[service.to_public(asset) for asset in assets],
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get("/media/counts", response_model=MediaCounts, summary="Media tab counts")
async def media_counts(session: DbSession, user: CurrentUser) -> MediaCounts:
    """Counts per kind, in one grouped query rather than four requests."""
    service = AssetService(session)
    counts = await service.repo.count_by_kind(user.id)
    return MediaCounts(
        all=sum(counts.values()),
        video=counts.get("video", 0),
        image=counts.get("image", 0),
        audio=counts.get("audio", 0),
    )
