"""Asset lifecycle — presigned uploads, confirmation, downloads.

The upload is two-phase, and the second phase is not a formality:

  1. `POST /assets/upload-url` validates the declared type and size, writes a
     `pending` asset row, and returns a presigned PUT.
  2. The browser uploads directly to object storage.
  3. `POST /assets/{id}/confirm` re-reads the object's real size and content
     type from storage and flips the row to `ready`.

Step 3 exists because everything in step 1 is a *claim*. A client can declare
`image/png` and 2 MB and upload a 4 GB file. Storage enforces the signed
Content-Type, and the confirm step enforces the size — against what actually
landed, not what was promised. Only `ready` assets can be used as generation
inputs, so nothing unverified reaches a worker.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import AssetKind, AssetSource, AssetStatus, ErrorCode
from app.core.errors import AppError, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.integrations.storage.s3 import get_storage
from app.models.asset import Asset
from app.repositories.asset import AssetRepository
from app.schemas.asset import AssetPublic

logger = get_logger(__name__)

#: Type allowlist per media kind.
#:
#: An allowlist, not a blocklist: an unknown type is refused rather than passed
#: through to a worker that has no idea what to do with it. Extending it is a
#: deliberate edit here.
ALLOWED_CONTENT_TYPES: dict[AssetKind, frozenset[str]] = {
    AssetKind.VIDEO: frozenset({"video/mp4", "video/quicktime", "video/webm"}),
    AssetKind.IMAGE: frozenset({"image/jpeg", "image/png", "image/webp"}),
    AssetKind.AUDIO: frozenset(
        {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg"}
    ),
}

#: Hard ceilings, independent of anything a workflow declares. A workflow may
#: be stricter; it can never be more permissive.
MAX_SIZE_BYTES: dict[AssetKind, int] = {
    AssetKind.VIDEO: 1024 * 1024 * 1024,  # 1 GB
    AssetKind.IMAGE: 50 * 1024 * 1024,
    AssetKind.AUDIO: 200 * 1024 * 1024,
}

_EXTENSION_BY_TYPE = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """Reduces a user-supplied name to something safe to embed in a storage key.

    Path separators and traversal sequences are stripped rather than escaped:
    a key is not a filesystem path, and `../` in one has no legitimate use.
    """
    cleaned = _UNSAFE.sub("_", name.strip().replace("..", "_"))
    cleaned = cleaned.strip("._-") or "file"
    return cleaned[:120]


def validate_upload(kind: AssetKind, content_type: str, size_bytes: int) -> None:
    normalized = content_type.split(";")[0].strip().lower()

    if normalized not in ALLOWED_CONTENT_TYPES[kind]:
        raise ValidationFailed(
            "That file type is not supported.",
            code=ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            details={
                "content_type": normalized,
                "allowed": sorted(ALLOWED_CONTENT_TYPES[kind]),
            },
        )

    ceiling = MAX_SIZE_BYTES[kind]
    if size_bytes > ceiling:
        raise ValidationFailed(
            f"That file is too large. The limit for {kind} is "
            f"{ceiling // (1024 * 1024)} MB.",
            code=ErrorCode.FILE_TOO_LARGE,
            details={"max_bytes": ceiling, "size_bytes": size_bytes},
        )


def upload_key(user_id: uuid.UUID, asset_id: uuid.UUID, filename: str) -> str:
    return f"users/{user_id}/uploads/{asset_id}/{sanitize_filename(filename)}"


def output_key(user_id: uuid.UUID, job_id: uuid.UUID, content_type: str) -> str:
    extension = _EXTENSION_BY_TYPE.get(content_type, ".bin")
    return f"users/{user_id}/generated/{job_id}/output{extension}"


def with_extension(name: str, content_type: str) -> str:
    """Ensures a download name ends in the extension its content type implies.

    The name travels as `Content-Disposition: attachment; filename=...`, so
    without this an extensionless name saves an mp4 as a file the operating
    system cannot identify: it will not open on a double-click, and a file
    picker filtering on `video/mp4` hides it, which makes a downloaded
    generation impossible to re-upload as an Extend source.
    """
    extension = _EXTENSION_BY_TYPE.get(content_type)
    if not extension or name.lower().endswith(extension):
        return name
    return f"{name}{extension}"


class AssetService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = AssetRepository(session)
        self.storage = get_storage()

    async def create_upload(
        self,
        *,
        user_id: uuid.UUID,
        filename: str,
        content_type: str,
        kind: AssetKind,
        size_bytes: int,
    ) -> tuple[Asset, object]:
        validate_upload(kind, content_type, size_bytes)

        asset_id = uuid.uuid4()
        key = upload_key(user_id, asset_id, filename)

        asset = await self.repo.create(
            asset_id=asset_id,
            user_id=user_id,
            kind=kind,
            source=AssetSource.UPLOAD,
            storage_bucket=settings.storage_bucket,
            storage_key=key,
            content_type=content_type,
            original_filename=filename[:255],
            size_bytes=size_bytes,
            status=AssetStatus.PENDING,
        )

        presigned = self.storage.presign_upload(
            key, content_type=content_type, max_size_bytes=MAX_SIZE_BYTES[kind]
        )
        logger.info(
            "upload_url_issued",
            extra={"asset_id": str(asset.id), "kind": str(kind), "size_bytes": size_bytes},
        )
        return asset, presigned

    async def confirm_upload(
        self,
        *,
        asset: Asset,
        duration_seconds: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Asset:
        """Verifies the object exists and matches what was promised."""
        if asset.status == AssetStatus.READY:
            return asset  # idempotent: a retried confirm is not an error

        # head_object is blocking network I/O — off the event loop it goes.
        stat = await asyncio.to_thread(self.storage.stat, asset.storage_key)

        if stat is None:
            raise ValidationFailed(
                "The upload has not finished. Please try uploading again.",
                code=ErrorCode.ASSET_NOT_READY,
            )

        ceiling = MAX_SIZE_BYTES[AssetKind(asset.kind)]
        if stat.size_bytes > ceiling:
            # The file that actually landed is too big. Remove it rather than
            # leaving an orphan occupying paid storage forever.
            await asyncio.to_thread(self.storage.delete, asset.storage_key)
            asset.status = AssetStatus.FAILED
            await self.session.flush()
            raise ValidationFailed(
                "That file is larger than the limit.",
                code=ErrorCode.FILE_TOO_LARGE,
                details={"max_bytes": ceiling, "size_bytes": stat.size_bytes},
            )

        asset.size_bytes = stat.size_bytes
        asset.status = AssetStatus.READY
        if duration_seconds is not None:
            asset.duration_seconds = duration_seconds
        if width is not None:
            asset.width = width
        if height is not None:
            asset.height = height

        await self.session.flush()
        logger.info(
            "upload_confirmed",
            extra={"asset_id": str(asset.id), "size_bytes": stat.size_bytes},
        )
        return asset

    async def register_generated(
        self,
        *,
        user_id: uuid.UUID,
        kind: AssetKind,
        storage_key: str,
        content_type: str,
        name: str,
        size_bytes: int | None = None,
        duration_seconds: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Asset:
        """Records a worker's output. Already in storage, so it starts READY.

        The stored name carries the content type's extension, because it is
        also the filename the browser saves on download.
        """
        asset = await self.repo.create(
            user_id=user_id,
            kind=kind,
            source=AssetSource.GENERATED,
            storage_bucket=settings.storage_bucket,
            storage_key=storage_key,
            content_type=content_type,
            original_filename=with_extension(name, content_type)[:255],
            size_bytes=size_bytes,
            status=AssetStatus.READY,
        )
        asset.duration_seconds = duration_seconds
        asset.width = width
        asset.height = height
        await self.session.flush()
        return asset

    def download_url(self, asset: Asset, *, as_attachment: bool = False) -> str:
        filename = None
        if as_attachment:
            # Applied here as well as at registration so rows created before
            # this rule still download with a usable extension.
            stem = asset.original_filename or f"{asset.kind}-{str(asset.id)[:8]}"
            filename = with_extension(stem, asset.content_type)
        return self.storage.presign_download(asset.storage_key, filename=filename)

    def to_public(self, asset: Asset, *, with_url: bool = True) -> AssetPublic:
        return AssetPublic(
            id=asset.id,
            kind=AssetKind(asset.kind),
            source=AssetSource(asset.source),
            status=AssetStatus(asset.status),
            name=asset.original_filename or f"{asset.kind}-{str(asset.id)[:8]}",
            content_type=asset.content_type,
            size_bytes=asset.size_bytes,
            duration_seconds=asset.duration_seconds,
            width=asset.width,
            height=asset.height,
            created_at=asset.created_at or datetime.now().astimezone(),
            # Only READY assets get a URL — a pending one points at nothing.
            url=self.download_url(asset) if with_url and asset.status == AssetStatus.READY else None,
        )

    async def require_for_user(self, asset_id: uuid.UUID, user_id: uuid.UUID) -> Asset:
        asset = await self.repo.get_for_user(asset_id, user_id)
        if asset is None:
            raise NotFound("That media item could not be found.")
        return asset


async def ensure_storage_ready() -> None:
    """Startup hook — creates the local bucket if it is missing."""
    storage = get_storage()
    try:
        await asyncio.to_thread(storage.ensure_bucket)
    except Exception as exc:  # noqa: BLE001 — startup must report, not crash silently
        logger.error("storage_bootstrap_failed", extra={"reason": type(exc).__name__})
        if settings.is_production:
            raise AppError("Object storage is unreachable.") from exc
