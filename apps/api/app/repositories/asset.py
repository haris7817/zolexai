"""Data access for media assets."""

from __future__ import annotations

import uuid
from typing import Sequence

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AssetKind, AssetSource, AssetStatus
from app.models.asset import Asset
from app.schemas.common import PageParams, encode_cursor


class AssetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        kind: AssetKind,
        source: AssetSource,
        storage_bucket: str,
        storage_key: str,
        content_type: str,
        original_filename: str = "",
        size_bytes: int | None = None,
        status: AssetStatus = AssetStatus.PENDING,
        asset_id: uuid.UUID | None = None,
    ) -> Asset:
        # The caller may pass an id it allocated in advance: the storage key
        # embeds the asset id, so the key cannot be built until the id is known
        # and the row must then be inserted with that exact id.
        asset = Asset(
            id=asset_id or uuid.uuid4(),
            user_id=user_id,
            kind=kind,
            source=source,
            status=status,
            storage_bucket=storage_bucket,
            storage_key=storage_key,
            content_type=content_type,
            original_filename=original_filename,
            size_bytes=size_bytes,
        )
        self.session.add(asset)
        await self.session.flush()
        return asset

    async def get_for_user(self, asset_id: uuid.UUID, user_id: uuid.UUID) -> Asset | None:
        return (
            await self.session.execute(
                select(Asset).where(Asset.id == asset_id, Asset.user_id == user_id)
            )
        ).scalar_one_or_none()

    async def get_many_for_user(
        self, asset_ids: Sequence[uuid.UUID], user_id: uuid.UUID
    ) -> dict[uuid.UUID, Asset]:
        """Fetches several assets in ONE query.

        Generation requests may reference multiple inputs (source video plus an
        optional reference image). Looking them up individually would be an
        N+1 against a table the media library also reads heavily.
        """
        if not asset_ids:
            return {}
        rows = (
            await self.session.execute(
                select(Asset).where(Asset.id.in_(list(asset_ids)), Asset.user_id == user_id)
            )
        ).scalars()
        return {asset.id: asset for asset in rows}

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        *,
        page: PageParams,
        kind: AssetKind | None = None,
        source: AssetSource | None = None,
        status: AssetStatus | None = AssetStatus.READY,
    ) -> tuple[list[Asset], str | None, bool]:
        """Media library page. Defaults to READY only.

        A pending row is an upload the browser has not finished; showing it
        would put a broken thumbnail in the library. Callers wanting everything
        pass `status=None` explicitly.
        """
        stmt = select(Asset).where(Asset.user_id == user_id)

        if kind is not None:
            stmt = stmt.where(Asset.kind == kind)
        if source is not None:
            stmt = stmt.where(Asset.source == source)
        if status is not None:
            stmt = stmt.where(Asset.status == status)

        position = page.decoded()
        if position is not None:
            created_at, last_id = position
            stmt = stmt.where(
                sa.tuple_(Asset.created_at, Asset.id) < sa.tuple_(created_at, last_id)
            )

        stmt = stmt.order_by(Asset.created_at.desc(), Asset.id.desc()).limit(page.limit + 1)
        rows = list((await self.session.execute(stmt)).scalars())

        has_more = len(rows) > page.limit
        items = rows[: page.limit]
        next_cursor = (
            encode_cursor(items[-1].created_at, items[-1].id) if has_more and items else None
        )
        return items, next_cursor, has_more

    async def count_by_kind(self, user_id: uuid.UUID) -> dict[str, int]:
        """Tab counts for the media library, in one grouped query."""
        rows = await self.session.execute(
            select(Asset.kind, sa.func.count())
            .where(Asset.user_id == user_id, Asset.status == AssetStatus.READY)
            .group_by(Asset.kind)
        )
        return {str(kind): count for kind, count in rows}
