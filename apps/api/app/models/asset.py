"""Media assets — uploads and generated outputs.

An asset row is a POINTER into object storage, never the bytes. Nothing here
holds a filesystem path on the API or worker host (scalability rule #2).

An upload is two-phase: the API issues a presigned PUT and writes a `pending`
row; the browser uploads straight to storage; the client then confirms and the
row flips to `ready`. Only `ready` assets may be used as generation inputs, so a
half-finished upload can never reach a worker.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import AssetKind, AssetSource, AssetStatus
from app.db.base import Base, created_at_col, enum_column, updated_at_col, uuid_pk


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = uuid_pk()

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    kind: Mapped[AssetKind] = mapped_column(enum_column(AssetKind), nullable=False)
    source: Mapped[AssetSource] = mapped_column(enum_column(AssetSource), nullable=False)
    status: Mapped[AssetStatus] = mapped_column(
        enum_column(AssetStatus), nullable=False, default=AssetStatus.PENDING
    )

    # ── Storage location ─────────────────────────────────────────────────
    storage_bucket: Mapped[str] = mapped_column(sa.String(120), nullable=False)
    storage_key: Mapped[str] = mapped_column(sa.String(512), nullable=False)

    content_type: Mapped[str] = mapped_column(sa.String(120), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    original_filename: Mapped[str] = mapped_column(sa.String(255), nullable=False, default="")

    # ── Optional media metadata, filled in when known ────────────────────
    duration_seconds: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    width: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)

    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()

    __table_args__ = (
        # Storage keys are namespaced per user and globally unique; the
        # constraint makes an accidental overwrite a database error rather than
        # silent data loss.
        sa.UniqueConstraint("storage_bucket", "storage_key", name="uq_assets_storage_location"),
        # The media library's default view: one user's assets, newest first.
        # DESC matches the ORDER BY exactly so keyset pagination is an index scan.
        sa.Index("ix_assets_user_created", "user_id", sa.text("created_at DESC")),
        # Tab filtering (All / Videos / Images / Audio) within one user.
        sa.Index("ix_assets_user_kind_created", "user_id", "kind", sa.text("created_at DESC")),
        sa.Index("ix_assets_user_status", "user_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<Asset {self.id} {self.kind} {self.status}>"
