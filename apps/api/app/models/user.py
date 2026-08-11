"""User account.

M1 has no authentication (that is M3.01). One seeded development user backs
every request so `generation_jobs.user_id` and `assets.user_id` are real foreign
keys from the very first migration — adding auth later changes how the identity
is *derived*, not the schema. See `app/core/security.py`.

Billing columns below are extension points, deliberately unenforced in M1
(directive §4, §18).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at_col, updated_at_col, uuid_pk


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = uuid_pk()

    email: Mapped[str] = mapped_column(sa.String(320), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(sa.String(120), nullable=False, default="")

    # ── Extension points for M3 billing ──────────────────────────────────
    # A plan is a code here rather than an FK to a `plans` table: M1 has no
    # plans to point at, and a nullable FK to a table that does not exist yet
    # is worse than a string that a later migration promotes.
    plan_code: Mapped[str] = mapped_column(sa.String(40), nullable=False, default="free")

    concurrency_limit: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    """Per-user override. NULL means the configured default applies."""

    is_active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()

    def __repr__(self) -> str:
        return f"<User {self.email}>"
