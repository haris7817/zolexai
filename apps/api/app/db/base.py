"""Declarative base and shared column conventions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum as PyEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Deterministic constraint names.
#:
#: Without this, PostgreSQL invents names and Alembic cannot autogenerate a
#: DROP for a constraint it did not create — the classic "migration works on my
#: machine" failure. Set before the first migration so every name is stable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {dict[str, Any]: JSONB}


def uuid_pk() -> Mapped[uuid.UUID]:
    """UUID primary key, generated application-side.

    Generated in Python rather than by the database so the caller knows the id
    before the INSERT commits — the API must return a job id in the same
    response that creates it, and a worker must be able to log against an id it
    is still writing.
    """
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=None
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Timestamps are generated PYTHON-side, with the server default kept only as a
# backstop for rows inserted outside the ORM.
#
# This is not a style preference. With a SQL-expression default or onupdate
# (`server_default=func.now()`, `onupdate=func.now()`), SQLAlchemy does not know
# the value the database computed, so it expires the attribute after the
# statement. The next attribute read then triggers a lazy refresh — and in async
# SQLAlchemy an attribute read is synchronous, so that refresh raises
# `MissingGreenlet`. It surfaces as a 500 the moment a mutated row is serialized
# into a response (cancel, complete, confirm). Generating in Python means the
# ORM already holds the value and no refresh is ever needed.


def created_at_col() -> Mapped[datetime]:
    # Deliberately NOT indexed here. Every real query filters by user first, so
    # the useful index is always composite — see each model's __table_args__. On
    # tables expected to grow very large, a bare created_at index would cost
    # write throughput and serve nothing.
    return mapped_column(
        sa.DateTime(timezone=True),
        default=_utcnow,
        server_default=sa.func.now(),
        nullable=False,
    )


def updated_at_col() -> Mapped[datetime]:
    return mapped_column(
        sa.DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=sa.func.now(),
        nullable=False,
    )


def enum_column(enum_cls: type[PyEnum], length: int = 32) -> sa.Enum:
    """A VARCHAR + CHECK constraint holding enum *values*.

    Two deliberate choices:

    `native_enum=False` — a PostgreSQL ENUM type cannot have a value removed and
    is awkward to reorder; a CHECK constraint is a one-line migration.

    `values_callable` — SQLAlchemy otherwise persists the member NAME
    ("QUEUED"), not the value ("queued"), which would put a different string in
    the database than the one the API and worker exchange over the wire.
    """
    return sa.Enum(
        enum_cls,
        native_enum=False,
        length=length,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )
