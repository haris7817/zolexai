"""Authentication seams.

Two distinct identities exist, and conflating them would be the security bug
that matters most here:

  **Workers** authenticate with a shared service token on `/api/v1/internal/*`.
  That token is not a user account, grants no access to public endpoints, and
  the internal routes are never exposed to a browser (directive §16).

  **Customers** have no real authentication in M1 — that is M3.01. Every public
  request resolves to one seeded development user. The seam is `get_current_user`
  and nothing else; when sessions arrive, only this function changes, because
  `user_id` is already a genuine foreign key on every row that needs one.
"""

from __future__ import annotations

import secrets
import uuid

from fastapi import Header
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import Forbidden, Unauthorized
from app.core.logging import get_logger
from app.models.user import User

logger = get_logger(__name__)

WORKER_TOKEN_HEADER = "X-Worker-Token"


async def require_worker_token(
    x_worker_token: str | None = Header(default=None, alias=WORKER_TOKEN_HEADER),
) -> None:
    """Guards every internal worker endpoint.

    An unset token is a hard failure rather than an open door: a deployment that
    forgot to configure one must not accidentally publish job claiming to the
    internet.
    """
    expected = settings.worker_api_token
    if not expected:
        logger.error("worker_token_not_configured")
        raise Forbidden("Worker access is not configured on this deployment.")

    # Constant-time: a naive `==` leaks the token prefix through timing.
    if not x_worker_token or not secrets.compare_digest(x_worker_token, expected):
        logger.warning("worker_token_rejected")
        raise Unauthorized("Invalid worker credentials.")


# The development user's id, cached after the first lookup.
#
# This is a CACHE of an immutable database fact, not a source of truth: the row
# lives in PostgreSQL and any instance can rebuild this by querying. It exists
# only to keep a per-request SELECT off the hot path, and it disappears entirely
# at M3 when the id comes from the session.
_dev_user_id: uuid.UUID | None = None


async def get_dev_user(session: AsyncSession) -> User:
    """Resolves (and on first run creates) the single M1 user.

    Creating it here rather than in a migration keeps the schema free of seed
    data, and makes a fresh database usable without a separate bootstrap step.
    """
    global _dev_user_id

    if _dev_user_id is not None:
        user = await session.get(User, _dev_user_id)
        if user is not None:
            return user
        _dev_user_id = None  # database was reset underneath us

    email = settings.dev_user_email
    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()

    if user is None:
        # INSERT ... ON CONFLICT DO NOTHING, then re-select.
        #
        # A plain add() loses a genuine race: several API instances booting at
        # once — or several concurrent first requests — all miss the SELECT and
        # all INSERT, and every one but the first fails on the unique email.
        # The upsert makes losing the race a no-op rather than a 500.
        await session.execute(
            pg_insert(User)
            .values(
                id=uuid.uuid4(),
                email=email,
                display_name="ZolexAI Demo",
                plan_code="free",
            )
            .on_conflict_do_nothing(index_elements=[User.email])
        )
        await session.flush()
        user = (
            await session.execute(select(User).where(User.email == email))
        ).scalar_one()
        logger.info("dev_user_resolved", extra={"user_id": str(user.id)})

    _dev_user_id = user.id
    return user


def reset_dev_user_cache() -> None:
    """Test hook — the cached id would otherwise outlive a rolled-back fixture."""
    global _dev_user_id
    _dev_user_id = None
