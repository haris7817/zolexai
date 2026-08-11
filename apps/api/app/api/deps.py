"""Shared FastAPI dependencies.

Route handlers stay thin by taking what they need from here: a session, the
current user, the registry, Redis. Business logic lives in `app/services`
(directive §2).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import bind_context
from app.core.security import get_dev_user
from app.db.redis import get_redis
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import PageParams
from app.services.workflow_registry import WorkflowRegistry, get_registry

DbSession = Annotated[AsyncSession, Depends(get_db)]


async def current_user(session: DbSession) -> User:
    """The requesting customer.

    M1 always resolves the seeded development user. At M3 this reads a verified
    session cookie instead; every downstream query already scopes by `user.id`,
    so nothing else moves.
    """
    user = await get_dev_user(session)
    # Bound for the remainder of the request so every log line is attributable.
    bind_context(user_id=str(user.id))
    return user


CurrentUser = Annotated[User, Depends(current_user)]
Registry = Annotated[WorkflowRegistry, Depends(get_registry)]
RedisClient = Annotated[Redis, Depends(get_redis)]


def page_params(
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
) -> PageParams:
    """Keyset pagination.

    An upper bound on `limit` is mandatory, not defensive: without it a single
    request could ask for an entire history table and defeat every index the
    schema has (directive §5).
    """
    return PageParams(limit=limit, cursor=cursor)


Pagination = Annotated[PageParams, Depends(page_params)]
