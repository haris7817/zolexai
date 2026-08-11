"""Duplicate-request suppression (directive §24).

**Why this is not optional.** A generation is the most expensive operation the
platform performs. A double-clicked Generate button, a client-side retry after a
timeout, or a proxy replaying a request would each start a second GPU job the
user never asked for and will still be billed for.

**How it works.** A client may send `Idempotency-Key: <opaque>` with
`POST /api/v1/generations`. The key is scoped to the user and recorded twice:

  * **Redis** — a short-lived reservation taken atomically with `SET NX`. This
    is what closes the race between two *simultaneous* requests, which a
    database uniqueness check alone cannot: both would pass their SELECT before
    either INSERT lands.

  * **PostgreSQL** — a partial unique index on `(user_id, idempotency_key)`.
    This is the durable guarantee, and it survives a Redis flush.

Redis makes the common case fast and race-free; PostgreSQL makes it correct.
Neither alone is sufficient, which is why both are here.

A repeat of a *completed* request returns the original job with 200 instead of
creating a new one — the retry gets the answer it was looking for.
"""

from __future__ import annotations

import uuid

from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_PENDING = "pending"


def _redis_key(user_id: uuid.UUID, key: str) -> str:
    return f"zx:idem:{user_id}:{key}"


async def reserve(redis: Redis, user_id: uuid.UUID, key: str) -> str | None:
    """Claims the key, or reports who already holds it.

    Returns None when the caller won the race and should proceed. Returns the
    stored value otherwise: a job id if the original request finished, or
    "pending" if it is still in flight.

    `SET NX` is a single atomic round trip — the check and the claim cannot be
    interleaved by a concurrent request.
    """
    redis_key = _redis_key(user_id, key)
    won = await redis.set(
        redis_key, _PENDING, nx=True, ex=settings.idempotency_ttl_seconds
    )
    if won:
        return None

    existing = await redis.get(redis_key)
    logger.info("idempotency_key_replayed", extra={"idempotency_state": existing or "unknown"})
    return existing or _PENDING


async def complete(redis: Redis, user_id: uuid.UUID, key: str, job_id: uuid.UUID) -> None:
    """Records which job the key produced, so a later retry can return it."""
    await redis.set(
        _redis_key(user_id, key), str(job_id), ex=settings.idempotency_ttl_seconds
    )


async def release(redis: Redis, user_id: uuid.UUID, key: str) -> None:
    """Frees a reservation whose request failed.

    Without this, a request that fails validation would lock its key for a full
    day and the user could not retry with the same one — turning a typo into a
    24-hour block.
    """
    await redis.delete(_redis_key(user_id, key))


def is_pending(value: str) -> bool:
    return value == _PENDING
