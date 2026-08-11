"""Fair-use foundation (directive §18).

The problem being designed against: **one user must never be able to occupy
every GPU worker.** Once generation is real, a script submitting a hundred jobs
would starve every other customer, and no amount of worker scaling fixes it —
the queue is simply fair-ordered by arrival.

Two independent limits, both enforced before a job is created:

  **Request rate** — a fixed-window counter in Redis. Cheap, approximate, and
  aimed at accidental floods (a retry loop) rather than a determined attacker.

  **Concurrency** — how many of this user's jobs are *simultaneously* running,
  counted in PostgreSQL. This is the one that protects worker capacity, and it
  must be exact, so it is counted against the durable record rather than a cache
  that could drift.

M1 enforces a single configured default for every user. The per-plan extension
point is `User.concurrency_limit` (an override column that already exists) plus
`User.plan_code`; M3 billing resolves a limit from the plan and nothing else in
this module changes.

Usage metering for billing is deliberately NOT implemented — `usage_records` is
named as a future table in the schema notes, not created here.
"""

from __future__ import annotations

import uuid

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import ACTIVE_STATUSES
from app.core.errors import ConcurrencyLimitReached, RateLimited
from app.core.logging import get_logger
from app.models.generation import GenerationJob
from app.models.user import User

logger = get_logger(__name__)


async def check_request_rate(
    redis: Redis, user_id: uuid.UUID, *, action: str, limit: int, window_seconds: int
) -> None:
    """Fixed-window counter.

    A fixed window can allow up to 2x `limit` across a boundary. That is
    acceptable here — this guards against runaway clients, and the concurrency
    limit below is the real protection for worker capacity. A sliding-window log
    would cost more Redis memory than the precision is worth at this stage.

    Fails OPEN: if Redis is unavailable, requests are allowed. A cache outage
    should not take generation offline, and the concurrency check still holds
    because it reads PostgreSQL.
    """
    key = f"zx:rl:{action}:{user_id}"
    try:
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds, nx=True)
        count, _ = await pipe.execute()
    except Exception:
        logger.warning("rate_limit_check_skipped", extra={"action": action})
        return

    if count > limit:
        logger.info("rate_limited", extra={"action": action, "count": count, "limit": limit})
        raise RateLimited(
            "Too many requests. Please wait a moment and try again.",
            details={"retry_after_seconds": window_seconds},
        )


def concurrency_limit_for(user: User) -> int:
    """Resolves the user's simultaneous-job allowance.

    Order of precedence — per-user override, then plan, then global default. The
    plan branch is where M3 billing plugs in; today every plan resolves to the
    default, and that is stated rather than hidden.
    """
    if user.concurrency_limit is not None:
        return user.concurrency_limit
    # M3 extension point: look up plan_code -> plan.concurrency_limit.
    return settings.default_user_concurrency_limit


async def check_generation_concurrency(session: AsyncSession, user: User) -> None:
    """Counts this user's in-flight jobs and refuses if they are at their limit.

    Counted in PostgreSQL, not Redis: this number decides whether expensive work
    starts, and a cache that drifted upward would let one user quietly exceed
    their share. The `(user_id, status, created_at DESC)` index makes it an
    index-only count.
    """
    limit = concurrency_limit_for(user)

    running = (
        await session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.user_id == user.id,
                GenerationJob.status.in_(list(ACTIVE_STATUSES)),
            )
        )
    ).scalar_one()

    if running >= limit:
        logger.info("concurrency_limit_reached", extra={"running": running, "limit": limit})
        raise ConcurrencyLimitReached(
            f"You already have {running} generations running. "
            "Please wait for one to finish before starting another.",
            details={"running": running, "limit": limit},
        )
