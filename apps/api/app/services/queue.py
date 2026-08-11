"""Queue signalling.

**PostgreSQL is the queue. Redis is only a doorbell.**

That split is deliberate and is the most important reliability decision in the
job system. A Redis list used as the queue loses work: a worker that pops an
item and then crashes has removed it from Redis and written it nowhere — the job
is simply gone. Recovering from that needs a durable copy anyway, at which point
the list is redundant.

So claiming is a single statement against PostgreSQL:

    SELECT ... WHERE status = 'queued'
    ORDER BY created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1

`SKIP LOCKED` lets any number of workers claim concurrently without blocking
each other or ever handing the same row to two of them, and a crash rolls the
transaction back so the job stays queued. See `repositories/generation.py`.

Redis's only job here is to remove polling latency: workers wait on a blocking
pop, and a new job wakes one immediately instead of after the poll interval.
If Redis is empty, stale or flushed, workers fall back to polling and nothing is
lost — the notification is an optimisation, never a source of truth.
"""

from __future__ import annotations

import uuid

from redis.asyncio import Redis

from app.core.logging import get_logger

logger = get_logger(__name__)

WAKE_LIST = "zx:queue:wake"

#: Bounds the doorbell so a burst cannot grow it without limit. Trimming loses
#: only wake-ups, and a worker that misses one finds the job by polling.
_MAX_WAKE_DEPTH = 1000


async def notify_job_available(redis: Redis, job_id: uuid.UUID) -> None:
    """Best-effort nudge that work exists. Never raises."""
    try:
        pipe = redis.pipeline()
        pipe.lpush(WAKE_LIST, str(job_id))
        pipe.ltrim(WAKE_LIST, 0, _MAX_WAKE_DEPTH - 1)
        await pipe.execute()
    except Exception:
        # Degrades to polling. Logged at debug because it is not an incident.
        logger.debug("queue_wake_failed", extra={"job_id": str(job_id)})


async def wake_workers(redis: Redis, count: int = 1) -> None:
    """Rings the doorbell without naming a job.

    Used by the lease reaper, which returns several jobs to the queue at once
    and has no single id to announce. `count` wake-ups let that many waiting
    workers start claiming immediately.
    """
    try:
        pipe = redis.pipeline()
        for _ in range(max(1, min(count, 50))):
            pipe.lpush(WAKE_LIST, "wake")
        pipe.ltrim(WAKE_LIST, 0, _MAX_WAKE_DEPTH - 1)
        await pipe.execute()
    except Exception:
        logger.debug("queue_wake_all_failed")


async def wait_for_wake(redis: Redis, timeout_seconds: int) -> str | None:
    """Blocks until a job is announced or the timeout elapses.

    The returned id is a HINT, not an assignment. Another worker may already
    have claimed it; the caller must always go through the real claim query.
    """
    try:
        result = await redis.brpop([WAKE_LIST], timeout=timeout_seconds)
    except Exception:
        return None
    return result[1] if result else None


async def queue_depth(redis: Redis) -> int:
    """Doorbell depth — an observability signal, not the true queue length.

    The authoritative count is `SELECT count(*) WHERE status='queued'`.
    """
    try:
        return int(await redis.llen(WAKE_LIST))
    except Exception:
        return 0
