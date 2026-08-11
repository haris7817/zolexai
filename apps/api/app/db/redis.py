"""Redis client.

Redis holds coordination and short-lived state only — never anything the system
cannot rebuild from PostgreSQL (directive §9):

  * idempotency records        (app/services/idempotency.py)
  * rate-limit counters        (app/services/rate_limit.py)
  * live SSE fan-out pub/sub   (app/services/events.py)
  * "work is available" wakeup (app/services/queue.py)

If Redis is flushed, no job is lost: the queue and the event history live in
PostgreSQL, and workers fall back to polling.

One connection pool per process, created lazily and shared. `decode_responses`
is on because every value written here is UTF-8 JSON or a small string.
"""

from __future__ import annotations

from redis.asyncio import ConnectionPool, Redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_pool: ConnectionPool | None = None
_client: Redis | None = None


def get_redis() -> Redis:
    global _pool, _client
    if _client is None:
        _pool = ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=50,
            health_check_interval=30,
        )
        _client = Redis(connection_pool=_pool)
    return _client


def new_redis_connection() -> Redis:
    """A dedicated connection, outside the shared pool.

    Required for pub/sub and blocking reads: once a connection enters subscribe
    mode it can serve no other command, so borrowing one from the shared pool
    would starve every concurrent request in the process.
    """
    return Redis.from_url(settings.redis_url, decode_responses=True, health_check_interval=30)


async def close_redis() -> None:
    global _pool, _client
    if _client is not None:
        await _client.aclose()
        logger.info("redis_client_closed")
    if _pool is not None:
        await _pool.aclose()
    _client = None
    _pool = None
