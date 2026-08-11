"""Async SQLAlchemy engine and session lifecycle.

The engine is process-wide; sessions are per-request and never shared. Nothing
here caches application state — an API instance holds a connection pool and
nothing else, which is what lets several run behind a load balancer
(scalability rules #1, #12, #13).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=5,
            # Recycles below the typical managed-Postgres idle timeout so a
            # pooled connection is never handed out already closed by the server.
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=False,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency — one transaction per request.

    Commit is explicit in the service layer. Anything that escapes without
    committing is rolled back, so a handler that raises halfway cannot leave a
    partially-written job behind.
    """
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("database_pool_closed")
    _engine = None
    _session_factory = None
