"""Test fixtures.

Tests run against a REAL PostgreSQL and Redis, not mocks or SQLite. Three of
the behaviours that matter most here cannot be tested any other way:

  * `FOR UPDATE SKIP LOCKED` — SQLite has no such thing, so a mocked claim would
    prove nothing about the property the whole worker design rests on.
  * Partial unique indexes — half the idempotency guarantee is enforced by
    PostgreSQL.
  * `SET NX` atomicity — the other half is enforced by Redis.

A suite that passed against SQLite while the production database behaved
differently would be worse than no suite at all.

Bring the dependencies up first:

    cd infrastructure/compose && docker compose --env-file ../../.env up -d
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Must be set before app modules import their settings.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("WORKER_API_TOKEN", "test_worker_token_0123456789abcdef")

import app.db.redis as redis_module  # noqa: E402
import app.db.session as session_module  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.security import reset_dev_user_cache  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402
from app.services.workflow_registry import init_registry  # noqa: E402

TEST_DATABASE_URL = settings.database_url.rsplit("/", 1)[0] + "/zolexai_test"
ADMIN_DATABASE_URL = settings.database_url.rsplit("/", 1)[0] + "/postgres"

_schema_ready = False


async def _ensure_test_database() -> None:
    """Creates the test database and schema once per session.

    Guarded by a module flag rather than a session-scoped async fixture: with
    function-scoped event loops (the pytest-asyncio default), a session-scoped
    async fixture runs on a loop that is closed before most tests execute, and
    every connection it cached becomes unusable.

    `create_all` rather than migrations — the migration is verified separately
    in `test_migrations.py`, so running it per suite would only make this slow.
    """
    global _schema_ready
    if _schema_ready:
        return

    admin = create_async_engine(ADMIN_DATABASE_URL, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.exec_driver_sql("DROP DATABASE IF EXISTS zolexai_test WITH (FORCE)")
        await conn.exec_driver_sql("CREATE DATABASE zolexai_test")
    await admin.dispose()

    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    _schema_ready = True


@pytest_asyncio.fixture(autouse=True)
async def _isolate(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Points the app at the test database and gives each test a clean slate.

    The engine and Redis client are reset around every test because both cache a
    connection pool bound to the event loop that created it. pytest-asyncio
    gives each test a fresh loop, so a pool carried over belongs to a closed one
    and every call through it fails.

    State is cleared by TRUNCATE rather than a rolled-back outer transaction:
    the claim tests need two concurrent sessions to see each other's COMMITTED
    rows, which a shared transaction would hide.
    """
    monkeypatch.setattr(settings, "database_url", TEST_DATABASE_URL, raising=False)

    session_module._engine = None
    session_module._session_factory = None
    redis_module._client = None
    redis_module._pool = None
    reset_dev_user_cache()

    await _ensure_test_database()

    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE generation_events, generation_job_outputs, generation_job_inputs, "
            "generation_jobs, assets, worker_nodes, users RESTART IDENTITY CASCADE"
        )
    await engine.dispose()

    redis = redis_module.get_redis()
    # Only this suite's key space — a shared Redis may hold a developer's data.
    for pattern in ("zx:idem:*", "zx:rl:*", "zx:queue:*"):
        keys = [key async for key in redis.scan_iter(match=pattern)]
        if keys:
            await redis.delete(*keys)

    yield

    await redis_module.close_redis()
    await session_module.dispose_engine()


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """The API, driven in-process.

    ASGITransport rather than a live server: no port to bind, no startup race,
    and a failing test points at a stack frame instead of a connection error.

    `create_app()` is called without running its lifespan, so the registry is
    initialised explicitly here and the background lease reaper never starts —
    a timer firing mid-test would make results depend on wall-clock timing.
    """
    init_registry()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Content-Type": "application/json"},
    ) as http:
        yield http


@pytest.fixture
def worker_headers() -> dict[str, str]:
    return {"X-Worker-Token": settings.worker_api_token}


@pytest.fixture
def text_to_video_request() -> dict:
    return {
        "workflow_id": "text-to-video",
        "prompt": "A cinematic drone shot over a neon city at dusk",
        "parameters": {"duration": "10s", "aspect_ratio": "16:9", "quality": "High"},
    }


@pytest.fixture
def idempotency_key() -> str:
    return uuid.uuid4().hex
