"""FastAPI application factory and process lifecycle.

Startup order is deliberate: configuration is validated, the workflow registry
is parsed, then dependencies are touched. A definition file with a typo aborts
the boot rather than producing a service that starts and then rejects every
generation — a container that will not start is a visible, cheap failure.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import bind, configure_logging, get_logger
from app.core.middleware import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.db.redis import close_redis, get_redis
from app.db.session import dispose_engine, get_session_factory
from app.repositories.generation import GenerationRepository
from app.repositories.worker import WorkerRepository
from app.services import queue
from app.services.storage import ensure_storage_ready
from app.services.workflow_registry import init_registry

logger = get_logger(__name__)

#: How often the lease reaper runs. Comfortably below `job_lease_seconds` so an
#: expired lease is noticed within a fraction of its lifetime.
_REAPER_INTERVAL_SECONDS = 30


async def _lease_reaper() -> None:
    """Background loop that recovers jobs from workers that stopped reporting.

    Safe to run in every API instance simultaneously: the recovery is a single
    conditional UPDATE, so concurrent runs contend on rows rather than
    duplicating work. Running it in-process means a single-node deployment needs
    no scheduler; larger deployments can disable it and drive
    `POST /internal/maintenance/reap-leases` from cron instead.

    Holds no state — everything it needs is in PostgreSQL, so an instance
    restarting mid-sweep changes nothing.
    """
    while True:
        try:
            await asyncio.sleep(_REAPER_INTERVAL_SECONDS)
            async with get_session_factory()() as session:
                repo = GenerationRepository(session)
                requeued, exhausted = await repo.requeue_expired_leases()
                await WorkerRepository(session).mark_stale_offline(stale_after_seconds=120)
                await session.commit()

            if requeued:
                with bind(component="reaper"):
                    logger.info("leases_requeued", extra={"count": requeued, "failed": exhausted})
                await queue.wake_workers(get_redis(), count=requeued)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let a transient database error kill the loop — the next
            # tick retries, and stranded jobs are exactly what it exists to fix.
            logger.exception("lease_reaper_iteration_failed")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings.assert_production_ready()

    # Parsed and validated BEFORE anything else — a bad definition should abort
    # the boot, not surface later as a broken settings panel.
    registry = init_registry()

    await ensure_storage_ready()

    logger.info(
        "api_starting",
        extra={
            "environment": settings.app_env,
            "workflow_count": len(registry),
            "cors_origins": settings.cors_origin_list,
        },
    )

    reaper = asyncio.create_task(_lease_reaper(), name="lease-reaper")
    try:
        yield
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper
        await close_redis()
        await dispose_engine()
        logger.info("api_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="ZolexAI generation platform API",
        lifespan=lifespan,
        # Interactive docs are a development convenience, not a production
        # surface: they advertise the full schema to anyone who finds the host.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app.add_middleware(
        CORSMiddleware,
        # Explicit origins only — `assert_production_ready` and the settings
        # validator both refuse a wildcard, because credentials will be sent
        # once sessions arrive at M3.
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Idempotency-Key", "Last-Event-ID", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
        max_age=600,
    )

    register_exception_handlers(app)
    # Imported here rather than at module scope so the registry is initialised
    # by lifespan before any route module resolves it.
    from app.api.v1.router import api_router

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
