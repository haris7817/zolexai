"""Health endpoints, separated the way container schedulers actually use them
(directive §20).

  `/health/live`   — is this process running? No dependencies touched.
  `/health/ready`  — can it serve traffic? Checks PostgreSQL, Redis and storage.
  `/health`        — human summary; same checks as ready, always 200.

The distinction is not cosmetic. If liveness also checked the database, a brief
Postgres blip would make the orchestrator *restart every API container* — a
dependency wobble escalated into an outage. Liveness must only fail when
restarting would help.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.db.redis import get_redis
from app.db.session import get_session_factory
from app.integrations.storage.s3 import get_storage
from app.services.workflow_registry import get_registry

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

_CHECK_TIMEOUT = 3.0


@router.get("/health/live", summary="Liveness — process is up")
async def live() -> dict[str, str]:
    return {"status": "ok"}


async def _check_database() -> bool:
    try:
        async with asyncio.timeout(_CHECK_TIMEOUT):
            async with get_session_factory()() as session:
                await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("healthcheck_database_failed", extra={"reason": type(exc).__name__})
        return False


async def _check_redis() -> bool:
    try:
        async with asyncio.timeout(_CHECK_TIMEOUT):
            await get_redis().ping()
        return True
    except Exception as exc:
        logger.warning("healthcheck_redis_failed", extra={"reason": type(exc).__name__})
        return False


async def _check_storage() -> bool:
    try:
        async with asyncio.timeout(_CHECK_TIMEOUT):
            return await asyncio.to_thread(get_storage().health)
    except Exception as exc:
        logger.warning("healthcheck_storage_failed", extra={"reason": type(exc).__name__})
        return False


async def _collect() -> dict[str, bool]:
    # Concurrently: three sequential 3s timeouts would make a readiness probe
    # take nine seconds to report a total outage.
    database, redis_ok, storage = await asyncio.gather(
        _check_database(), _check_redis(), _check_storage()
    )
    workflows = False
    try:
        workflows = len(get_registry()) > 0
    except Exception:
        workflows = False
    return {
        "database": database,
        "redis": redis_ok,
        "storage": storage,
        "workflows": workflows,
    }


@router.get("/health/ready", summary="Readiness — dependencies reachable")
async def ready(response: Response) -> dict[str, object]:
    checks = await _collect()
    # Storage is excluded from the readiness verdict on purpose: an API that
    # cannot presign uploads can still serve workflows, history and SSE. Taking
    # every instance out of the load balancer over it would turn a partial
    # degradation into a full one. It is still reported.
    healthy = checks["database"] and checks["redis"] and checks["workflows"]
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": checks}


@router.get("/health", summary="Health summary")
async def health() -> dict[str, object]:
    checks = await _collect()
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "environment": settings.app_env,
        "checks": checks,
    }
