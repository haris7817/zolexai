"""HTTP client for the internal worker API.

The worker's ONLY channel to the platform. It has no database connection and no
standing storage credentials — every capability it has arrives as a presigned
URL scoped to one job (directive §16).

Every call carries the service token. Nothing here retries blindly: a failed
progress report is logged and the job continues, because the lease reaper will
recover a job whose worker genuinely went away, and a retry storm against a
struggling API helps nobody.
"""

from __future__ import annotations

from typing import Any

import httpx

from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)


class ApiUnavailable(RuntimeError):
    """The API could not be reached or returned a server error."""


class WorkerApiClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_v1,
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={"X-Worker-Token": settings.worker_api_token},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise ApiUnavailable(f"{type(exc).__name__} calling {path}") from exc

        if response.status_code >= 500:
            raise ApiUnavailable(f"HTTP {response.status_code} from {path}")
        if response.status_code >= 400:
            # 4xx is the API telling the worker it is wrong — a lost lease, an
            # unknown worker. Surfaced as data, not an exception, so the runner
            # can drop the job cleanly rather than treat it as an outage.
            logger.warning(
                "worker_api_rejected",
                extra={"path": path, "status_code": response.status_code},
            )
            return {"_rejected": True, "status_code": response.status_code}

        return response.json()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def register(self, *, workflows: list[str] | None = None) -> dict[str, Any]:
        return await self._post(
            "/internal/workers/register",
            {
                "name": settings.worker_name,
                "runtime": settings.runtime,
                "version": settings.worker_version,
                "workflows": workflows or [],
                "max_concurrency": settings.max_concurrency,
            },
        )

    async def heartbeat(
        self, worker_id: str, *, status: str = "online", active_job_ids: list[str] | None = None
    ) -> dict[str, Any]:
        return await self._post(
            "/internal/workers/heartbeat",
            {
                "worker_id": worker_id,
                "status": status,
                "active_job_ids": active_job_ids or [],
            },
        )

    async def claim(self, worker_id: str, workflows: list[str] | None = None) -> dict[str, Any]:
        return await self._post(
            "/internal/jobs/claim",
            {"worker_id": worker_id, "workflows": workflows or []},
        )

    # ── Job reporting ────────────────────────────────────────────────────

    async def report_progress(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        status: str,
        progress: int,
        message: str,
    ) -> dict[str, Any]:
        return await self._post(
            f"/internal/jobs/{job_id}/progress",
            {
                "worker_id": worker_id,
                "lease_token": lease_token,
                "status": status,
                "progress": progress,
                "message": message,
            },
        )

    async def report_complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        output_key: str,
        output_kind: str,
        output_content_type: str,
        size_bytes: int,
        duration_seconds: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            f"/internal/jobs/{job_id}/complete",
            {
                "worker_id": worker_id,
                "lease_token": lease_token,
                "output_key": output_key,
                "output_kind": output_kind,
                "output_content_type": output_content_type,
                "size_bytes": size_bytes,
                "duration_seconds": duration_seconds,
                "width": width,
                "height": height,
            },
        )

    async def report_failure(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        user_message: str,
        internal_detail: str,
        retriable: bool,
        error_code: str = "generation_failed",
    ) -> dict[str, Any]:
        return await self._post(
            f"/internal/jobs/{job_id}/fail",
            {
                "worker_id": worker_id,
                "lease_token": lease_token,
                "error_code": error_code,
                "user_message": user_message,
                "internal_detail": internal_detail,
                "retriable": retriable,
            },
        )
