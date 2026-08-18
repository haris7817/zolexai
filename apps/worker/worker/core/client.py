"""HTTP client for the internal worker API.

The worker's ONLY channel to the platform. It has no database connection and no
standing storage credentials — every capability it has arrives as a presigned
URL scoped to one job (directive §16).

Every call carries the service token. Nothing here retries blindly: a failed
progress report is logged and the job continues, because the lease reaper will
recover a job whose worker genuinely went away, and a retry storm against a
struggling API helps nobody.

TRANSPORT failures are the one exception, and they are not the same thing as an
API that is struggling — see `_RETRYABLE`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)


class ApiUnavailable(RuntimeError):
    """The API could not be reached or returned a server error."""


#: Transport failures worth one more attempt: the request never reached the API,
#: or the connection died underneath it. These say nothing about the API's
#: health, so the "no retry storms" rule above does not apply — asking again on
#: a fresh socket is the correct and complete fix.
#:
#: This is not hypothetical. The worker reaches the API down a long-lived
#: tunnel, and a pooled connection that the far end (or the tunnel carrying it)
#: has already closed raises `RemoteProtocolError` on the next write. On
#: 2026-08-17 that single error, on a routine progress update, discarded SIX
#: jobs — one of them a video-to-video that had already rendered seven of its
#: eight sections, six minutes of GPU work thrown away because a status message
#: did not land. Every one of those jobs was healthy.
#:
#: `ReadTimeout` is deliberately absent: there the request DID reach the API and
#: may still be executing, so repeating it would be a second instruction rather
#: than a retry.
_RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)

#: Attempts per call, including the first. Three covers the observed failure —
#: one dead pooled socket, one tunnel reconnect — and stops well short of the
#: hammering this module has always warned against.
_TRANSPORT_ATTEMPTS = 3

#: Waits before the second and third attempts. Short on purpose: a job is
#: holding a GPU while this sleeps.
_TRANSPORT_BACKOFF_SECONDS = (0.5, 1.5)

#: How long an idle connection may be kept in the pool, in seconds.
#:
#: This is the ROOT CAUSE of the dropped jobs above, and it is a race we were
#: losing by default. The API runs uvicorn with no `--timeout-keep-alive`, so
#: the server closes an idle connection after 5s — and httpx's own default
#: `keepalive_expiry` is also 5.0s. Both ends therefore expire the same socket
#: at the same instant, and any call landing on that boundary finds a
#: connection the client still trusts and the server has already closed.
#:
#: Expiring well before the server does removes the race rather than surviving
#: it: the pool never offers a socket old enough to have been closed. The cost
#: is a TCP handshake on calls more than two seconds apart, which over a
#: loopback tunnel is nothing next to the job it protects.
_KEEPALIVE_EXPIRY_SECONDS = 2.0


#: Mirrors `JobFailRequest.internal_detail` in the API's internal schema. Over
#: this and the whole report is rejected — see `fit_detail`.
MAX_INTERNAL_DETAIL = 2000

#: Same, for `JobFailRequest.user_message`. The API substitutes generic copy
#: for anything it considers unsuitable, but only after the request validates.
MAX_USER_MESSAGE = 400


def fit_detail(detail: str, limit: int = MAX_INTERNAL_DETAIL) -> str:
    """Trims a diagnostic to what the API will accept, keeping both ends.

    Load-bearing, and the failure it prevents is worse than losing detail. A
    CUDA traceback runs to several thousand characters; the API caps this
    field at 2000 and rejects the whole report with 422. The job is then never
    marked failed — it sits until its lease expires and is retried, at full
    GPU cost, with the same input and the same deterministic crash. Observed
    on 2026-08-16: two identical six-minute music-video failures, neither of
    which the platform ever recorded as a failure.

    Both ends are kept because they carry different things: the head says
    which stage failed and with what exit code, the tail carries the actual
    exception. Cutting either one loses the diagnosis.
    """
    if len(detail) <= limit:
        return detail

    marker = "\n  …[trimmed]…\n"
    room = max(0, limit - len(marker))
    head = room * 2 // 5
    return detail[:head] + marker + detail[len(detail) - (room - head):]


class WorkerApiClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_v1,
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={"X-Worker-Token": settings.worker_api_token},
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY_SECONDS),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(self, path: str, payload: dict[str, Any], *, attempts: int) -> httpx.Response:
        """POSTs once, or a few times if the transport — not the API — failed."""
        for attempt in range(1, attempts + 1):
            try:
                return await self._client.post(path, json=payload)
            except _RETRYABLE as exc:
                if attempt == attempts:
                    raise ApiUnavailable(f"{type(exc).__name__} calling {path}") from exc
                delay = _TRANSPORT_BACKOFF_SECONDS[
                    min(attempt - 1, len(_TRANSPORT_BACKOFF_SECONDS) - 1)
                ]
                logger.warning(
                    "api_transport_retry",
                    extra={
                        "path": path,
                        "attempt": attempt,
                        "attempts": attempts,
                        "reason": type(exc).__name__,
                    },
                )
                await asyncio.sleep(delay)
            except httpx.HTTPError as exc:
                # Everything else — a timeout the API may still be serving, a
                # malformed response — is reported as-is rather than repeated.
                raise ApiUnavailable(f"{type(exc).__name__} calling {path}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def _post(
        self, path: str, payload: dict[str, Any], *, retry: bool = True
    ) -> dict[str, Any]:
        response = await self._send(
            path, payload, attempts=_TRANSPORT_ATTEMPTS if retry else 1
        )

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
                "runtimes": settings.runtime_list,
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
        # The ONE call that is not retried. Every other endpoint here can be
        # repeated safely — progress clamps forward, completion and failure are
        # terminal transitions the API validates, register and heartbeat are
        # upserts — but a claim whose RESPONSE was lost has already taken a job
        # on the server, and asking again would take a second one this worker
        # will never run. It needs no retry anyway: the poll loop is the retry,
        # and it comes round in a second.
        return await self._post(
            "/internal/jobs/claim",
            {
                "worker_id": worker_id,
                "workflows": workflows or [],
                # Sent on every claim, not just at registration: a node's
                # capability is a property of the running process, and a
                # restarted worker with a different runtime must not inherit
                # what the old row said it could do.
                "runtimes": settings.runtime_list,
            },
            retry=False,
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
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            f"/internal/jobs/{job_id}/progress",
            {
                "worker_id": worker_id,
                "lease_token": lease_token,
                "status": status,
                "progress": progress,
                "message": message,
                **(details or {}),
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
                # Bounded HERE rather than at each call site: this is the layer
                # that knows the API's contract, and a report the API refuses
                # leaves the job unfailed and silently retried.
                "user_message": user_message[:MAX_USER_MESSAGE],
                "internal_detail": fit_detail(internal_detail),
                "retriable": retriable,
            },
        )
