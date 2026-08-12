"""Worker entry point.

    python -m worker.main

**Workers pull; the platform never pushes.** This process registers, then loops:
wait for a signal, claim a job, run it, report. It exposes no port and needs no
inbound connectivity — only the ability to reach the API. That is what makes
"add another worker" an operational action rather than an architectural one, and
what will let M2 put a GPU node behind any provider's NAT without changing a
line of platform code.

Concurrency is bounded by `max_concurrency`, and claiming is atomic on the API
side (`FOR UPDATE SKIP LOCKED`), so any number of these may run against the same
database with no coordination between them:

    docker compose --profile apps up -d --scale worker=3
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys

from redis.asyncio import Redis

from worker.core.client import ApiUnavailable, WorkerApiClient
from worker.core.config import settings
from worker.core.logging import bind, configure_logging, get_logger
from worker.jobs.runner import JobRunner

logger = get_logger(__name__)

WAKE_LIST = "zx:queue:wake"

#: Backoff bounds for when the API is unreachable. Capped so a worker recovers
#: promptly after an outage instead of sleeping for minutes.
_BACKOFF_START = 2.0
_BACKOFF_MAX = 30.0


class WorkerService:
    def __init__(self) -> None:
        self.client = WorkerApiClient()
        self.worker_id: str | None = None
        self.redis: Redis | None = None
        self._stopping = asyncio.Event()
        self._active: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def register(self) -> None:
        """Registers, retrying until the API answers.

        A worker that started before the API is normal in a compose stack, so
        this waits rather than exiting — a crash-loop would be noisier and no
        faster.
        """
        delay = _BACKOFF_START
        while not self._stopping.is_set():
            try:
                response = await self.client.register()
                if response.get("_rejected"):
                    raise ApiUnavailable(f"registration rejected: {response}")
                self.worker_id = str(response["worker_id"])
                logger.info(
                    "worker_ready",
                    extra={
                        "worker_name": settings.worker_name,
                        "runtime": settings.runtime,
                        "runtimes": settings.runtime_list,
                        "workflows": response.get("accepted_workflows", []),
                        "max_concurrency": settings.max_concurrency,
                    },
                )
                return
            except (ApiUnavailable, KeyError) as exc:
                logger.warning(
                    "registration_retry",
                    extra={"reason": str(exc), "retry_in_seconds": delay},
                )
                await self._sleep(delay)
                delay = min(delay * 2, _BACKOFF_MAX)

    async def heartbeat_loop(self) -> None:
        """Proof of life, and the trigger to re-register if identity is lost."""
        while not self._stopping.is_set():
            await self._sleep(settings.heartbeat_interval_seconds)
            if self._stopping.is_set() or self.worker_id is None:
                continue
            try:
                response = await self.client.heartbeat(
                    self.worker_id,
                    status="busy" if self._active else "online",
                    active_job_ids=[],
                )
                if not response.get("acknowledged", True):
                    # The API does not know us — a reset database, a pruned row.
                    # Re-register rather than keep reporting into the void.
                    logger.warning("worker_identity_lost")
                    await self.register()
            except ApiUnavailable as exc:
                logger.warning("heartbeat_failed", extra={"reason": str(exc)})

    # ── Claim loop ───────────────────────────────────────────────────────

    async def claim_loop(self) -> None:
        runner = JobRunner(self.client, str(self.worker_id))
        delay = _BACKOFF_START

        while not self._stopping.is_set():
            # Blocks until a slot frees, so a worker never claims more than it
            # can run. The job stays queued for someone else instead.
            await self._semaphore.acquire()
            if self._stopping.is_set():
                self._semaphore.release()
                break

            try:
                response = await self.client.claim(str(self.worker_id))
                delay = _BACKOFF_START
            except ApiUnavailable as exc:
                self._semaphore.release()
                logger.warning("claim_failed", extra={"reason": str(exc)})
                await self._sleep(delay)
                delay = min(delay * 2, _BACKOFF_MAX)
                continue

            if response.get("_rejected"):
                self._semaphore.release()
                # Most likely our worker row is gone; re-register and retry.
                await self.register()
                runner = JobRunner(self.client, str(self.worker_id))
                continue

            claim = response.get("job")
            if claim is None:
                self._semaphore.release()
                await self._wait_for_work()
                continue

            task = asyncio.create_task(self._run_and_release(runner, claim))
            self._active.add(task)
            task.add_done_callback(self._active.discard)

    async def _run_and_release(self, runner: JobRunner, claim: dict) -> None:
        try:
            await runner.run(claim)
        finally:
            self._semaphore.release()

    async def _wait_for_work(self) -> None:
        """Waits for a wake-up, or falls back to polling.

        Redis is an optimisation only — the queue lives in PostgreSQL. With
        Redis down or disabled this polls every `idle_poll_seconds` and behaves
        identically, just less promptly.
        """
        if not settings.use_redis_wakeup:
            await self._sleep(settings.idle_poll_seconds)
            return

        try:
            if self.redis is None:
                self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
            # Blocking pop: returns the moment a job is announced.
            await self.redis.brpop([WAKE_LIST], timeout=settings.wake_timeout_seconds)
        except Exception:
            self.redis = None
            await self._sleep(settings.idle_poll_seconds)

    async def _sleep(self, seconds: float) -> None:
        """Interruptible sleep — shutdown does not wait out the full delay."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)

    # ── Shutdown ─────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Stops claiming, then lets in-flight jobs finish.

        Draining rather than killing: a job abandoned mid-flight would sit until
        its lease expired before another worker could retry it. Waiting here
        removes that delay for the user watching it.

        The window is `shutdown_drain_seconds` — minutes, not the 30 seconds M1
        used. Thirty was sized for a six-second mock; against a real render it
        guaranteed the cancellation it was meant to avoid.
        """
        if self._stopping.is_set():
            return
        self._stopping.set()
        logger.info(
            "worker_draining",
            extra={
                "active_jobs": len(self._active),
                "drain_seconds": settings.shutdown_drain_seconds,
            },
        )

        if self._active:
            _, pending = await asyncio.wait(
                self._active, timeout=settings.shutdown_drain_seconds
            )
            for task in pending:
                task.cancel()
            if pending:
                logger.warning("jobs_cancelled_on_shutdown", extra={"count": len(pending)})

        if self.redis is not None:
            with contextlib.suppress(Exception):
                await self.redis.aclose()

        if self.worker_id:
            with contextlib.suppress(ApiUnavailable):
                await self.client.heartbeat(self.worker_id, status="offline")

        await self.client.aclose()
        logger.info("worker_stopped")


async def run() -> int:
    configure_logging()

    if not settings.worker_api_token:
        logger.error("worker_token_missing")
        return 1

    service = WorkerService()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, AttributeError):
            # Windows' ProactorEventLoop has no add_signal_handler; KeyboardInterrupt
            # covers Ctrl+C there instead.
            loop.add_signal_handler(sig, lambda: asyncio.create_task(service.shutdown()))

    await service.register()
    if service.worker_id is None:
        return 1

    with bind(worker_id=service.worker_id):
        heartbeat = asyncio.create_task(service.heartbeat_loop(), name="heartbeat")
        try:
            await service.claim_loop()
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await service.shutdown()

    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
