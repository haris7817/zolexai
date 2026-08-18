"""Executes one claimed job, start to finish.

The contract with the platform, in order:

  1. Report `preparing` — the job is now visibly ours.
  2. Stage any inputs from object storage onto local disk (presigned GETs).
  3. Run the adapter, forwarding its progress reports and keeping the lease
     alive underneath it.
  4. Stream the result to the presigned output URL.
  5. Report completion, or report failure with a customer-safe message.

Four behaviours are worth understanding before changing anything here.

**A rejected progress report aborts the job immediately.** If the API answers
`accepted: false`, this worker no longer owns the job — its lease expired and
another worker took it, or the user cancelled. Continuing would waste compute on
a result nobody will accept and risk two workers writing the same output.

**The lease is kept alive while the adapter is silent.** A lease lasts
`JOB_LEASE_SECONDS` and only a progress report renews it. A real render goes
quiet for far longer than that, so a background task re-reports the last known
progress on the adapter's behalf. Without it the reaper hands a perfectly
healthy job to a second worker while the first is still rendering it.

**Cancellation and timeouts reach the adapter as exceptions.** The keepalive
sets a flag the adapter polls through `job.raise_if_cancelled()`; the timeout is
a wall-clock budget. Both unwind through the adapter's own `finally` blocks so
subprocesses die and files close before the workspace is removed.

**Failure is always reported.** A worker that dies silently is recovered by the
lease reaper, but only after the lease expires — up to two minutes of a user
watching a stalled progress bar. Reporting immediately turns that into instant
feedback, and the reaper stays as the backstop for the case this cannot cover:
the process disappearing entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace
from typing import Any

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    JobCancelled,
    JobTimedOut,
)
from worker.core.client import ApiUnavailable, WorkerApiClient
from worker.core.config import settings
from worker.core.logging import bind, get_logger
from worker.jobs.workspace import job_workspace
from worker.storage.transfer import download_input_to, upload_output_file
from worker.workflows.resolver import build_adapter_job, resolve_adapter

logger = get_logger(__name__)


class LeaseLost(RuntimeError):
    """This worker no longer owns the job. Stop work; report nothing further."""


class JobRunner:
    def __init__(self, client: WorkerApiClient, worker_id: str) -> None:
        self.client = client
        self.worker_id = worker_id

    async def run(self, claim: dict[str, Any]) -> None:
        job_id = str(claim["job_id"])
        lease_token = str(claim["lease_token"])
        attempt = int(claim.get("attempt", 1))

        with bind(job_id=job_id, attempt=attempt, workflow_id=claim.get("workflow_id")):
            try:
                await self._execute(claim, job_id=job_id, lease_token=lease_token)
            except LeaseLost as exc:
                # Expected, not an incident: the platform reassigned or
                # cancelled the job while we held it.
                logger.info("job_abandoned", extra={"reason": str(exc)})
            except JobCancelled:
                # The adapter stopped because we asked it to. Whoever cancelled
                # the job already moved it to a terminal state.
                logger.info("job_stopped_after_cancellation")
            except JobTimedOut as exc:
                logger.warning("job_timed_out", extra={"detail": str(exc)})
                await self._report_failure(
                    job_id,
                    lease_token,
                    AdapterError(
                        "This generation took too long and was stopped.",
                        internal_detail=str(exc),
                        # Bounded by max_attempts, so a workflow that always
                        # exceeds its budget fails for good after three tries
                        # rather than looping.
                        retriable=True,
                    ),
                )
            except AdapterError as exc:
                logger.warning(
                    "job_failed",
                    extra={"internal_detail": exc.internal_detail, "retriable": exc.retriable},
                )
                await self._report_failure(job_id, lease_token, exc)
            except ApiUnavailable as exc:
                # Nothing can be reported — the API is the only channel. The
                # lease will expire and the reaper requeues the job.
                logger.error("job_orphaned_api_unavailable", extra={"reason": str(exc)})
            except asyncio.CancelledError:
                logger.info("job_cancelled_on_shutdown")
                raise
            except Exception as exc:  # noqa: BLE001 — an adapter bug must still fail cleanly
                logger.exception("job_crashed", extra={"exception_type": type(exc).__name__})
                await self._report_failure(
                    job_id,
                    lease_token,
                    AdapterError(
                        "This generation could not be completed. Please try again.",
                        internal_detail=f"unhandled {type(exc).__name__}: {exc}",
                        retriable=True,
                    ),
                )

    async def _execute(self, claim: dict[str, Any], *, job_id: str, lease_token: str) -> None:
        budget = float(
            claim.get("execution", {}).get("timeout_seconds") or settings.job_timeout_seconds
        )
        cancelled = asyncio.Event()

        with job_workspace(job_id) as workspace:
            job = build_adapter_job(
                claim,
                workspace=workspace,
                cancelled=cancelled,
                deadline_monotonic=time.monotonic() + budget,
            )
            adapter = resolve_adapter(job)
            keeper = _LeaseKeeper(self, job_id, lease_token, cancelled)

            async def on_progress(
                status: str,
                progress: int,
                message: str,
                details: dict[str, Any] | None = None,
            ) -> None:
                await keeper.report(status, progress, message, details)

            await on_progress("preparing", 8, "Setting up your generation…")
            job = await self._stage_inputs(job)

            logger.info("job_started", extra={"adapter": adapter.name, "budget_seconds": budget})
            async with keeper.running():
                try:
                    result = await asyncio.wait_for(adapter.run(job, on_progress), timeout=budget)
                except TimeoutError as exc:
                    raise JobTimedOut(f"adapter exceeded {budget:.0f}s") from exc

            # Raised after the adapter unwound cleanly, so its `finally` blocks
            # have already run.
            keeper.raise_if_lost()

            size_bytes = await upload_output_file(
                claim["output_upload_url"], result.path, claim["output_content_type"]
            )

            response = await self.client.report_complete(
                job_id,
                worker_id=self.worker_id,
                lease_token=lease_token,
                output_key=claim["output_upload_key"],
                output_kind=result.kind,
                output_content_type=result.content_type,
                size_bytes=size_bytes,
                duration_seconds=result.duration_seconds,
                width=result.width,
                height=result.height,
            )
            if not response.get("accepted", False):
                raise LeaseLost(response.get("reason", "completion rejected"))

            logger.info("job_completed", extra={"size_bytes": size_bytes})

    async def _stage_inputs(self, job: AdapterJob) -> AdapterJob:
        """Downloads every input into the workspace and records its path.

        Deliberately eager, for two reasons. Discovering an unreadable input
        after two minutes of GPU time has been spent is the expensive ordering;
        discovering it in the first second is free. And an adapter that receives
        a local file cannot accidentally re-download the source — which is what
        happened in M1, where the runner fetched the bytes purely to check
        reachability and then discarded them.
        """
        if not job.inputs:
            return job

        staged = []
        for item in job.inputs:
            destination = job.workspace / "inputs" / f"{item.role}{_suffix_for(item.content_type)}"
            await download_input_to(item.download_url, destination, role=item.role)
            staged.append(
                AdapterInput(
                    role=item.role,
                    kind=item.kind,
                    content_type=item.content_type,
                    download_url=item.download_url,
                    path=destination,
                )
            )

        return replace(job, inputs=staged)

    async def _report_progress(
        self,
        job_id: str,
        lease_token: str,
        status: str,
        progress: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        response = await self.client.report_progress(
            job_id,
            worker_id=self.worker_id,
            lease_token=lease_token,
            status=status,
            progress=progress,
            message=message,
            details=details,
        )
        if response.get("_rejected") or not response.get("accepted", False):
            raise LeaseLost(response.get("reason", "progress rejected"))

    async def _report_failure(
        self, job_id: str, lease_token: str, error: AdapterError
    ) -> None:
        try:
            await self.client.report_failure(
                job_id,
                worker_id=self.worker_id,
                lease_token=lease_token,
                user_message=error.user_message,
                internal_detail=error.internal_detail,
                retriable=error.retriable,
            )
        except ApiUnavailable:
            # Best effort. The reaper is the backstop.
            logger.error("failure_report_undeliverable")


class _LeaseKeeper:
    """Keeps the job's lease alive while the adapter is working.

    It re-sends the last progress the adapter reported rather than inventing a
    new one: the user's bar should reflect real work, and a lease renewal is not
    progress. The API clamps progress forward anyway, so a repeat is a no-op to
    the customer and proof of life to the platform.

    When a renewal is refused — the user cancelled, or the lease was already
    reassigned — it flips the shared cancellation flag. The adapter notices at
    its next `raise_if_cancelled()` and stops, which is the difference between
    releasing a GPU in seconds and holding it until the render finishes for
    nobody.
    """

    def __init__(
        self,
        runner: JobRunner,
        job_id: str,
        lease_token: str,
        cancelled: asyncio.Event,
    ) -> None:
        self._runner = runner
        self._job_id = job_id
        self._lease_token = lease_token
        self._cancelled = cancelled
        self._last: tuple[str, int, str, dict[str, Any] | None] = (
            "preparing", 0, "", None
        )
        self._lost: LeaseLost | None = None

    async def report(
        self,
        status: str,
        progress: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._last = (status, progress, message, details)
        try:
            await self._runner._report_progress(
                self._job_id, self._lease_token, status, progress, message, details
            )
        except LeaseLost:
            self._cancelled.set()
            raise
        except ApiUnavailable as exc:
            # A status update is telemetry; the job is the work. Letting this
            # propagate discarded six healthy jobs on 2026-08-17 — one of them
            # seven-eighths rendered — because a progress message did not land.
            #
            # Nothing is lost by continuing. `_loop` below is still renewing the
            # lease and retrying this same report, `self._last` already holds the
            # value it will send, and a lease that is genuinely gone comes back
            # as a REJECTION (LeaseLost, above), which is a different thing from
            # a socket that broke. The keepalive loop has always treated an
            # outage this way; the foreground path disagreeing with it was the
            # bug, not the policy.
            logger.warning("progress_report_failed", extra={"reason": str(exc)})

    def raise_if_lost(self) -> None:
        if self._lost is not None:
            raise self._lost

    @contextlib.asynccontextmanager
    async def running(self):
        task = asyncio.create_task(self._loop(), name=f"lease-keepalive-{self._job_id}")
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(settings.lease_keepalive_seconds)
            status, progress, message, details = self._last
            try:
                await self._runner._report_progress(
                    self._job_id, self._lease_token, status, progress, message, details
                )
                logger.debug("lease_renewed", extra={"progress": progress})
            except LeaseLost as exc:
                logger.info("lease_lost_during_generation", extra={"reason": str(exc)})
                self._lost = exc
                self._cancelled.set()
                return
            except ApiUnavailable as exc:
                # Transient. Keep trying: the lease may still be ours, and
                # giving up here would abandon a job that is going fine.
                logger.warning("lease_renewal_failed", extra={"reason": str(exc)})


def _suffix_for(content_type: str) -> str:
    """A file extension ffmpeg and friends can recognise the input by."""
    return _SUFFIXES.get(content_type.split(";")[0].strip().lower(), ".bin")


_SUFFIXES: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}
