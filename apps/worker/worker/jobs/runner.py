"""Executes one claimed job, start to finish.

The contract with the platform, in order:

  1. Report `preparing` — the job is now visibly ours.
  2. Fetch any inputs from object storage (presigned GETs).
  3. Run the adapter, forwarding its progress reports.
  4. PUT the result to the presigned output URL.
  5. Report completion, or report failure with a customer-safe message.

Two behaviours are worth understanding before changing anything here.

**A rejected progress report aborts the job immediately.** If the API answers
`accepted: false`, this worker no longer owns the job — its lease expired and
another worker took it, or the user cancelled. Continuing would waste compute on
a result nobody will accept and risk two workers writing the same output. So the
runner raises `LeaseLost` and unwinds.

**Failure is always reported.** A worker that dies silently is recovered by the
lease reaper, but only after the lease expires — up to two minutes of a user
watching a stalled progress bar. Reporting immediately turns that into instant
feedback, and the reaper stays as the backstop for the case this cannot cover:
the process disappearing entirely.
"""

from __future__ import annotations

import asyncio
from typing import Any

from worker.adapters.base import AdapterError, AdapterJob
from worker.core.client import ApiUnavailable, WorkerApiClient
from worker.core.logging import bind, get_logger
from worker.storage.transfer import upload_output
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
        job = build_adapter_job(claim)
        adapter = resolve_adapter(job)

        async def on_progress(status: str, progress: int, message: str) -> None:
            await self._report_progress(job_id, lease_token, status, progress, message)

        await on_progress("preparing", 8, "Setting up your generation…")

        await self._fetch_inputs(job)

        logger.info("job_started", extra={"adapter": adapter.name})
        result = await adapter.run(job, on_progress)

        await upload_output(
            claim["output_upload_url"], result.content, claim["output_content_type"]
        )

        response = await self.client.report_complete(
            job_id,
            worker_id=self.worker_id,
            lease_token=lease_token,
            output_key=claim["output_upload_key"],
            output_kind=result.kind,
            output_content_type=result.content_type,
            size_bytes=len(result.content),
            duration_seconds=result.duration_seconds,
            width=result.width,
            height=result.height,
        )
        if not response.get("accepted", False):
            raise LeaseLost(response.get("reason", "completion rejected"))

        logger.info("job_completed", extra={"size_bytes": len(result.content)})

    async def _fetch_inputs(self, job: AdapterJob) -> None:
        """Verifies every input is reachable before generation starts.

        Deliberately eager. Discovering an unreadable input after two minutes of
        GPU time has been spent is the expensive ordering; discovering it in the
        first second is free.
        """
        if not job.inputs:
            return

        from worker.storage.transfer import download_input

        for item in job.inputs:
            data = await download_input(item.download_url, role=item.role)
            logger.info(
                "input_fetched", extra={"role": item.role, "size_bytes": len(data)}
            )
            # M1's mock adapter does not consume the bytes; a real adapter in M2
            # will stage them to its working directory here.

    async def _report_progress(
        self, job_id: str, lease_token: str, status: str, progress: int, message: str
    ) -> None:
        response = await self.client.report_progress(
            job_id,
            worker_id=self.worker_id,
            lease_token=lease_token,
            status=status,
            progress=progress,
            message=message,
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
