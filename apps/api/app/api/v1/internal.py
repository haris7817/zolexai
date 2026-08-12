"""Internal worker coordination — NOT a public API.

Every route here requires the worker service token and must be blocked at the
edge so it is reachable only on the private network (directive §16). The nginx
config under `infrastructure/nginx/` does exactly that.

**Control flows one way: workers pull.** The API never calls out to a worker, so
workers need no inbound port, no public address and no certificate — they only
need to reach the API. That is what makes scaling to N workers, on any host or
GPU provider, an operational decision rather than an architectural one.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, status

from app.api.deps import DbSession, RedisClient, Registry
from app.core.config import settings
from app.core.enums import AssetStatus, JobStatus, WorkerStatus
from app.core.errors import NotFound, ValidationFailed
from app.core.logging import bind_context, get_logger
from app.core.security import require_worker_token
from app.integrations.storage.s3 import get_storage
from app.repositories.generation import GenerationRepository
from app.repositories.worker import WorkerRepository
from app.schemas.internal import (
    ClaimedInput,
    ClaimedJob,
    JobAckResponse,
    JobClaimRequest,
    JobClaimResponse,
    JobCompleteRequest,
    JobFailRequest,
    JobProgressRequest,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)
from app.services import queue
from app.services.generation import GenerationService
from app.services.storage import output_key

logger = get_logger(__name__)

router = APIRouter(
    prefix="/internal",
    tags=["internal"],
    dependencies=[Depends(require_worker_token)],
    # Kept out of the public schema so the customer-facing OpenAPI document
    # never advertises the worker protocol.
    include_in_schema=False,
)

#: Content type a worker must produce, by workflow output kind.
_OUTPUT_CONTENT_TYPE = {"video": "video/mp4", "audio": "audio/mpeg", "image": "image/png"}

_HEARTBEAT_INTERVAL = 30


# ── Worker lifecycle ─────────────────────────────────────────────────────


@router.post(
    "/workers/register",
    response_model=WorkerRegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_worker(
    payload: WorkerRegisterRequest,
    session: DbSession,
    registry: Registry,
) -> WorkerRegisterResponse:
    """Registers (or re-registers) a node.

    Unknown workflow ids are rejected rather than silently dropped: a worker
    that registered for a mistyped id would sit idle forever and look like a
    capacity problem instead of a configuration one.
    """
    requested = payload.workflows or registry.ids()
    unknown = [wid for wid in requested if wid not in registry]
    if unknown:
        raise ValidationFailed(
            "Unknown workflow ids in worker registration.",
            details={"unknown": unknown, "known": registry.ids()},
        )

    repo = WorkerRepository(session)
    worker = await repo.register(
        name=payload.name,
        runtime=payload.runtime,
        version=payload.version,
        capabilities={
            "workflows": requested,
            # Recorded for operators — "which nodes can run this workflow?" is
            # the first question when a queue stops draining. Enforcement is at
            # claim time, where the worker re-asserts it, not from this row.
            "runtimes": payload.runtimes or [payload.runtime],
            "max_concurrency": payload.max_concurrency,
        },
        max_concurrency=payload.max_concurrency,
    )
    await session.commit()

    bind_context(worker_id=str(worker.id))
    logger.info(
        "worker_registered",
        extra={"worker_name": worker.name, "workflow_count": len(requested)},
    )
    return WorkerRegisterResponse(
        worker_id=worker.id,
        lease_seconds=settings.job_lease_seconds,
        heartbeat_interval_seconds=_HEARTBEAT_INTERVAL,
        accepted_workflows=requested,
    )


@router.post("/workers/heartbeat", response_model=WorkerHeartbeatResponse)
async def worker_heartbeat(
    payload: WorkerHeartbeatRequest, session: DbSession
) -> WorkerHeartbeatResponse:
    """Proof of life.

    `acknowledged: false` tells a worker its identity is gone (a reset database,
    a pruned row) so it re-registers instead of reporting against an id nothing
    recognises.
    """
    bind_context(worker_id=str(payload.worker_id))
    repo = WorkerRepository(session)
    worker = await repo.heartbeat(payload.worker_id, status=payload.status)
    await session.commit()

    return WorkerHeartbeatResponse(
        acknowledged=worker is not None, lease_seconds=settings.job_lease_seconds
    )


# ── Job claiming ─────────────────────────────────────────────────────────


@router.post("/jobs/claim", response_model=JobClaimResponse)
async def claim_job(
    payload: JobClaimRequest,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
) -> JobClaimResponse:
    """Hands the oldest eligible queued job to this worker, or reports none.

    Atomic via `FOR UPDATE SKIP LOCKED`, so any number of workers may call this
    simultaneously and no two receive the same job. An empty response is normal,
    not an error.

    The response carries presigned URLs for every input and for the output, so
    the worker streams media directly to and from object storage — media never
    passes through the API (scalability rule #2).
    """
    bind_context(worker_id=str(payload.worker_id))

    worker_repo = WorkerRepository(session)
    worker = await worker_repo.get(payload.worker_id)
    if worker is None:
        raise NotFound("Unknown worker. Register before claiming work.")

    registered: list[str] = list(worker.capabilities.get("workflows") or registry.ids())
    wanted = [w for w in (payload.workflows or registered) if w in registered]

    # A node may only claim work whose workflow is routed to a runtime it can
    # actually execute. Without this, a mock worker claims a job destined for a
    # real provider, finds no adapter, and fails it permanently — the user just
    # sees the tool break. Empty means a pre-M2 worker that cannot assert its
    # runtimes; it keeps the old behaviour rather than being starved.
    if payload.runtimes:
        servable = set(registry.ids_for_runtimes(payload.runtimes))
        wanted = [w for w in wanted if w in servable]

    if not wanted:
        return JobClaimResponse(job=None)

    repo = GenerationRepository(session)
    job = await repo.claim_next(
        worker_id=worker.id,
        workflow_ids=wanted,
        lease_seconds=settings.job_lease_seconds,
    )
    if job is None:
        await session.rollback()
        return JobClaimResponse(job=None)

    definition = registry.get(job.workflow_id)

    # Allocate the output location now so the worker never invents storage keys
    # — key layout stays entirely a backend concern.
    #
    # The runtime may declare what it actually produces (M1's mock emits a
    # placeholder image). The upload is signed for THAT type, so a worker cannot
    # upload something other than what the workflow agreed to.
    content_type = definition.execution.output_content_type or _OUTPUT_CONTENT_TYPE.get(
        definition.output_type, "application/octet-stream"
    )
    out_key = output_key(job.user_id, job.id, content_type)

    storage = get_storage()
    # Presigning is local HMAC, but it is CPU work per input; off-thread so a
    # burst of claims cannot stall the event loop.
    upload = await asyncio.to_thread(
        storage.presign_upload, out_key, content_type=content_type, max_size_bytes=0
    )

    inputs: list[ClaimedInput] = []
    for link, asset in await repo.load_inputs(job.id):
        if asset.status != AssetStatus.READY:
            continue
        inputs.append(
            ClaimedInput(
                role=link.role,
                asset_id=asset.id,
                kind=str(asset.kind),
                content_type=asset.content_type,
                download_url=storage.presign_download(asset.storage_key),
            )
        )

    lease_token = job.lease_token
    lease_expires_at = job.lease_expires_at
    await session.commit()

    bind_context(job_id=str(job.id))
    logger.info(
        "job_claimed",
        extra={"workflow_id": job.workflow_id, "attempt": job.attempt_count},
    )

    return JobClaimResponse(
        job=ClaimedJob(
            job_id=job.id,
            user_id=job.user_id,
            workflow_id=job.workflow_id,
            workflow_version=job.workflow_version,
            prompt=job.prompt,
            parameters=job.request_params or {},
            inputs=inputs,
            # The private execution block — internal recipients only.
            execution=definition.execution.model_dump(),
            lease_token=lease_token,  # type: ignore[arg-type]
            lease_expires_at=lease_expires_at,  # type: ignore[arg-type]
            attempt=job.attempt_count,
            max_attempts=job.max_attempts,
            output_upload_key=out_key,
            output_upload_url=upload.url,
            output_content_type=content_type,
        ),
        poll_after_seconds=5,
    )


# ── Job reporting ────────────────────────────────────────────────────────


@router.post("/jobs/{job_id}/progress", response_model=JobAckResponse)
async def report_progress(
    job_id: uuid.UUID,
    payload: JobProgressRequest,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
) -> JobAckResponse:
    """Records a progress update and fans it out to any SSE listener.

    A rejected report (`accepted: false`) means the worker lost its lease or
    attempted an illegal transition; it should stop working on the job rather
    than retry.
    """
    bind_context(job_id=str(job_id), worker_id=str(payload.worker_id))
    service = GenerationService(session, redis, registry)
    job, reason = await service.report_progress(
        job_id=job_id,
        worker_id=payload.worker_id,
        lease_token=payload.lease_token,
        status=payload.status,
        progress=payload.progress,
        message=payload.message,
    )
    if job is None:
        logger.warning("worker_progress_rejected", extra={"reason": reason})
        return JobAckResponse(
            job_id=job_id, status=JobStatus.CANCELLED, accepted=False, reason=reason
        )
    return JobAckResponse(job_id=job.id, status=JobStatus(job.status), accepted=True)


@router.post("/jobs/{job_id}/complete", response_model=JobAckResponse)
async def complete_job(
    job_id: uuid.UUID,
    payload: JobCompleteRequest,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
) -> JobAckResponse:
    """Registers the worker's output and finishes the job."""
    bind_context(job_id=str(job_id), worker_id=str(payload.worker_id))
    service = GenerationService(session, redis, registry)
    job, reason = await service.complete(
        job_id=job_id,
        worker_id=payload.worker_id,
        lease_token=payload.lease_token,
        output_key=payload.output_key,
        output_kind=payload.output_kind,
        output_content_type=payload.output_content_type,
        size_bytes=payload.size_bytes,
        duration_seconds=payload.duration_seconds,
        width=payload.width,
        height=payload.height,
    )
    if job is None:
        logger.warning("worker_complete_rejected", extra={"reason": reason})
        return JobAckResponse(
            job_id=job_id, status=JobStatus.CANCELLED, accepted=False, reason=reason
        )
    return JobAckResponse(job_id=job.id, status=JobStatus(job.status), accepted=True)


@router.post("/jobs/{job_id}/fail", response_model=JobAckResponse)
async def fail_job(
    job_id: uuid.UUID,
    payload: JobFailRequest,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
) -> JobAckResponse:
    """Reports a failure. Retries while attempts remain, then fails for good.

    `internal_detail` is written to the log and nowhere else — the customer sees
    only the sanitized message (directive §23).
    """
    bind_context(job_id=str(job_id), worker_id=str(payload.worker_id))
    service = GenerationService(session, redis, registry)
    job, reason = await service.fail(
        job_id=job_id,
        worker_id=payload.worker_id,
        lease_token=payload.lease_token,
        error_code=payload.error_code,
        user_message=payload.user_message,
        internal_detail=payload.internal_detail,
        retriable=payload.retriable,
    )
    if job is None:
        logger.warning("worker_fail_rejected", extra={"reason": reason})
        return JobAckResponse(
            job_id=job_id, status=JobStatus.CANCELLED, accepted=False, reason=reason
        )
    return JobAckResponse(job_id=job.id, status=JobStatus(job.status), accepted=True)


# ── Maintenance ──────────────────────────────────────────────────────────


@router.post("/maintenance/reap-leases")
async def reap_leases(session: DbSession, redis: RedisClient) -> dict[str, int]:
    """Requeues jobs whose worker stopped reporting.

    Exposed as an endpoint so it can be driven by an external scheduler in
    production. The API also runs it on a timer (see `main.py`) so a single-node
    deployment needs no cron.
    """
    repo = GenerationRepository(session)
    requeued, exhausted = await repo.requeue_expired_leases()
    stale = await WorkerRepository(session).mark_stale_offline(
        stale_after_seconds=_HEARTBEAT_INTERVAL * 4
    )
    await session.commit()

    if requeued:
        # Wake workers for the jobs just returned to the queue.
        await queue.wake_workers(redis, count=requeued)

    return {"requeued": requeued, "exhausted": exhausted, "workers_offline": stale}
