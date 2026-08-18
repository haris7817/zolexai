"""Generation orchestration — the business logic behind every job endpoint.

Route handlers do argument binding and nothing else; everything that decides
what happens lives here (directive §2).

**One ordering rule governs this whole module: commit to PostgreSQL, then
publish to Redis.** An event published before its transaction commits can reach
a browser describing a state that then rolls back — the UI shows "Completed" for
a job the database still has as `generating`, and no amount of client retry
fixes it. So every method below writes, commits, and only then announces.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import (
    STATUS_LABELS,
    AssetKind,
    AssetStatus,
    ErrorCode,
    EventType,
    JobStatus,
    can_transition,
    is_terminal,
)
from app.core.errors import Conflict, NotFound, ValidationFailed
from app.core.logging import bind, get_logger
from app.models.generation import GenerationJob
from app.models.user import User
from app.repositories.generation import GenerationRepository
from app.schemas.generation import (
    GenerationCreateRequest,
    GenerationError,
    GenerationInput,
    GenerationJobPublic,
    GenerationOutput,
)
from app.services import events as event_bus
from app.services import queue, rate_limit
from app.services.storage import AssetService
from app.services.workflow_registry import WorkflowRegistry

logger = get_logger(__name__)

#: Generic replacement for anything a worker sends that is unsuitable to show a
#: customer. Failure reasons must never leak worker or model internals (§23).
GENERIC_FAILURE = "This generation could not be completed. Please try again."

_GENERATION_RATE_LIMIT = 30
_GENERATION_RATE_WINDOW = 60


def _now() -> datetime:
    return datetime.now(UTC)


class GenerationService:
    def __init__(
        self,
        session: AsyncSession,
        redis: Redis,
        registry: WorkflowRegistry,
    ) -> None:
        self.session = session
        self.redis = redis
        self.registry = registry
        self.repo = GenerationRepository(session)
        self.assets = AssetService(session)

    # ── Creation ─────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        user: User,
        request: GenerationCreateRequest,
        idempotency_key: str | None = None,
    ) -> GenerationJob:
        """Validates, persists and queues a job. Returns immediately.

        Nothing here waits on generation — the job is a row and a notification,
        and the HTTP response is a 202 (scalability rule #3).
        """
        await rate_limit.check_request_rate(
            self.redis,
            user.id,
            action="generation_create",
            limit=_GENERATION_RATE_LIMIT,
            window_seconds=_GENERATION_RATE_WINDOW,
        )

        params = request.parameters
        definition = self.registry.validate_request(
            request.workflow_id,
            prompt=request.prompt,
            duration=params.duration,
            aspect_ratio=params.aspect_ratio,
            quality=params.quality,
            input_roles=set(request.inputs),
            lyrics=params.lyrics,
            lyrics_language=params.lyrics_language,
            prompt_mode=params.prompt_mode,
            dialogue_language=params.dialogue_language,
        )

        await self._validate_inputs(user, request, definition)

        # Checked LAST among the guards, and deliberately: a user at their limit
        # should still get a precise validation error for a malformed request
        # rather than a misleading "too many running".
        await rate_limit.check_generation_concurrency(self.session, user)

        job = await self.repo.create_job(
            user_id=user.id,
            workflow_id=definition.id,
            workflow_version=definition.version,
            prompt=request.prompt.strip(),
            request_params=params.model_dump(mode="json"),
            max_attempts=settings.job_max_attempts,
            idempotency_key=idempotency_key,
            inputs=request.inputs,
        )
        job.stage_hint = "Waiting for an available slot…"

        event = await self.repo.append_event(
            job, event_type=EventType.STATUS, message=job.stage_hint
        )
        published = event_bus.JobEvent.from_row(event)

        try:
            await self.session.commit()
        except IntegrityError as exc:
            # The partial unique index on (user_id, idempotency_key) fired: a
            # concurrent duplicate won. Return the original rather than failing
            # the user's retry.
            await self.session.rollback()
            if idempotency_key:
                existing = await self.repo.find_by_idempotency_key(user.id, idempotency_key)
                if existing is not None:
                    return existing
            raise Conflict("That request could not be accepted. Please try again.") from exc

        with bind(job_id=str(job.id)):
            logger.info(
                "generation_created",
                extra={"workflow_id": job.workflow_id, "input_count": len(request.inputs)},
            )
            # Committed — safe to announce.
            await event_bus.publish_event(self.redis, job.id, published)
            await queue.notify_job_available(self.redis, job.id)

        return job

    async def _validate_inputs(
        self, user: User, request: GenerationCreateRequest, definition: Any
    ) -> None:
        """Checks every referenced asset belongs to the user, is ready, and is
        the kind the role expects.

        Ownership is verified here rather than assumed: without it, a caller
        could name any asset id and have a worker fetch someone else's media.
        """
        if not request.inputs:
            return

        found = await self.assets.repo.get_many_for_user(
            list(request.inputs.values()), user.id
        )

        problems: list[dict[str, Any]] = []
        for role, asset_id in request.inputs.items():
            asset = found.get(asset_id)
            spec = definition.input_for(role)

            if asset is None:
                # Same message whether the asset does not exist or belongs to
                # someone else — distinguishing them would confirm the existence
                # of another user's media.
                problems.append({"field": f"inputs.{role}", "reason": "Media not found."})
                continue
            if asset.status != AssetStatus.READY:
                problems.append(
                    {
                        "field": f"inputs.{role}",
                        "reason": "That upload has not finished processing yet.",
                    }
                )
                continue
            if spec is not None and AssetKind(asset.kind) != AssetKind(spec.kind):
                problems.append(
                    {
                        "field": f"inputs.{role}",
                        "reason": f"This input expects {spec.kind}.",
                        "received": str(asset.kind),
                    }
                )

        if problems:
            raise ValidationFailed(
                "One or more of the selected media items cannot be used.",
                code=ErrorCode.MISSING_REQUIRED_INPUT,
                details={"fields": problems},
            )

    # ── Reads ────────────────────────────────────────────────────────────

    async def require_for_user(self, job_id: uuid.UUID, user_id: uuid.UUID) -> GenerationJob:
        job = await self.repo.get_for_user(job_id, user_id)
        if job is None:
            raise NotFound("That generation could not be found.")
        return job

    async def to_public(
        self, job: GenerationJob, *, with_urls: bool = True
    ) -> GenerationJobPublic:
        status = JobStatus(job.status)

        inputs = [
            GenerationInput(role=link.role, asset_id=asset.id, kind=str(asset.kind))
            for link, asset in await self.repo.load_inputs(job.id)
        ]
        outputs = [
            GenerationOutput(
                asset_id=asset.id,
                kind=str(asset.kind),
                is_primary=link.is_primary,
                url=self.assets.download_url(asset) if with_urls else None,
                # Already loaded with the row, so this costs no extra query.
                width=asset.width,
                height=asset.height,
                duration_seconds=asset.duration_seconds,
            )
            for link, asset in await self.repo.load_outputs(job.id)
        ]

        error = (
            GenerationError(code=str(job.error_code), message=job.error_message or GENERIC_FAILURE)
            if job.error_code
            else None
        )

        # The workflow may have been renamed since the job ran; fall back to the
        # id rather than failing a history page over a definition that moved.
        try:
            workflow_name = self.registry.get_public(job.workflow_id).name
        except NotFound:
            workflow_name = job.workflow_id

        return GenerationJobPublic(
            id=job.id,
            workflow_id=job.workflow_id,
            workflow_name=workflow_name,
            status=status,
            stage_label=STATUS_LABELS[status],
            progress=job.progress,
            hint=job.stage_hint,
            prompt=job.prompt,
            parameters=job.request_params or {},
            inputs=inputs,
            outputs=outputs,
            error=error,
            attempt_count=job.attempt_count,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            last_event_seq=job.last_event_seq,
            is_terminal=is_terminal(status),
        )

    # ── Cancellation ─────────────────────────────────────────────────────

    async def cancel(self, job: GenerationJob) -> GenerationJob:
        """Marks a job cancelled.

        Clearing the lease is what actually stops the work: the worker's next
        progress report presents a token the job no longer holds, is refused,
        and the worker abandons the job. There is no need to reach out to the
        worker — and no way to, since control only ever flows worker-to-API.
        """
        if is_terminal(JobStatus(job.status)):
            raise Conflict("That generation has already finished.")

        locked = await self.repo.get(job.id, for_update=True)
        if locked is None:
            raise NotFound("That generation could not be found.")
        if is_terminal(JobStatus(locked.status)):
            raise Conflict("That generation has already finished.")

        locked.status = JobStatus.CANCELLED
        locked.completed_at = _now()
        locked.stage_hint = "Cancelled."
        locked.lease_token = None
        locked.lease_expires_at = None

        event = await self.repo.append_event(
            locked, event_type=EventType.CANCELLED, message="Cancelled."
        )
        published = event_bus.JobEvent.from_row(event)
        await self.session.commit()

        with bind(job_id=str(locked.id)):
            logger.info("generation_cancelled")
            await event_bus.publish_event(self.redis, locked.id, published)

        return locked

    # ── Worker reports ───────────────────────────────────────────────────

    async def _lock_and_authorize(
        self, job_id: uuid.UUID, worker_id: uuid.UUID, lease_token: uuid.UUID
    ) -> GenerationJob | tuple[None, str]:
        """Loads the job under a row lock and verifies the caller still owns it.

        The lease token is the authorization. A worker whose lease expired had
        its token rotated by whoever claimed the job next, so its late report is
        refused here — which is precisely what prevents a zombie process from
        overwriting live state.
        """
        job = await self.repo.get(job_id, for_update=True)
        if job is None:
            return None, "unknown job"
        if is_terminal(JobStatus(job.status)):
            return None, f"job already {job.status}"
        if job.worker_id != worker_id or job.lease_token != lease_token:
            return None, "lease no longer held"
        return job

    async def report_progress(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: uuid.UUID,
        lease_token: uuid.UUID,
        status: JobStatus,
        progress: int,
        message: str,
        event_payload: dict[str, Any] | None = None,
    ) -> tuple[GenerationJob | None, str]:
        result = await self._lock_and_authorize(job_id, worker_id, lease_token)
        if isinstance(result, tuple):
            await self.session.rollback()
            return None, result[1]
        job = result

        if not can_transition(JobStatus(job.status), status):
            await self.session.rollback()
            return None, f"illegal transition {job.status} -> {status}"

        job.status = status
        # Progress never moves backwards: an out-of-order report would make the
        # bar jump left, which reads as a fault even when the job is healthy.
        job.progress = max(job.progress, progress)
        job.stage_hint = message or job.stage_hint
        if job.started_at is None:
            job.started_at = _now()

        # Any report is also proof of life, so it renews the lease. A worker
        # doing genuine long work never needs a separate heartbeat.
        await self.repo.extend_lease(job, lease_seconds=settings.job_lease_seconds)

        event = await self.repo.append_event(
            job,
            event_type=EventType.PROGRESS,
            message=job.stage_hint,
            payload=event_payload,
        )
        published = event_bus.JobEvent.from_row(event)
        await self.session.commit()

        await event_bus.publish_event(self.redis, job.id, published)
        return job, ""

    async def complete(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: uuid.UUID,
        lease_token: uuid.UUID,
        output_key: str,
        output_kind: str,
        output_content_type: str,
        size_bytes: int | None,
        duration_seconds: float | None,
        width: int | None,
        height: int | None,
    ) -> tuple[GenerationJob | None, str]:
        result = await self._lock_and_authorize(job_id, worker_id, lease_token)
        if isinstance(result, tuple):
            await self.session.rollback()
            return None, result[1]
        job = result

        asset = await self.assets.register_generated(
            user_id=job.user_id,
            kind=AssetKind(output_kind),
            storage_key=output_key,
            content_type=output_content_type,
            name=f"{job.workflow_id}-{str(job.id)[:8]}",
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
        )
        await self.repo.attach_output(job, asset=asset, is_primary=True)

        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.stage_hint = ""
        job.completed_at = _now()
        job.lease_token = None
        job.lease_expires_at = None

        event = await self.repo.append_event(
            job,
            event_type=EventType.COMPLETED,
            message="",
            payload={"asset_id": str(asset.id)},
        )
        published = event_bus.JobEvent.from_row(event)
        await self.session.commit()

        with bind(job_id=str(job.id), worker_id=str(worker_id)):
            logger.info("generation_completed", extra={"asset_id": str(asset.id)})
            await event_bus.publish_event(self.redis, job.id, published)

        return job, ""

    async def fail(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: uuid.UUID,
        lease_token: uuid.UUID,
        error_code: ErrorCode,
        user_message: str,
        internal_detail: str,
        retriable: bool,
    ) -> tuple[GenerationJob | None, str]:
        """Records a failure, retrying only while attempts remain.

        Retries are bounded by `max_attempts` — no infinite loop is possible
        (directive §23). `internal_detail` is logged and discarded; only the
        sanitized message reaches the database and therefore the customer.
        """
        result = await self._lock_and_authorize(job_id, worker_id, lease_token)
        if isinstance(result, tuple):
            await self.session.rollback()
            return None, result[1]
        job = result

        with bind(job_id=str(job.id), worker_id=str(worker_id)):
            logger.error(
                "generation_failed_report",
                extra={
                    "error_code": error_code.value,
                    "attempt": job.attempt_count,
                    "max_attempts": job.max_attempts,
                    "retriable": retriable,
                    # Worker detail lives HERE and nowhere else.
                    "internal_detail": internal_detail[:2000],
                },
            )

        should_retry = retriable and job.attempt_count < job.max_attempts

        if should_retry:
            job.status = JobStatus.QUEUED
            job.progress = 0
            job.worker_id = None
            job.lease_token = None
            job.lease_expires_at = None
            job.stage_hint = "Retrying…"
            event_type, message = EventType.STATUS, "Retrying…"
        else:
            job.status = JobStatus.FAILED
            job.completed_at = _now()
            job.lease_token = None
            job.lease_expires_at = None
            job.error_code = error_code
            job.error_message = _safe_failure_message(user_message)
            job.stage_hint = ""
            event_type, message = EventType.FAILED, job.error_message

        event = await self.repo.append_event(job, event_type=event_type, message=message)
        published = event_bus.JobEvent.from_row(event)
        await self.session.commit()

        await event_bus.publish_event(self.redis, job.id, published)
        if should_retry:
            await queue.notify_job_available(self.redis, job.id)

        return job, ""


def _safe_failure_message(candidate: str) -> str:
    """Guards the one path where worker text could reach a customer.

    A worker is trusted to run generation, not to write customer-facing copy. A
    message that looks like a traceback, a file path or a stack frame is
    replaced wholesale rather than trimmed.
    """
    text = (candidate or "").strip()
    if not text:
        return GENERIC_FAILURE

    lowered = text.lower()
    leak_markers = (
        "traceback",
        "exception",
        "  file ",
        ".py",
        "cuda",
        "torch",
        "errno",
        "0x",
        "\n",
    )
    if any(marker in lowered for marker in leak_markers) or len(text) > 300:
        return GENERIC_FAILURE
    return text
