"""Data access for generation jobs, their inputs/outputs and their event log.

All SQL for the job system lives here — routes and services never build queries
(directive §2). That is what keeps the indexing story reviewable in one file
rather than scattered across handlers.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ACTIVE_STATUSES, TERMINAL_STATUSES, ErrorCode, EventType, JobStatus
from app.core.logging import get_logger
from app.models.asset import Asset
from app.models.generation import (
    GenerationEvent,
    GenerationJob,
    GenerationJobInput,
    GenerationJobOutput,
)
from app.schemas.common import PageParams, encode_cursor

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class GenerationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Creation ─────────────────────────────────────────────────────────

    async def create_job(
        self,
        *,
        user_id: uuid.UUID,
        workflow_id: str,
        workflow_version: str,
        prompt: str,
        request_params: dict[str, Any],
        max_attempts: int,
        idempotency_key: str | None,
        inputs: dict[str, uuid.UUID],
    ) -> GenerationJob:
        job = GenerationJob(
            user_id=user_id,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            prompt=prompt,
            request_params=request_params,
            status=JobStatus.QUEUED,
            progress=0,
            max_attempts=max_attempts,
            idempotency_key=idempotency_key,
        )
        self.session.add(job)
        await self.session.flush()

        for role, asset_id in inputs.items():
            self.session.add(
                GenerationJobInput(job_id=job.id, asset_id=asset_id, role=role)
            )
        await self.session.flush()
        return job

    # ── Reads ────────────────────────────────────────────────────────────

    async def get(self, job_id: uuid.UUID, *, for_update: bool = False) -> GenerationJob | None:
        stmt = select(GenerationJob).where(GenerationJob.id == job_id)
        if for_update:
            # Serialises concurrent writers on one job. Required whenever
            # `last_event_seq` is incremented, otherwise two workers reporting
            # at the same instant would allocate the same sequence number and
            # violate uq_generation_events_job_seq.
            stmt = stmt.with_for_update()
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_for_user(
        self, job_id: uuid.UUID, user_id: uuid.UUID
    ) -> GenerationJob | None:
        # Ownership is part of the WHERE clause, never a check afterwards: a
        # filter cannot be forgotten the way a follow-up `if` can.
        return (
            await self.session.execute(
                select(GenerationJob).where(
                    GenerationJob.id == job_id, GenerationJob.user_id == user_id
                )
            )
        ).scalar_one_or_none()

    async def producing_job_for_asset(
        self, asset_id: uuid.UUID, user_id: uuid.UUID
    ) -> GenerationJob | None:
        """The user's own completed job whose output this asset is, if any.

        This is video lineage: an Extend whose source came from the library
        was uploaded by hand and has no ancestry, while one reached through
        the result's Extend button is the child of the job that rendered it —
        and everything Director-aware extension needs (mode, language, the
        original idea, the original inputs) is already on that ancestor's row.
        Ownership is in the WHERE clause for the same reason it is everywhere.
        """
        return (
            await self.session.execute(
                select(GenerationJob)
                .join(
                    GenerationJobOutput,
                    GenerationJobOutput.job_id == GenerationJob.id,
                )
                .where(
                    GenerationJobOutput.asset_id == asset_id,
                    GenerationJob.user_id == user_id,
                    GenerationJob.status == JobStatus.COMPLETED,
                )
                .limit(1)
            )
        ).scalar_one_or_none()

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        *,
        page: PageParams,
        statuses: Sequence[JobStatus] | None = None,
        workflow_id: str | None = None,
    ) -> tuple[list[GenerationJob], str | None, bool]:
        """One page of history, newest first, via keyset seek.

        Fetches `limit + 1` rows: the extra row answers "is there more?" without
        a second COUNT query over the whole filtered set.
        """
        stmt = select(GenerationJob).where(GenerationJob.user_id == user_id)

        if statuses:
            stmt = stmt.where(GenerationJob.status.in_(list(statuses)))
        if workflow_id:
            stmt = stmt.where(GenerationJob.workflow_id == workflow_id)

        position = page.decoded()
        if position is not None:
            created_at, last_id = position
            # Row-value comparison, which PostgreSQL can satisfy directly from
            # the composite index — unlike the equivalent OR-expansion.
            stmt = stmt.where(
                sa.tuple_(GenerationJob.created_at, GenerationJob.id)
                < sa.tuple_(created_at, last_id)
            )

        stmt = stmt.order_by(
            GenerationJob.created_at.desc(), GenerationJob.id.desc()
        ).limit(page.limit + 1)

        rows = list((await self.session.execute(stmt)).scalars())

        has_more = len(rows) > page.limit
        items = rows[: page.limit]
        next_cursor = (
            encode_cursor(items[-1].created_at, items[-1].id) if has_more and items else None
        )
        return items, next_cursor, has_more

    async def count_active_for_user(self, user_id: uuid.UUID) -> int:
        return (
            await self.session.execute(
                select(sa.func.count())
                .select_from(GenerationJob)
                .where(
                    GenerationJob.user_id == user_id,
                    GenerationJob.status.in_(list(ACTIVE_STATUSES)),
                )
            )
        ).scalar_one()

    async def find_by_idempotency_key(
        self, user_id: uuid.UUID, key: str
    ) -> GenerationJob | None:
        return (
            await self.session.execute(
                select(GenerationJob).where(
                    GenerationJob.user_id == user_id, GenerationJob.idempotency_key == key
                )
            )
        ).scalar_one_or_none()

    # ── Event log ────────────────────────────────────────────────────────

    async def append_event(
        self,
        job: GenerationJob,
        *,
        event_type: EventType,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> GenerationEvent:
        """Records one lifecycle event against the job's CURRENT state.

        The caller must already hold the job row lock (`get(for_update=True)`)
        — `seq` is allocated by incrementing a column, which is only safe under
        that lock.
        """
        job.last_event_seq += 1
        event = GenerationEvent(
            job_id=job.id,
            seq=job.last_event_seq,
            event_type=event_type,
            status=job.status,
            progress=job.progress,
            message=message or job.stage_hint,
            payload=payload or {},
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def list_events(self, job_id: uuid.UUID, after_seq: int = 0) -> list[GenerationEvent]:
        return list(
            (
                await self.session.execute(
                    select(GenerationEvent)
                    .where(GenerationEvent.job_id == job_id, GenerationEvent.seq > after_seq)
                    .order_by(GenerationEvent.seq)
                )
            ).scalars()
        )

    # ── Worker leasing ───────────────────────────────────────────────────

    async def claim_next(
        self,
        *,
        worker_id: uuid.UUID,
        workflow_ids: Sequence[str],
        lease_seconds: int,
    ) -> GenerationJob | None:
        """Atomically hands the oldest eligible queued job to one worker.

        `FOR UPDATE SKIP LOCKED` is the whole mechanism. Concurrent workers each
        lock a different row instead of queueing behind one another, so claiming
        scales with worker count. A worker that dies mid-transaction releases its
        lock on disconnect and the job is untouched — still `queued`, claimable
        by the next worker.

        The SELECT and the UPDATE are separate statements, but they run inside
        the caller's single transaction while the row lock is held, so no other
        worker can observe or take the row in between.
        """
        stmt = (
            select(GenerationJob)
            .where(
                GenerationJob.status == JobStatus.QUEUED,
                GenerationJob.workflow_id.in_(list(workflow_ids)),
            )
            .order_by(GenerationJob.created_at)  # fair: oldest first
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = (await self.session.execute(stmt)).scalar_one_or_none()
        if job is None:
            return None

        job.status = JobStatus.ASSIGNED
        job.worker_id = worker_id
        job.lease_token = uuid.uuid4()  # rotated so a superseded worker is locked out
        job.lease_expires_at = _now() + timedelta(seconds=lease_seconds)
        job.attempt_count += 1
        job.started_at = job.started_at or _now()
        job.stage_hint = "Waiting for an available slot…"
        await self.session.flush()
        return job

    async def extend_lease(
        self, job: GenerationJob, *, lease_seconds: int
    ) -> None:
        job.lease_expires_at = _now() + timedelta(seconds=lease_seconds)
        await self.session.flush()

    async def requeue_expired_leases(self, *, limit: int = 100) -> tuple[int, int]:
        """Recovers jobs whose worker stopped reporting.

        Returns (requeued, exhausted). A job under its attempt ceiling goes back
        to `queued` for another worker; one that has used every attempt is failed
        with a customer-safe message rather than retried forever (directive §23).

        Nothing here trusts the worker to notice its own death — that is the
        point. A crashed process, a killed container and a network partition all
        look identical from PostgreSQL, and all resolve the same way.
        """
        now = _now()
        expired_filter = (
            GenerationJob.lease_expires_at.is_not(None),
            GenerationJob.lease_expires_at < now,
            GenerationJob.status.not_in(list(TERMINAL_STATUSES)),
        )

        requeue = (
            sa.update(GenerationJob)
            .where(
                *expired_filter,
                GenerationJob.attempt_count < GenerationJob.max_attempts,
                GenerationJob.id.in_(
                    select(GenerationJob.id).where(*expired_filter).limit(limit)
                ),
            )
            .values(
                status=JobStatus.QUEUED,
                worker_id=None,
                lease_token=None,
                lease_expires_at=None,
                progress=0,
                stage_hint="Waiting for an available slot…",
            )
            .returning(GenerationJob.id)
        )
        requeued = len((await self.session.execute(requeue)).scalars().all())

        exhaust = (
            sa.update(GenerationJob)
            .where(
                *expired_filter,
                GenerationJob.attempt_count >= GenerationJob.max_attempts,
            )
            .values(
                status=JobStatus.FAILED,
                lease_token=None,
                lease_expires_at=None,
                completed_at=now,
                error_code=ErrorCode.MAX_ATTEMPTS_EXCEEDED,
                error_message=(
                    "This generation could not be completed after several attempts. "
                    "Please try again."
                ),
            )
            .returning(GenerationJob.id)
        )
        exhausted = len((await self.session.execute(exhaust)).scalars().all())

        if requeued or exhausted:
            logger.warning(
                "leases_reaped", extra={"requeued": requeued, "exhausted": exhausted}
            )
        return requeued, exhausted

    # ── Outputs ──────────────────────────────────────────────────────────

    async def attach_output(
        self, job: GenerationJob, *, asset: Asset, is_primary: bool = True
    ) -> GenerationJobOutput:
        output = GenerationJobOutput(
            job_id=job.id, asset_id=asset.id, kind=asset.kind, is_primary=is_primary
        )
        self.session.add(output)
        await self.session.flush()
        return output

    async def load_inputs(self, job_id: uuid.UUID) -> list[tuple[GenerationJobInput, Asset]]:
        rows = await self.session.execute(
            select(GenerationJobInput, Asset)
            .join(Asset, Asset.id == GenerationJobInput.asset_id)
            .where(GenerationJobInput.job_id == job_id)
        )
        return [(row[0], row[1]) for row in rows]

    async def load_outputs(self, job_id: uuid.UUID) -> list[tuple[GenerationJobOutput, Asset]]:
        rows = await self.session.execute(
            select(GenerationJobOutput, Asset)
            .join(Asset, Asset.id == GenerationJobOutput.asset_id)
            .where(GenerationJobOutput.job_id == job_id)
            .order_by(GenerationJobOutput.is_primary.desc())
        )
        return [(row[0], row[1]) for row in rows]
