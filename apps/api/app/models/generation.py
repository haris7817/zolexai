"""Generation job, its inputs, its outputs and its event log.

This is the heart of the system. Four tables:

  generation_jobs          one row per submitted request
  generation_job_inputs    the assets a job consumes (0..n, by role)
  generation_job_outputs   the assets a job produced (0..n)
  generation_events        append-only lifecycle log, and the SSE replay source

Design notes worth knowing before changing anything here:

* **Leasing, not assignment.** A worker does not own a job; it holds a lease
  that expires. If the worker dies, `lease_expires_at` passes and the reaper
  requeues the job. This is what removes the single-permanent-worker assumption
  (directive §8).

* **`last_event_seq` is the SSE cursor.** Every event gets a per-job sequence
  number that becomes the SSE `id:` field. A reconnecting browser sends
  `Last-Event-ID`, and the API replays from the database before subscribing to
  live updates — so no event is lost in the gap between the two.

* **Nothing is deleted.** Cancel and failure are states, not row removals, so
  history and future usage accounting stay intact.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import AssetKind, ErrorCode, EventType, JobStatus
from app.db.base import Base, created_at_col, enum_column, updated_at_col, uuid_pk


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    id: Mapped[uuid.UUID] = uuid_pk()

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Workflows live in version-controlled YAML, not in a table, so this is a
    # plain string. `workflow_version` records which revision produced the job,
    # which is what makes an old result explainable after the definition moves on.
    workflow_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    workflow_version: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="1")

    status: Mapped[JobStatus] = mapped_column(
        enum_column(JobStatus), nullable=False, default=JobStatus.QUEUED
    )
    progress: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False, default=0)
    stage_hint: Mapped[str] = mapped_column(sa.String(200), nullable=False, default="")
    """Supporting line under the status, authored by the worker."""

    prompt: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    request_params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sa.text("'{}'::jsonb")
    )
    """The validated request exactly as accepted — duration, aspect, quality, advanced settings."""

    # ── Failure ──────────────────────────────────────────────────────────
    error_code: Mapped[ErrorCode | None] = mapped_column(enum_column(ErrorCode, 48), nullable=True)
    error_message: Mapped[str | None] = mapped_column(sa.String(400), nullable=True)
    """Customer-safe copy only. Internal detail belongs in the logs (rule §23)."""

    # ── Worker leasing ───────────────────────────────────────────────────
    worker_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("worker_nodes.id", ondelete="SET NULL"),
        nullable=True,
    )
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    """
    Rotated on every claim. A worker must present it to report progress, so a
    zombie that wakes after its lease expired cannot overwrite the state of the
    worker that took over.
    """
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    attempt_count: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False, default=3)

    # ── Duplicate suppression (directive §24) ────────────────────────────
    idempotency_key: Mapped[str | None] = mapped_column(sa.String(128), nullable=True)

    # ── SSE cursor ───────────────────────────────────────────────────────
    last_event_seq: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()
    started_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    inputs: Mapped[list[GenerationJobInput]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    outputs: Mapped[list[GenerationJobOutput]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        # ── The four indexes that matter ─────────────────────────────────
        #
        # 1. History list: WHERE user_id = ? ORDER BY created_at DESC.
        #    DESC in the index matches the ORDER BY so pagination never sorts.
        sa.Index("ix_generation_jobs_user_created", "user_id", sa.text("created_at DESC")),
        # 2. Status filter within a user's history, and the concurrency check
        #    ("how many of this user's jobs are still running?").
        sa.Index(
            "ix_generation_jobs_user_status_created",
            "user_id",
            "status",
            sa.text("created_at DESC"),
        ),
        # 3. Workflow filter within a user's history.
        sa.Index(
            "ix_generation_jobs_user_workflow_created",
            "user_id",
            "workflow_id",
            sa.text("created_at DESC"),
        ),
        # 4. The worker claim: WHERE status='queued' ORDER BY created_at
        #    FOR UPDATE SKIP LOCKED. Partial, because the queue is a tiny
        #    fraction of a large table and only queued rows are ever scanned.
        sa.Index(
            "ix_generation_jobs_claimable",
            "created_at",
            postgresql_where=sa.text("status = 'queued'"),
        ),
        # The lease reaper. Also partial — only leased rows have an expiry.
        sa.Index(
            "ix_generation_jobs_lease_expiry",
            "lease_expires_at",
            postgresql_where=sa.text("lease_expires_at IS NOT NULL"),
        ),
        # Idempotency is scoped per user so two customers may reuse a key.
        # Partial, so the overwhelming majority of rows (no key) cost nothing.
        sa.Index(
            "uq_generation_jobs_user_idempotency",
            "user_id",
            "idempotency_key",
            unique=True,
            postgresql_where=sa.text("idempotency_key IS NOT NULL"),
        ),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="progress_range"),
        sa.CheckConstraint("attempt_count >= 0", name="attempt_count_positive"),
    )

    def __repr__(self) -> str:
        return f"<GenerationJob {self.id} {self.workflow_id} {self.status}>"


class GenerationJobInput(Base):
    """An asset a job consumes, tagged by the role the workflow expects.

    `role` is what makes the optional video-to-video reference image possible
    (directive §14) without a schema change per workflow: a job simply has a
    `source_video` input and, when supplied, a `reference_image` input. Adding a
    workflow that takes three inputs adds rows, not columns.
    """

    __tablename__ = "generation_job_inputs"

    id: Mapped[uuid.UUID] = uuid_pk()

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("generation_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("assets.id", ondelete="RESTRICT"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(sa.String(48), nullable=False)
    created_at: Mapped[datetime] = created_at_col()

    job: Mapped[GenerationJob] = relationship(back_populates="inputs")

    __table_args__ = (
        sa.UniqueConstraint("job_id", "role", name="uq_generation_job_inputs_job_role"),
        sa.Index("ix_generation_job_inputs_job", "job_id"),
    )


class GenerationJobOutput(Base):
    __tablename__ = "generation_job_outputs"

    id: Mapped[uuid.UUID] = uuid_pk()

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("generation_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("assets.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[AssetKind] = mapped_column(enum_column(AssetKind), nullable=False)
    is_primary: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)
    """The result the UI shows. A job may also emit a thumbnail or preview."""

    created_at: Mapped[datetime] = created_at_col()

    job: Mapped[GenerationJob] = relationship(back_populates="outputs")

    __table_args__ = (sa.Index("ix_generation_job_outputs_job", "job_id"),)


class GenerationEvent(Base):
    """Append-only lifecycle log — the durable half of SSE.

    Live delivery goes through Redis pub/sub, which is fire-and-forget: a
    subscriber that is not connected at that instant misses the message. This
    table is what makes reconnection lossless. The API replays
    `seq > Last-Event-ID` from here, then attaches to the live channel.

    `seq` is per-job and allocated under the job row's lock, so it is dense and
    strictly increasing without a global sequence becoming a write bottleneck.
    """

    __tablename__ = "generation_events"

    id: Mapped[uuid.UUID] = uuid_pk()

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("generation_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    event_type: Mapped[EventType] = mapped_column(enum_column(EventType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(enum_column(JobStatus), nullable=False)
    progress: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False, default=0)
    message: Mapped[str] = mapped_column(sa.String(200), nullable=False, default="")

    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sa.text("'{}'::jsonb")
    )

    created_at: Mapped[datetime] = created_at_col()

    __table_args__ = (
        sa.UniqueConstraint("job_id", "seq", name="uq_generation_events_job_seq"),
        # Replay is always "this job, after this sequence" — one index serves
        # both the uniqueness guarantee and the range scan.
        sa.Index("ix_generation_events_job_seq", "job_id", "seq"),
    )

    def __repr__(self) -> str:
        return f"<GenerationEvent job={self.job_id} seq={self.seq} {self.event_type}>"
