"""Worker coordination contracts — INTERNAL, never a public frontend API.

Everything in this module is served under `/api/v1/internal/*`, guarded by the
service token, and must not be reachable from a browser (directive §16). The
deployment rule that goes with it: the internal prefix is blocked at the edge,
so these endpoints are reachable only on the private network.

The claim response is the one place execution detail crosses a process
boundary — and it crosses to a worker, never to a client.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.enums import ErrorCode, JobStatus, WorkerStatus


class WorkerRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=120)
    """Stable per deployment slot, so a restart re-registers rather than
    creating a new node."""
    runtime: str = Field(default="mock", max_length=40)
    runtimes: list[str] = Field(default_factory=list, max_length=16)
    """
    Every runtime this node can actually execute.

    Empty means "trust `runtime` alone", which is also what a pre-M2 worker
    sends — so an old node keeps working while a fleet is mid-upgrade.
    """
    version: str = Field(default="", max_length=40)
    workflows: list[str] = Field(default_factory=list)
    """Workflow ids this node can execute. Unknown ids are rejected at
    registration, so a typo cannot silently make a worker idle forever."""
    max_concurrency: int = Field(default=1, ge=1, le=64)


class WorkerRegisterResponse(BaseModel):
    worker_id: uuid.UUID
    lease_seconds: int
    heartbeat_interval_seconds: int
    accepted_workflows: list[str]


class WorkerHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: uuid.UUID
    status: WorkerStatus = WorkerStatus.ONLINE
    active_job_ids: list[uuid.UUID] = Field(default_factory=list)


class WorkerHeartbeatResponse(BaseModel):
    acknowledged: bool
    """False when the API does not recognise the worker — the worker must
    re-register rather than keep reporting against a dead identity."""
    lease_seconds: int


class JobClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: uuid.UUID
    workflows: list[str] = Field(default_factory=list)
    """Narrows the claim. Empty means "anything I registered for"."""

    runtimes: list[str] = Field(default_factory=list, max_length=16)
    """
    Runtimes this node can execute, re-asserted on every claim.

    Sent per claim rather than trusted from registration because capability is a
    property of the running process: a node restarted with a different runtime
    must not inherit what its previous incarnation recorded.
    """


class ClaimedInput(BaseModel):
    role: str
    asset_id: uuid.UUID
    kind: str
    content_type: str
    download_url: str
    """Presigned GET so the worker pulls bytes straight from object storage,
    never through the API (scalability rule #2)."""


class ClaimedJob(BaseModel):
    job_id: uuid.UUID
    user_id: uuid.UUID
    workflow_id: str
    workflow_version: str

    prompt: str
    parameters: dict[str, Any]

    inputs: list[ClaimedInput] = Field(default_factory=list)

    execution: dict[str, Any] = Field(default_factory=dict)
    """
    The workflow's PRIVATE execution block — runtime, and from M2 the model and
    graph reference. This is the only place it leaves the backend, and it goes
    to an authenticated worker on the private network. It must never appear in
    any response under `/api/v1/generations` or `/api/v1/workflows`.
    """

    lease_token: uuid.UUID
    lease_expires_at: datetime
    attempt: int
    max_attempts: int

    output_upload_key: str
    """Where the worker must PUT its result, allocated by the API so key layout
    stays a backend concern."""
    output_upload_url: str
    output_content_type: str


class JobClaimResponse(BaseModel):
    job: ClaimedJob | None = None
    """None means "nothing available" — a normal, expected answer, not an error."""

    poll_after_seconds: int = 5


class JobProgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: uuid.UUID
    lease_token: uuid.UUID
    """Proves this worker still owns the job. A worker whose lease expired and
    was reassigned is rejected here, which is what stops a zombie process from
    overwriting the state of the worker that took over."""

    status: JobStatus
    progress: int = Field(ge=0, le=100)
    message: str = Field(default="", max_length=200)
    phase: str | None = Field(default=None, max_length=40)
    section_index: int | None = Field(default=None, ge=1)
    section_total: int | None = Field(default=None, ge=1)
    section_start_seconds: float | None = Field(default=None, ge=0)
    section_end_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _coherent_section(self) -> JobProgressRequest:
        fields = (
            self.section_index,
            self.section_total,
            self.section_start_seconds,
            self.section_end_seconds,
        )
        if any(value is not None for value in fields) and any(
            value is None for value in fields
        ):
            raise ValueError("section progress fields must be supplied together")
        if self.section_index is not None and self.section_total is not None:
            if self.section_index > self.section_total:
                raise ValueError("section_index cannot exceed section_total")
        if (
            self.section_start_seconds is not None
            and self.section_end_seconds is not None
            and self.section_end_seconds < self.section_start_seconds
        ):
            raise ValueError("section_end_seconds cannot precede section_start_seconds")
        return self


class JobCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: uuid.UUID
    lease_token: uuid.UUID

    output_key: str = Field(max_length=512)
    output_kind: str = Field(max_length=16)
    output_content_type: str = Field(max_length=120)
    size_bytes: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class JobFailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: uuid.UUID
    lease_token: uuid.UUID

    error_code: ErrorCode = ErrorCode.GENERATION_FAILED
    user_message: str = Field(default="", max_length=400)
    """Customer-safe copy. If a worker sends something unsuitable the API
    substitutes a generic message — the customer never receives worker
    internals (directive §23)."""
    internal_detail: str = Field(default="", max_length=2000)
    """Logged, never stored on the job and never returned to a client."""
    retriable: bool = True


class JobAckResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    accepted: bool
    """False when the report was ignored — a lost lease or an illegal
    transition. The worker should stop working on the job."""
    reason: str = ""
