"""Public request/response contracts for generation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import JobStatus

RoleName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=48)]


class GenerationParameters(BaseModel):
    """The creative settings. Which of these apply is decided by the workflow.

    Every field is optional at the schema level because whether it applies is
    decided by the workflow, not by the shape: an automatic-duration workflow
    takes its length from the uploaded file and REJECTS a supplied duration,
    while a fixed-duration workflow requires one. Sending a parameter a
    workflow does not use is rejected rather than ignored, so a client bug
    surfaces instead of silently producing something the user did not ask for.
    """

    model_config = ConfigDict(extra="forbid")

    duration: str | None = Field(default=None, max_length=16)
    aspect_ratio: str | None = Field(default=None, max_length=16)
    quality: str | None = Field(default=None, max_length=32)

    motion_strength: int = Field(default=60, ge=0, le=100)
    prompt_adherence: int = Field(default=75, ge=0, le=100)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class GenerationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(max_length=64)
    prompt: str = Field(default="", max_length=20_000)
    parameters: GenerationParameters

    inputs: dict[RoleName, uuid.UUID] = Field(default_factory=dict)
    """
    Role -> asset id. Roles come from the workflow definition, so Video to Video
    can accept `{"source_video": ..., "reference_image": ...}` with the second
    optional, and no other endpoint or column changes (directive §14).
    """


class GenerationOutput(BaseModel):
    asset_id: uuid.UUID
    kind: str
    is_primary: bool
    url: str | None = None
    """Short-lived presigned URL. Responses carrying one are marked no-store."""


class GenerationInput(BaseModel):
    role: str
    asset_id: uuid.UUID
    kind: str


class GenerationError(BaseModel):
    code: str
    message: str
    """Customer-safe copy only. Never a stack trace or worker detail (§23)."""


class GenerationJobPublic(BaseModel):
    id: uuid.UUID
    workflow_id: str
    workflow_name: str

    status: JobStatus
    stage_label: str
    """Human label for `status`, resolved server-side so every client agrees."""
    progress: int
    hint: str

    prompt: str
    parameters: dict[str, Any]

    inputs: list[GenerationInput] = Field(default_factory=list)
    outputs: list[GenerationOutput] = Field(default_factory=list)

    error: GenerationError | None = None

    attempt_count: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    last_event_seq: int
    """Where to resume an SSE stream from — pass as `Last-Event-ID`."""

    is_terminal: bool


class GenerationAccepted(BaseModel):
    """202 body — the job exists and is queued; nothing has been generated yet.

    Returned immediately so the HTTP request never waits on generation
    (directive §6, scalability rule #3).
    """

    job_id: uuid.UUID
    status: JobStatus
    stage_label: str
    events_url: str
    """Where to subscribe for progress, so the client does not construct URLs."""
