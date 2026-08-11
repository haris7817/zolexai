"""Centralized contracts — the single definition of every status string.

Nothing in the codebase may write a lifecycle string literal. The worker, the
API, the database and the SSE stream all import from here, so adding a state is
one edit rather than a search-and-replace across three applications
(directive §6, §26).
"""

from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    """The generation lifecycle.

    Order matters: `ORDERED` below relies on declaration order to reject
    backwards transitions (a late progress report from a superseded worker must
    never drag a completed job back to `generating`).
    """

    QUEUED = "queued"
    ASSIGNED = "assigned"
    PREPARING = "preparing"
    GENERATING = "generating"
    POST_PROCESSING = "post_processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)

ACTIVE_STATUSES: frozenset[JobStatus] = frozenset(set(JobStatus) - set(TERMINAL_STATUSES))

#: Statuses a worker may claim. Only `queued` — an expired lease is requeued to
#: `queued` by the reaper rather than being claimable in place, so there is
#: exactly one path into a worker's hands.
CLAIMABLE_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.QUEUED})

_PROGRESSION: tuple[JobStatus, ...] = (
    JobStatus.QUEUED,
    JobStatus.ASSIGNED,
    JobStatus.PREPARING,
    JobStatus.GENERATING,
    JobStatus.POST_PROCESSING,
    JobStatus.UPLOADING,
)

_RANK: dict[JobStatus, int] = {status: index for index, status in enumerate(_PROGRESSION)}


def is_terminal(status: JobStatus) -> bool:
    return status in TERMINAL_STATUSES


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    """Whether `current -> target` is a legal move.

    Terminal is terminal: nothing leaves `completed`/`failed`/`cancelled`. This
    is what makes duplicate or delayed worker reports harmless — a retry that
    arrives after cancellation is rejected rather than resurrecting the job.
    """
    if current is target:
        return True
    if is_terminal(current):
        return False
    if is_terminal(target):
        return True
    return _RANK.get(target, -1) > _RANK.get(current, -1)


#: Customer-facing label for each internal status.
#:
#: The public API returns BOTH: `status` (this enum, the machine contract) and
#: `stage_label` (this map). Internal granularity the customer does not need —
#: `assigned` vs `queued`, `post_processing` vs `uploading` — collapses here, so
#: the worker can gain states without changing what the user reads.
STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.QUEUED: "Queued",
    JobStatus.ASSIGNED: "Queued",
    JobStatus.PREPARING: "Preparing",
    JobStatus.GENERATING: "Generating",
    JobStatus.POST_PROCESSING: "Finalizing",
    JobStatus.UPLOADING: "Finalizing",
    JobStatus.COMPLETED: "Completed",
    JobStatus.FAILED: "Failed",
    JobStatus.CANCELLED: "Cancelled",
}


class EventType(StrEnum):
    """SSE event names, also stored on `generation_events.event_type`."""

    STATUS = "status"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCode(StrEnum):
    """Stable machine-readable error codes.

    Returned to clients and written to `generation_jobs.error_code`. Distinct
    from the human message, which may be reworded freely without breaking a
    client that branches on the code.
    """

    VALIDATION_FAILED = "validation_failed"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    RATE_LIMITED = "rate_limited"
    CONCURRENCY_LIMIT_REACHED = "concurrency_limit_reached"
    IDEMPOTENCY_KEY_REUSED = "idempotency_key_reused"
    UNSUPPORTED_WORKFLOW = "unsupported_workflow"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    MISSING_REQUIRED_INPUT = "missing_required_input"
    ASSET_NOT_READY = "asset_not_ready"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    FILE_TOO_LARGE = "file_too_large"
    LEASE_LOST = "lease_lost"
    MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"
    GENERATION_FAILED = "generation_failed"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    INTERNAL_ERROR = "internal_error"


class AssetKind(StrEnum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class AssetSource(StrEnum):
    UPLOAD = "upload"
    GENERATED = "generated"


class AssetStatus(StrEnum):
    """An upload is only usable once the browser has confirmed the PUT."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class WorkerStatus(StrEnum):
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
