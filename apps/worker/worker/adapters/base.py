"""The provider abstraction (directive §12).

    Frontend → ZolexAI API → Workflow Service → **Workflow Adapter** → Provider

This module is that fourth arrow, and it is the single seam that keeps a model
change from becoming a product change. Everything above it — the frontend, the
public API, the job schema, the SSE contract — is expressed in ZolexAI's own
vocabulary. Everything below it is provider-specific and invisible.

Adding a real provider in M2 means writing one class that satisfies
`GenerationAdapter` and registering it. No route, schema, migration or component
changes, and no provider name appears anywhere a customer can reach.

Four rules an adapter must honour:

  * Report progress through `on_progress`, in ZolexAI's own lifecycle states.
    An adapter never invents a status the platform does not know.
  * Raise `AdapterError` for anything that fails. `user_message` is customer-safe
    copy; `internal_detail` is for the log and never leaves the backend.
  * Write every intermediate and final file inside `job.workspace`. The runner
    deletes that directory afterwards; anything written elsewhere leaks.
  * Call `job.raise_if_cancelled()` between units of work, and clean up in a
    `finally`. Cancellation and timeouts both arrive as exceptions through the
    adapter's own call stack.

## Why the result is a path and not bytes

M1 returned `content: bytes`, which was correct for a placeholder PNG and wrong
for anything real: a minute of 1080p video is hundreds of megabytes, held in RAM
per concurrent job, and then handed to a single buffered PUT. Returning a path
lets the file be produced incrementally by an external process — which is what
every real provider does — and streamed to storage without ever being resident.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Called by an adapter to report progress. (status, progress 0-100, message)
ProgressCallback = Callable[[str, int, str], Awaitable[None]]


def parse_duration_seconds(value: object) -> float | None:
    """Turns a requested duration string into seconds: "10s" → 10.0, "3m" → 180.0.

    Durations travel as opaque display strings ("5s", "3m" — music is chosen in
    minutes), and the worker is the first place they must become numbers. Kept
    here because every adapter needs the same reading; None means the request
    carried no usable duration, which is normal for automatic-duration
    workflows where the source file decides.
    """
    text = str(value or "").strip().lower()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("m"):
        multiplier, text = 60.0, text[:-1]
    elif text.endswith("s"):
        text = text[:-1]
    try:
        seconds = float(text) * multiplier
    except ValueError:
        return None
    return seconds if seconds > 0 else None


class JobCancelled(Exception):
    """The job should stop: the user cancelled it, or the lease was lost.

    Raised inside the adapter's call stack so ordinary `finally` blocks run and
    subprocesses get killed. An adapter should let it propagate.
    """


class JobTimedOut(Exception):
    """The adapter exceeded its wall-clock budget."""


@dataclass(frozen=True)
class AdapterInput:
    role: str
    kind: str
    content_type: str
    download_url: str
    """Presigned GET. Media never travels through the API."""

    path: Path | None = None
    """
    Where the runner staged this input on local disk.

    Populated before `run` is called, so an adapter reads a file rather than
    re-fetching a URL the runner has already downloaded once.
    """

    def require_path(self) -> Path:
        """The staged file, or a clear internal error if staging was skipped."""
        if self.path is None:
            raise AdapterError(
                "One of the selected files could not be read.",
                internal_detail=f"input '{self.role}' was not staged to disk",
                retriable=False,
            )
        return self.path


@dataclass(frozen=True)
class AdapterJob:
    job_id: str
    workflow_id: str
    workflow_version: str
    prompt: str
    parameters: dict[str, Any]
    inputs: list[AdapterInput] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    """The workflow's private execution block — runtime, and in M2 the model
    and graph reference."""

    output_content_type: str = "application/octet-stream"

    workspace: Path = field(default_factory=Path)
    """Scratch directory owned by this job. Deleted when the job ends."""

    _cancelled: asyncio.Event | None = None
    _deadline_monotonic: float | None = None

    def input_for(self, role: str) -> AdapterInput | None:
        return next((item for item in self.inputs if item.role == role), None)

    # ── Cooperative cancellation ─────────────────────────────────────────

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled is not None and self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        """Call between units of work — segments, polls, encode steps.

        Cheap enough to call in a tight loop, and the only thing that makes a
        cancelled job stop consuming GPU before it finishes on its own.
        """
        if self.is_cancelled:
            raise JobCancelled(f"job {self.job_id} cancelled")
        if self.seconds_remaining is not None and self.seconds_remaining <= 0:
            raise JobTimedOut(f"job {self.job_id} exceeded its time budget")

    @property
    def seconds_remaining(self) -> float | None:
        if self._deadline_monotonic is None:
            return None
        return self._deadline_monotonic - time.monotonic()

    @property
    def cancellation_event(self) -> asyncio.Event | None:
        """The runner's cancel signal, for racing long operations against.

        `raise_if_cancelled()` covers work that checkpoints naturally; an
        adapter awaiting one long external operation (a big ffmpeg re-encode)
        can instead wait on this event concurrently and abandon the operation
        the moment the job dies. None when no runner attached one (tests,
        tooling) — cancellation then simply cannot happen.
        """
        return self._cancelled

    # ── Convenience for adapters ─────────────────────────────────────────

    def execution_int(self, key: str, default: int) -> int:
        """Reads a tuning value from the private execution block.

        `ExecutionSpec` is `extra="allow"`, so a workflow can carry
        `max_segment_seconds`, `timeout_seconds` or a model reference without
        any schema change — which is exactly how M2 tunes per workflow.
        """
        try:
            return int(self.execution[key])
        except (KeyError, TypeError, ValueError):
            return default

    def execution_float(self, key: str, default: float) -> float:
        """The same, for conditioning strengths and other fractional dials."""
        try:
            return float(self.execution[key])
        except (KeyError, TypeError, ValueError):
            return default


@dataclass(frozen=True)
class AdapterResult:
    """What an adapter produces: a file on disk plus what it is."""

    path: Path
    content_type: str
    kind: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


class AdapterError(Exception):
    """A generation failure with a clean split between the two audiences."""

    def __init__(
        self,
        user_message: str,
        *,
        internal_detail: str = "",
        retriable: bool = True,
    ) -> None:
        self.user_message = user_message
        self.internal_detail = internal_detail or user_message
        self.retriable = retriable
        super().__init__(internal_detail or user_message)


async def cancellable(job: AdapterJob, operation: Awaitable[Any]) -> Any:
    """Awaits a long operation, abandoning it the moment the job dies.

    `raise_if_cancelled()` covers work that checkpoints naturally — between
    segments, between polls. This covers the other case: one external call that
    runs for minutes without returning, which on a long-form job is most of the
    wall clock. The media helpers kill their child process when their task is
    cancelled, so racing the operation against the runner's cancel event stops a
    re-encode within milliseconds instead of at its end.

    Without an event (tests, tooling) this is a plain await, because
    cancellation then cannot happen.
    """
    event = job.cancellation_event
    if event is None:
        return await operation

    op = asyncio.ensure_future(operation)
    watcher = asyncio.ensure_future(event.wait())
    try:
        done, _ = await asyncio.wait({op, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if op in done:
            return op.result()
        op.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await op
        job.raise_if_cancelled()
        raise JobCancelled(f"job {job.job_id} cancelled")
    finally:
        watcher.cancel()


@runtime_checkable
class GenerationAdapter(Protocol):
    name: str

    def supports(self, workflow_id: str) -> bool: ...

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        """Executes one generation.

        Must call `on_progress` at least once per lifecycle stage. Between calls
        the runner keeps the lease alive on the adapter's behalf, so a long
        silent stage is safe — but frequent reports are still what the user sees
        as a moving progress bar.

        Write output into `job.workspace` and return its path.
        """
        ...
