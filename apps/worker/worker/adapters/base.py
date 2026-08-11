"""The provider abstraction (directive §12).

    Frontend → ZolexAI API → Workflow Service → **Workflow Adapter** → Provider

This module is that fourth arrow, and it is the single seam that keeps a model
change from becoming a product change. Everything above it — the frontend, the
public API, the job schema, the SSE contract — is expressed in ZolexAI's own
vocabulary. Everything below it is provider-specific and invisible.

Adding a real provider in M2 means writing one class that satisfies
`GenerationAdapter` and registering it. No route, schema, migration or component
changes, and no provider name appears anywhere a customer can reach.

Two rules an adapter must honour:

  * Report progress through `on_progress`, in ZolexAI's own lifecycle states.
    An adapter never invents a status the platform does not know.
  * Raise `AdapterError` for anything that fails. `user_message` is customer-safe
    copy; `internal_detail` is for the log and never leaves the backend.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Called by an adapter to report progress. (status, progress 0-100, message)
ProgressCallback = Callable[[str, int, str], Awaitable[None]]


@dataclass(frozen=True)
class AdapterInput:
    role: str
    kind: str
    content_type: str
    download_url: str
    """Presigned GET. The adapter streams from object storage directly — media
    never travels through the API."""


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

    def input_for(self, role: str) -> AdapterInput | None:
        return next((item for item in self.inputs if item.role == role), None)


@dataclass(frozen=True)
class AdapterResult:
    """What an adapter produces: bytes plus what they are."""

    content: bytes
    content_type: str
    kind: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None


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


@runtime_checkable
class GenerationAdapter(Protocol):
    name: str

    def supports(self, workflow_id: str) -> bool: ...

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        """Executes one generation.

        Must call `on_progress` at least once per lifecycle stage. Between calls
        the runner renews the job's lease, so a slow stage is safe as long as
        the adapter keeps reporting.
        """
        ...
