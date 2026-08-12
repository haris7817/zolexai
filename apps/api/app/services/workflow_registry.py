"""Workflow registry — loads and validates `workflow-definitions/*.yaml`.

**These YAML files are the single source of truth for what ZolexAI can do.**
The API serves them, the worker dispatches on them, and the frontend builds its
tool surfaces from them. Nothing anywhere hard-codes the list of six workflows.

Validation happens at application startup (directive §11). A malformed or
inconsistent definition aborts the boot rather than surfacing as a broken
settings panel at runtime — a container that will not start is a far cheaper
failure than one that starts and quietly offers a control the backend rejects.

The registry is immutable once loaded, so it is safe to share across requests
and carries no per-instance state.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.core.config import settings
from app.core.enums import ErrorCode
from app.core.errors import AppError, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.schemas.workflow import WorkflowDefinition, WorkflowPublic

logger = get_logger(__name__)


class WorkflowRegistryError(RuntimeError):
    """Raised at startup when a definition file is unusable."""


class WorkflowRegistry:
    def __init__(self, definitions: dict[str, WorkflowDefinition]) -> None:
        self._definitions = definitions
        # Cached because the public list is requested on every app page load and
        # the projection is pure. Definitions never change after load.
        self._public: dict[str, WorkflowPublic] = {
            key: value.to_public() for key, value in definitions.items()
        }
        # File order is alphabetical and meaningless; display order is a product
        # decision, so it is declared here next to the reason it exists.
        self._order = [wid for wid in DISPLAY_ORDER if wid in definitions]
        self._order += sorted(set(definitions) - set(self._order))

    # ── Reads ────────────────────────────────────────────────────────────

    def ids(self) -> list[str]:
        return list(self._order)

    def ids_for_runtimes(self, runtimes: Sequence[str]) -> list[str]:
        """Workflows whose private `execution.runtime` a node can actually run.

        Routing has always been per workflow — the YAML decides which runtime
        should execute a job — but nothing checked that the claiming worker
        agreed. That was harmless while every definition said `mock`. The moment
        one says something else, a mock node claims that job, finds no adapter,
        and fails it with `retriable=False`: a permanently dead job, and the
        user is simply told the tool is unavailable.

        Intersecting here keeps the check on the side that owns the definitions,
        so a worker cannot claim work by asserting a capability the workflow
        never asked for.
        """
        wanted = set(runtimes)
        return [
            wid for wid in self._order if self._definitions[wid].execution.runtime in wanted
        ]

    def get(self, workflow_id: str) -> WorkflowDefinition:
        """Internal lookup — includes the private execution block."""
        definition = self._definitions.get(workflow_id)
        if definition is None:
            raise NotFound(
                "That creation tool is not available.",
                code=ErrorCode.UNSUPPORTED_WORKFLOW,
                details={"workflow_id": workflow_id},
            )
        return definition

    def get_public(self, workflow_id: str) -> WorkflowPublic:
        self.get(workflow_id)  # raises NotFound with a customer-safe message
        return self._public[workflow_id]

    def list_public(self) -> list[WorkflowPublic]:
        return [self._public[wid] for wid in self._order]

    def __contains__(self, workflow_id: object) -> bool:
        return workflow_id in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)

    # ── Request validation ───────────────────────────────────────────────

    def validate_request(
        self,
        workflow_id: str,
        *,
        prompt: str,
        duration: str | None,
        aspect_ratio: str | None,
        quality: str | None,
        input_roles: set[str],
    ) -> WorkflowDefinition:
        """Checks a submitted generation request against its workflow.

        Every rejection names the offending field and the permitted values, so a
        client can render a useful message without a lookup table of its own.

        This is the only place request shape is judged. The worker trusts what
        it claims, because nothing reaches the queue without passing here.
        """
        definition = self.get(workflow_id)
        problems: list[dict[str, Any]] = []

        cleaned_prompt = prompt.strip()
        if definition.prompt.required and not cleaned_prompt:
            problems.append({"field": "prompt", "reason": "A prompt is required."})
        elif len(cleaned_prompt) > definition.prompt.max_length:
            problems.append(
                {
                    "field": "prompt",
                    "reason": f"Keep the prompt under {definition.prompt.max_length} characters.",
                }
            )

        # Duration is judged by the workflow's mode. `source` sets the length
        # from the uploaded file, so a supplied duration is a client bug — the
        # user was never offered the choice, and honouring a stray value would
        # quietly contradict what the UI promised ("Same as source").
        if definition.duration_mode == "source":
            if duration is not None:
                problems.append(
                    {
                        "field": "duration",
                        "reason": "This tool sets the duration from your file automatically.",
                    }
                )
        elif duration is None:
            problems.append(
                {
                    "field": "duration",
                    "reason": "A duration is required.",
                    "allowed": definition.supported_durations,
                }
            )
        elif duration not in definition.supported_durations:
            problems.append(
                {
                    "field": "duration",
                    "reason": "Unsupported duration for this tool.",
                    "allowed": definition.supported_durations,
                }
            )

        # Absent-and-unsupported is correct (audio has no frame); present-and-
        # unsupported is a client bug worth reporting rather than ignoring.
        if definition.supported_aspect_ratios:
            if aspect_ratio not in definition.supported_aspect_ratios:
                problems.append(
                    {
                        "field": "aspect_ratio",
                        "reason": "Unsupported aspect ratio for this tool.",
                        "allowed": definition.supported_aspect_ratios,
                    }
                )
        elif aspect_ratio is not None:
            problems.append(
                {"field": "aspect_ratio", "reason": "This tool does not use an aspect ratio."}
            )

        if definition.supported_quality_levels:
            if quality not in definition.supported_quality_levels:
                problems.append(
                    {
                        "field": "quality",
                        "reason": "Unsupported quality level for this tool.",
                        "allowed": definition.supported_quality_levels,
                    }
                )
        elif quality is not None:
            problems.append(
                {"field": "quality", "reason": "This tool does not use a quality level."}
            )

        missing = [role for role in definition.required_roles if role not in input_roles]
        if missing:
            problems.append(
                {
                    "field": "inputs",
                    "reason": "Required input missing.",
                    "missing_roles": missing,
                }
            )

        unknown = sorted(input_roles - definition.known_roles)
        if unknown:
            problems.append(
                {
                    "field": "inputs",
                    "reason": "This tool does not accept those inputs.",
                    "unknown_roles": unknown,
                    "allowed_roles": sorted(definition.known_roles),
                }
            )

        if problems:
            raise ValidationFailed(
                "Some settings are not valid for this tool.",
                details={"fields": problems},
                code=ErrorCode.UNSUPPORTED_PARAMETER
                if all(p["field"] != "inputs" for p in problems)
                else ErrorCode.MISSING_REQUIRED_INPUT,
            )

        return definition


#: Product display order. Every workflow surface renders in this sequence so the
#: sidebar, All Tools, the dashboard and the landing grid can never disagree.
DISPLAY_ORDER: tuple[str, ...] = (
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "extend-video",
    "music",
    "music-video",
)


def load_registry(directory: Path | None = None) -> WorkflowRegistry:
    """Parses and validates every definition. Raises on the first problem."""
    root = directory or settings.workflow_definitions_dir

    if not root.is_dir():
        raise WorkflowRegistryError(f"Workflow definitions directory not found: {root}")

    paths = sorted(p for p in root.glob("*.yaml") if not p.name.startswith("_"))
    if not paths:
        raise WorkflowRegistryError(f"No workflow definitions found in {root}")

    definitions: dict[str, WorkflowDefinition] = {}

    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise WorkflowRegistryError(f"{path.name}: not valid YAML — {exc}") from exc

        if not isinstance(raw, dict):
            raise WorkflowRegistryError(f"{path.name}: expected a mapping at the top level")

        try:
            definition = WorkflowDefinition.model_validate(raw)
        except ValidationError as exc:
            raise WorkflowRegistryError(f"{path.name}: {exc}") from exc

        # Filename and id must agree, otherwise `text-to-video.yaml` could
        # define `music` and every reader would disagree about which file to
        # edit.
        if definition.id != path.stem:
            raise WorkflowRegistryError(
                f"{path.name}: id '{definition.id}' does not match the filename"
            )
        if definition.id in definitions:
            raise WorkflowRegistryError(f"duplicate workflow id '{definition.id}'")

        definitions[definition.id] = definition

    logger.info(
        "workflow_registry_loaded",
        extra={"workflow_count": len(definitions), "workflow_ids": sorted(definitions)},
    )
    return WorkflowRegistry(definitions)


_registry: WorkflowRegistry | None = None


def init_registry(directory: Path | None = None) -> WorkflowRegistry:
    global _registry
    _registry = load_registry(directory)
    return _registry


def get_registry() -> WorkflowRegistry:
    if _registry is None:
        raise AppError("The workflow registry is not initialised.")
    return _registry
