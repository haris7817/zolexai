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
        lyrics: str | None = None,
        lyrics_language: str | None = None,
        prompt_mode: str | None = None,
        dialogue_language: str | None = None,
        sound: bool | None = None,
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
        else:
            # The Fast/Best round (27 Aug): a quality level may narrow the
            # duration ladder (Best's engine sells 5-30s, not 60s). The
            # narrowed list applies only when the submitted quality actually
            # names one; an invalid quality is its own problem below, and
            # duration then judges against the full ladder rather than
            # compounding the error message.
            offered = (
                definition.supported_durations_by_quality.get(quality or "")
                or definition.supported_durations
            )
            if duration is None:
                problems.append(
                    {
                        "field": "duration",
                        "reason": "A duration is required.",
                        "allowed": offered,
                    }
                )
            elif duration not in offered:
                problems.append(
                    {
                        "field": "duration",
                        "reason": "Unsupported duration for this tool.",
                        "allowed": offered,
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
            # Absent means the FIRST level — the default engine — exactly the
            # absence-is-default contract prompt_mode and sound follow, so a
            # client from before the toggle existed keeps its behaviour.
            if quality is not None and quality not in definition.supported_quality_levels:
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

        # Sound follows the lyrics policy: a workflow that does not declare
        # the control rejects the parameter. Absent means "with sound" — the
        # worker's own default — so every existing client is untouched.
        if sound is not None and not definition.settings.sound:
            problems.append(
                {"field": "sound", "reason": "This tool does not offer a sound choice."}
            )

        # Lyrics on a video workflow is a client bug, same policy as the rest:
        # present-and-unsupported is reported, never silently dropped.
        if not definition.settings.lyrics:
            if lyrics is not None:
                problems.append(
                    {"field": "lyrics", "reason": "This tool does not take lyrics."}
                )
            if lyrics_language is not None:
                problems.append(
                    {
                        "field": "lyrics_language",
                        "reason": "This tool does not take a lyrics language.",
                    }
                )

        # Prompt modes follow the lyrics policy exactly: a workflow that does
        # not declare the control rejects the parameter, and the dependent
        # language choice is only meaningful inside Director mode.
        if not definition.settings.prompt_modes:
            if prompt_mode is not None:
                problems.append(
                    {"field": "prompt_mode", "reason": "This tool does not offer prompt modes."}
                )
            if dialogue_language is not None:
                problems.append(
                    {
                        "field": "dialogue_language",
                        "reason": "This tool does not take a dialogue language.",
                    }
                )
        else:
            if prompt_mode is not None and prompt_mode not in PROMPT_MODES:
                problems.append(
                    {
                        "field": "prompt_mode",
                        "reason": "Unsupported prompt mode for this tool.",
                        "allowed": list(PROMPT_MODES),
                    }
                )
            if dialogue_language is not None:
                if prompt_mode != "director":
                    problems.append(
                        {
                            "field": "dialogue_language",
                            "reason": "A dialogue language only applies to Director mode.",
                        }
                    )
                elif dialogue_language not in DIALOGUE_LANGUAGES:
                    problems.append(
                        {
                            "field": "dialogue_language",
                            "reason": "Unsupported dialogue language.",
                            "allowed": list(DIALOGUE_LANGUAGES),
                        }
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


#: The two ways a prompt can be read on workflows that declare
#: `settings.prompt_modes`. Absent means `standard`, so existing clients are
#: untouched by the feature existing.
PROMPT_MODES: tuple[str, ...] = ("standard", "director")

#: Languages Director mode will write dialogue in. "auto" follows the idea's
#: own language; the named five are the set the video runtime's vendor
#: documents as validated for generated speech. Mirrored by the worker
#: (`worker.director.provider.DIALOGUE_LANGUAGES`) and the frontend selector.
DIALOGUE_LANGUAGES: tuple[str, ...] = (
    "auto",
    "english",
    "spanish",
    "french",
    "german",
    "russian",
)

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


#: What a runtime producing this `output_type` actually writes. Mirrors
#: `_OUTPUT_CONTENT_TYPE` in api/v1/internal.py, which is what the claim
#: presigns with when a definition declares nothing.
_NATIVE_CONTENT_TYPE = {"video": "video/mp4", "audio": "audio/mpeg", "image": "image/png"}


def _reject_mock_output_on_a_real_runtime(path: Path, definition: WorkflowDefinition) -> None:
    """`output_content_type` may only contradict `output_type` under the mock.

    The mock runtime writes a placeholder PNG whatever the workflow produces,
    so a video workflow declaring `image/png` is correct there — the claim
    presigns for what will actually be uploaded. Under a real runtime the same
    two lines are a silent data corruption: the worker uploads an MP4 to a key
    ending `.png`, signed as an image, and every customer's finished video is
    served with `Content-Type: image/png`. Browsers will not play it.

    Measured in production on 28 Aug 2026: 64 finished generations tagged
    `image/png` while their asset rows said `video/mp4`. The customer's report
    was "I can't download anything", and it had been true for weeks. The lines
    are deleted in the deployment's YAML and had been restored by a
    `git stash pop` — which is not a mistake anyone makes once.

    So it fails at load. An API that will not boot is discovered by whoever is
    deploying, in the minute they deploy; a mistagged asset is discovered by a
    customer, and only if they complain.
    """
    execution = definition.execution
    declared = execution.output_content_type
    if declared is None or execution.runtime == "mock":
        return
    native = _NATIVE_CONTENT_TYPE.get(definition.output_type)
    if native is None or declared == native:
        return
    raise WorkflowRegistryError(
        f"{path.name}: runtime '{execution.runtime}' produces {definition.output_type} "
        f"({native}) but execution.output_content_type says '{declared}'. That pairing "
        "signs every upload as the wrong type and the delivered media will not play. "
        "Delete output_content_type and output_kind — they belong to the mock runtime."
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

        _reject_mock_output_on_a_real_runtime(path, definition)

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
