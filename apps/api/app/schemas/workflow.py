"""Workflow definition schema and its public projection.

Two models, and the distinction is the whole point (directive §11, §12):

  `WorkflowDefinition` — the full parsed YAML, including the optional
      `execution` block. Backend and worker only.

  `WorkflowPublic` — what `GET /api/v1/workflows` returns. Built by explicit
      projection, never by `.model_dump()` of the definition, so a field added
      to `execution` in M2 cannot leak to the browser by accident.

`execution` is empty in every M1 definition; the block exists so that adding a
provider in M2 does not require reshaping the registry.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WorkflowCategory = Literal["video", "audio"]
OutputType = Literal["video", "audio", "image"]
AssetKindLiteral = Literal["video", "image", "audio"]

DurationMode = Literal["fixed", "source", "minutes"]
"""How a workflow's output duration is decided (M2, client requirement §5/8/10).

  `fixed`   — the user picks from `supported_durations` ("5s", "10s", …).
  `source`  — the duration comes from the uploaded source file; the user picks
              nothing and a request supplying a duration is rejected. Video to
              Video and Music Video work this way: a 45-second source yields a
              45-second result.
  `minutes` — music: the user picks a length in minutes from
              `supported_durations` ("1m" … "5m"). The list's ceiling is a
              product decision pending model benchmarking, so raising it later
              is a YAML edit, not a code change.

One list, three meanings. `supported_durations` stays the single membership
check for every mode that has choices, so validation did not grow a second
mechanism — the mode only changes what the strings look like and how the UI
renders them.
"""

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)]
RoleName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=48)]


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = True
    placeholder: str = ""
    max_length: int = Field(default=2000, ge=1, le=20_000)


class InputSpec(BaseModel):
    """One asset a workflow consumes, addressed by role.

    Roles rather than positional fields are what let Video to Video gain an
    OPTIONAL reference image (directive §14) without a schema migration: the
    request simply carries one more `role -> asset_id` pair. A workflow needing
    three inputs adds a list entry, not a column.
    """

    model_config = ConfigDict(extra="forbid")

    role: RoleName
    kind: AssetKindLiteral
    required: bool = True

    label: str
    """Section heading in the settings panel, e.g. "SOURCE VIDEO"."""
    drop_hint: str
    """Reads inside "Drop {drop_hint} here"."""
    help: str = ""
    """Optional clarifying line — used to set expectations on optional inputs."""

    accept: list[str] = Field(default_factory=list)
    """Exact MIME types permitted. Empty means the kind's default set applies."""
    max_size_mb: int = Field(default=512, ge=1, le=10_240)


class SettingsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: bool = False
    motion_strength: bool = False
    prompt_adherence: bool = False
    seed: bool = False
    lyrics: bool = False
    """Whether the panel offers a lyrics box and a lyric-language choice.
    Music only; a video workflow receiving lyrics is a client bug and is
    rejected in `validate_request`."""


class CapabilitiesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    download: bool = True
    extend: bool = False
    reuse_settings: bool = True
    variation: bool = False


class UiSpec(BaseModel):
    """Presentation metadata.

    This lives in the YAML rather than the frontend so the icon, gradient and
    marketing copy for a workflow cannot drift from its behaviour — one file
    describes a tool completely. It is public by design: "UI settings" is on the
    directive's list of publishable workflow information.
    """

    model_config = ConfigDict(extra="forbid")

    icon: Literal["sparkles", "image", "repeat", "extend", "music", "clapper"]
    thumb: str
    """CSS gradient used as a placeholder surface for cards and thumbnails."""


class ExecutionSpec(BaseModel):
    """PRIVATE. Never projected into a public response.

    M2 fills this with runner, graph file, model reference and hardware
    requirements. Kept permissive (`extra="allow"`) so adding a field then does
    not require touching this class.
    """

    model_config = ConfigDict(extra="allow")

    runtime: str = "mock"

    output_content_type: str | None = None
    """
    What this runtime ACTUALLY produces, when it differs from `output_type`.

    In M1 every workflow runs the mock runtime, which emits a placeholder image
    rather than rendered video or audio — so the declared output stays `video`
    (the product truth, and what the UI is designed around) while the bytes on
    disk are a PNG. The API signs the upload for this type, so the worker cannot
    quietly upload something other than what was agreed.

    Deleting this line is part of wiring up a real provider in M2.
    """
    output_kind: Literal["video", "image", "audio"] | None = None
    """Asset kind for what the runtime produces. Pairs with the field above."""


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Slug
    version: str = "1"
    name: str
    category: WorkflowCategory
    output_type: OutputType

    description: str
    short_description: str = ""
    marketing_description: str = ""

    prompt: PromptSpec = Field(default_factory=PromptSpec)
    inputs: list[InputSpec] = Field(default_factory=list)

    duration_mode: DurationMode = "fixed"
    supported_durations: list[str] = Field(default_factory=list)
    """Non-empty for `fixed`/`minutes`; must be empty for `source` — enforced in
    the consistency validator rather than by `min_length`, because the
    constraint depends on the mode."""
    supported_aspect_ratios: list[str] = Field(default_factory=list)
    supported_quality_levels: list[str] = Field(default_factory=list)

    settings: SettingsSpec = Field(default_factory=SettingsSpec)
    capabilities: CapabilitiesSpec = Field(default_factory=CapabilitiesSpec)
    ui: UiSpec

    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> WorkflowDefinition:
        roles = [spec.role for spec in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("input roles must be unique within a workflow")

        # A workflow that advertises a quality control but offers no levels
        # would render an empty segmented control. Caught at startup rather
        # than discovered as a blank space in the UI.
        if self.settings.quality and not self.supported_quality_levels:
            raise ValueError("settings.quality is true but supported_quality_levels is empty")
        if not self.settings.quality and self.supported_quality_levels:
            raise ValueError("supported_quality_levels is set but settings.quality is false")

        # Audio has no frame. Advertising an aspect ratio for it would be a
        # control the result cannot honour.
        if self.output_type == "audio" and self.supported_aspect_ratios:
            raise ValueError("audio output cannot declare aspect ratios")

        # Extending audio is not meaningful, and the result actions are driven
        # straight off capabilities — so this would render an Extend button
        # that produces nothing.
        if self.output_type == "audio" and self.capabilities.extend:
            raise ValueError("audio output cannot declare the extend capability")

        if not self.prompt.required and not self.inputs:
            raise ValueError("a workflow must require either a prompt or at least one input")

        # ── Duration mode coherence ───────────────────────────────────
        # Each mode has a shape the UI and the validator rely on; a file that
        # mixes them would render a broken control or validate the wrong thing,
        # so it must not load at all.
        if self.duration_mode == "source":
            if self.supported_durations:
                raise ValueError(
                    "duration_mode 'source' takes its length from the uploaded file; "
                    "supported_durations must be empty"
                )
            # The length has to come from somewhere: a required media input.
            if not any(s.required and s.kind in ("video", "audio") for s in self.inputs):
                raise ValueError(
                    "duration_mode 'source' requires a required video or audio input "
                    "to derive the duration from"
                )
        else:
            if not self.supported_durations:
                raise ValueError(f"duration_mode '{self.duration_mode}' needs supported_durations")
            suffix = "m" if self.duration_mode == "minutes" else "s"
            bad = [d for d in self.supported_durations if not _is_duration(d, suffix)]
            if bad:
                raise ValueError(
                    f"duration_mode '{self.duration_mode}' entries must look like "
                    f"'30{suffix}'; got {bad}"
                )

        return self

    # ── Lookups used by request validation ───────────────────────────────

    def input_for(self, role: str) -> InputSpec | None:
        return next((spec for spec in self.inputs if spec.role == role), None)

    @property
    def required_roles(self) -> list[str]:
        return [spec.role for spec in self.inputs if spec.required]

    @property
    def known_roles(self) -> set[str]:
        return {spec.role for spec in self.inputs}

    def to_public(self) -> WorkflowPublic:
        """Explicit field-by-field projection.

        Deliberately NOT `model_dump(exclude={"execution"})`: an exclude list is
        a denylist, and a denylist silently leaks the next private field
        somebody adds. This is an allowlist.
        """
        return WorkflowPublic(
            id=self.id,
            version=self.version,
            name=self.name,
            category=self.category,
            output_type=self.output_type,
            description=self.description,
            short_description=self.short_description,
            marketing_description=self.marketing_description,
            prompt=self.prompt,
            inputs=self.inputs,
            duration_mode=self.duration_mode,
            supported_durations=self.supported_durations,
            supported_aspect_ratios=self.supported_aspect_ratios,
            supported_quality_levels=self.supported_quality_levels,
            settings=self.settings,
            capabilities=self.capabilities,
            ui=self.ui,
        )


def _is_duration(value: str, suffix: str) -> bool:
    body = value.removesuffix(suffix)
    return value.endswith(suffix) and body.isdigit() and int(body) > 0


class WorkflowPublic(BaseModel):
    """The customer-facing shape. Contains no provider, model or runtime detail."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    name: str
    category: WorkflowCategory
    output_type: OutputType

    description: str
    short_description: str
    marketing_description: str

    prompt: PromptSpec
    inputs: list[InputSpec]

    duration_mode: DurationMode
    supported_durations: list[str]
    supported_aspect_ratios: list[str]
    supported_quality_levels: list[str]

    settings: SettingsSpec
    capabilities: CapabilitiesSpec
    ui: UiSpec


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowPublic]

    def model_dump_public(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
