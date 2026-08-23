"""The dry-run manifest: what a provider WOULD do, without running a model.

Two jobs, one shape. It is the fair-comparison artifact for the LTX/H3
benchmark — every generation records the manifest it ran under, so a quality
score can always be traced back to the exact plan that produced it — and it is
the LTX regression net: the manifest is deterministic, so a golden copy of it
fails the moment provider work changes what LTX would send.

Nothing here runs, downloads, or allocates. Compilation is pure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReferenceSpec:
    """One input asset as the provider will present it to the model.

    `role` is ZolexAI vocabulary (identity, structure, seam, source_audio…),
    not a provider's — the whole point is that two providers describe the same
    customer intent in the same words and differ only in `native`.
    """

    role: str
    kind: str
    """image | video | audio."""

    native: str
    """How the provider actually carries it: an LTX `--image` triple, an H3
    `conditions[]` entry, and so on. Free text, provider-owned."""

    frame_index: int | None = None
    strength: float | None = None
    source: str = ""
    """A path placeholder or upload role. Never a signed URL."""


@dataclass(frozen=True)
class AudioWindow:
    """The slice of a track a section is given, in the track's own timeline."""

    start_seconds: float
    duration_seconds: float
    mode: str
    """How the model treats it — `frozen_latent` (LTX a2vid), `fully_copy` or
    `timbre_reference` (H3 Ref2VA), or `none`."""

    returns_input_waveform: bool = False


@dataclass(frozen=True)
class SectionPlan:
    index: int
    start_seconds: float
    duration_seconds: float
    frames_requested: int
    frames_rendered: int
    seed: int | None = None
    prompt: str = ""
    references: list[ReferenceSpec] = field(default_factory=list)
    audio: AudioWindow | None = None


@dataclass(frozen=True)
class GenerationManifest:
    provider: str
    workflow_id: str
    pipeline: str
    total_seconds: float
    width: int
    height: int
    fps: float
    sections: list[SectionPlan] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    """Anything a reader of the manifest must know that the numbers do not say
    — a dropped identity anchor, an unmeasured frame count, a capability the
    provider is emulating rather than supporting natively."""

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def seam_count(self) -> int:
        return max(0, len(self.sections) - 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
