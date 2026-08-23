"""LTX draft → decoded RGB → H3 reference. Compiled, never run.

The hybrid is a benchmark strategy, not a provider. It has no entry in the
routing table, `auto` cannot reach it, and nothing here generates anything:
what it produces is a plan, a handoff manifest per section, and a final H3
plan that carries the draft alongside — never instead of — the customer's own
assets.

That last point is the rule the module exists to enforce. The interesting
hypothesis is not "LTX made a video, give it to H3". It is:

    the customer's photograph owns WHO,
    the LTX draft owns approximately HOW IT MOVES,
    the prompt owns WHAT HAPPENS,
    and H3 regenerates from all three.

Collapse that into "hand H3 the draft" and the original identity asset is
gone, the model is conditioned on our own invention, and an identity score
measures how well H3 reproduced an LTX hallucination. So every reference
carries its provenance (`user_asset` or `generated_intermediate`), a hybrid
plan is refused outright if it would drop an original asset, and the
handoff manifest names both halves separately.

Whether any of this beats plain H3 is unknown and unclaimed. It costs two
inference passes and a model switch; it has to earn that back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from worker.adapters.base import AdapterJob
from worker.providers.base import ProviderRefusal
from worker.providers.h3 import H3_MAX_CLIP_SECONDS, H3_MAX_FILES, H3Provider
from worker.providers.ltx import LtxProvider
from worker.providers.manifest import GenerationManifest, ReferenceSpec, SectionPlan
from worker.providers.strategy import (
    HYBRID_EXCLUDED,
    HYBRID_RATIONALE,
    GenerationStrategy,
    HandoffForm,
    default_handoff_form,
    hybrid_allowed,
)

GENERATED = "generated_intermediate"
USER = "user_asset"


@dataclass(frozen=True)
class HandoffManifest:
    """One section's worth of handoff, with provenance on its face.

    Deliberately provider-neutral and deliberately explicit about which side
    of the boundary each asset came from — a reader who cannot tell the
    customer's photograph from our draft cannot audit the comparison.
    """

    strategy: str
    source_job_id: str
    workflow_id: str
    draft_provider: str
    final_provider: str
    section_index: int
    duration_seconds: float
    handoff_form: str

    original_references: list[ReferenceSpec] = field(default_factory=list)
    """The customer's own assets, carried through untouched."""

    generated_references: list[ReferenceSpec] = field(default_factory=list)
    """What LTX produced. Every entry has origin=generated_intermediate."""

    draft_window: tuple[float, float] | None = None
    """The slice of the decoded draft this section consumes, in the draft's
    own timeline. `None` when no video is handed over."""

    selected_frames: list[float] = field(default_factory=list)
    """Timestamps, in the draft's timeline, of any stills handed over."""

    audio_reference: str = ""
    """The customer's track, when the workflow has one. Never the draft's own
    audio: a music video must be measured against the real song."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridPlan:
    """Both passes and the bridge between them."""

    strategy: str
    workflow_id: str
    draft: GenerationManifest
    final: GenerationManifest
    handoffs: list[HandoffManifest]
    rationale: str
    notes: list[str] = field(default_factory=list)

    @property
    def providers(self) -> tuple[str, ...]:
        return (self.draft.provider, self.final.provider)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "workflow_id": self.workflow_id,
            "rationale": self.rationale,
            "draft": self.draft.to_dict(),
            "final": self.final.to_dict(),
            "handoffs": [h.to_dict() for h in self.handoffs],
            "notes": self.notes,
        }


def _original_references(job: AdapterJob) -> list[ReferenceSpec]:
    """The customer's own assets, in ZolexAI vocabulary, provenance stamped."""
    roles = {
        "source_image": ("first_frame", "image"),
        "reference_image": ("identity", "image"),
        "source_video": ("structure", "video"),
        "source_audio": ("audio", "audio"),
        "identity_image": ("identity", "image"),
    }
    out: list[ReferenceSpec] = []
    for item in job.inputs:
        role, kind = roles.get(item.role, (item.role, item.kind))
        out.append(
            ReferenceSpec(
                role=role,
                kind=kind,
                native="conditions[] (H3) — carried through from the request",
                source=item.role,
                origin=USER,
            )
        )
    return out


def _draft_references(
    form: HandoffForm, section: SectionPlan, *, has_original_image: bool
) -> tuple[list[ReferenceSpec], tuple[float, float] | None, list[float]]:
    """What of the draft this section shows H3, as references + provenance."""
    start = section.start_seconds
    end = start + section.duration_seconds
    window: tuple[float, float] | None = None
    frames: list[float] = []
    refs: list[ReferenceSpec] = []

    def still(role: str, at: float) -> ReferenceSpec:
        frames.append(round(at, 4))
        return ReferenceSpec(
            role=role,
            kind="image",
            native="conditions[] {type: image, role: reference}",
            source=f"ltx draft @ {at:.3f}s",
            origin=GENERATED,
        )

    def clip() -> ReferenceSpec:
        return ReferenceSpec(
            role="motion",
            kind="video",
            native="conditions[] {type: video, role: reference}",
            source=f"ltx draft [{start:.3f}s, {end:.3f}s]",
            origin=GENERATED,
        )

    if form is HandoffForm.FULL_VIDEO:
        refs.append(clip())
        window = (round(start, 4), round(end, 4))
    elif form is HandoffForm.FIRST_FRAME:
        refs.append(still("motion", start))
    elif form is HandoffForm.LAST_FRAME:
        refs.append(still("motion", end))
    elif form is HandoffForm.FIRST_AND_LAST:
        refs.append(still("motion", start))
        refs.append(still("motion", end))
    elif form is HandoffForm.KEYFRAMES:
        for fraction in (0.0, 0.33, 0.66, 1.0):
            refs.append(still("motion", start + fraction * section.duration_seconds))
    elif form is HandoffForm.VIDEO_PLUS_ORIGINAL_IMAGE:
        if not has_original_image:
            raise ProviderRefusal(
                "video_plus_original_image needs the customer's own image, and "
                "this request has none",
                capability="handoff_form",
            )
        refs.append(clip())
        window = (round(start, 4), round(end, 4))
    return refs, window, frames


def compile_hybrid(
    job: AdapterJob,
    *,
    form: HandoffForm | None = None,
    ltx: LtxProvider | None = None,
    h3: H3Provider | None = None,
) -> HybridPlan:
    """Both plans and the bridge, or a refusal explaining why not.

    Pure. Compiles two provider manifests and the handoff between them; runs
    nothing, decodes nothing, and writes nothing.

    `form` defaults per workflow rather than globally — image-to-video's
    hypothesis needs the draft AND the photograph, while a reference-person
    restyle already holds its identity asset and wants only the draft's
    structure. The other forms are benchmark variables, swept explicitly.
    """
    if not hybrid_allowed(job.workflow_id):
        raise ProviderRefusal(
            HYBRID_EXCLUDED.get(
                job.workflow_id,
                f"no hybrid cell is defined for {job.workflow_id}",
            ),
            capability="hybrid_scope",
        )

    reference_image = job.input_for("reference_image")
    if job.workflow_id == "video-to-video" and reference_image is None:
        # The rationale for a v2v hybrid is reference-person replacement. A
        # plain restyle would be a restyle of a restyle.
        raise ProviderRefusal(
            HYBRID_EXCLUDED["video-to-video-standard"],
            capability="hybrid_scope",
        )

    form = form or default_handoff_form(job.workflow_id)
    ltx = ltx or LtxProvider()
    h3 = h3 or H3Provider()

    draft = ltx.compile(job)
    final = h3.compile(job)

    notes: list[str] = [
        "the handoff is DECODED RGB — LTX and H3 share no latent space, and a "
        "shape-compatible tensor is not a compatible representation",
        "hybrid is a benchmark strategy only: it is not in the routing table "
        "and `auto` cannot reach it",
    ]

    originals = _original_references(job)
    if not originals and job.workflow_id != "text-to-video":
        notes.append("this request carries no customer assets to preserve")

    # "the customer's own image" means whichever image THIS workflow was
    # given: the still for image-to-video, the person for a reference restyle.
    has_image = (
        job.input_for("source_image") is not None
        or job.input_for("reference_image") is not None
    )
    audio = job.input_for("source_audio")

    handoffs: list[HandoffManifest] = []
    sections: list[SectionPlan] = []

    for section in final.sections:
        draft_refs, window, frames = _draft_references(
            form, section, has_original_image=has_image
        )
        if window is not None:
            span = window[1] - window[0]
            if span > H3_MAX_CLIP_SECONDS + 1e-6:
                raise ProviderRefusal(
                    f"a {span:.2f}s draft clip exceeds H3's {H3_MAX_CLIP_SECONDS:g}s "
                    "per-clip reference limit",
                    capability="reference_video",
                )

        # The customer's assets stay in front of the model, and the draft is
        # added to them — never substituted for them.
        combined = [*section.references, *draft_refs]
        if len(combined) > H3_MAX_FILES:
            raise ProviderRefusal(
                f"section {section.index} would carry {len(combined)} references; "
                f"H3 accepts {H3_MAX_FILES}",
                capability="multimodal_references",
            )
        sections.append(replace(section, references=combined))

        handoffs.append(
            HandoffManifest(
                strategy=GenerationStrategy.LTX_TO_H3_REFERENCE.value,
                source_job_id=job.job_id,
                workflow_id=job.workflow_id,
                draft_provider=draft.provider,
                final_provider=final.provider,
                section_index=section.index,
                duration_seconds=section.duration_seconds,
                handoff_form=form.value,
                original_references=originals,
                generated_references=draft_refs,
                draft_window=window,
                selected_frames=frames,
                # The ORIGINAL track, never the draft's own generated audio:
                # a music video is measured against the customer's song.
                audio_reference=audio.role if audio is not None else "",
            )
        )

    if audio is not None:
        notes.append(
            "the customer's own track is the audio reference on the final pass; "
            "the draft's generated audio is discarded"
        )

    final_with_draft = replace(
        final,
        sections=sections,
        notes=[
            *final.notes,
            f"hybrid: each section additionally carries the LTX draft as "
            f"{form.value}, marked generated_intermediate",
        ],
    )

    return HybridPlan(
        strategy=GenerationStrategy.LTX_TO_H3_REFERENCE.value,
        workflow_id=job.workflow_id,
        draft=draft,
        final=final_with_draft,
        handoffs=handoffs,
        rationale=HYBRID_RATIONALE[job.workflow_id],
        notes=notes,
    )
