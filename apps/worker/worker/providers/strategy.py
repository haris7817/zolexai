"""Which engine, or engines, a benchmark cell runs — and where the line is.

Three strategies. Two of them are providers; the third is an experiment.

`LTX_TO_H3_REFERENCE` runs LTX first, decodes its output to ordinary RGB, and
hands that to H3 **as one reference among the customer's own**. It exists to
answer one question and no other:

    does running LTX first improve H3's result enough to justify a second
    generation pass?

The honest default answer is no. A hybrid costs both engines' inference, both
model loads, and a switch between them; it has to earn that back in quality,
identity or adherence before it is worth anything. Nothing here assumes it
does, and nothing here routes a customer through it — `_AUTO_ROUTES` is
untouched and hybrid is not a provider.

## Why decoded RGB and not latents

LTX and H3 are different models with different VAEs, different latent
geometries and different temporal compression. LTX's video VAE is 32x32
spatial / 8x temporal; H3 ships its own visual and audio VAEs. There is no
shared space, no published mapping between them, and a tensor that happens to
have a compatible shape is not a compatible representation. Wiring one's
latents into the other's sampler would produce confident nonsense.

Decoded RGB is the only interface the two models genuinely share — it is what
H3's reference conditioning is documented to take, and it is what our chain
already produces at every seam. So the handoff boundary is pixels.
"""

from __future__ import annotations

from enum import StrEnum


class GenerationStrategy(StrEnum):
    LTX_ONLY = "ltx_only"
    H3_ONLY = "h3_only"
    LTX_TO_H3_REFERENCE = "ltx_to_h3_reference"

    @property
    def is_hybrid(self) -> bool:
        return self is GenerationStrategy.LTX_TO_H3_REFERENCE

    @property
    def providers(self) -> tuple[str, ...]:
        """Every engine a run of this strategy pays for, in order.

        Hybrid returns both, which is what stops a cost table from comparing
        H3's half of a hybrid against a whole LTX run.
        """
        if self is GenerationStrategy.LTX_ONLY:
            return ("ltx",)
        if self is GenerationStrategy.H3_ONLY:
            return ("h3",)
        return ("ltx", "h3")

    @property
    def final_provider(self) -> str:
        return self.providers[-1]


class HandoffForm(StrEnum):
    """What of the LTX draft is actually shown to H3.

    A benchmark variable, not a setting. More references are not obviously
    better: H3 counts every file against its 12-file ceiling, a full draft
    clip spends one of only three video slots, and a draft that carries the
    whole shot may pin H3 to LTX's mistakes as firmly as to its virtues.
    Which of these wins is a measurement.
    """

    FULL_VIDEO = "full_video"
    """The whole decoded draft as a video reference. Costs one of H3's three
    video slots and is capped at 15s per clip by H3's own limits."""

    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    FIRST_AND_LAST = "first_and_last"
    """Cheap structural anchors. H3's FL2VA takes exactly these."""

    KEYFRAMES = "keyframes"
    """Several stills across the draft's window — motion as a sequence of
    positions rather than as a clip."""

    VIDEO_PLUS_ORIGINAL_IMAGE = "video_plus_original_image"
    """The draft for motion, the customer's own photograph for identity. The
    form the I2V hypothesis actually proposes, and the one that keeps the
    original asset in front of the model."""


#: Where a hybrid is worth spending GPU time on at all, and why. A workflow
#: absent from this table gets no hybrid cell — testing everything would burn
#: the budget on cells whose answer is already "no".
HYBRID_RATIONALE: dict[str, str] = {
    "image-to-video": (
        "the customer's photograph owns WHO, an LTX draft may own motion and "
        "camera structure, and H3 regenerates from both. The one case where "
        "the two engines' strengths are cleanly separable"
    ),
    "video-to-video": (
        "reference-person replacement only. LTX has no identity input at all "
        "and composites the reference into a frame; H3 takes the person as a "
        "subject reference. Whether LTX's structural draft adds anything to "
        "H3's own reading of the source is the question"
    ),
    "music-video": (
        "H3 caps a generation and its audio at 15s, so a song is already a "
        "chain; an LTX draft may carry performance staging that H3 then "
        "regenerates against the real vocal in fully_copy mode"
    ),
    "text-to-video": (
        "premium cells only — a difficult camera move, fast action or a "
        "multi-shot sequence, where an LTX draft could supply motion "
        "structure H3 would otherwise have to invent from text alone"
    ),
}

#: Workflows deliberately excluded, and the reason, so the omission reads as a
#: decision rather than an oversight.
HYBRID_EXCLUDED: dict[str, str] = {
    "extend-video": (
        "H3 documents video continuation as a native Ref2VA task type; the "
        "source clip is already the reference, and an LTX draft of a "
        "continuation is a draft of the very thing under test"
    ),
    "video-to-video-standard": (
        "a plain restyle already hands H3 the source video. An LTX draft "
        "would be a restyle of a restyle, and neither engine's result would "
        "be attributable"
    ),
}


#: The handoff form each workflow starts from. Not a tuning constant — it is
#: the shape of that workflow's hypothesis, and picking one default for
#: everything is wrong in a way the dry run makes obvious: image-to-video
#: wants the draft for motion AND the photograph for identity, while a
#: reference-person restyle already has its identity asset and only needs the
#: draft's structure. The benchmark sweeps the others.
DEFAULT_HANDOFF: dict[str, str] = {
    "image-to-video": "video_plus_original_image",
    "video-to-video": "full_video",
    "music-video": "full_video",
    "text-to-video": "full_video",
}


def hybrid_allowed(workflow_id: str) -> bool:
    return workflow_id in HYBRID_RATIONALE


def default_handoff_form(workflow_id: str) -> HandoffForm:
    return HandoffForm(DEFAULT_HANDOFF.get(workflow_id, "full_video"))


def parse(value: str | None, *, default: GenerationStrategy | None = None) -> GenerationStrategy:
    """Strategy from a benchmark field or a QA override.

    Unknown values raise rather than falling back: a typo that silently ran
    `ltx_only` while the results table said `hybrid` would poison every
    conclusion drawn from that row.
    """
    if value is None or not str(value).strip():
        return default or GenerationStrategy.LTX_ONLY
    text = str(value).strip().lower()
    try:
        return GenerationStrategy(text)
    except ValueError as exc:
        raise ValueError(
            f"unknown generation strategy {value!r}; "
            f"expected one of {[s.value for s in GenerationStrategy]}"
        ) from exc
