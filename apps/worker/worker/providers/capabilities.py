"""What each engine can structurally do — researched, cited, and comparable.

This table is the only place a routing decision may read a capability from,
and every row is a fact from an official source, not a benchmark result. A
quality opinion ("H3 looks better") never belongs here; a structural fact
("H3 cannot render more than 15 seconds in one pass") does, because that is
the kind of difference that decides an architecture before any GPU exists.

Support levels are deliberately coarse. `NATIVE` means the model takes the
thing as an input of its own; `EMULATED` means the platform builds it out of
something the model does support (our long-form chain out of single passes,
for instance); `NONE` means it cannot be had at all. Anything whose answer is
a measurement rather than a capability is `UNKNOWN` and belongs in the GPU
checklist, never in a router.

Sources, read 2026-08-22:
  * LTX — the official Lightricks/LTX-2 tree at upstream commit 400fd31, plus
    this project's own dated GPU measurements (docs/internal/*.md).
  * H3 — MiniMaxAI/MiniMax-H3 model card, its two official prompt-writing
    guides (docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and _ref_en.md), the
    MiniMax open-source announcement, and the official ComfyUI integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Support(StrEnum):
    NATIVE = "native"
    EMULATED = "emulated"
    NONE = "none"
    UNKNOWN = "unknown"
    """Requires a render to answer. Never a routing input."""


@dataclass(frozen=True)
class Capability:
    ltx: Support
    h3: Support
    note: str = ""
    gpu_test: bool = False
    """True when the two are structurally comparable and only a measurement
    can separate them — i.e. this row becomes a benchmark case, not a
    decision."""

    @property
    def structural_winner(self) -> str | None:
        """A winner ONLY where one engine supports something the other cannot.

        `EMULATED` never loses to `NATIVE` here: our long-form chain produces
        60 seconds and so, differently, would H3's — which of them looks
        better is a measurement, and this property refuses to guess it.
        """
        if self.ltx is Support.UNKNOWN or self.h3 is Support.UNKNOWN:
            return None
        if self.ltx is not Support.NONE and self.h3 is Support.NONE:
            return "ltx"
        if self.h3 is not Support.NONE and self.ltx is Support.NONE:
            return "h3"
        return None


#: One row per capability the product actually sells or plans to sell.
MATRIX: dict[str, Capability] = {
    "t2v": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX: ltx_pipelines.distilled. H3: FL2VA with zero input images.",
        gpu_test=True,
    ),
    "i2v": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX: --image PATH 0 1.0. H3: FL2VA with a first frame.",
        gpu_test=True,
    ),
    "first_frame": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX pins any frame index; H3 has an explicit first-frame mode.",
        gpu_test=True,
    ),
    "last_frame": Capability(
        Support.NATIVE, Support.NATIVE,
        "H3 FL2VA takes a last frame directly. LTX has no last-frame FLAG, but "
        "--image accepts any frame index, and keyframe_interpolation exists "
        "upstream — neither is wired in this repo today.",
        gpu_test=True,
    ),
    "first_and_last_frame": Capability(
        Support.EMULATED, Support.NATIVE,
        "H3 FL2VA is built for it. LTX would need two --image triples (or the "
        "unwired keyframe_interpolation pipeline); untested here.",
        gpu_test=True,
    ),
    "v2v_structural": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX: ic_lora + Union Control on an edge map. H3: Ref2VA video "
        "reference, documented for 'video editing' and motion reference.",
        gpu_test=True,
    ),
    "reference_image": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX carries a reference only as pixels in a conditioned frame — no "
        "identity input exists in the LTX family. H3 takes up to 9 reference "
        "images with Subject/Picture roles.",
        gpu_test=True,
    ),
    "multiple_image_references": Capability(
        Support.EMULATED, Support.NATIVE,
        "LTX can pin several --image frames but they are timeline anchors, not "
        "a reference set. H3: up to 9 images, max 12 files across all types.",
        gpu_test=True,
    ),
    "reference_video": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX: one --video-conditioning control channel (IC-LoRA). H3: up to 3 "
        "clips, each 2-15s, total 15s.",
        gpu_test=True,
    ),
    "reference_audio": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX a2vid takes --audio-path and FREEZES the encoded audio latent in "
        "both stages. H3 Ref2VA takes up to 3 clips (2-15s each, total 15s) in "
        "two documented modes — see `audio_conditioned_video`.",
        gpu_test=True,
    ),
    "multimodal_references": Capability(
        Support.NONE, Support.NATIVE,
        "One request carrying images AND video AND audio references together. "
        "H3 documents it (12 files total). LTX has no equivalent request "
        "shape: audio lives only on a2vid, control video only on ic_lora, and "
        "the two are different entry points.",
    ),
    "audio_conditioned_video": Capability(
        Support.NATIVE, Support.NATIVE,
        "The customer's OWN track drives the picture. LTX a2vid: audio latent "
        "frozen, and the pipeline returns the input waveform. H3: only the "
        "'fully_copy' mode does this — the official guide states lip-sync "
        "occurs when audio is directly copied, and that the copied source "
        "audio becomes the target's final track. H3 timbre-reference mode does "
        "NOT sync to the supplied signal.",
        gpu_test=True,
    ),
    "native_audio_generation": Capability(
        Support.NATIVE, Support.NATIVE,
        "Both generate video and audio jointly. LTX: joint latents, distilled "
        "included. H3: 32 kHz stereo in the same pass.",
        gpu_test=True,
    ),
    "dialogue_generation": Capability(
        Support.NATIVE, Support.NATIVE,
        "LTX: measured verbatim delivery of planned lines; the Dub-It "
        "validated set is 5 languages. H3: dialogue tags with stable speaker "
        "ids, 11 languages stated stable.",
        gpu_test=True,
    ),
    "lip_sync_to_supplied_audio": Capability(
        Support.NATIVE, Support.NATIVE,
        "Structurally available on both, by different mechanisms (LTX frozen "
        "audio latent; H3 fully_copy). NEITHER is proven phoneme-accurate: "
        "LTX measures at goal-B (mouth follows vocal energy, -125..-208ms, "
        "r~0.45); H3 is unmeasured by us.",
        gpu_test=True,
    ),
    "single_pass_15s": Capability(
        Support.NATIVE, Support.NATIVE,
        "Both. LTX measured to 60s single-pass post-NATTEN; H3 caps at 15s.",
    ),
    "single_pass_over_15s": Capability(
        Support.NATIVE, Support.NONE,
        "THE structural difference for long-form. LTX renders one pass up to "
        "the measured grid ceiling (60s at product grids, though the model's "
        "own story coherence caps our default at 30s). H3's documented output "
        "range is 4-15 seconds, full stop.",
    ),
    "long_form_60s": Capability(
        Support.EMULATED, Support.EMULATED,
        "Neither model generates 60s in one diffusion pass by design — "
        "upstream LTX ships no extension pipeline either. Both reach it "
        "through our chain; the difference is seam COUNT: 2 sections / 1 seam "
        "for LTX at 30s passes, at least 4 sections / 3 seams for H3 at 15s.",
        gpu_test=True,
    ),
    "extend_continuation": Capability(
        Support.EMULATED, Support.NATIVE,
        "LTX: our chain conditions on the source's final frame; upstream has "
        "no extension inference pipeline (only a trainable LoRA mode). H3 "
        "documents 'video continuation' as a Ref2VA task type — new content "
        "continues, extends or resumes from a source video.",
        gpu_test=True,
    ),
    "camera_control_structured": Capability(
        Support.NONE, Support.NONE,
        "Neither takes a camera embedding, pose matrix or trajectory. Both are "
        "prompt-text only. H3 publishes a CLOSED motion vocabulary (zoom, "
        "push, pan, truck, tilt, pedestal, arc, tracking, static, shake, POV, "
        "roll, with amplitude and speed modifiers); LTX publishes prose "
        "guidance and 2.3-generation camera LoRAs with no 2.5 build.",
        gpu_test=True,
    ),
    "camera_adherence": Capability(
        Support.UNKNOWN, Support.UNKNOWN,
        "A measurement, not a capability. Benchmark group G.",
        gpu_test=True,
    ),
    "person_identity_transfer": Capability(
        Support.EMULATED, Support.NATIVE,
        "LTX has no identity input at all: our path composites the reference "
        "into a frame and anchors it as pixels. H3 takes the person as a "
        "Subject reference image, which is what the mechanism is for.",
        gpu_test=True,
    ),
    "music_video": Capability(
        Support.NATIVE, Support.NATIVE,
        "Comparable but not equivalent: LTX a2vid takes a 20.04s window of the "
        "master per section; H3 caps any one generation — and its audio input "
        "— at 15s, so a 5-minute song is at least 20 H3 sections against 15 "
        "LTX ones.",
        gpu_test=True,
    ),
    "prompt_timestamps": Capability(
        Support.NONE, Support.NATIVE,
        "A real compilation difference. LTX's own enhancer prompts FORBID "
        "timestamps and no official source supports them. H3's guide "
        "prescribes them for shot changes after the first shot.",
    ),
    "negative_prompt": Capability(
        Support.NATIVE, Support.UNKNOWN,
        "LTX: guided family only (distilled and ic_lora have no such flag), "
        "with a long default negative prompt applied automatically. H3: not "
        "documented in the open-weight material we could read.",
    ),
    "resolution_4_5": Capability(
        Support.NATIVE, Support.NONE,
        "Our 4:5 product grid is 512x640 on LTX. H3's documented aspect list "
        "is 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 — no 4:5.",
    ),
    "resolution_4_3": Capability(
        Support.EMULATED, Support.NATIVE,
        "H3 lists 4:3 and 3:4 natively. LTX has no measured 4:3 product grid; "
        "grid_for_source would synthesise one and take the pessimistic "
        "unmeasured pass ceiling.",
    ),
    "resolution_2k": Capability(
        Support.EMULATED, Support.NONE,
        "H3's 2K path is H3-Regenerate-2K, explicitly NOT in the open-source "
        "release; open weights default to a 768px short edge. LTX renders at "
        "measured grids and the platform upscales or normalises on delivery.",
    ),
}


def gpu_test_rows() -> list[str]:
    """Rows whose comparison is a benchmark case rather than a fact."""
    return sorted(name for name, cap in MATRIX.items() if cap.gpu_test)


def structural_winners() -> dict[str, str]:
    """Only the rows where one engine can do something the other cannot."""
    return {
        name: winner
        for name, cap in MATRIX.items()
        if (winner := cap.structural_winner) is not None
    }
