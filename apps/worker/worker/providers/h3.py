"""MiniMax H3 as a provider — compile-only until a GPU exists.

`compile` is complete and exercisable today. `generate` refuses, loudly,
because no weights are on any node, no licence application has been made, and
a provider that pretended otherwise would eventually be routed to.

Everything encoded here is an OFFICIAL limit, read 2026-08-22 from the model
card, the open-source announcement and the two prompt-writing guides:

  * output 4-15 seconds, 24 fps, 32 kHz stereo audio generated in the same pass;
  * aspect ratios 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 — note there is no 4:5;
  * open weights default to a 768-pixel short edge (2K lives in
    H3-Regenerate-2K, which is NOT part of the open-source release);
  * FL2VA takes zero, one or two images (text / first / last / first+last);
  * Ref2VA takes at most 9 images, 3 video clips and 3 audio clips, each clip
    2-15 seconds, each modality totalling at most 15 seconds, and at most 12
    files across all types;
  * references travel as a `conditions[]` array of {type, uri, role,
    frame_index} entries.

Nothing about steps, CFG, quantization, offload or section length is decided
here. Those are measurements, the GPU does not exist, and inventing them is
exactly the failure this whole exercise is meant to avoid.
"""

from __future__ import annotations

from worker.adapters.base import AdapterJob, AdapterResult, ProgressCallback
from worker.longform import plan_chain_segments
from worker.providers.base import ProviderRefusal, ProviderUnavailable
from worker.providers.capabilities import MATRIX, Capability
from worker.providers.h3_prompt import compile_h3_text_section
from worker.providers.manifest import (
    AudioWindow,
    GenerationManifest,
    ReferenceSpec,
    SectionPlan,
)

#: Official output bounds, in seconds, for ONE generation.
H3_MIN_SECONDS = 4.0
H3_MAX_SECONDS = 15.0

H3_FPS = 24.0

#: Short edge of the open-weight release. The 2K path is a separate model that
#: was not open-sourced.
H3_DEFAULT_SHORT_EDGE = 768

#: aspect ratio -> (width, height) at the default short edge. 4:5 is absent
#: from H3's documented list and is refused rather than silently reshaped —
#: a benchmark that quietly changed the customer's frame would be comparing
#: two different products.
H3_GRIDS: dict[str, tuple[int, int]] = {
    "21:9": (1792, 768),
    "16:9": (1366, 768),
    "4:3": (1024, 768),
    "1:1": (768, 768),
    "3:4": (768, 1024),
    "9:16": (768, 1366),
}

#: Reference ceilings, per modality and overall.
H3_MAX_IMAGES = 9
H3_MAX_VIDEOS = 3
H3_MAX_AUDIO = 3
H3_MAX_FILES = 12
H3_MIN_CLIP_SECONDS = 2.0
H3_MAX_CLIP_SECONDS = 15.0

#: The two documented audio behaviours. Only the first drives the mouth from
#: the supplied waveform; the second copies a voice's character and generates
#: new speech, which is NOT what a music video needs.
AUDIO_FULLY_COPY = "fully_copy"
AUDIO_TIMBRE_REFERENCE = "timbre_reference"

_TASK_LINES = {
    "t2v": "",
    "i2v": "The first frame of the video is the provided image.",
    "flf2v": "The first and last frames of the video are the provided images.",
    "ref2va": "Use the provided references as described below.",
}


class H3Provider:
    name = "h3"

    def capabilities(self) -> dict[str, Capability]:
        return MATRIX

    def health(self) -> tuple[bool, str]:
        # Deliberately hard-wired until a node actually carries H3. The
        # licence is a real gate, not a formality, and it turns on WHERE the
        # node physically is: the Applicable Territory is worldwide EXCEPT the
        # EU, UK, South Korea and the US, and an organisation deploying inside
        # one of those four must APPLY and be authorised first.
        # (Corrected 24 Aug 2026 — earlier wording here had the polarity
        # inverted; see docs/internal/h3-rtxpro6000-runtime-research.md §1.1.)
        return False, (
            "H3 is not installed on any node. Before it may be, the MiniMax H3 "
            "Community Licence requires the node's physical location to be "
            "confirmed: deployment inside the EU, UK, South Korea or the US "
            "needs an approved application; elsewhere is licensed by default"
        )

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self, job: AdapterJob) -> list[str]:
        problems: list[str] = []
        aspect = str(job.parameters.get("aspect_ratio") or "16:9")
        if aspect not in H3_GRIDS:
            problems.append(
                f"H3 does not document the {aspect} aspect ratio "
                f"(documented: {', '.join(H3_GRIDS)})"
            )
        total = self._total_seconds(job)
        if total is not None and total < H3_MIN_SECONDS:
            problems.append(
                f"H3 generates {H3_MIN_SECONDS:g}-{H3_MAX_SECONDS:g}s clips; "
                f"{total:g}s is below its floor"
            )
        images = sum(1 for i in job.inputs if i.kind == "image")
        videos = sum(1 for i in job.inputs if i.kind == "video")
        audio = sum(1 for i in job.inputs if i.kind == "audio")
        if images > H3_MAX_IMAGES:
            problems.append(f"H3 accepts at most {H3_MAX_IMAGES} reference images")
        if videos > H3_MAX_VIDEOS:
            problems.append(f"H3 accepts at most {H3_MAX_VIDEOS} reference video clips")
        if audio > H3_MAX_AUDIO:
            problems.append(f"H3 accepts at most {H3_MAX_AUDIO} reference audio clips")
        if images + videos + audio > H3_MAX_FILES:
            problems.append(f"H3 accepts at most {H3_MAX_FILES} reference files in total")
        return problems

    # ── Compilation ──────────────────────────────────────────────────────

    def compile(self, job: AdapterJob) -> GenerationManifest:
        problems = self.validate(job)
        if problems:
            raise ProviderRefusal(problems[0], capability="h3_limits")

        total = self._total_seconds(job)
        if total is None:
            raise ProviderRefusal(
                f"{job.workflow_id} takes its length from the uploaded file; "
                "supply execution.dry_run_source_seconds to compile a dry run",
                capability="duration",
            )

        aspect = str(job.parameters.get("aspect_ratio") or "16:9")
        width, height = H3_GRIDS[aspect]
        notes: list[str] = []

        # H3's own ceiling is the section length. Even windows (the same
        # planner LTX uses) keep every section inside 4-15s for any total at
        # or above the floor, and keep the two engines' section boundaries
        # comparable in kind even when they differ in count.
        per_pass = min(H3_MAX_SECONDS, total)
        segments = plan_chain_segments(total, per_pass)
        if len(segments) > 1:
            notes.append(
                f"{total:g}s exceeds H3's documented 15s single-generation ceiling, "
                f"so it becomes {len(segments)} generations with {len(segments) - 1} "
                "seams — the platform's chain, not an H3 feature"
            )
        shortest = min(s.duration_seconds for s in segments)
        if shortest < H3_MIN_SECONDS:
            raise ProviderRefusal(
                f"a section of {shortest:.2f}s falls below H3's {H3_MIN_SECONDS:g}s floor",
                capability="duration",
            )

        task, task_note = self._task(job)
        if task_note:
            notes.append(task_note)

        sections: list[SectionPlan] = []
        for segment in segments:
            frames = int(round(segment.duration_seconds * H3_FPS))
            references = self._references(job, segment.index, task)
            audio = self._audio(job, segment.start_seconds, segment.duration_seconds)
            sections.append(
                SectionPlan(
                    index=segment.index,
                    start_seconds=round(segment.start_seconds, 4),
                    duration_seconds=round(segment.duration_seconds, 4),
                    frames_requested=frames,
                    frames_rendered=frames,
                    seed=None,
                    prompt=compile_h3_text_section(
                        job.prompt,
                        index=segment.index,
                        total=len(segments),
                        task_line=_TASK_LINES.get(task, ""),
                    ),
                    references=references,
                    audio=audio,
                )
            )

        notes.append(
            "steps, CFG, quantization, offload and the sparse-attention path are "
            "deliberately unset — every one of them is a measurement this project "
            "has not made"
        )

        return GenerationManifest(
            provider=self.name,
            workflow_id=job.workflow_id,
            pipeline=f"MiniMax-H3/{'Ref2VA' if task == 'ref2va' else 'FL2VA'}",
            total_seconds=round(total, 4),
            width=width,
            height=height,
            fps=H3_FPS,
            sections=sections,
            settings={
                "task": task,
                "short_edge": H3_DEFAULT_SHORT_EDGE,
                "audio_output": "32 kHz stereo, generated in the same pass",
                "max_single_generation_seconds": H3_MAX_SECONDS,
                "steps": "UNKNOWN — GPU validation required",
                "guidance": "UNKNOWN — CFG-distilled weights exist; defaults unread",
                "quantization": "UNKNOWN — bf16 weights are 61.7 GB + 48 GB text "
                "encoder; int8/nvfp4 community builds exist",
                "attention": "full attention only in the initial open-source release",
            },
            notes=notes,
        )

    # ── Pieces ───────────────────────────────────────────────────────────

    def _total_seconds(self, job: AdapterJob) -> float | None:
        from worker.adapters.base import parse_duration_seconds

        if job.workflow_id in ("video-to-video", "music-video"):
            declared = job.execution_float("dry_run_source_seconds", 0.0)
            return declared if declared > 0 else None
        return parse_duration_seconds(job.parameters.get("duration"))

    def _task(self, job: AdapterJob) -> tuple[str, str]:
        """Which H3 head serves this workflow, and anything worth recording."""
        has_reference = job.input_for("reference_image") is not None
        has_video = any(i.kind == "video" for i in job.inputs)
        has_audio = any(i.kind == "audio" for i in job.inputs)

        if has_audio or has_video or has_reference:
            return "ref2va", ""
        if job.input_for("source_image") is not None:
            return "i2v", ""
        if job.workflow_id == "extend-video":
            # H3 documents continuation as a Ref2VA task type, so an extension
            # is a reference job even though the customer supplied no image.
            return "ref2va", (
                "extend compiles as Ref2VA video continuation — H3's documented "
                "task type — rather than as our seam-frame chain"
            )
        return "t2v", ""

    def _references(self, job: AdapterJob, index: int, task: str) -> list[ReferenceSpec]:
        items: list[ReferenceSpec] = []
        if index:
            items.append(
                ReferenceSpec(
                    role="seam",
                    kind="image",
                    native="conditions[] {type: image, role: first_frame}",
                    frame_index=0,
                    source="previous section final frame",
                )
            )
        still = job.input_for("source_image")
        if still is not None and index == 0:
            items.append(
                ReferenceSpec(
                    role="first_frame",
                    kind="image",
                    native="conditions[] {type: image, role: first_frame}",
                    frame_index=0,
                    source="source_image",
                )
            )
        reference = job.input_for("reference_image")
        if reference is not None:
            # The difference that matters against LTX: the person rides in
            # EVERY section as a subject reference, not as pixels in a frame.
            items.append(
                ReferenceSpec(
                    role="identity",
                    kind="image",
                    native="conditions[] {type: image, role: subject}",
                    source="reference_image",
                )
            )
        source_video = job.input_for("source_video")
        if source_video is not None:
            items.append(
                ReferenceSpec(
                    role="structure",
                    kind="video",
                    native="conditions[] {type: video, role: source}",
                    source="source_video window for this section (2-15s)",
                )
            )
        return items

    def _audio(
        self, job: AdapterJob, start: float, duration: float
    ) -> AudioWindow | None:
        track = job.input_for("source_audio")
        if track is None:
            return None
        # `fully_copy` is the only mode that syncs the mouth to the supplied
        # signal and makes it the final track — the timbre-reference mode
        # generates new speech and would silently discard the customer's song.
        return AudioWindow(
            start_seconds=round(start, 4),
            duration_seconds=round(min(duration, H3_MAX_CLIP_SECONDS), 4),
            mode=AUDIO_FULLY_COPY,
            returns_input_waveform=True,
        )

    # ── Running ──────────────────────────────────────────────────────────

    async def generate(
        self, job: AdapterJob, on_progress: ProgressCallback
    ) -> AdapterResult:
        usable, reason = self.health()
        raise ProviderUnavailable(reason if not usable else "H3 generation is not wired")
