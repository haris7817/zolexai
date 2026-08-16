"""LTX-2.5 runtime — real GPU generation for every video workflow.

## How it runs

The worker process never imports torch. It shells out to the LTX repository's
own `uv` environment (`settings.ltx_repo_dir`) exactly the way the benchmark
did, which keeps CUDA, model weights and their 40-GB dependency tree entirely
outside this codebase. The subprocess is supervised the same way the media
tools are: killed on cancellation, on timeout, and on any exception path, so a
dead job can never leave a render burning VRAM.

## The measurements this is built on

RTX 5090, 2026-08-12 — facts, not guesses:

  * NVFP4 works end to end; BF16's transformer alone is ~40 GB against 32 GB
    of VRAM and cannot load.
  * A 30s pass completes; a 60s pass hard-OOMs at 29.6/31.4 GiB mid-denoise.
    **Nothing here ever asks the GPU for more than one pass can survive** —
    longer durations become several passes, chained, and the ceiling is read
    from `settings.ltx_max_seconds` rather than written down anywhere.
  * The `distilled` entry point cannot emit audio-only output (it tries to
    attach an H.264 stream to an MP3 container and dies), so music generation
    is a different runtime entirely — see `adapters/music.py`.
  * Conditioning uses the pipeline's `--image PATH FRAME_IDX STRENGTH` input.
    A still pinned at frame 0 at full strength becomes the first frame; the
    same argument at other indices and lower strengths is how a restyle keeps
    the source's composition.

## One chain, five workflows

`worker.longform.render_chain` is the whole long-form mechanism, and every
workflow is a way of choosing what conditions its passes and what happens to
the parts afterwards:

  | workflow       | total length      | conditioning per pass          | assembled as            |
  |----------------|-------------------|--------------------------------|-------------------------|
  | text-to-video  | requested         | previous pass's final frame    | parts                   |
  | image-to-video | requested         | the still, then final frames   | parts                   |
  | extend-video   | requested         | source's final frame, then …   | source + parts          |
  | video-to-video | the SOURCE's own  | source keyframes + continuity  | parts + source audio    |
  | music-video    | the SONG's own    | previous pass's final frame    | parts + the whole song  |

Video-to-video and music video take their length from the uploaded file, never
from a request field — that is the client's automatic-duration requirement, and
it is enforced by reading the probe rather than by trusting a parameter.

## The prompt

The user's prompt reaches the model **verbatim**. It is passed as a single
argv element, so nothing quotes, escapes, truncates or reflows it, and this
module never rewrites, prefixes or "improves" it — a generation that does not
match what someone typed is a bad enough experience without the system having
silently typed something else. `tests/test_ltx.py` pins that byte-for-byte.

The distilled entry point exposes no guidance scale, step count or negative
prompt (its non-distilled sibling does), so there is no adherence dial to turn
from here. LTX's own prompt enhancer is the one available lever and it *does*
rewrite the prompt, so it is opt-in per workflow via `execution.enhance_prompt`
and off everywhere by default.

## Testability

Everything except the model itself is provable without a GPU: `_command()` is
pure, `_execute()` accepts any argv (tests substitute a stub script that writes
a real MP4), and progress parsing is a pure function over output lines. The
only thing the GPU-node run adds is the model.

Each refusal is `retriable=False` with the real reason in `internal_detail` — a
mis-routed job should fail once with a clear log line, not burn three attempts.
"""

from __future__ import annotations

import asyncio
import math
import zlib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    cancellable,
    parse_duration_seconds,
)
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import (
    GENERATE_FROM,
    GENERATE_TO,
    ChainStep,
    RenderStep,
    StageReporter,
    plan_musical_boundaries,
    plan_section_prompts,
    render_chain,
)
from worker.media import (
    AudioMode,
    FfmpegError,
    MediaInfo,
    OutputExpectation,
    audio_onsets,
    concat_segments,
    duration_tolerance,
    extract_final_frame,
    extract_frames_at,
    ffmpeg,
    mux_audio,
    normalize_clip,
    probe_media,
    verify_output,
)

logger = get_logger(__name__)

#: Weight files relative to `settings.ltx_models_root`, exactly as the
#: benchmark laid them out. The transformer is chosen by quantization mode.
_MODEL_FILES: dict[str, str] = {
    "transformer_nvfp4": "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "transformer_bf16": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "duration_head": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upsampler": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    ),
}

#: LTX's two-stage pipeline requires dimensions divisible by 64 — 480x848 was
#: rejected outright in benchmarking.
#:
#: These grids were re-measured on the RTX PRO 6000 on 16 Aug 2026, after NATTEN
#: replaced the failing Triton fallback kernel (see `_GRID_CEILINGS`). The
#: previous set was inherited from the RTX 5090's 32 GB and was roughly a third
#: smaller than this card sustains.
_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (768, 768),
    # 4:5 is the odd one out. Exact 4:5 on a /64 lattice is only 512x640, then
    # 768x960, then 1024x1280 — there is no intermediate, and 768x960 fails.
    "4:5": (512, 640),
}
_DEFAULT_DIMENSIONS = (1024, 576)

#: Single-pass ceilings **measured per grid**, in seconds. Not derived.
#:
#: The VAE fails on a *set of bad shapes*, not above a size threshold, and the
#: set follows no rule anyone has been able to predict. Measured on this card at
#: 60s: 1024x576 passes, 896x512 fails, 1152x640 fails, 768x960 fails. A larger
#: grid passing where a smaller one fails rules out every "budget" model, so
#: nothing here may be interpolated or extrapolated — a grid that is not in this
#: table has not been run, and gets `_UNMEASURED_CEILING`.
#:
#: Before NATTEN this whole table was effectively 10s, because one global value
#: had to satisfy the worst aspect ratio. That cost every 60s render five seams.
_GRID_CEILINGS: dict[tuple[int, int], float] = {
    # current product grids — all proven to 60s, the longest length offered
    (1024, 576): 60.0,
    (576, 1024): 60.0,
    (768, 768): 60.0,
    (512, 640): 60.0,
    # previous grids, kept because a source's own aspect can still select them
    (896, 512): 30.0,  # 60s FAILS: CUBLAS_STATUS_INTERNAL_ERROR
    (512, 896): 60.0,
    (640, 640): 60.0,
}

#: What an unmeasured grid is allowed. Deliberately pessimistic: 10s was the
#: last value proven safe across every shape the product offered, and a grid
#: absent from `_GRID_CEILINGS` is by definition one nobody has run.
_UNMEASURED_CEILING = 10.0

#: Frame counts the VAE decoder cannot decode, and where to land instead.
#:
#: The decoder dies in a cuBLAS batched GEMM (`CUBLAS_STATUS_INTERNAL_ERROR`
#: from `cublasGemmStridedBatchedEx`, all dims cast to int32) at specific
#: (grid, conditioned, frame-count) triples. The failing set follows no rule
#: anyone has produced: at 1024x576 unconditioned, 240 fails while 232, 248 and
#: 1440 pass; WITH a conditioning image the same 1440 fails and 240 passes.
#: Every entry below is a measurement from 16 Aug 2026 — nothing interpolated.
#:
#: Structure: bad band (inclusive) → first measured-safe landing at or above
#: the band. Rendering a few extra frames costs seconds; the output is trimmed
#: back to the exact requested duration afterwards, so the customer's video is
#: bit-for-bit the length they asked for. This removes the failure without a
#: ceiling, without chaining and without seams.
#:
#: Landings marked with their evidence:
#:   248  measured ✅ at 1024x576 and 576x1024
#:   736  measured ✅ at 1024x576, 576x1024 and 768x768
#:   1528 measured ✅ at 1024x576 conditioned; applied to the other grids
#:        because they fail identically at 1440 and are re-verified by the
#:        matrix before any deploy
_BAD_FRAME_BANDS: dict[bool, dict[tuple[int, int], list[tuple[int, int, int]]]] = {
    # unconditioned: (band_lo, band_hi, safe_landing)
    False: {
        (1024, 576): [(233, 247, 248), (714, 735, 736)],
        (576, 1024): [(233, 247, 248), (714, 735, 736)],
        (768, 768): [(714, 735, 736)],  # 240 passes on 1:1 — measured
    },
    # conditioned (any --image): the band moves entirely. 240 and 720 PASS
    # conditioned (the image/extend matrix cells), 1440-1464 fail.
    True: {
        (1024, 576): [(1436, 1520, 1528)],
        (576, 1024): [(1436, 1520, 1528)],
        (768, 768): [(1436, 1520, 1528)],
        (512, 640): [(1436, 1520, 1528)],  # extend 4:5 failed via its 16:9 source
    },
}


def safe_frame_count(
    dimensions: tuple[int, int], frames: int, *, conditioned: bool
) -> int:
    """The frame count actually sent to the pipeline for this shape.

    Returns `frames` untouched unless it falls inside a measured bad band for
    this exact (grid, conditioned) pair, in which case it returns the measured
    safe landing above the band. The caller renders the substitute and trims
    the output back to the requested duration — the delivered video is exactly
    the length asked for, in one pass, with no seam.

    Unknown grids return `frames` unchanged: without a measurement there is no
    landing to prefer, and the pessimistic per-shape ceiling already keeps
    unmeasured shapes on short passes.
    """
    for lo, hi, landing in _BAD_FRAME_BANDS[conditioned].get(dimensions, ()):
        if lo <= frames <= hi:
            return landing
    return frames

#: The largest frame measured on this card (1024x576 == 768x768 == 589,824 px).
#: A source's own aspect may still synthesise a grid that is not in
#: `_GRID_CEILINGS` — a 4:3 upload has no measured grid, and forcing it to 16:9
#: would make the model keep the style and replace the subject, which is worse
#: than a shorter pass. Such a grid renders at its true aspect and takes
#: `_UNMEASURED_CEILING`, so it is chained rather than gambled on.
_PIXEL_BUDGET = 1024 * 576

#: Delivery is at the source's own resolution (a user's 1080p clip must not
#: come back as 512p), capped at full HD — beyond that the normalization
#: re-encode cost stops being worth invisible extra pixels.
_MAX_OUTPUT_LONG_SIDE = 1920
_MAX_OUTPUT_SHORT_SIDE = 1080

#: Video-to-video conditioning defaults. Every one is overridable per workflow
#: through the private `execution` block, because the right values are a
#: quality judgement made against real footage on a real GPU, and baking them
#: in would make that judgement a code change.
_V2V_KEYFRAMES = 3
"""Source stills shown to the model per pass. One locks the opening
composition and lets everything after it drift; too many turn a restyle into a
slideshow of the original. Three across a window keeps subject placement and
the direction of movement without pinning every frame."""

_V2V_STRUCTURE_STRENGTH = 0.7
"""How hard those stills pull. At 1.0 the source frame IS the output frame and
the prompt does nothing; near 0 the prompt wins and the source is a suggestion.
This is the dial between "restyled" and "unrelated"."""

_V2V_CONTINUITY_STRENGTH = 0.85
"""Frame 0 of every pass after the first, taken from the previous pass's last
frame. High, because this is the seam: the two sides of it must be the same
video."""

_V2V_REFERENCE_STRENGTH = 0.3
"""The optional reference image, first pass only, at frame 0. Low on purpose —
the contract the customer reads says it "guides the look", and a strength that
made it the opening frame would be replacing the source's intent with it.
Setting this to 0 in a workflow drops reference conditioning entirely."""


def grid_for_source(width: int | None, height: int | None) -> tuple[int, int]:
    """The /64 generation grid closest to the source's aspect, within budget.

    The I2V benchmark made this non-negotiable: conditioning survives only
    when the render's aspect matches the conditioning image's. Grids are the
    model's constraint (each side divisible by 64), the budget is the GPU's.
    Ties in aspect error go to the larger frame.
    """
    if not width or not height:
        return _DEFAULT_DIMENSIONS
    aspect = math.log(width / height)

    def error(grid: tuple[int, int]) -> float:
        return abs(math.log(grid[0] / grid[1]) - aspect)

    grids = [
        (w, h)
        for h in range(256, 1025, 64)
        for w in range(256, 1025, 64)
        if w * h <= _PIXEL_BUDGET
    ]
    # Within a small aspect error the crop is invisible and more pixels win —
    # otherwise 576x320 (1.80) would beat 1024x576 (1.78) for a 16:9 source on
    # a 1% aspect technicality while shrinking the frame.
    #
    # Grids reached this way may have no entry in `_GRID_CEILINGS`. That is
    # deliberate: matching the source's aspect matters more than a long pass,
    # and an unmeasured grid is chained at `_UNMEASURED_CEILING` rather than
    # run at a length nobody has proven for that shape.
    close = [grid for grid in grids if error(grid) <= 0.08]
    if close:
        return max(close, key=lambda grid: (grid[0] * grid[1], -error(grid)))
    return min(grids, key=error)


def output_dimensions(width: int | None, height: int | None) -> tuple[int, int]:
    """The stitched file's resolution: the source's own, capped, made even.

    Even dimensions because yuv420p subsamples chroma 2x2 — libx264 refuses
    odd sizes outright.
    """
    if not width or not height:
        return _DEFAULT_DIMENSIONS
    long_side, short_side = max(width, height), min(width, height)
    scale = min(1.0, _MAX_OUTPUT_LONG_SIDE / long_side, _MAX_OUTPUT_SHORT_SIDE / short_side)

    def even(value: float) -> int:
        return max(2, int(value * scale) // 2 * 2)

    return even(width), even(height)


@dataclass(frozen=True)
class ConditioningFrame:
    """One `--image PATH FRAME_IDX STRENGTH` triple.

    A list of these is the entire conditioning vocabulary of the distilled
    entry point, and every workflow's identity is which list it builds.
    """

    path: Path
    frame_index: int
    strength: float

    def as_args(self) -> list[str]:
        # `round` then `str` rather than a %g format: the pipeline's own
        # documentation writes full strength as "1.0", and "1" is a different
        # token to argument parsers that type-check positionally.
        return ["--image", str(self.path), str(self.frame_index), str(round(self.strength, 3))]


#: Ordered milestones matched against the pipeline's log output. Matching is
#: forward-only (an index walks down this list), which is what makes the twice-
#: occurring "Running denoising loop" line map to two different steps. The
#: messages are customer copy: no model, provider or tensor vocabulary.
_MARKERS: list[tuple[str, int, str]] = [
    ("Building text encoder", 20, "Understanding your prompt…"),
    ("Running denoising loop", 40, "Generating your video…"),
    ("Building video encoder + spatial upsampler", 55, "Adding detail…"),
    ("Running denoising loop", 70, "Refining your video…"),
    ("Building video decoder", 80, "Rendering the final video…"),
    ("saved to", 85, "Almost done…"),
]

#: How long the output may be silent before we recheck cancellation. Denoising
#: prints nothing for stretches; without this poll a cancelled job would not
#: stop until the next log line happened to arrive.
_CANCEL_POLL_SECONDS = 2.0

#: Kept for diagnostics when the pipeline fails — the whole log would be huge.
_OUTPUT_TAIL_LINES = 40


def match_marker(line: str, start: int) -> int | None:
    """Index of the first milestone at or after `start` that `line` announces.

    Forward-only so repeated phrases advance rather than repeat, and so a
    skipped milestone (variant pipelines log different stages) cannot wedge
    the whole sequence.
    """
    for index in range(start, len(_MARKERS)):
        if _MARKERS[index][0] in line:
            return index
    return None


class LtxAdapter:
    name = "ltx"

    #: Every workflow whose output is video. Music is a different runtime; the
    #: benchmark proved this entry point cannot write an audio-only file.
    _SUPPORTED = frozenset(
        {"text-to-video", "image-to-video", "extend-video", "video-to-video", "music-video"}
    )

    def supports(self, workflow_id: str) -> bool:
        return workflow_id in self._SUPPORTED

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing()
        self._require_models()

        # Dispatch on the WORKFLOW, never on which inputs happen to be present:
        # a mis-routed job must fail loudly rather than being quietly treated as
        # whichever workflow its inputs resemble.
        handlers = {
            "extend-video": self._run_extension,
            "video-to-video": self._run_restyle,
            "music-video": self._run_music_video,
        }
        handler = handlers.get(job.workflow_id, self._run_generation)
        return await handler(job, reporter)

    # ── text-to-video / image-to-video ───────────────────────────────────

    async def _run_generation(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """A requested duration, optionally starting from an uploaded still."""
        self._require_generation_shape(job)
        seconds = self._requested_seconds(job)
        still = await self._conditioning_image(job, "source_image")
        dimensions = self._requested_dimensions(job)
        self._record_audio_mode(job, AudioMode.GENERATED_PER_SECTION_AUDIO)
        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                prompt_plan = plan_section_prompts(job.prompt, step.total)
            return prompt_plan[step.index]

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            if step.is_first:
                return [ConditioningFrame(still, 0, 1.0)] if still else []

            items: list[ConditioningFrame] = []
            if step.previous_frame:
                items.append(ConditioningFrame(step.previous_frame, 0, 1.0))
            # A predecessor frame carries temporal state but is a weak identity
            # anchor. Keep the original upload in every I2V pass at low strength
            # and away from frame zero, so it guides identity without resetting
            # the section to the composition of the first image.
            if still:
                frames = self._frame_count(step.seconds)
                reference_frame = min(frames - 1, max(1, frames // 3))
                strength = job.execution_float("i2v_reference_strength", 0.2)
                if strength > 0 and reference_frame > 0:
                    items.append(ConditioningFrame(still, reference_frame, strength))
            return items

        rendered = await render_chain(
            job,
            seconds,
            per_pass_seconds=self._per_pass_seconds(job, dimensions),
            render=self._renderer(job, reporter, dimensions=dimensions,
                                  conditioning=conditioning,
                                  prompt_for_step=prompt_for_step,
                                  require_audio=True),
            reporter=reporter,
        )

        await reporter.stitching()
        output = job.workspace / "output.mp4"
        info = await self._assemble(
            job,
            # Normalize every pass to one FPS/timebase/stream layout before
            # concat. Equal requested dimensions do not guarantee equal media
            # timestamps, and a mismatch becomes a visible seam.
            lambda: self._assemble_generated_sections(
                job, rendered, output, dimensions=dimensions, audio=True
            ),
            output,
            OutputExpectation(
                expect_video=True, expect_audio=True, expected_seconds=seconds
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    # ── extend-video ─────────────────────────────────────────────────────

    async def _run_extension(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """source → final frame → continuation → normalize → stitch.

        The finished file is the untouched-in-content source plus the
        continuation, both normalized to one set of encoder parameters at the
        source's own (capped) resolution.
        """
        staged, source = await self._staged_source(job, "source_video", kind="video")
        extension_seconds = self._requested_seconds(job)
        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                prompt_plan = plan_section_prompts(job.prompt, step.total)
            return prompt_plan[step.index]

        await reporter.probing("Reading your video…")
        seed_frame = await self._final_frame_of(job, staged)

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            frame = step.previous_frame
            return [ConditioningFrame(frame, 0, 1.0)] if frame else []

        # The grid follows the SOURCE's aspect, not the request's: the I2V
        # benchmark showed a mismatched aspect makes the model keep the style
        # and replace the subject, which at a seam means a different video
        # after the join. Bound once so the pass ceiling is derived from the
        # shape actually rendered rather than the requested aspect's.
        grid = grid_for_source(source.width, source.height)

        rendered = await render_chain(
            job,
            extension_seconds,
            per_pass_seconds=self._per_pass_seconds(job, grid),
            render=self._renderer(
                job, reporter,
                dimensions=grid,
                conditioning=conditioning,
                prompt_for_step=prompt_for_step,
            ),
            reporter=reporter,
            prefix="continuation",
            seed_frame=seed_frame,
        )

        await reporter.stitching()
        width, height = output_dimensions(source.width, source.height)
        fps = _delivery_fps(source)
        continuation_infos = [await probe_media(part) for part in rendered]
        keep_audio = source.has_audio or any(info.has_audio for info in continuation_infos)
        self._record_audio_mode(
            job,
            AudioMode.SOURCE_AUDIO
            if source.has_audio
            else (
                AudioMode.GENERATED_PER_SECTION_AUDIO
                if keep_audio
                else AudioMode.NO_AUDIO
            ),
        )
        output = job.workspace / "output.mp4"
        expected = (source.duration_seconds or 0.0) + extension_seconds

        async def assemble() -> Path:
            continuation = await self._assemble_generated_sections(
                job,
                rendered,
                job.workspace / "continuation.mp4",
                dimensions=(width, height),
                fps=fps,
                audio=keep_audio,
            )
            source_part = await normalize_clip(
                staged,
                job.workspace / "part-0000.mp4",
                width=width,
                height=height,
                fps=fps,
                audio=keep_audio,
            )
            return await concat_segments(
                [source_part, continuation], output
            )

        info = await self._assemble(
            job,
            assemble,
            output,
            OutputExpectation(
                expect_video=True,
                expect_audio=keep_audio,
                expected_seconds=expected,
                # Each re-timed part can drift a little; scale with length
                # instead of failing honest 60s extensions on frame rounding.
                tolerance_seconds=duration_tolerance(expected, floor=1.5),
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    # ── video-to-video ───────────────────────────────────────────────────

    async def _run_restyle(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """Restyle footage, matching the source's own duration exactly.

        The customer picks no duration for this workflow — the API rejects one
        — so the target is whatever the probe measures, and a 42-second upload
        produces a 42-second result whether that is one pass or two.

        What keeps it a restyle rather than an unrelated generation is the
        conditioning: several stills lifted from the *same window of the source*
        that the pass is about to generate, so subject placement, framing and
        the direction of movement carry over while the prompt supplies the look.
        Passes after the first also take their frame 0 from the previous pass's
        last frame, which is what makes the joins invisible.
        """
        staged, source = await self._staged_source(job, "source_video", kind="video")
        await reporter.probing("Reading your video…")

        target_seconds = source.duration_seconds or 0.0
        reference = await self._conditioning_image(job, "reference_image")
        grid = grid_for_source(source.width, source.height)

        keyframes = max(1, min(8, job.execution_int("v2v_keyframes", _V2V_KEYFRAMES)))
        structure = job.execution_float("v2v_structure_strength", _V2V_STRUCTURE_STRENGTH)
        continuity = job.execution_float("v2v_continuity_strength", _V2V_CONTINUITY_STRENGTH)
        reference_strength = job.execution_float(
            "v2v_reference_strength", _V2V_REFERENCE_STRENGTH
        )

        async def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            frames = self._frame_count(step.seconds)
            items: list[ConditioningFrame] = []

            # Frame 0 is the seam (or, on the first pass, the one place a
            # reference image can guide the look without displacing the
            # source's structure). Exactly one thing may own it.
            anchored = False
            if step.previous_frame is not None:
                items.append(ConditioningFrame(step.previous_frame, 0, continuity))
                anchored = True
            elif reference is not None and reference_strength > 0:
                items.append(ConditioningFrame(reference, 0, reference_strength))
                anchored = True

            # Source stills spread across the window this pass covers. The
            # half-step offsets keep them off both ends, so they never collide
            # with the anchor above and never fight the next pass's opening.
            offsets = [(index + 0.5) / keyframes for index in range(keyframes)]
            if not anchored:
                offsets.insert(0, 0.0)

            window = step.segment.start_seconds
            timestamps = [window + offset * step.seconds for offset in offsets]
            stills = await cancellable(
                job,
                extract_frames_at(
                    staged,
                    timestamps,
                    job.workspace / "keyframes",
                    prefix=f"pass-{step.index:04d}",
                ),
            )
            items += [
                ConditioningFrame(still, min(frames - 1, round(offset * (frames - 1))), structure)
                for still, offset in zip(stills, offsets, strict=False)
            ]
            return items

        rendered = await render_chain(
            job,
            target_seconds,
            per_pass_seconds=self._per_pass_seconds(job, grid),
            render=self._renderer(job, reporter, dimensions=grid, conditioning=conditioning),
            reporter=reporter,
            prefix="restyled",
        )

        await reporter.stitching()
        width, height = output_dimensions(source.width, source.height)
        fps = _delivery_fps(source)
        keep_audio = source.has_audio
        self._record_audio_mode(
            job, AudioMode.SOURCE_AUDIO if keep_audio else AudioMode.NO_AUDIO
        )
        output = job.workspace / "output.mp4"

        async def assemble() -> Path:
            # `audio=False` on purpose: the model generates its own soundtrack,
            # and a restyle that replaced the user's audio with an invented one
            # would be a bug nobody asked for. The source's own track goes back
            # on below, whole.
            picture = await self._assemble_generated_sections(
                job,
                rendered,
                job.workspace / "picture.mp4",
                dimensions=(width, height),
                fps=fps,
                audio=False,
            )
            if not keep_audio:
                return picture.replace(output)
            await reporter.muxing("Restoring your audio…")
            return await mux_audio(picture, staged, output)

        info = await self._assemble(
            job,
            assemble,
            output,
            OutputExpectation(
                expect_video=True,
                expect_audio=keep_audio,
                expected_seconds=target_seconds,
                tolerance_seconds=duration_tolerance(target_seconds, floor=1.0),
                expected_width=width,
                expected_height=height,
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    # ── music-video ──────────────────────────────────────────────────────

    async def _run_music_video(
        self, job: AdapterJob, reporter: StageReporter
    ) -> AdapterResult:
        """Visuals for the whole uploaded track, with the track laid over once.

        The client's requirement is specific and it is about the audio, not the
        picture: the finished file carries the COMPLETE song, continuous, not
        restarting per visual section. That shape is enforced structurally —
        the chain produces silent picture, and `mux_audio` attaches the user's
        original file exactly once, at the end, as the only soundtrack.
        """
        staged, track = await self._staged_source(job, "source_audio", kind="audio")
        await reporter.probing("Listening to your track…")

        target_seconds = track.duration_seconds or 0.0
        self._record_audio_mode(job, AudioMode.SOURCE_AUDIO)
        dimensions = self._requested_dimensions(job)
        per_pass = self._per_pass_seconds(job, dimensions)
        prompt_plan: list[str] | None = None

        def prompt_for_step(step: ChainStep) -> str:
            nonlocal prompt_plan
            if prompt_plan is None:
                prompt_plan = plan_section_prompts(job.prompt, step.total)
            return prompt_plan[step.index]

        boundaries = await self._musical_boundaries(job, staged, target_seconds, per_pass)

        def conditioning(step: ChainStep) -> list[ConditioningFrame]:
            frame = step.previous_frame
            return [ConditioningFrame(frame, 0, 1.0)] if frame else []

        rendered = await render_chain(
            job,
            target_seconds,
            per_pass_seconds=per_pass,
            render=self._renderer(job, reporter, dimensions=dimensions,
                                  conditioning=conditioning,
                                  prompt_for_step=prompt_for_step),
            reporter=reporter,
            prefix="scene",
            boundaries=boundaries,
        )

        await reporter.stitching()
        width, height = dimensions
        output = job.workspace / "output.mp4"

        async def assemble() -> Path:
            picture = await self._assemble_generated_sections(
                job,
                rendered,
                job.workspace / "picture.mp4",
                dimensions=(width, height),
                audio=False,
            )
            await reporter.muxing("Adding your track…")
            return await mux_audio(picture, staged, output)

        info = await self._assemble(
            job,
            assemble,
            output,
            OutputExpectation(
                expect_video=True,
                # Both halves of the client's promise, checked on the real file:
                # the song is present, and the result is the song's length.
                expect_audio=True,
                expected_seconds=target_seconds,
                tolerance_seconds=duration_tolerance(target_seconds, floor=1.0),
            ),
            reporter,
        )

        await reporter.uploading()
        return _video_result(output, info)

    async def _musical_boundaries(
        self, job: AdapterJob, track: Path, total_seconds: float, per_pass: float
    ) -> list[float]:
        """Cut points taken from the music, or none — never a failed job.

        Timing analysis is an improvement to where the seams land, not a
        prerequisite for producing the video. A track this cannot measure still
        gets even windows, which is exactly what it would have got anyway.
        """
        if not job.execution.get("align_cuts_to_audio", True):
            return []
        try:
            onsets = await cancellable(job, audio_onsets(track))
        except FfmpegError as exc:
            logger.info("onset_analysis_skipped", extra={"detail": str(exc)})
            return []
        return plan_musical_boundaries(
            total_seconds, per_pass_seconds=per_pass, onsets=onsets
        )

    # ── Shared assembly and validation ───────────────────────────────────

    async def _assemble(
        self,
        job: AdapterJob,
        build,
        output: Path,
        expectation: OutputExpectation,
        reporter: StageReporter,
    ) -> MediaInfo:
        """Runs an assembly step, then refuses to ship what it produced unless
        the file is genuinely deliverable.

        Every workflow ends here. The whole point of a single exit is that
        "never mark a job successful unless the artifact validates" cannot be
        forgotten in one branch — there is only one branch.
        """
        try:
            await cancellable(job, build())
            await reporter.finalizing("Verifying your videoâ€¦")
            return await verify_output(output, expectation)
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"assembly or validation failed: {exc}",
            ) from exc

    async def _assemble_generated_sections(
        self,
        job: AdapterJob,
        rendered: list[Path],
        output: Path,
        *,
        dimensions: tuple[int, int],
        fps: float | None = None,
        audio: bool,
    ) -> Path:
        """Normalize FPS/timebase/streams before any generated-section concat."""
        width, height = dimensions
        normalized: list[Path] = []
        for index, part in enumerate(rendered):
            job.raise_if_cancelled()
            normalized.append(
                await normalize_clip(
                    part,
                    job.workspace / f"normalized-section-{index:04d}.mp4",
                    width=width,
                    height=height,
                    fps=fps or float(settings.ltx_frame_rate),
                    audio=audio,
                )
            )
        return await concat_segments(normalized, output)

    def _record_audio_mode(self, job: AdapterJob, mode: AudioMode) -> None:
        logger.info(
            "audio_mode_selected",
            extra={"workflow_id": job.workflow_id, "audio_mode": mode.value},
        )

    # ── Guardrails ───────────────────────────────────────────────────────

    def _require_generation_shape(self, job: AdapterJob) -> None:
        """Refuses job shapes this runtime cannot honestly produce.

        The failure mode to avoid is the quiet one: an audio job that produces
        a broken file three attempts later, or a source-conditioned job that
        ignores its source and returns unrelated text-to-video footage.
        """
        if job.execution.get("output_kind") == "audio":
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    "audio-only output is not supported by the LTX distilled entry "
                    "point (benchmarked 2026-08-12: it attaches libx264 to an mp3 "
                    "container and fails); music must route to the `music` runtime"
                ),
                retriable=False,
            )
        unsupported = [item.role for item in job.inputs if item.role != "source_image"]
        if unsupported:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"inputs {unsupported} reached the plain generation path on "
                    f"workflow '{job.workflow_id}'; source-conditioned workflows have "
                    "their own handlers and must not silently drop their inputs"
                ),
                retriable=False,
            )

    async def _staged_source(
        self, job: AdapterJob, role: str, *, kind: str
    ) -> tuple[Path, MediaInfo]:
        """The uploaded file and its measurements, or a clear refusal.

        Probing here rather than at first use is what turns "the GPU burned two
        minutes and produced nothing" into "that file could not be read",
        before any compute is spent. A corrupt upload is corrupt on every
        attempt, hence `retriable=False`.
        """
        item = job.input_for(role)
        if item is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"{job.workflow_id} job arrived without a {role} input",
                retriable=False,
            )
        staged = item.require_path()

        noun = "video" if kind == "video" else "audio file"
        try:
            info = await probe_media(staged)
        except FfmpegError as exc:
            raise AdapterError(
                f"That {noun} could not be read. Please try another.",
                internal_detail=f"probe of {role} failed: {exc}",
                retriable=False,
            ) from exc

        usable = info.has_video if kind == "video" else info.has_audio
        if not usable or not info.duration_seconds:
            raise AdapterError(
                f"That {noun} could not be read. Please try another.",
                internal_detail=f"{role} is not usable {kind}: {info}",
                retriable=False,
            )
        return staged, info

    async def _conditioning_image(self, job: AdapterJob, role: str) -> Path | None:
        """A staged still, verified decodable, or None when not supplied.

        This decodes one frame rather than probing: ffprobe's metadata pass
        accepts garbage (the `tty` demuxer will even claim ASCII text as
        "video"), while an actual decode rejects a truncated or mislabelled
        upload in milliseconds.
        """
        item = job.input_for(role)
        if item is None:
            return None
        staged = item.require_path()
        try:
            await ffmpeg(
                ["-i", str(staged), "-frames:v", "1", "-f", "null", "-"], timeout=30
            )
        except FfmpegError as exc:
            raise AdapterError(
                "That image could not be read. Please try another.",
                internal_detail=f"decode check of {role} failed: {exc}",
                retriable=False,
            ) from exc
        return staged

    async def _final_frame_of(self, job: AdapterJob, source: Path) -> Path:
        try:
            return await cancellable(
                job, extract_final_frame(source, job.workspace / "seed-frame.png")
            )
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"final-frame extraction from source failed: {exc}",
                retriable=False,
            ) from exc

    def _require_models(self) -> None:
        shared = ("text_encoder", "video_vae", "audio_vae", "duration_head", "spatial_upsampler")
        needed = [self._transformer_file(), *(_MODEL_FILES[key] for key in shared)]
        root = settings.ltx_models_root
        missing = [name for name in needed if not (root / name).exists()]
        if missing:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"LTX weights missing under {root}: {missing}",
                retriable=False,
            )

    def _requested_seconds(self, job: AdapterJob) -> float:
        """The requested length. NOT capped at the single-pass ceiling.

        Anything longer than one pass is chained, so the ceiling constrains
        each GPU invocation rather than the product. Which lengths a customer
        may ask for is the workflow definition's job, validated by the API
        before the request ever reaches a worker.

        Automatic-duration workflows never call this — their length comes from
        a probe of the uploaded file, which is the point of `duration_mode:
        source`.
        """
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"no usable duration in {job.parameters!r}",
                retriable=False,
            )
        return seconds

    def _requested_dimensions(self, job: AdapterJob) -> tuple[int, int]:
        return _DIMENSIONS.get(
            str(job.parameters.get("aspect_ratio") or ""), _DEFAULT_DIMENSIONS
        )

    def _per_pass_seconds(
        self, job: AdapterJob, dimensions: tuple[int, int] | None = None
    ) -> float:
        """The most this GPU may be asked for in one invocation, for THIS shape.

        The ceiling is a property of the grid, not of the product. 1024x576
        sustains 60s while 896x512 — fewer pixels — dies at 60s and survives 30s.
        A single global value therefore has to satisfy the worst shape offered,
        which is how every 60s render came to be split into six passes with five
        seams when only one aspect ratio needed it.

        `dimensions` is passed by every caller that knows the grid, which is all
        of them; the default exists so the ceiling can still be asked for
        generically. A workflow may lower it further via
        `execution.max_segment_seconds` — nothing raises it above what the grid
        was measured at, because that is where the kernel failure lives.
        """
        grid = dimensions or self._requested_dimensions(job)
        measured = _GRID_CEILINGS.get(grid, _UNMEASURED_CEILING)
        requested = float(job.execution_int("max_segment_seconds", settings.ltx_max_seconds))
        # Three clamps, all lowering: the workflow's own override, the
        # operational brake, and what this grid was actually measured at.
        # `settings.ltx_max_seconds` is kept in the chain so one environment
        # variable can still pull every shape down mid-incident without a
        # deploy — that lever saved the product on 14 Aug and must not be
        # quietly removed by making the ceiling per-shape.
        return max(1.0, min(requested, float(settings.ltx_max_seconds), measured))

    def _frame_count(self, seconds: float) -> int:
        return max(1, round(seconds * settings.ltx_frame_rate))

    # ── The renderer handed to the chain ─────────────────────────────────

    def _renderer(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        *,
        dimensions: tuple[int, int],
        conditioning,
        prompt_for_step: Callable[[ChainStep], str] | None = None,
        require_audio: bool = False,
    ) -> RenderStep:
        """Binds this job's fixed choices into the callable the chain drives.

        `conditioning` may be sync or async — building a restyle's conditioning
        means extracting stills from the source, which is I/O, while a
        text-to-video's is a one-line decision.
        """

        async def render(step: ChainStep) -> None:
            items = conditioning(step)
            if asyncio.iscoroutine(items):
                items = await items
            # Dodge the decoder's measured bad shapes: render a few extra
            # frames where the exact count would crash the VAE, then trim back
            # so the delivered pass is exactly the planned length.
            requested_frames = self._frame_count(step.seconds)
            frames = safe_frame_count(
                dimensions, requested_frames, conditioned=bool(items)
            )
            if frames != requested_frames:
                logger.info(
                    "frame_count_nudged",
                    extra={
                        "workflow_id": job.workflow_id,
                        "grid": list(dimensions),
                        "requested_frames": requested_frames,
                        "rendered_frames": frames,
                        "conditioned": bool(items),
                    },
                )
            await self._execute(
                job=job,
                cmd=self._command(
                    job,
                    step.seconds,
                    step.output,
                    conditioning=items,
                    dimensions=dimensions,
                    prompt=(prompt_for_step(step) if prompt_for_step else None),
                    # Distinct per pass or every chained render replays the
                    # same noise; still deterministic so a retry reproduces.
                    seed=self._seed_for_step(job, step.index),
                    num_frames=frames,
                ),
                reporter=reporter,
                band=step.band,
                section=step.section_progress,
            )
            if frames != requested_frames:
                # BEFORE verification and before any continuity frame is
                # extracted, so the seam frame sits at the planned timestamp.
                await self._trim_to(job, step.output, step.seconds)
            if require_audio:
                try:
                    await verify_output(
                        step.output,
                        OutputExpectation(
                            expect_video=True,
                            expect_audio=True,
                            expected_seconds=step.seconds,
                        ),
                    )
                except FfmpegError as exc:
                    raise AdapterError(
                        "This generation could not be completed. Please try again.",
                        internal_detail=(
                            f"section {step.index + 1}/{step.total} failed media "
                            f"validation: {exc}"
                        ),
                    ) from exc

        return render

    # ── The command (pure — this is what unit tests pin) ─────────────────

    def _transformer_file(self) -> str:
        key = "transformer_nvfp4" if "nvfp4" in settings.ltx_quantization else "transformer_bf16"
        return _MODEL_FILES[key]

    def _launcher(self) -> list[str]:
        """The argv prefix that reaches the LTX environment.

        A separate seam so tests can substitute a plain Python stub and still
        exercise every real flag `_command` produces.
        """
        return ["uv", "run", "python", "-m", "ltx_pipelines.distilled"]

    def _command(
        self,
        job: AdapterJob,
        seconds: float,
        output: Path,
        *,
        conditioning: Sequence[ConditioningFrame] = (),
        dimensions: tuple[int, int] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
        num_frames: int | None = None,
    ) -> list[str]:
        root = settings.ltx_models_root
        # Explicit dimensions (extension and restyle: the source's aspect
        # decides) beat the requested aspect ratio's lookup.
        width, height = dimensions or self._requested_dimensions(job)
        # `num_frames` lets the renderer substitute a measured-safe count for a
        # frame count the VAE cannot decode (see `safe_frame_count`); the extra
        # material is trimmed after the render, never delivered.
        frames = num_frames if num_frames is not None else self._frame_count(seconds)
        if seed is None:
            # The pipeline's default seed is fixed, which would hand two users
            # with the same prompt the same video. CRC of the job id:
            # deterministic per job (a retried job reproduces), distinct
            # across jobs.
            seed = zlib.crc32(job.job_id.encode())

        cmd = [
            *self._launcher(),
            "--transformer-path", str(root / self._transformer_file()),
            "--text-encoder-path", str(root / _MODEL_FILES["text_encoder"]),
            "--video-vae-path", str(root / _MODEL_FILES["video_vae"]),
            "--audio-vae-path", str(root / _MODEL_FILES["audio_vae"]),
            "--duration-head-path", str(root / _MODEL_FILES["duration_head"]),
            "--spatial-upsampler-path", str(root / _MODEL_FILES["spatial_upsampler"]),
            "--quantization", settings.ltx_quantization,
            # Exactly what the user typed, as one argv element: no quoting,
            # escaping, truncation or rewriting between the text field and
            # the model. Pinned by test_the_users_prompt_reaches_the_model_verbatim.
            "--prompt", job.prompt if prompt is None else prompt,
            "--num-frames", str(frames),
            "--height", str(height),
            "--width", str(width),
            "--frame-rate", str(settings.ltx_frame_rate),
            "--seed", str(seed),
            "--output-path", str(output),
        ]
        if job.execution.get("enhance_prompt"):
            # LTX's own enhancer expands a terse prompt into a detailed one,
            # which is the only adherence lever the distilled entry point
            # offers. It REWRITES the prompt, so it stays opt-in per workflow
            # and off by default: a user who typed something specific should
            # not silently get a machine's paraphrase of it.
            cmd.append("--enhance-prompt")
        for item in conditioning:
            # `--image PATH FRAME_IDX STRENGTH`, once per conditioning frame,
            # in ascending frame order so the pipeline reads them as a timeline
            # rather than as an unordered set.
            cmd += item.as_args()
        return cmd

    async def _trim_to(self, job: AdapterJob, path: Path, seconds: float) -> None:
        """Cuts a nudged render back to the requested length, in place.

        A re-encode rather than a stream copy: `-c copy` can only cut on a
        keyframe, and the whole point is landing on the exact requested
        duration. Settings match the section normalizer so a trimmed pass is
        indistinguishable from an untrimmed one downstream.
        """
        trimmed = path.with_name(path.stem + ".trimmed.mp4")
        try:
            await cancellable(
                job,
                ffmpeg([
                    "-i", str(path),
                    "-t", f"{seconds:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(trimmed),
                ]),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"trimming a nudged render failed: {exc}",
            ) from exc
        trimmed.replace(path)

    def _seed_for_step(self, job: AdapterJob, index: int) -> int:
        requested = job.parameters.get("seed")
        try:
            base = int(requested) if requested is not None else None
        except (TypeError, ValueError):
            base = None
        if base is None:
            return zlib.crc32(f"{job.job_id}:{index}".encode())
        return (base + index) % (2**31)

    # ── Supervision ──────────────────────────────────────────────────────

    async def _execute(
        self,
        cmd: list[str],
        job: AdapterJob,
        reporter: StageReporter,
        *,
        band: tuple[int, int] = (GENERATE_FROM, GENERATE_TO),
        section: tuple[int, int, float, float] | None = None,
    ) -> None:
        """Runs the pipeline, streaming its output for progress and diagnostics.

        The contract with the rest of the platform:

          * `job.raise_if_cancelled()` is honoured within `_CANCEL_POLL_SECONDS`
            even while the pipeline is silent — cancellation and timeout both
            surface here as exceptions.
          * The child is killed on *every* non-completion path (the `finally`),
            because an orphaned render holds VRAM, and the runner is about to
            delete the workspace the child is writing into.

        `band` compresses the markers' sweep into a slice of it, so N chained
        passes produce one monotonic ramp instead of N restarts. `section`
        swaps the stage messages for "Generating section i of N…" — the
        machinery is only named when there are several.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(settings.ltx_repo_dir),
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"could not launch LTX pipeline ({cmd[0]!r}): {exc}",
                retriable=False,
            ) from exc

        tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
        marker_from = 0
        try:
            assert process.stdout is not None
            while True:
                job.raise_if_cancelled()
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(), timeout=_CANCEL_POLL_SECONDS
                    )
                except TimeoutError:
                    continue  # silence — loop back to the cancellation check
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    tail.append(text)
                matched = match_marker(text, marker_from)
                if matched is None:
                    continue
                _, nominal, message = _MARKERS[matched]
                marker_from = matched + 1
                low, high = band
                progress = low + (nominal - GENERATE_FROM) * (high - low) // (
                    GENERATE_TO - GENERATE_FROM
                )
                if section is None:
                    await reporter.generating(progress, message)
                else:
                    await reporter.section(
                        section[0],
                        section[1],
                        progress,
                        start_seconds=section[2],
                        end_seconds=section[3],
                    )

            returncode = await process.wait()
            if returncode != 0:
                raise AdapterError(
                    "This generation could not be completed. Please try again.",
                    internal_detail=(
                        f"LTX pipeline exited {returncode}; output tail: "
                        + " | ".join(tail)
                    ),
                )
        finally:
            if process.returncode is None:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except TimeoutError:
                    logger.warning("ltx_kill_timeout", extra={"pid": process.pid})


def _delivery_fps(source: MediaInfo) -> float:
    """The rate a stitched result is delivered at.

    The source's own where it is sane, so a 30fps upload does not come back
    retimed. Clamped because ffprobe reports nonsense for some variable-rate
    phone recordings, and normalizing to a nonsense rate produces a file that
    plays at the wrong speed.
    """
    return min(60.0, max(10.0, source.fps or float(settings.ltx_frame_rate)))


def _video_result(output: Path, info: MediaInfo) -> AdapterResult:
    return AdapterResult(
        path=output,
        content_type="video/mp4",
        kind="video",
        duration_seconds=info.duration_seconds,
        width=info.width,
        height=info.height,
    )
