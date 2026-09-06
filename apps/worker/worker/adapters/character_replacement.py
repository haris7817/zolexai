"""Character Replacement — a separate module on the client's LTX 2.5 graph.

Runtime `character_replacement`, one workflow (`character-replacement`),
one graph (`ltx25_character_replacement.json`). Kept apart from Video to
Video by design: that workflow restyles footage on the CLI runtime and is
untouched; this one regenerates a performance with a new person, and the
two share nothing but the ComfyUI service object.

## What the product is (client decision, 5 Sep 2026)

Exactly the delivered sample. The reference image provides the new
character AND the environment; the source video provides motion, camera and
timing. No background preservation, no compositing — the graph's Ripple
LoRA propagates the reference picture through the source's motion, and the
first frame of the result is the reference picture itself (measured on the
sample: frame 0 is the photo, frame 4 onward is the source's motion).

## What a job supplies to the graph

The source clip (uploaded as-is; the graph resamples it to 24 fps), the
reference still (PNG-normalised), the length in whole seconds (the source's,
capped by `character_replacement_max_seconds`), the canvas (the pack's
736×1280 budget, oriented like the source), the prompt (the pack's lead
sentence plus the customer's description of the new character — the sample
prompt shows that description is what carries identity), the seed and the
output prefix. Sampler, schedule, LoRA strength, patches and switches are
the pack's.

STATUS: WAITING FOR GPU VALIDATION (runtime, VRAM, and a side-by-side with
the delivered sample).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
)
from worker.comfy.client import ComfyError, evict_comfy_vram
from worker.comfy.ltx_graphs import (
    GraphError,
    ReplacementEdits,
    character_frames_for_seconds,
    compile_character_replacement,
    oriented_canvas,
)
from worker.comfy.ltx_prompts import character_replacement_prompt, negative_for
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import GENERATE_FROM, GENERATE_TO, StageReporter
from worker.media import (
    FfmpegError,
    MediaInfo,
    OutputExpectation,
    ffmpeg,
    probe_media,
    verify_output,
)
from worker.providers.ltx_comfy import LtxComfyService

logger = get_logger(__name__)

WORKFLOW_ID = "character-replacement"

#: Below this the graph's own frame formula gives fewer than two latent
#: frames of motion to follow; the customer is told to upload a longer clip.
_MIN_SOURCE_SECONDS = 1.0


class CharacterReplacementAdapter:
    name = "character_replacement"

    def __init__(self, service: LtxComfyService | None = None) -> None:
        self._service = service

    def supports(self, workflow_id: str) -> bool:
        return workflow_id == WORKFLOW_ID

    def service(self) -> LtxComfyService:
        if self._service is None:
            self._service = LtxComfyService()
        return self._service

    # ── Entry ────────────────────────────────────────────────────────────

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        if not self.supports(job.workflow_id):
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"character_replacement does not serve '{job.workflow_id}'; "
                    "check this deployment's execution.runtime"
                ),
                retriable=False,
            )
        reporter = StageReporter(on_progress)
        await reporter.preparing("Setting up your video…")
        await evict_comfy_vram(exclude=settings.ltx_comfy_base_url)

        source = job.input_for("source_video")
        reference = job.input_for("reference_image")
        if source is None:
            raise AdapterError(
                "Please add the video to work from.",
                internal_detail="character-replacement without source_video",
                retriable=False,
            )
        if reference is None:
            raise AdapterError(
                "Please add a picture of the new character.",
                internal_detail="character-replacement without reference_image",
                retriable=False,
            )

        await reporter.probing("Reading your video…")
        info = await self._probe(source.require_path())
        seconds = self.window_seconds(info, job)
        width, height = oriented_canvas(
            tuple(settings.character_replacement_canvas),
            source_width=info.width,
            source_height=info.height,
        )

        service = self.service()
        video_name = await self._upload_source(job, source, info)
        image_name = await self._upload_still(job, reference)

        edits = ReplacementEdits(
            positive=character_replacement_prompt(job.prompt),
            negative=negative_for(WORKFLOW_ID, job.execution),
            video=video_name,
            image=image_name,
            seconds=seconds,
            width=width,
            height=height,
            seed_base=self.seed_base(job),
            filename_prefix=f"zolexai/{job.job_id}/output",
        )
        try:
            api = compile_character_replacement(service.load("character_replacement"), edits)
        except (GraphError, ComfyError) as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"graph compile failed: {exc}",
                retriable=False,
            ) from exc

        expected_wall = max(1.0, seconds * settings.ltx_comfy_expected_wall_per_output_second)
        await reporter.generating(GENERATE_FROM, "Starting your video…")

        async def tick(elapsed: float) -> None:
            fraction = min(1.0, elapsed / (expected_wall * 1.2))
            await reporter.generating(
                GENERATE_FROM + int((GENERATE_TO - GENERATE_FROM - 1) * fraction),
                "Generating your video…",
            )

        output = job.workspace / "output.mp4"
        started = time.monotonic()
        prompt_id = ""
        try:
            prompt_id = await service.generate(api, client_id=f"zolex-{job.job_id}")
            logger.info(
                "character_replacement_submitted",
                extra={
                    "job_id": job.job_id,
                    "prompt_id": prompt_id,
                    "seconds": seconds,
                    "canvas": [width, height],
                    "source_seconds": info.duration_seconds,
                    "nodes": len(api),
                },
            )
            remaining = job.seconds_remaining
            timeout = settings.ltx_comfy_generation_timeout
            if remaining is not None:
                timeout = max(1.0, min(timeout, remaining))
            history = await service.progress(job, prompt_id, timeout_seconds=timeout, on_tick=tick)
            await service.collect(history, output)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc
        finally:
            if settings.ltx_comfy_free_after_job:
                await service.free_memory()
        wall = time.monotonic() - started

        # The graph renders round((fps·s − 1)/8)·8 + 1 frames, capped by the
        # frames the resampled source actually has; the sample (8 s) is 193
        # frames = 8.04 s. Audio is the source's own, passed through.
        fps = settings.ltx_comfy_frame_rate
        planned_frames = min(
            character_frames_for_seconds(seconds, fps),
            int(math.floor((info.duration_seconds or seconds) * fps)),
        )
        expected_seconds = planned_frames / fps
        try:
            result_info = await verify_output(
                output,
                OutputExpectation(
                    expect_video=True,
                    expect_audio=True,
                    expected_seconds=expected_seconds,
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "The finished video failed its check.", internal_detail=str(exc)
            ) from exc

        logger.info(
            "character_replacement_finished",
            extra={
                "job_id": job.job_id,
                "prompt_id": prompt_id,
                "wall_seconds": round(wall, 1),
                "duration_seconds": result_info.duration_seconds,
                "width": result_info.width,
                "height": result_info.height,
            },
        )
        await reporter.uploading()
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=result_info.duration_seconds,
            width=result_info.width,
            height=result_info.height,
        )

    # ── Readings ─────────────────────────────────────────────────────────

    @staticmethod
    def window_seconds(info: MediaInfo, job: AdapterJob) -> int:
        """Whole seconds of the source the graph is asked to follow.

        The source's own length, floored to whole seconds (the graph's
        length input is an integer), capped by the deployment's ceiling.
        The delivered length is therefore the source's minus any fraction of
        a second, minus anything above the cap — stated in the log and the
        history rather than silently.
        """
        seconds = info.duration_seconds or 0.0
        if seconds < _MIN_SOURCE_SECONDS:
            raise AdapterError(
                "Please upload a video at least one second long.",
                internal_detail=f"source is {seconds:.2f}s",
                retriable=False,
            )
        cap = job.execution_int("max_seconds", int(settings.character_replacement_max_seconds))
        window = min(int(math.floor(seconds)), max(1, cap))
        if window < seconds - 1e-6:
            logger.info(
                "character_replacement_window",
                extra={
                    "job_id": job.job_id,
                    "source_seconds": round(seconds, 3),
                    "window_seconds": window,
                    "cap_seconds": cap,
                },
            )
        return window

    @staticmethod
    def seed_base(job: AdapterJob) -> int | None:
        """A customer seed wins; otherwise None — the graph's own fixed seeds.

        The ZIP's character graph carries fixed seeds and the client asked
        for the tool to behave exactly like the ZIP (6 Sep 2026). So a job
        without a seed runs those seeds, and "Regenerate" with the same
        inputs returns the same result — the seed control is the way to a
        different take.
        """
        raw = job.parameters.get("seed")
        try:
            if raw is not None and str(raw).strip() != "":
                return abs(int(raw)) % (2**48)
        except (TypeError, ValueError):
            pass
        return None

    async def _probe(self, path: Path) -> MediaInfo:
        try:
            info = await probe_media(path)
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"probe failed: {exc}",
                retriable=False,
            ) from exc
        if not info.has_video or not info.duration_seconds:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"source probe: {info!r}",
                retriable=False,
            )
        return info

    async def _upload_source(self, job: AdapterJob, source: AdapterInput, info: MediaInfo) -> str:
        """The clip as uploaded — plus a silent track when it has none.

        The graph passes the source's audio through to the output; a source
        with no audio stream would hand the combiner nothing. Silence is the
        honest stand-in and costs a stream copy.
        """
        staged = source.require_path()
        name = f"zolex_{job.job_id}_source.mp4"
        local = job.workspace / name
        try:
            if info.has_audio:
                await ffmpeg(
                    ["-i", str(staged), "-c", "copy", "-movflags", "+faststart", str(local), "-y"]
                )
            else:
                await ffmpeg(
                    [
                        "-i",
                        str(staged),
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=48000:cl=stereo",
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-shortest",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        str(local),
                        "-y",
                    ]
                )
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"source repack failed: {exc}",
                retriable=False,
            ) from exc
        try:
            return await self.service().upload(local, name=name)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc

    async def _upload_still(self, job: AdapterJob, reference: AdapterInput) -> str:
        staged = reference.require_path()
        name = f"zolex_{job.job_id}_reference.png"
        local = job.workspace / name
        try:
            await ffmpeg(["-i", str(staged), "-frames:v", "1", str(local), "-y"])
        except FfmpegError as exc:
            raise AdapterError(
                "That picture could not be read. Please try another.",
                internal_detail=f"reference normalise failed: {exc}",
                retriable=False,
            ) from exc
        try:
            return await self.service().upload(local, name=name)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc
