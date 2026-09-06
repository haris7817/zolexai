"""LTX 2.5 through the client's ComfyUI graphs — Text to Video, First/Last
Frame Video and Extend Video.

Runtime `ltx_comfy`. The graphs under `benchmarks/client-pack/ltx25/` are the
product (client decision 5 Sep 2026); this adapter supplies what a job has to
supply and nothing else: the prompt, the seed, the length, the input stills,
where the file goes. `worker/comfy/ltx_graphs.py` is the compiler,
`worker/providers/ltx_comfy.py` the service; this module is the job flow.

## One pass, one submission

Every product length (5/10/15/30 s) is one graph submission: the pack's own
slider runs to 30 s and the client's sample is a single 30 s pass. Longer
results are not single generations — they are chained continuations through
the extension engine (`worker/longform/continuation.py`), which drives
`render_pass` once per section and stitches. This adapter never asks the
graph for more than `ltx_comfy_max_segment_seconds`.

## What is unchanged from the CLI adapter

The user's prompt reaches the graph verbatim as the first block. The same
deterministic structuring (`execution.prompt_structuring`) and the same
soundtrack-owner clause (the 28 Aug "reads the prompt aloud" fix) apply,
because they are prompt text and the prompt is a job input. The sound on/off
parameter is honoured the same way: a silent file drops the audio stream
after the render is verified.

## Verification

Nothing here is proven on a GPU yet (the node is unavailable in Sep 2026).
What is proven without one: the compiled prompts, every edit, the HTTP
conversation (a fake ComfyUI in the tests serves a real MP4), cancellation,
failure handling, and the output validation. STATUS: WAITING FOR GPU
VALIDATION for runtime, VRAM and quality.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    parse_duration_seconds,
)
from worker.comfy.client import ComfyError, evict_comfy_vram
from worker.comfy.ltx_graphs import (
    GenerationEdits,
    GraphError,
    aspect_label_for,
    compile_first_last_frame,
    compile_text_to_video,
    frames_for_seconds,
)
from worker.comfy.ltx_prompts import negative_for
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import GENERATE_FROM, GENERATE_TO, StageReporter, structure_prompt
from worker.longform.chain import ChainStep
from worker.longform.continuation import continue_video
from worker.longform.language import soundscape_clause
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


@dataclass(frozen=True)
class PassSpec:
    """Everything one graph submission needs, and nothing about the job."""

    seconds: float
    positive: str
    negative: str
    aspect_label: str
    seed_base: int
    first_image: str | None = None
    """Filename already uploaded to the service; None selects graph 01."""
    last_image: str | None = None
    band: tuple[int, int] = (GENERATE_FROM, GENERATE_TO)
    section: tuple[int, int, float, float] | None = None
    """(index, total, start, end) for section copy on chained renders."""


class LtxComfyAdapter:
    name = "ltx_comfy"

    #: Grows per phase: Text to Video (Phase 1), First/Last Frame (Phase 2),
    #: Extend Video (Phase 4). Video to Video is never here — it stays on
    #: the CLI runtime untouched.
    _SUPPORTED = frozenset({"text-to-video", "image-to-video", "extend-video"})

    def __init__(self, service: LtxComfyService | None = None) -> None:
        self._service = service

    def supports(self, workflow_id: str) -> bool:
        return workflow_id in self._SUPPORTED

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
                    f"ltx_comfy does not serve '{job.workflow_id}'; check this "
                    "deployment's execution.runtime"
                ),
                retriable=False,
            )
        reporter = StageReporter(on_progress)
        await reporter.preparing("Setting up your video…")
        # Another engine's ComfyUI may hold the card; this one's stays warm.
        await evict_comfy_vram(exclude=settings.ltx_comfy_base_url)
        if job.workflow_id == "extend-video":
            return await self._run_extension(job, reporter)
        return await self._run_generation(job, reporter)

    # ── text-to-video / image-to-video ───────────────────────────────────

    async def _run_generation(self, job: AdapterJob, reporter: StageReporter) -> AdapterResult:
        seconds = self._requested_seconds(job)
        per_pass = self.per_pass_seconds(job)
        if seconds > per_pass + 1e-6:
            raise AdapterError(
                f"This tool makes videos up to {int(per_pass)} seconds long. "
                "Use Extend Video to continue one.",
                internal_detail=f"{seconds}s requested; single pass ceiling is {per_pass}s",
                retriable=False,
            )
        self._require_lattice(seconds)

        first = job.input_for("source_image")
        last = job.input_for("last_frame")
        if job.workflow_id == "image-to-video" and first is None:
            raise AdapterError(
                "Please add the first frame image.",
                internal_detail="image-to-video without source_image",
                retriable=False,
            )
        if job.workflow_id == "text-to-video":
            first = last = None

        aspect_label = await self._aspect_label(job)
        spec = PassSpec(
            seconds=seconds,
            positive=self.positive_prompt(job),
            negative=negative_for(job.workflow_id, job.execution),
            aspect_label=aspect_label,
            seed_base=self.seed_base(job),
            first_image=await self.upload_still(job, first, "first") if first else None,
            last_image=await self.upload_still(job, last, "last") if last else None,
        )
        output = job.workspace / "output.mp4"
        info = await self.render_pass(job, spec, output, reporter)
        return await self.deliver(job, output, info, reporter)

    # ── extend-video ─────────────────────────────────────────────────────

    async def _run_extension(self, job: AdapterJob, reporter: StageReporter) -> AdapterResult:
        """The customer's clip plus 5/10/15/30 s of chained continuation.

        The extension engine (`worker/longform/continuation.py`) owns the
        chain: it takes the source's final frame, drives `render_pass` once
        per section with the previous part's last picture as the first-frame
        still, drops the overlap frame at every seam, and stitches the
        source in front. Each section is one submission of the client's
        First/Last Frame graph.

        Two optional stills (client request, 6 Sep 2026), both through the
        same graph: `first_frame` replaces the source's final frame as pass
        0's conditioning picture; `last_frame` goes to the FINAL pass as the
        graph's second image, so the continuation ends on it. With neither
        the run is what it was before — first frame only, from the source's
        own final frame. Every extension is its own job with its own output;
        the chain has no counter and the source is never rewritten.
        """
        source = job.input_for("source_video")
        if source is None:
            raise AdapterError(
                "Please add the video to continue.",
                internal_detail="extend-video without source_video",
                retriable=False,
            )
        staged = source.require_path()
        seconds = self._requested_seconds(job)
        per_pass = self.per_pass_seconds(job)
        try:
            info = await probe_media(staged)
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"source probe failed: {exc}",
                retriable=False,
            ) from exc
        aspect_label = await self._aspect_label_for_source(info)
        positive = self.positive_prompt(job)
        negative = negative_for(job.workflow_id, job.execution)
        seed = self.seed_base(job)

        first_still = job.input_for("first_frame")
        last_still = job.input_for("last_frame")
        # The last frame is uploaded once, up front, so a bad picture fails
        # the job before any GPU time is spent rather than at the final pass.
        last_name = await self.upload_still(job, last_still, "last") if last_still else None
        logger.info(
            "extension_framing",
            extra={
                "job_id": job.job_id,
                "first_frame": first_still is not None,
                "last_frame": last_still is not None,
                "source_seconds": info.duration_seconds,
                "added_seconds": seconds,
            },
        )

        async def render_pass(step: ChainStep, frame: Path | None) -> MediaInfo:
            if frame is None:
                raise AdapterError(
                    "This generation could not be completed. Please try again.",
                    internal_detail=f"continuation pass {step.index} has no conditioning frame",
                    retriable=False,
                )
            self._require_lattice(step.seconds)
            # `frame` is the customer's first frame on pass 0 when they gave
            # one (the engine seeds the chain with it), otherwise the
            # previous part's last picture — the same upload either way.
            first = await self.upload_still(job, frame, f"continue{step.index:02d}")
            final_pass = step.index + 1 == step.total
            spec = PassSpec(
                seconds=step.seconds,
                positive=positive,
                negative=negative,
                aspect_label=aspect_label,
                seed_base=seed + step.index,
                first_image=first,
                last_image=last_name if final_pass else None,
                band=step.band,
                section=step.section_progress,
            )
            return await self.render_pass(job, spec, step.output, reporter)

        output, _metadata = await continue_video(
            job,
            source=staged,
            seconds=seconds,
            per_pass_seconds=per_pass,
            fps=float(settings.ltx_comfy_frame_rate),
            render_pass=render_pass,
            reporter=reporter,
            first_frame=first_still.require_path() if first_still else None,
            last_frame=last_still.require_path() if last_still else None,
        )
        result_info = await probe_media(output)
        return await self.deliver(job, output, result_info, reporter)

    async def _aspect_label_for_source(self, info: MediaInfo) -> str:
        """The product ratio closest to the source's own frame.

        The graph renders on its own canvas; the engine then fits every part
        to the source's dimensions, so the closest ratio is what keeps that
        fit a slight crop rather than a heavy one.
        """
        if info.width and info.height:
            actual = info.width / info.height
            ratio = min(
                ("16:9", "9:16", "1:1"),
                key=lambda r: abs(actual - (lambda a, b: a / b)(*map(int, r.split(":")))),
            )
        else:
            ratio = "16:9"
        options = await self.service().aspect_options()
        try:
            return aspect_label_for(ratio, options)
        except GraphError as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=str(exc),
                retriable=False,
            ) from exc

    # ── One graph submission ─────────────────────────────────────────────

    async def render_pass(
        self, job: AdapterJob, spec: PassSpec, output: Path, reporter: StageReporter
    ) -> MediaInfo:
        """Compiles, submits, waits, collects and verifies one pass.

        Public because the extension engine drives it once per section. The
        returned probe is of the file at `output`.
        """
        service = self.service()
        edits = GenerationEdits(
            positive=spec.positive,
            negative=spec.negative,
            seconds=spec.seconds,
            aspect_label=spec.aspect_label,
            seed_base=spec.seed_base,
            filename_prefix=f"zolexai/{job.job_id}/{output.stem}",
            first_image=spec.first_image,
            last_image=spec.last_image,
        )
        try:
            if spec.first_image is None:
                api = compile_text_to_video(service.load("text_to_video"), edits)
                graph = "text_to_video"
            else:
                api = compile_first_last_frame(service.load("first_last_frame"), edits)
                graph = "first_last_frame"
        except (GraphError, ComfyError) as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"graph compile failed: {exc}",
                retriable=False,
            ) from exc

        # Pacing only — an elapsed-time bar between the band's ends that
        # never claims completion. The rate is a placeholder until the GPU
        # benchmark records one (WAITING FOR GPU VALIDATION).
        expected_wall = max(1.0, spec.seconds * settings.ltx_comfy_expected_wall_per_output_second)
        low, high = spec.band

        async def announce(progress: int, message: str) -> None:
            if spec.section is not None:
                index, total, start, end = spec.section
                await reporter.section(index, total, progress, start_seconds=start, end_seconds=end)
            else:
                await reporter.generating(progress, message)

        await announce(low, "Starting your video…")

        async def tick(elapsed: float) -> None:
            fraction = min(1.0, elapsed / (expected_wall * 1.2))
            await announce(low + int((high - low - 1) * fraction), "Generating your video…")

        started = time.monotonic()
        prompt_id = ""
        try:
            prompt_id = await service.generate(api, client_id=f"zolex-{job.job_id}")
            logger.info(
                "ltx_comfy_submitted",
                extra={
                    "job_id": job.job_id,
                    "workflow": job.workflow_id,
                    "graph": graph,
                    "prompt_id": prompt_id,
                    "seconds": spec.seconds,
                    "aspect": spec.aspect_label,
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

        # The graphs render fps·s+1 frames, so the file is one frame longer
        # than the nominal length (the ZIP's 30 s sample probes at 30.042 s).
        fps = settings.ltx_comfy_frame_rate
        expected_seconds = spec.seconds + 1.0 / fps
        try:
            info = await verify_output(
                output,
                OutputExpectation(
                    expect_video=True,
                    expect_audio=True,
                    expected_seconds=expected_seconds,
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "The finished video failed its check.",
                internal_detail=str(exc),
            ) from exc

        logger.info(
            "ltx_comfy_pass_finished",
            extra={
                "job_id": job.job_id,
                "prompt_id": prompt_id,
                "wall_seconds": round(wall, 1),
                "duration_seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "graph": graph,
            },
        )
        return info

    # ── Delivery ─────────────────────────────────────────────────────────

    async def deliver(
        self, job: AdapterJob, output: Path, info: MediaInfo, reporter: StageReporter
    ) -> AdapterResult:
        wants_sound = str(job.parameters.get("sound", True)).strip().lower() not in (
            "false",
            "no",
            "off",
            "0",
        )
        if info.has_audio and not wants_sound:
            await reporter.finalizing("Removing the soundtrack…")
            muted = output.with_name(f"{output.stem}_muted.mp4")
            await ffmpeg(["-i", str(output), "-an", "-c:v", "copy", str(muted), "-y"])
            output = muted
        await reporter.uploading()
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=info.duration_seconds,
            width=info.width,
            height=info.height,
        )

    # ── Job readings ─────────────────────────────────────────────────────

    @staticmethod
    def _requested_seconds(job: AdapterJob) -> float:
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None:
            raise AdapterError(
                "Please choose a video length.",
                internal_detail=f"no usable duration in {job.parameters.get('duration')!r}",
                retriable=False,
            )
        return seconds

    @staticmethod
    def per_pass_seconds(job: AdapterJob) -> float:
        ceiling = float(settings.ltx_comfy_max_segment_seconds)
        override = job.execution.get("max_segment_seconds")
        try:
            if override is not None:
                ceiling = min(ceiling, float(override))
        except (TypeError, ValueError):
            pass
        return ceiling

    @staticmethod
    def _require_lattice(seconds: float) -> None:
        try:
            frames_for_seconds(seconds, settings.ltx_comfy_frame_rate)
        except GraphError as exc:
            raise AdapterError(
                "This tool supports these lengths: 5s, 10s, 15s, 30s.",
                internal_detail=str(exc),
                retriable=False,
            ) from exc

    async def _aspect_label(self, job: AdapterJob) -> str:
        ratio = str(job.parameters.get("aspect_ratio") or "16:9").strip()
        options = await self.service().aspect_options()
        try:
            return aspect_label_for(ratio, options)
        except GraphError as exc:
            raise AdapterError(
                "This tool supports these aspect ratios: 16:9, 9:16, 1:1.",
                internal_detail=str(exc),
                retriable=False,
            ) from exc

    @staticmethod
    def positive_prompt(job: AdapterJob) -> str:
        """The customer's words first and verbatim, then the house additions.

        `structure_prompt` appends derived continuity rules when the
        deployment asks for it; the soundtrack-owner clause is the 28 Aug
        fix for the model narrating its own prompt. Both are prompt text,
        which is a job input the graph exposes.
        """
        text = job.prompt.strip()
        if job.execution.get("prompt_structuring"):
            text = structure_prompt(text, v2=bool(job.execution.get("prompt_structuring_v2")))
        clause = soundscape_clause(job.prompt, job.parameters, job.execution)
        if clause:
            text = f"{text.rstrip()} {clause}"
        return text

    @staticmethod
    def seed_base(job: AdapterJob) -> int:
        """A customer seed wins; otherwise the job id, stable across retries."""
        raw = job.parameters.get("seed")
        try:
            if raw is not None and str(raw).strip() != "":
                return abs(int(raw)) % (2**48)
        except (TypeError, ValueError):
            pass
        digest = hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    async def upload_still(self, job: AdapterJob, source: AdapterInput | Path, tag: str) -> str:
        """Normalises a still to PNG and uploads it to the service's input dir."""
        staged = source.require_path() if isinstance(source, AdapterInput) else source
        name = f"zolex_{job.job_id}_{tag}.png"
        local = job.workspace / name
        try:
            await ffmpeg(["-i", str(staged), "-frames:v", "1", str(local), "-y"])
        except FfmpegError as exc:
            raise AdapterError(
                "One of the selected images could not be read.",
                internal_detail=f"still normalise failed for {tag}: {exc}",
                retriable=False,
            ) from exc
        try:
            return await self.service().upload(local, name=name)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc

    def with_section(
        self,
        spec: PassSpec,
        *,
        band: tuple[int, int],
        section: tuple[int, int, float, float] | None,
    ) -> PassSpec:
        return replace(spec, band=band, section=section)
