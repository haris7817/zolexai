"""Text to Video HD — the client's own FAST 1080 workflow (7 Sep 2026).

A fourth graph the client sent after the pack: NVFP4 transformer, no LoRAs,
no detailer, one 8-step stage, generated at 1920x1088 and delivered at
1920x1080 with its own soundtrack. Benchmarked at 8, 15 and 30 s
(`docs/internal/ltx25-highres-benchmark.md`) and byte-identical through this
path against a direct ComfyUI run at every length.

**Deliberately its own adapter.** It shares only the ComfyUI service object
with `ltx_comfy`; nothing here can change what Text to Video, Image to Video
or Extend Video render. Their graphs, their adapter and their routing are
untouched, so this ships as an addition and rolls back by removing one line
of YAML.

The graph exposes exactly what a job needs — the two prompt boxes, a duration
in seconds from which it computes its own frame count, a seed, the output
location, and an optional conditioning picture — and the compiler
(`compile_fast_1080`) sets those and nothing else.
"""

from __future__ import annotations

import time

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    cancellable,
    parse_duration_seconds,
)
from worker.comfy.client import ComfyError
from worker.comfy.ltx_graphs import (
    Fast1080Edits,
    GraphError,
    compile_fast_1080,
)
from worker.comfy.ltx_prompts import negative_for
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import GENERATE_FROM, GENERATE_TO, StageReporter
from worker.media import FfmpegError, OutputExpectation, ffmpeg, verify_output
from worker.providers.ltx_comfy import LtxComfyService

logger = get_logger(__name__)

WORKFLOW_ID = "text-to-video-hd"

#: The graph's frame arithmetic, `1 + floor(fps*seconds/8)*8`, reproduced so a
#: length can be checked before any GPU time is spent.
def frames_for(seconds: float, fps: int = 24) -> int:
    return 1 + int(fps * seconds / 8) * 8


class LtxHdAdapter:
    """One pass of the client's FAST 1080 graph."""

    name = "ltx_hd"

    def __init__(self, service: LtxComfyService | None = None) -> None:
        self._service = service

    def service(self) -> LtxComfyService:
        if self._service is None:
            self._service = LtxComfyService()
        return self._service

    def supports(self, workflow_id: str) -> bool:
        return workflow_id == WORKFLOW_ID

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing()

        seconds = self._seconds(job)
        frames = frames_for(seconds, settings.ltx_comfy_frame_rate)
        service = self.service()

        catalogue = await service.object_info()
        if catalogue is None:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail="the ComfyUI server offered no /object_info",
            )

        # The graph's LoadImage slot must name a real file even when the
        # picture is discarded: ComfyUI refuses an empty one at validation.
        image = await self._placeholder(job)

        try:
            api = compile_fast_1080(
                service.load("fast_1080"),
                Fast1080Edits(
                    positive=job.prompt.strip(),
                    negative=negative_for(WORKFLOW_ID, job.execution),
                    seconds=seconds,
                    seed=self._seed(job),
                    filename_prefix=f"zolexai/{job.job_id}/output",
                    image=image,
                    condition_on_image=False,
                ),
                catalogue,
            )
        except (GraphError, ComfyError) as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"graph compile failed: {exc}",
                retriable=False,
            ) from exc

        logger.info(
            "ltx_hd_submitting",
            extra={
                "job_id": job.job_id,
                "seconds": seconds,
                "frames": frames,
                "nodes": len(api),
            },
        )

        started = time.monotonic()
        expected_wall = max(1.0, seconds * settings.ltx_hd_expected_wall_per_output_second)

        async def tick(elapsed: float) -> None:
            fraction = min(1.0, elapsed / (expected_wall * 1.2))
            await reporter.generating(
                GENERATE_FROM + int((GENERATE_TO - GENERATE_FROM - 1) * fraction),
                "Generating your video…",
            )

        await reporter.generating(GENERATE_FROM, "Starting your video…")
        output = job.workspace / "output.mp4"
        try:
            prompt_id = await service.generate(api, client_id=f"zolex-{job.job_id}")
            timeout = settings.ltx_comfy_generation_timeout
            remaining = job.seconds_remaining
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
        try:
            info = await verify_output(
                output,
                OutputExpectation(
                    expect_video=True,
                    expect_audio=True,
                    expected_seconds=frames / settings.ltx_comfy_frame_rate,
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "The finished video failed its check.", internal_detail=str(exc)
            ) from exc

        logger.info(
            "ltx_hd_finished",
            extra={
                "job_id": job.job_id,
                "prompt_id": prompt_id,
                "wall_seconds": round(wall, 1),
                "duration_seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
            },
        )
        await reporter.uploading()
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=info.duration_seconds,
            width=info.width,
            height=info.height,
        )

    # ── Readings ─────────────────────────────────────────────────────────

    @staticmethod
    def _seconds(job: AdapterJob) -> float:
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None:
            raise AdapterError(
                "Please choose a video length.",
                internal_detail=f"no usable duration in {job.parameters.get('duration')!r}",
                retriable=False,
            )
        ceiling = float(settings.ltx_hd_max_seconds)
        if seconds > ceiling + 1e-6:
            raise AdapterError(
                f"This tool makes videos up to {int(ceiling)} seconds long.",
                internal_detail=f"{seconds}s requested; ceiling is {ceiling}s",
                retriable=False,
            )
        return seconds

    @staticmethod
    def _seed(job: AdapterJob) -> int:
        raw = job.parameters.get("seed")
        try:
            if raw is not None and str(raw).strip() != "":
                return abs(int(raw)) % (2**48)
        except (TypeError, ValueError):
            pass
        return abs(hash(job.job_id)) % (2**48)

    async def _placeholder(self, job: AdapterJob) -> str:
        """A small grey PNG for the graph's unused image slot.

        `condition_on_image` is false for this tool, so the picture never
        reaches a sampler — but ComfyUI validates `LoadImage` before it runs
        anything and refuses an empty filename.
        """
        local = job.workspace / "zolex_placeholder.png"
        if not local.exists():
            try:
                await cancellable(
                    job,
                    ffmpeg(
                        ["-f", "lavfi", "-i", "color=c=0x808080:s=64x64:d=0.1",
                         "-frames:v", "1", str(local), "-y"]
                    ),
                )
            except FfmpegError as exc:
                raise AdapterError(
                    "This tool is temporarily unavailable.",
                    internal_detail=f"placeholder could not be made: {exc}",
                ) from exc
        try:
            return await self.service().upload(local, name=f"zolex_{job.job_id}_placeholder.png")
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc


__all__ = ["LtxHdAdapter", "WORKFLOW_ID", "frames_for"]
