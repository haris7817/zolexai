"""Text to Video HD — the client's own FAST 1080 workflow (7 Sep 2026).

A fourth graph the client sent after the pack: NVFP4 transformer, no LoRAs,
no detailer, one 8-step stage, generated at 1920x1088 and delivered at
1920x1080 with its own soundtrack. Benchmarked at 8, 15 and 30 s
(`docs/internal/ltx25-highres-benchmark.md`) and byte-identical through this
path against a direct ComfyUI run at every length.

**Its own adapter, and in client-test the engine behind Text to Video.**
It shares only the ComfyUI service object with `ltx_comfy`. It first shipped
as a separate "Text to Video HD" tool; the client's ask (7 Sep 2026) was for
their graph to BE Text to Video, so the deploy overlay now routes
`text-to-video` here in client-test and the HD tool is hidden. Image to Video
and Extend Video stay on `ltx_comfy`, untouched.

What the graph cannot serve, it refuses rather than approximates: it pins a
16:9 canvas, so a 9:16 or 1:1 request gets a clear error before any GPU time
rather than a landscape video it did not ask for.

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
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.dialogue import add_auto_dialogue
from worker.longform import GENERATE_FROM, GENERATE_TO, StageReporter
from worker.media import FfmpegError, OutputExpectation, ffmpeg, verify_output
from worker.providers.ltx_comfy import LtxComfyService

logger = get_logger(__name__)

WORKFLOW_ID = "text-to-video-hd"

#: What this graph renders. Text to Video itself in client-test — the client
#: asked for their FAST 1080 graph to BE Text to Video, not sit beside it —
#: and the HD id, kept so the definition, its tests and anything in flight
#: keep a stable name.
SUPPORTED = frozenset({"text-to-video", WORKFLOW_ID})


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
        return workflow_id in SUPPORTED

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing()

        seconds = self._seconds(job)
        self._require_landscape(job)
        frames = frames_for(seconds, settings.ltx_comfy_frame_rate)
        # Spoken lines, when this deployment asks for them and the prompt has
        # none. This graph writes its own soundtrack in one pass, which is
        # exactly the shape generated speech is honest in.
        # No soundscape clause runs on this path — the graph is driven from
        # the job's own text — so the anti-repeat rule comes with the lines.
        job = await add_auto_dialogue(job, seconds, carries_soundscape_clause=False)
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
                    negative=self._negative(job),
                    seconds=seconds,
                    seed=self._seed(job),
                    filename_prefix=f"zolexai/{job.job_id}/output",
                    image=image,
                    condition_on_image=False,
                    canvas=self._canvas(job),
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
                "canvas": self._canvas(job) or "native",
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

    @staticmethod
    def _canvas(job: AdapterJob) -> tuple[int, int] | None:
        """The generation canvas: a job's `execution.canvas`, else the
        deployment's `ltx_hd_canvas`, else the graph's own. "native" and an
        empty value both mean the graph's own; "1280x736" means that."""
        raw = str(job.execution.get("canvas") or settings.ltx_hd_canvas or "native").strip().lower()
        if raw in ("", "native"):
            return None
        try:
            width, height = (int(part) for part in raw.split("x", 1))
        except ValueError as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"ltx_hd canvas {raw!r} is not WxH",
                retriable=False,
            ) from exc
        return width, height

    @staticmethod
    def _negative(job: AdapterJob) -> str | None:
        """The client's own negative, unless a deployment overrides it.

        None keeps what they wrote into the graph — a long, specific list this
        adapter has no business replacing. Until 7 Sep 2026 it did:
        `negative_for` had no entry for this workflow and fell back to the
        first/last-frame negative, silently. The claim that the graph ran as
        delivered was wrong for that one widget. It is not any more.
        """
        raw = job.execution.get("negative_prompt")
        return raw.strip() if isinstance(raw, str) and raw.strip() else None

    @staticmethod
    def _require_landscape(job: AdapterJob) -> None:
        """The graph pins its canvas at 1920x1088, delivered as 1920x1080.

        Text to Video also offers 9:16 and 1:1. Rendering those as 16:9 would
        be a silently wrong result; changing the graph's canvas and its crop
        would be redesigning the client's workflow. The honest answer is a
        clear refusal before any GPU time is spent.
        """
        ratio = str(job.parameters.get("aspect_ratio") or "16:9").strip()
        if ratio != "16:9":
            raise AdapterError(
                "This workflow renders 16:9 (landscape) only. Please choose 16:9.",
                internal_detail=(
                    f"aspect {ratio!r} requested; the FAST 1080 graph is pinned to 16:9"
                ),
                retriable=False,
            )

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


__all__ = ["SUPPORTED", "LtxHdAdapter", "WORKFLOW_ID", "frames_for"]
