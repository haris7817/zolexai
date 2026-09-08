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

The graph has no aspect selector — it is written for 1920x1088 generated and
1920x1080 delivered — so an orientation is those same two size widgets turned
on their side (`ASPECTS`, client request 8 Sep 2026): portrait generates
1088x1920 and delivers 1080x1920, and the 8 spare pixels come off the width
instead of the height, which is the graph's own crop, mirrored. 16:9 leaves
both widgets alone, so the landscape render is still the workflow exactly as
delivered. A ratio with no mapping is refused before any GPU time rather than
returned as a video the customer did not ask for.

The graph exposes exactly what a job needs — the two prompt boxes, a duration
in seconds from which it computes its own frame count, a seed, the output
location, and an optional conditioning picture — and the compiler
(`compile_fast_1080`) sets those and nothing else.
"""

from __future__ import annotations

import time
from pathlib import Path

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
from worker.comfy.seedvr2 import compile_upscale
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


#: What each offered aspect ratio means in this graph's two size widgets:
#: (generation canvas, delivered size). The graph has no aspect selector — it
#: is written for 1920x1088 generated, 1920x1080 delivered — so an
#: orientation is those same two numbers turned on their side (client
#: request, 8 Sep 2026). Generation sides stay on the model's 32 grid; the
#: delivery drops the 8 spare pixels exactly as the graph already does for
#: 16:9, on whichever side carries them.
#:
#: 16:9 maps to (None, None) on purpose: the graph's own widgets are left
#: untouched, so the landscape render stays bit-for-bit the workflow as
#: delivered and every earlier measurement still describes it.
ASPECTS: dict[str, tuple[tuple[int, int] | None, tuple[int, int] | None]] = {
    "16:9": (None, None),
    "9:16": ((1088, 1920), (1080, 1920)),
    "1:1": ((1088, 1088), (1080, 1080)),
}

#: The "720p" generation canvas per ratio — the client's speed plan (8 Sep
#: 2026): "generate at the LTX-compatible 720p size, 1280x704 for landscape
#: or 704x1280 for vertical, then upscale to 1080p on the GPU keeping the
#: same audio". The upscale is the graph's own closing `ImageScale` (lanczos,
#: crop=center), so it runs on the GPU inside ComfyUI and the soundtrack —
#: which never passes through that node — is untouched.
#:
#: 1:1 is not in the client's sentence and cannot be derived by transposing
#: a landscape size, so it gets its own square canvas at the same ~0.9 MP
#: budget. Without it, a square job would generate a 16:9 frame and have its
#: sides cropped off.
#:
#: What this costs is not in dispute and is not small: measured 7 Sep 2026,
#: a 1280-wide generation upscaled to 1080p carries **0.29x the fine detail**
#: of a native 1920x1088 render. See docs/internal/text-to-video-speed.md.
DRAFT_CANVAS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 704),
    "9:16": (704, 1280),
    "1:1": (960, 960),
}

#: The 4K frame per ratio (client request, 8 Sep 2026), reached from the
#: generated frame by one lanczos resize in ffmpeg — see `_upscale_4k`.
DELIVERY_4K: dict[str, tuple[int, int]] = {
    "16:9": (3840, 2160),
    "9:16": (2160, 3840),
    "1:1": (2160, 2160),
}


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
        aspect = self._aspect(job)
        ai_upscale = self._ai_upscale(job)
        four_k = self._delivery_tier(job) == "4k"
        canvas = self._canvas(job, aspect)
        # Delivered size: the ratio's 1080p frame. With the AI upscaler on,
        # the graph's own lanczos node is parked at the canvas size (an
        # identity resize) and SeedVR2 does the enlarging afterwards. With 4K
        # on, the same parking, and ffmpeg does the enlarging afterwards —
        # once, from the generated frame, never from an already-resized one.
        delivered = ASPECTS[aspect][1] or (1920, 1080)
        delivery = canvas if ((ai_upscale or four_k) and canvas) else ASPECTS[aspect][1]
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
                    canvas=canvas,
                    delivery=delivery,
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
                "aspect": aspect,
                "canvas": canvas or "native",
                "ai_upscale": ai_upscale,
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

        if ai_upscale and canvas:
            await reporter.generating(GENERATE_TO - 1, "Upscaling your video…")
            output = await self._upscale(job, service, output, canvas, delivered)
        if four_k:
            await reporter.generating(GENERATE_TO - 1, "Upscaling your video to 4K…")
            output = await self._upscale_4k(job, output, DELIVERY_4K[aspect])

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
    def _canvas(job: AdapterJob, aspect: str = "16:9") -> tuple[int, int] | None:
        """The generation canvas: a job's `execution.canvas`, else the
        deployment's `ltx_hd_canvas`, else what the ratio asks for.

        "native" and an empty value both mean "whatever this ratio needs",
        which for 16:9 is the graph's own widget and so None. **"720p"** is
        the client's speed plan — `DRAFT_CANVAS` for this ratio, upscaled to
        1080p by the graph's own closing node. An explicit "1280x736" means
        that size.

        An override is a SIZE lever (the 7 Sep speed work), not an
        orientation choice, and a deployment sets one string for every job.
        So a landscape override on a portrait request is transposed rather
        than sent as a frame the graph would centre-crop down its sides."""
        raw = str(job.execution.get("canvas") or settings.ltx_hd_canvas or "native").strip().lower()
        wanted = ASPECTS[aspect][0]
        if raw in ("", "native"):
            return wanted
        if raw in ("720p", "draft"):
            return DRAFT_CANVAS[aspect]
        try:
            width, height = (int(part) for part in raw.split("x", 1))
        except ValueError as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"ltx_hd canvas {raw!r} is not WxH",
                retriable=False,
            ) from exc
        if wanted is not None and wanted[1] > wanted[0] and width > height:
            width, height = height, width
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
    def _aspect(job: AdapterJob) -> str:
        """Which offered ratio this job wants, refusing any this graph has no
        size mapping for, before any GPU time is spent.

        Until 8 Sep 2026 this refused everything but 16:9, because the graph
        has no aspect selector and rendering portrait as landscape would be a
        silently wrong video. The client asked for portrait, so the two size
        widgets now follow the ratio (`ASPECTS`) — which is the same pair of
        numbers the graph already carries, not a redesign of it.
        """
        ratio = str(job.parameters.get("aspect_ratio") or "16:9").strip()
        if ratio not in ASPECTS:
            raise AdapterError(
                f"This workflow renders {', '.join(ASPECTS)}. Please choose one of those.",
                internal_detail=(
                    f"aspect {ratio!r} has no canvas mapping on the FAST 1080 graph"
                ),
                retriable=False,
            )
        return ratio

    @staticmethod
    def _delivery_tier(job: AdapterJob) -> str:
        """"1080p" (the ratio's frame from `ASPECTS`) or "4k" (`DELIVERY_4K`).

        `execution.delivery` on the job, else the deployment's
        `ltx_hd_delivery`. Client request, 8 Sep 2026: 4K "in the same way we
        do 1920x1080" — a lanczos resize, no model. Anything unrecognised is
        1080p, so a typo cannot silently quadruple every file."""
        raw = str(job.execution.get("delivery") or settings.ltx_hd_delivery or "1080p")
        return "4k" if raw.strip().lower() in ("4k", "2160p", "uhd") else "1080p"

    async def _upscale_4k(self, job: AdapterJob, clip: Path, target: tuple[int, int]) -> Path:
        """Lanczos to 4K with ffmpeg, the way the client's own package does
        1080p (`upscale_to_1080` in `integration_example.py`): scale to cover,
        centre-crop to the exact frame, NVENC, and the soundtrack copied
        through untouched (`-c:a copy`).

        Done here rather than in the graph's `ImageScale` on purpose: a 4K
        frame batch of 721 frames is ~72 GB as a tensor inside ComfyUI plus a
        CPU encode, where ffmpeg streams it and NVENC does it in seconds —
        measured 2.3 s for a 10 s clip (8 Sep 2026). H.264 rather than HEVC
        because browsers play it; the file is ~2.5x larger for that.
        """
        width, height = target
        out = job.workspace / "output_4k.mp4"
        scale = (
            f"scale=w={width}:h={height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height}"
        )
        common = ["-i", str(clip), "-vf", scale, "-pix_fmt", "yuv420p", "-c:a", "copy",
                  "-movflags", "+faststart"]
        try:
            try:
                await cancellable(
                    job,
                    ffmpeg([*common, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", str(out)],
                           timeout=settings.ltx_comfy_transfer_timeout),
                )
            except FfmpegError as exc:
                # No NVENC on this box: the CPU encoder, slower and identical.
                logger.warning(
                    "ltx_hd_4k_nvenc_unavailable",
                    extra={"job_id": job.job_id, "detail": str(exc)[-300:]},
                )
                await cancellable(
                    job,
                    ffmpeg([*common, "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out)],
                           timeout=settings.ltx_comfy_generation_timeout),
                )
        except FfmpegError as exc:
            raise AdapterError(
                "The finished video could not be upscaled to 4K.",
                internal_detail=str(exc)[-600:],
            ) from exc
        logger.info(
            "ltx_hd_upscaled_4k",
            extra={"job_id": job.job_id, "target": f"{width}x{height}"},
        )
        return out

    @staticmethod
    def _ai_upscale(job: AdapterJob) -> bool:
        """Whether the finished clip goes through SeedVR2 on its way to 1080p.

        `execution.upscaler` on the job, else the deployment's
        `ltx_hd_upscaler`: "seedvr2" turns it on; "lanczos" (the graph's own
        closing node) or anything else leaves it off. Only meaningful with a
        generation canvas smaller than the delivery — at native there is
        nothing to enlarge."""
        raw = str(job.execution.get("upscaler") or settings.ltx_hd_upscaler or "lanczos")
        return raw.strip().lower() == "seedvr2"

    async def _upscale(
        self,
        job: AdapterJob,
        service: LtxComfyService,
        clip: Path,
        source: tuple[int, int],
        target: tuple[int, int],
    ) -> Path:
        """SeedVR2 on the finished clip: a second, separate ComfyUI prompt.

        Separate on purpose. The client's graph stays exactly as delivered
        (the upscaler is not spliced into it), the generation can be kept as
        a fallback, and a failure here is reported as an upscale failure
        rather than a generation failure. The soundtrack rides through
        `GetVideoComponents` → `CreateVideo` untouched.
        """
        try:
            name = await service.upload(clip, name=f"zolex_{job.job_id}_720p.mp4")
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc
        api = compile_upscale(
            input_file=name,
            source=source,
            target=target,
            filename_prefix=f"zolexai/{job.job_id}/upscaled",
            seed=self._seed(job),
        )
        upscaled = job.workspace / "output_upscaled.mp4"
        try:
            prompt_id = await service.generate(api, client_id=f"zolex-{job.job_id}-upscale")
            timeout = settings.ltx_comfy_generation_timeout
            remaining = job.seconds_remaining
            if remaining is not None:
                timeout = max(1.0, min(timeout, remaining))
            history = await service.progress(job, prompt_id, timeout_seconds=timeout)
            await service.collect(history, upscaled)
        except ComfyError as exc:
            raise AdapterError(
                "The finished video could not be upscaled.",
                internal_detail=exc.internal_detail,
                retriable=exc.retriable,
            ) from exc
        finally:
            if settings.ltx_comfy_free_after_job:
                await service.free_memory()
        logger.info(
            "ltx_hd_upscaled",
            extra={"job_id": job.job_id, "prompt_id": prompt_id,
                   "source": f"{source[0]}x{source[1]}", "target": f"{target[0]}x{target[1]}"},
        )
        return upscaled

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
