"""LTX-2.5 runtime — real GPU generation, staged rollout.

## How it runs

The worker process never imports torch. It shells out to the LTX repository's
own `uv` environment (`settings.ltx_repo_dir`) exactly the way the benchmark
did, which keeps CUDA, model weights and their 40-GB dependency tree entirely
outside this codebase. The subprocess is supervised the same way the media
tools are: killed on cancellation, on timeout, and on any exception path, so a
dead job can never leave a render burning VRAM.

## What is enabled today, and why

Text-to-video and image-to-video, 30 seconds max, NVFP4 quantization. Every
one of those limits is a measurement, not a guess (RTX 5090, 2026-08-12):

  * NVFP4 works end to end; BF16's transformer alone is ~40 GB against 32 GB
    of VRAM and cannot load.
  * 30s completes; 60s hard-OOMs at 29.6/31.4 GiB mid-denoise.
  * The `distilled` entry point cannot emit audio-only output (it tries to
    attach an H.264 stream to an MP3 container and dies), so music stays on
    its current runtime until the audio pipeline is wired separately.
  * Image conditioning uses the pipeline's `--image PATH FRAME_IDX STRENGTH`
    input pinned at frame 0, full strength — the still becomes the first
    frame and the prompt says how it moves.
  * Video extension is that same seam driven in a loop: the source's final
    frame conditions the first continuation, each further segment chains off
    the previous segment's final frame (`plan_segments` keeps every pass
    inside the measured ceiling, so a 60s extension is two 30s renders), and
    the media layer normalizes and stitches source + continuations into one
    file at the source's own resolution. The generation grid follows the
    SOURCE's aspect, not the request's — the I2V benchmark showed a
    mismatched aspect makes the model keep the style and replace the
    subject, which for an extension means a different video after the seam.
    Video-to-video (restyling) stays refused until it is wired separately.

Each refusal is `retriable=False` with the real reason in `internal_detail` —
a mis-routed job should fail once with a clear log line, not burn three
attempts.

## Testability

Everything except the model itself is provable without a GPU: `_command()` is
pure, `_execute()` accepts any argv (tests substitute a stub script that writes
a real MP4), and progress parsing is a pure function over output lines. The
only thing the GPU-node run adds is the model.

No shipped workflow points at `runtime: ltx` yet; routing it is a YAML change
made deliberately, after the GPU-side smoke test passes.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import zlib
from collections import deque
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    JobCancelled,
    ProgressCallback,
    parse_duration_seconds,
)
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media import (
    FfmpegError,
    concat_segments,
    extract_final_frame,
    ffmpeg,
    normalize_clip,
    plan_segments,
    probe_media,
    verify_duration,
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
#: rejected outright in benchmarking. These are the closest /64 grids to the
#: product's aspect ratios; 896x512 is the exact configuration measured on the
#: RTX 5090, the others scale within the same pixel budget.
_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (896, 512),
    "9:16": (512, 896),
    "1:1": (640, 640),
    "4:5": (512, 640),
}
_DEFAULT_DIMENSIONS = (896, 512)

#: The largest frame the RTX 5090 benchmark proved (896x512). Grids chosen for
#: arbitrary source aspects must stay within it or a 30s pass OOMs again.
_PIXEL_BUDGET = 896 * 512

#: Extension output is delivered at the source's own resolution (a user's
#: 1080p clip must not come back as 512p), capped at full HD — beyond that the
#: normalization re-encode cost stops being worth invisible extra pixels.
_MAX_OUTPUT_LONG_SIDE = 1920
_MAX_OUTPUT_SHORT_SIDE = 1080


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
        for h in range(256, 897, 64)
        for w in range(256, 897, 64)
        if w * h <= _PIXEL_BUDGET
    ]
    # Within a small aspect error the crop is invisible and more pixels win —
    # otherwise 576x320 (1.80) would beat 896x512 (1.75) for a 16:9 source on
    # a 1% aspect technicality while halving the frame.
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

#: Progress stays inside the 15–85 `generating` band for the same reason the
#: harness's does: the API ranks statuses strictly forward, so hopping out of
#: `generating` per stage would be an illegal transition.
_GENERATE_FROM = 15
_GENERATE_TO = 85

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

    def supports(self, workflow_id: str) -> bool:
        return workflow_id in {"text-to-video", "image-to-video", "extend-video"}

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        await on_progress("preparing", 10, "Setting up your generation…")
        self._require_models()

        # Extension is its own orchestration (probe, chain, stitch) built from
        # the same seams — dispatched on the workflow, not on input presence,
        # so a mis-routed video-to-video job cannot be quietly "extended".
        if job.workflow_id == "extend-video":
            return await self._run_extension(job, on_progress)

        self._require_supported_shape(job)
        seconds = self._target_seconds(job)
        conditioning_image = await self._conditioning_image(job)

        output = job.workspace / "output.mp4"
        await self._execute(
            self._command(job, seconds, output, conditioning_image=conditioning_image),
            job,
            on_progress,
        )

        await on_progress("post_processing", 90, "Polishing and encoding…")
        try:
            measured = await verify_duration(output, expected_seconds=seconds)
            info = await probe_media(output)
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"output failed verification: {exc}",
            ) from exc

        await on_progress("uploading", 95, "Almost ready…")
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=measured,
            width=info.width,
            height=info.height,
        )

    # ── Guardrails ───────────────────────────────────────────────────────

    def _require_supported_shape(self, job: AdapterJob) -> None:
        """Refuses job shapes this runtime cannot honestly produce yet.

        These are routing mistakes, and the failure mode to avoid is the quiet
        one: an audio job that produces a broken file three attempts later, or
        a video-to-video job that ignores its source and returns unrelated
        text-to-video footage.
        """
        if job.execution.get("output_kind") == "audio":
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    "audio-only output is not supported by the LTX distilled entry "
                    "point (benchmarked 2026-08-12: it attaches libx264 to an mp3 "
                    "container and fails); music must not route to `ltx` yet"
                ),
                retriable=False,
            )
        unsupported = [item.role for item in job.inputs if item.role != "source_image"]
        if unsupported:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"inputs {unsupported} are not wired to the LTX runtime yet; only "
                    "text-to-video and image-to-video (source_image) are enabled"
                ),
                retriable=False,
            )

    async def _conditioning_image(self, job: AdapterJob) -> Path | None:
        """The staged still that becomes the first frame, verified decodable.

        This decodes one frame rather than probing: ffprobe's metadata pass
        accepts garbage (the `tty` demuxer will even claim ASCII text as
        "video"), while an actual decode rejects a truncated or mislabelled
        upload in milliseconds. That turns "the GPU burned a minute and
        produced garbage" into "that upload is not a readable image" — the
        answer the user can act on. A corrupt file is corrupt on every retry,
        hence `retriable=False`.

        Video extension will feed this same conditioning path a frame it
        extracted itself, which is why the command seam takes a plain Path
        rather than an AdapterInput.
        """
        item = job.input_for("source_image")
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
                internal_detail=f"decode check of source_image failed: {exc}",
                retriable=False,
            ) from exc
        return staged

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

    def _target_seconds(self, job: AdapterJob) -> float:
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"no usable duration in {job.parameters!r}",
                retriable=False,
            )
        limit = float(job.execution_int("max_segment_seconds", settings.ltx_max_seconds))
        if seconds > limit:
            raise AdapterError(
                "This length is not available for this tool yet.",
                internal_detail=(
                    f"requested {seconds}s exceeds the single-pass ceiling of {limit}s "
                    "measured on the RTX 5090; the segmentation layer handles longer "
                    "requests and is not enabled for this runtime yet"
                ),
                retriable=False,
            )
        return seconds

    # ── Video extension ──────────────────────────────────────────────────

    async def _run_extension(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        """source → final frame → continuation segment(s) → normalize → stitch.

        Every pass stays inside the measured per-pass ceiling; `plan_segments`
        decides how many passes a request needs (5–30s: one; 60s: two), and
        each pass after the first chains off the previous segment's final
        frame. The finished file is the untouched-in-content source plus the
        continuation, both normalized to one set of encoder parameters at the
        source's own (capped) resolution.
        """
        item = job.input_for("source_video")
        if item is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail="extend-video job arrived without a source_video input",
                retriable=False,
            )
        staged = item.require_path()

        try:
            source = await probe_media(staged)
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"probe of source_video failed: {exc}",
                retriable=False,
            ) from exc
        if not source.has_video or not source.duration_seconds:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"source_video is not usable video: {source}",
                retriable=False,
            )

        extension_seconds = parse_duration_seconds(job.parameters.get("duration"))
        if extension_seconds is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"no usable duration in {job.parameters!r}",
                retriable=False,
            )

        per_pass = float(job.execution_int("max_segment_seconds", settings.ltx_max_seconds))
        segments = plan_segments(extension_seconds, max_segment_seconds=per_pass)
        grid = grid_for_source(source.width, source.height)

        try:
            conditioning = await self._cancellable(
                job,
                extract_final_frame(staged, job.workspace / "condition-0000.png"),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"final-frame extraction from source failed: {exc}",
                retriable=False,
            ) from exc

        # ── Continuation segments (all inside the `generating` band) ─────
        total = len(segments)
        span = _GENERATE_TO - _GENERATE_FROM
        rendered: list[Path] = []
        for segment in segments:
            job.raise_if_cancelled()
            part = job.workspace / f"continuation-{segment.index:04d}.mp4"
            band = (
                _GENERATE_FROM + span * segment.index // total,
                _GENERATE_FROM + span * (segment.index + 1) // total,
            )
            command = self._command(
                job,
                segment.duration_seconds,
                part,
                conditioning_image=conditioning,
                dimensions=grid,
                # Distinct per segment or every chained pass replays the same
                # noise; still deterministic so a retried job reproduces.
                seed=zlib.crc32(f"{job.job_id}:{segment.index}".encode()),
            )
            await self._execute(
                job=job,
                cmd=command,
                on_progress=on_progress,
                band=band,
                section=(segment.index + 1, total) if total > 1 else None,
            )
            rendered.append(part)

            if segment.index + 1 < total:
                try:
                    conditioning = await self._cancellable(
                        job,
                        extract_final_frame(
                            part, job.workspace / f"condition-{segment.index + 1:04d}.png"
                        ),
                    )
                except FfmpegError as exc:
                    # The GENERATED segment being unreadable is a generation
                    # flake, not a bad upload — retrying can genuinely help.
                    raise AdapterError(
                        "This generation could not be completed. Please try again.",
                        internal_detail=f"segment {segment.index} unreadable: {exc}",
                    ) from exc

        # ── Assembly: normalize both parts, then a deterministic concat ──
        await on_progress("post_processing", 88, "Stitching your video…")
        out_width, out_height = output_dimensions(source.width, source.height)
        out_fps = min(60.0, max(10.0, source.fps or float(settings.ltx_frame_rate)))
        keep_audio = source.has_audio

        output = job.workspace / "output.mp4"
        try:
            continuation = await self._cancellable(
                job, concat_segments(rendered, job.workspace / "continuation.mp4")
            )
            parts = []
            for index, clip in enumerate((staged, continuation)):
                job.raise_if_cancelled()
                parts.append(
                    await self._cancellable(
                        job,
                        normalize_clip(
                            clip,
                            job.workspace / f"part-{index:04d}.mp4",
                            width=out_width,
                            height=out_height,
                            fps=out_fps,
                            audio=keep_audio,
                        ),
                    )
                )
            await self._cancellable(job, concat_segments(parts, output))
            expected = source.duration_seconds + extension_seconds
            measured = await verify_duration(
                output,
                expected_seconds=expected,
                # Each re-timed part can drift a little; scale with length
                # instead of failing honest 60s extensions on frame rounding.
                tolerance_seconds=max(1.5, 0.03 * expected),
            )
            info = await probe_media(output)
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"extension assembly failed: {exc}",
            ) from exc

        await on_progress("uploading", 95, "Almost ready…")
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=measured,
            width=info.width,
            height=info.height,
        )

    async def _cancellable(self, job: AdapterJob, operation: Coroutine[Any, Any, Path]) -> Path:
        """Awaits a media operation, abandoning it the moment the job dies.

        The ffmpeg helper kills its child when its task is cancelled, so
        racing the operation against the runner's cancel event means a
        cancelled job stops a long re-encode within milliseconds instead of
        at its end. Without an event (tests, tooling) this is a plain await.
        """
        event = job.cancellation_event
        if event is None:
            return await operation
        op = asyncio.ensure_future(operation)
        watcher = asyncio.ensure_future(event.wait())
        try:
            done, _ = await asyncio.wait({op, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if op in done:
                return op.result()
            op.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await op
            job.raise_if_cancelled()
            raise JobCancelled(f"job {job.job_id} cancelled")
        finally:
            watcher.cancel()

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
        conditioning_image: Path | None = None,
        dimensions: tuple[int, int] | None = None,
        seed: int | None = None,
    ) -> list[str]:
        root = settings.ltx_models_root
        # Explicit dimensions (extension: the source's aspect decides) beat
        # the requested aspect ratio's lookup.
        width, height = dimensions or _DIMENSIONS.get(
            str(job.parameters.get("aspect_ratio") or ""), _DEFAULT_DIMENSIONS
        )
        frames = max(1, round(seconds * settings.ltx_frame_rate))
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
            "--prompt", job.prompt,
            "--num-frames", str(frames),
            "--height", str(height),
            "--width", str(width),
            "--frame-rate", str(settings.ltx_frame_rate),
            "--seed", str(seed),
            "--output-path", str(output),
        ]
        if conditioning_image is not None:
            # `--image PATH FRAME_IDX STRENGTH`: pin the still as frame 0 at
            # full strength — the image is the shot's first frame and the
            # prompt describes its motion. Extension will pass an extracted
            # final frame through this same argument.
            cmd += ["--image", str(conditioning_image), "0", "1.0"]
        return cmd

    # ── Supervision ──────────────────────────────────────────────────────

    async def _execute(
        self,
        cmd: list[str],
        job: AdapterJob,
        on_progress: ProgressCallback,
        *,
        band: tuple[int, int] = (_GENERATE_FROM, _GENERATE_TO),
        section: tuple[int, int] | None = None,
    ) -> None:
        """Runs the pipeline, streaming its output for progress and diagnostics.

        The contract with the rest of the platform:

          * `job.raise_if_cancelled()` is honoured within `_CANCEL_POLL_SECONDS`
            even while the pipeline is silent — cancellation and timeout both
            surface here as exceptions.
          * The child is killed on *every* non-completion path (the `finally`),
            because an orphaned render holds VRAM, and the runner is about to
            delete the workspace the child is writing into.

        `band` compresses the markers' 15–85 sweep into a slice of it, so N
        chained segments produce one monotonic ramp instead of N restarts.
        `section` swaps the stage messages for "Generating section i of N…" —
        the machinery is only named when there are several (harness rule).
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
                if matched is not None:
                    _, nominal, message = _MARKERS[matched]
                    marker_from = matched + 1
                    low, high = band
                    progress = low + (nominal - _GENERATE_FROM) * (high - low) // (
                        _GENERATE_TO - _GENERATE_FROM
                    )
                    if section is not None:
                        message = f"Generating section {section[0]} of {section[1]}…"
                    await on_progress("generating", progress, message)

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
