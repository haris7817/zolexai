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
    frame and the prompt says how it moves. Video extension will reuse this
    exact seam later: extract a source's final frame, condition on it,
    stitch. Video/audio-conditioned modes stay refused until then.

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
import zlib
from collections import deque
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    parse_duration_seconds,
)
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media import FfmpegError, ffmpeg, probe_media, verify_duration

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
        return workflow_id in {"text-to-video", "image-to-video"}

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        await on_progress("preparing", 10, "Setting up your generation…")

        self._require_supported_shape(job)
        self._require_models()
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
    ) -> list[str]:
        root = settings.ltx_models_root
        width, height = _DIMENSIONS.get(
            str(job.parameters.get("aspect_ratio") or ""), _DEFAULT_DIMENSIONS
        )
        frames = max(1, round(seconds * settings.ltx_frame_rate))
        # The pipeline's default seed is fixed, which would hand two users with
        # the same prompt the same video. CRC of the job id: deterministic per
        # job (reruns of a retried job reproduce), distinct across jobs.
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
        self, cmd: list[str], job: AdapterJob, on_progress: ProgressCallback
    ) -> None:
        """Runs the pipeline, streaming its output for progress and diagnostics.

        The contract with the rest of the platform:

          * `job.raise_if_cancelled()` is honoured within `_CANCEL_POLL_SECONDS`
            even while the pipeline is silent — cancellation and timeout both
            surface here as exceptions.
          * The child is killed on *every* non-completion path (the `finally`),
            because an orphaned render holds VRAM, and the runner is about to
            delete the workspace the child is writing into.
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
                    _, progress, message = _MARKERS[matched]
                    marker_from = matched + 1
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
