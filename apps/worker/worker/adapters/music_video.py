"""Music Video on the client's music-video worker (v1.8.0, 8 Sep 2026).

The client delivered a complete orchestrator for this tool — song analysis,
timestamped lyric transcription, up to five performers with reference
pictures and instrument roles, a shot plan of 2–5 second scenes cut on the
music, an anchor still per shot, an audio-conditioned LTX pass per shot,
assembly, one 4K finishing pass and delivery QA — as an installable package.
It is vendored verbatim (`zolex_music_worker/`, beside `worker/`) and never
edited; this adapter is the platform's side of the seam.

**Its own runtime, beside the CLI adapter, not inside it.** `music-video`
still resolves to `ltx` wherever the deployment says so — that path is the
rollback and is untouched. The client-test overlay routes the workflow here
instead. Nothing in this module can reach any other workflow.

What happens per job:

  1. the runner has staged the song and any `performer_N` pictures;
  2. `build_request` turns the job into the package's request JSON and
     `build_config` turns this node's settings into its `WorkerConfig`
     (worker/musicvideo/);
  3. the package's `run_job` runs in a worker thread — it is synchronous,
     supervises its own subprocesses and writes `status.json` as it goes —
     while this adapter polls that file for the customer's progress bar and
     drops the package's `CANCEL` marker if the job is cancelled;
  4. the delivered 4K master is verified and handed back.

Everything the package leaves to the deployer runs on this node: the
official `ltx_pipelines.a2vid_two_stage` for each shot (the same audio
tier the platform's own Music Video used), Qwen-Image-Edit on ComfyUI for
anchor stills (`scripts/mv_anchor.py`), faster-whisper for lyrics, FFmpeg
CUDA for the 4K finish.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    JobCancelled,
    ProgressCallback,
)
from worker.comfy.client import evict_comfy_vram
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import StageReporter
from worker.media import FfmpegError, OutputExpectation, verify_output
from worker.musicvideo import (
    build_config,
    build_request,
    ensure_vendored_package,
    prepare_whisper_libraries,
    progress_for,
)

logger = get_logger(__name__)

WORKFLOW_ID = "music-video"
SUPPORTED = frozenset({WORKFLOW_ID})

#: How often the package's status file is read for the progress bar.
_POLL_SECONDS = 2.0


class MusicVideoAdapter:
    name = "music_video"

    def supports(self, workflow_id: str) -> bool:
        return workflow_id in SUPPORTED

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing()

        audio = job.input_for("source_audio")
        if audio is None:
            raise AdapterError(
                "Please add an audio track.",
                internal_detail="no source_audio input on the job",
                retriable=False,
            )
        audio_path = audio.require_path()

        ensure_vendored_package()
        try:
            from zolex_music_worker.errors import (
                ExternalCommandError,
                QualityError,
                ValidationError,
                WorkerError,
            )
            from zolex_music_worker.job_state import CancelledError
            from zolex_music_worker.models import MusicVideoRequest
            from zolex_music_worker.worker import run_job
        except ImportError as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"the music-video package or its dependencies are missing: {exc}",
                retriable=False,
            ) from exc

        work_root = job.workspace / "music-video"
        payload = build_request(job, audio_path, lyric_mode=settings.music_video_lyric_mode)
        try:
            request = MusicVideoRequest.from_dict(payload)
            config = build_config(
                job,
                work_root=work_root,
                settings=settings,
                anchor_command=self._script_command("mv_anchor.py"),
                render_command=self._script_command("mv_render.py"),
                ltx_python=self._ltx_python(),
                ltx_models_root=settings.ltx_models_root,
            )
            config.validate()
        except (ValidationError, ValueError) as exc:
            raise AdapterError(
                "Some settings are not valid for this tool.",
                internal_detail=f"music-video request rejected: {exc}",
                retriable=False,
            ) from exc

        aspect = request.formats[0]
        logger.info(
            "music_video_starting",
            extra={
                "job_id": job.job_id,
                "aspect": aspect,
                "performers": len(request.performers),
                "with_pictures": sum(1 for p in request.performers if p.reference_images),
                "lyric_mode": request.lyric_mode,
                "lyrics_supplied": bool(request.lyrics),
                "render_backend": config.render_backend,
                "anchor_backend": config.anchor_backend,
                "steps": config.ltx_num_inference_steps,
            },
        )

        # The render and anchor commands are subprocesses that inherit this
        # environment: the allocator setting is what lets the unquantized
        # audio tier survive its VAE-decode spike (27 Aug 2026), and a
        # per-job step count reaches the render command the same way the
        # node's own does — as the setting's environment variable.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["MUSIC_VIDEO_INFERENCE_STEPS"] = str(config.ltx_num_inference_steps)
        if config.transcription_backend == "faster_whisper":
            prepare_whisper_libraries()
        # ComfyUI keeps the last graph's models warm; the anchor stage needs
        # its own model there and the render stage needs the card.
        await evict_comfy_vram()

        await reporter.probing("Listening to your track…")
        started = time.monotonic()
        status_path = work_root / job.job_id / "status.json"
        cancel_path = work_root / job.job_id / "CANCEL"

        task = asyncio.ensure_future(asyncio.to_thread(run_job, request, config))
        cancelled = False
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=_POLL_SECONDS)
                if job.is_cancelled and not cancelled:
                    cancelled = True
                    cancel_path.parent.mkdir(parents=True, exist_ok=True)
                    cancel_path.write_text("cancel requested\n", encoding="utf-8")
                await self._report(reporter, status_path)
            result = task.result()
        except CancelledError as exc:
            raise JobCancelled(str(exc)) from exc
        except ValidationError as exc:
            raise AdapterError(
                self._customer_message(exc),
                internal_detail=f"music-video validation: {exc}",
                retriable=False,
            ) from exc
        except QualityError as exc:
            raise AdapterError(
                "The finished video failed its check.",
                internal_detail=f"music-video quality: {exc}",
            ) from exc
        except ExternalCommandError as exc:
            raise AdapterError(
                "Generation failed. Please try again.",
                internal_detail=f"music-video command: {exc}",
            ) from exc
        except WorkerError as exc:
            raise AdapterError(
                "Generation failed. Please try again.",
                internal_detail=f"music-video: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - the package's own libraries can raise anything
            raise AdapterError(
                "Generation failed. Please try again.",
                internal_detail=f"music-video {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if not task.done():
                task.cancel()
            if settings.ltx_comfy_free_after_job:
                await evict_comfy_vram()

        if job.is_cancelled:
            raise JobCancelled(f"job {job.job_id} cancelled")

        delivery = result.get("formats", {}).get(aspect) or {}
        output = Path(str(delivery.get("output") or ""))
        if not output.is_file():
            raise AdapterError(
                "The finished video could not be found.",
                internal_detail=f"music-video result carries no output for {aspect}: {result}",
            )

        expected_seconds = float(delivery.get("seconds") or 0.0) or None
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
                "The finished video failed its check.", internal_detail=str(exc)
            ) from exc

        wall = time.monotonic() - started
        logger.info(
            "music_video_finished",
            extra={
                "job_id": job.job_id,
                "wall_seconds": round(wall, 1),
                "duration_seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
                "shots": delivery.get("shots"),
                "render_resolution": delivery.get("render_resolution"),
                "lyric_driven": delivery.get("lyric_driven"),
                "lipsync_mode": delivery.get("singing_lipsync_mode"),
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

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    async def _report(reporter: StageReporter, status_path: Path) -> None:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        step = progress_for(status)
        if step is None:
            return
        state, progress, message, details = step
        await reporter.report(state, progress, message, details)  # type: ignore[arg-type]

    @staticmethod
    def _customer_message(exc: Exception) -> str:
        text = str(exc)
        if "No usable lyrics" in text or "transcription is disabled" in text:
            return (
                "We could not make out the lyrics in this track. "
                "Paste the lyrics, or describe the video without asking to follow them."
            )
        if "Decoded audio is" in text or "limit is" in text:
            return "This tool makes videos for songs up to 5 minutes long."
        if "at least one second" in text:
            return "The audio track is too short."
        return "Some settings are not valid for this tool."

    @staticmethod
    def _script_command(name: str) -> list[str]:
        """A script shipped beside this package, run by the worker's own
        interpreter, in the package's placeholder shape."""
        script = Path(__file__).resolve().parents[2] / "scripts" / name
        return [sys.executable, str(script), "--request", "{request_json}", "--output", "{output}"]

    @staticmethod
    def _ltx_python() -> str:
        """The LTX environment's interpreter — the same one the CLI adapter
        reaches through `uv run`, addressed directly so the package's own
        render adapter can launch it."""
        candidate = settings.ltx_repo_dir / ".venv" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        return "python"


__all__ = ["SUPPORTED", "WORKFLOW_ID", "MusicVideoAdapter"]
