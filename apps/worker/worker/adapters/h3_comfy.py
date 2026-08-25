"""MiniMax H3 through the pinned ComfyUI INT8 stack — the client-test runtime.

This adapter is the production face of the path proven on 25 Aug 2026
(`docs/internal/client-h3-comfyui-results.md`): the client's frozen workflow
graphs, official Comfy-Org INT8 weights, ComfyUI v0.33.3 with two pinned
custom nodes, and the ZolexAI per-segment prompt discipline that held one
subject, one wardrobe and one room across a full 60-second continuation.

Scope, deliberately narrow:

  * `image-to-video`  → the client I2V graph (FL2VA; the customer's image IS
    the first frame, so identity is pinned by pixels, not by reference).
  * `video-to-video` WITH a reference image → the client R2V graph, mapped
    the way the proven D1 run mapped it: reference image → Picture 1
    (identity), first frame of the source video → Picture 2 (environment /
    composition), Picture 3 disconnected. The R2V graph consumes IMAGES —
    the source video's motion is not re-enacted; that remains LTX transform's
    job, and a plain restyle without a reference image is refused here rather
    than half-served.

Everything else — T2V, music video, extend — stays on LTX per the measured
routing. Turbo is not reachable from here at all: rejected on quality.

The ComfyUI service is treated exactly like ACE-Step: a long-lived local
process the worker connects to and never manages. Worker and service share a
filesystem (same GPU node), which is what lets the Final Decode node write the
assembled MP4 straight into the job's own workspace.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    parse_duration_seconds,
)
from worker.comfy import ComfyClient, ComfyError, GraphEdits, load_graph, to_api_prompt
from worker.comfy.graph import (
    DURATION_PRESETS,
    PROMPTS_PER_INDEX,
    duration_index_for,
    nearest_duration_index,
)
from worker.core.config import settings
from worker.longform.h3_prompts import discipline_prompts, plan_from_prompt
from worker.longform.progress import GENERATE_FROM, GENERATE_TO, StageReporter
from worker.media.ffmpeg import ffmpeg
from worker.media.probe import probe_media

logger = logging.getLogger("zolexai.worker.h3_comfy")

#: workflow id → graph filename in the frozen client pack.
_GRAPHS = {
    "image-to-video": "minimax_h3_i2v_extender.json",
    "video-to-video": "minimax_h3_r2v_extender.json",
}

#: Measured wall-clock per second of output on the RTX PRO 6000, used ONLY to
#: pace the progress bar, never to declare completion. 25 Aug measurements:
#: R2V draft 11x, R2V 960x544 33x, I2V 1280x736 ~55x warm.
_EXPECTED_RATE = {
    ("video-to-video", "draft"): 12.0,
    ("video-to-video", "quality"): 34.0,
    ("image-to-video", "quality"): 60.0,
    ("image-to-video", "draft"): 60.0,  # I2V has one proven canvas; same pace
}


class H3ComfyAdapter:
    name = "h3_comfy"

    def __init__(self, client: ComfyClient | None = None) -> None:
        self._client = client

    def supports(self, workflow_id: str) -> bool:
        return workflow_id in _GRAPHS

    # ── Wiring ───────────────────────────────────────────────────────────

    def _service(self) -> ComfyClient:
        if self._client is None:
            self._client = ComfyClient(
                settings.h3_comfy_base_url,
                request_timeout=settings.h3_comfy_request_timeout,
                poll_seconds=settings.h3_comfy_poll_seconds,
            )
        return self._client

    @staticmethod
    def _graph_path(workflow_id: str) -> Path:
        path = settings.h3_comfy_workflows_dir / _GRAPHS[workflow_id]
        if not path.is_file():
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"frozen client graph missing: {path}",
                retriable=False,
            )
        return path

    @staticmethod
    def _tier(job: AdapterJob) -> str:
        tier = str(job.execution.get("h3_tier") or "").strip().lower()
        if tier in ("draft", "quality"):
            return tier
        # The public `quality` parameter, when the product starts sending one.
        if str(job.parameters.get("quality") or "").lower() in ("draft", "fast", "low"):
            return "draft"
        return "quality"

    @staticmethod
    async def _duration_index(job: AdapterJob) -> int:
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None and job.workflow_id == "video-to-video":
            # `duration_mode: source`: the API sends no duration because the
            # source clip's length IS the request. H3 generates preset lengths,
            # so the nearest preset to the source is the honest reading.
            source = job.input_for("source_video")
            if source is not None and source.path is not None:
                info = await probe_media(source.require_path())
                if info.duration_seconds:
                    return nearest_duration_index(info.duration_seconds)
        if seconds is None:
            raise AdapterError(
                "Please choose a video length.",
                internal_detail=f"no usable duration in {job.parameters.get('duration')!r}",
                retriable=False,
            )
        index = duration_index_for(seconds)
        if index is None:
            supported = ", ".join(f"{int(v)}s" for v in DURATION_PRESETS.values())
            raise AdapterError(
                f"This tool supports these lengths: {supported}.",
                internal_detail=f"{seconds}s does not match a client-pack preset",
                retriable=False,
            )
        return index

    # ── Input staging ────────────────────────────────────────────────────

    @staticmethod
    def _input_dir() -> Path:
        directory = settings.h3_comfy_input_dir
        if directory is None:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail="h3_comfy_input_dir is not configured on this node",
                retriable=False,
            )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    async def _stage_image(self, job: AdapterJob, source: AdapterInput, tag: str) -> str:
        """Copies a staged input into ComfyUI's input directory, PNG-encoded.

        LoadImage validates the file at submit time, so the copy happens before
        the prompt is posted. Re-encoding through ffmpeg normalises whatever
        the customer uploaded into something the node reliably reads.
        """
        staged = source.require_path()
        name = f"zolex_{job.job_id}_{tag}.png"
        target = self._input_dir() / name
        await ffmpeg(["-i", str(staged), "-frames:v", "1", str(target), "-y"])
        return name

    async def _stage_video_first_frame(
        self, job: AdapterJob, source: AdapterInput, tag: str
    ) -> str:
        staged = source.require_path()
        name = f"zolex_{job.job_id}_{tag}.png"
        target = self._input_dir() / name
        await ffmpeg(
            ["-i", str(staged), "-vf", "select=eq(n\\,0)", "-frames:v", "1", str(target), "-y"]
        )
        return name

    def _cleanup_staged(self, job: AdapterJob) -> None:
        directory = settings.h3_comfy_input_dir
        if directory is None:
            return
        for leftover in directory.glob(f"zolex_{job.job_id}_*.png"):
            try:
                leftover.unlink()
            except OSError:
                logger.warning("h3_comfy_input_cleanup_failed", extra={"path": str(leftover)})

    # ── The run ──────────────────────────────────────────────────────────

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        reporter = StageReporter(on_progress)
        await reporter.preparing("Setting up your video…")

        graph = load_graph(self._graph_path(job.workflow_id))
        index = await self._duration_index(job)
        tier = self._tier(job)
        segments = PROMPTS_PER_INDEX[index]
        nominal_seconds = DURATION_PRESETS[index]

        # ── Inputs ───────────────────────────────────────────────────
        images: dict[str, str] = {}
        drop_ref3 = False
        reference_labels: tuple[str, ...] = ()
        if job.workflow_id == "image-to-video":
            source = job.input_for("source_image")
            if source is None:
                raise AdapterError(
                    "Please add the image to animate.",
                    internal_detail="image-to-video without source_image",
                    retriable=False,
                )
            images["I2V SOURCE IMAGE"] = await self._stage_image(job, source, "src")
        else:
            reference = job.input_for("reference_image")
            source_video = job.input_for("source_video")
            if reference is None:
                # A plain restyle has no identity reference; the H3 R2V graph
                # would generate an unrelated scene. LTX transform owns that
                # job — refuse rather than half-serve.
                raise AdapterError(
                    "This tool needs a reference photo of the person.",
                    internal_detail="video-to-video routed to h3_comfy without reference_image",
                    retriable=False,
                )
            images["REFERENCE IMAGE 1"] = await self._stage_image(job, reference, "ref1")
            if source_video is not None:
                images["REFERENCE IMAGE 2"] = await self._stage_video_first_frame(
                    job, source_video, "ref2"
                )
                reference_labels = ("<Picture 1>", "<Picture 2>")
            else:
                # Identity-only request: Picture 1 alone, mirroring the pack's
                # own "supply all three or disconnect" instruction.
                images["REFERENCE IMAGE 2"] = images["REFERENCE IMAGE 1"]
                reference_labels = ("<Picture 1>",)
            drop_ref3 = True

        # ── Prompt discipline ────────────────────────────────────────
        plan = plan_from_prompt(job.prompt, reference_labels=reference_labels)
        prompts = dict(enumerate(discipline_prompts(plan, segments), start=1))

        # ── Canvas tier (R2V only; I2V's one proven canvas is the graph's) ──
        width = height = None
        if job.workflow_id == "video-to-video":
            if tier == "draft":
                width, height = settings.h3_comfy_draft_canvas
            else:
                width, height = settings.h3_comfy_quality_canvas

        audio_context = job.execution.get("h3_audio_context")
        edits = GraphEdits(
            duration_index=index,
            prompts=prompts,
            images=images,
            drop_reference_3=drop_ref3,
            width=width,
            height=height,
            filename_prefix=f"zolex_{job.job_id}",
            output_directory=str(job.workspace),
            audio_context_length=(
                int(audio_context) if audio_context is not None else None
            ),
        )
        api_prompt = to_api_prompt(graph, edits)

        # ── Submit and wait ──────────────────────────────────────────
        service = self._service()
        expected = nominal_seconds * _EXPECTED_RATE.get((job.workflow_id, tier), 40.0)
        await reporter.generating(GENERATE_FROM, "Starting your video…")

        async def tick(elapsed: float) -> None:
            # Elapsed-paced bar between the generation bounds; the +20% head-
            # room keeps it moving on a slow run without ever claiming done.
            span = GENERATE_TO - GENERATE_FROM - 1
            fraction = min(1.0, elapsed / (expected * 1.2))
            progress = GENERATE_FROM + int(span * fraction)
            if segments > 1:
                approx = min(segments, 1 + int(segments * fraction))
                await reporter.section(approx, segments, progress)
            else:
                await reporter.generating(progress)

        started = time.monotonic()
        try:
            prompt_id = await service.submit(api_prompt, client_id=f"zolex-{job.job_id}")
            logger.info(
                "h3_comfy_submitted",
                extra={
                    "job_id": job.job_id,
                    "workflow": job.workflow_id,
                    "prompt_id": prompt_id,
                    "duration_index": index,
                    "tier": tier,
                    "segments": segments,
                },
            )
            timeout = min(
                settings.h3_comfy_generation_timeout,
                job.seconds_remaining or settings.h3_comfy_generation_timeout,
            )
            await service.wait(job, prompt_id, timeout_seconds=timeout, on_tick=tick)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message,
                internal_detail=exc.internal_detail,
                retriable=exc.retriable,
            ) from exc
        finally:
            self._cleanup_staged(job)
            if settings.h3_comfy_free_after_job:
                await service.free_memory()
        wall = time.monotonic() - started

        # ── Collect and validate — a file is not a result until probed ──
        await reporter.report("post_processing", 88, "Assembling your video…")
        candidates = sorted(
            job.workspace.glob(f"zolex_{job.job_id}*.mp4"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise AdapterError(
                "The finished video could not be found.",
                internal_detail=(
                    f"no zolex_{job.job_id}*.mp4 in {job.workspace} after success"
                ),
            )
        output = candidates[-1]
        info = await probe_media(output)
        if not info.has_video or not info.duration_seconds:
            raise AdapterError(
                "The finished video failed its check.",
                internal_detail=f"probe of {output.name}: {info!r}",
            )
        if abs(info.duration_seconds - nominal_seconds) > 1.5:
            raise AdapterError(
                "The finished video failed its check.",
                internal_detail=(
                    f"duration {info.duration_seconds:.3f}s vs preset {nominal_seconds}s"
                ),
            )
        logger.info(
            "h3_comfy_finished",
            extra={
                "job_id": job.job_id,
                "wall_seconds": round(wall, 1),
                "duration_seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
                "segments": segments,
            },
        )
        await reporter.report("uploading", 96, "Almost ready…")
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=info.duration_seconds,
            width=info.width,
            height=info.height,
        )


# ── Health, shared with the provider layer ──────────────────────────────────

#: Node classes the frozen graphs require; missing any means the pinned
#: custom-node checkout is wrong, not merely that a model is absent.
REQUIRED_NODE_CLASSES = (
    "MiniMaxH3Extender",
    "MiniMaxH3MotionContextDiskFinalDecode",
    "MiniMaxH3MotionContextDiskJoin",
    "MiniMaxH3MotionContextRAM",
    "MiniMaxH3PromptPackBridge",
    "MiniMaxH3ReferencePackBridge",
    "MiniMaxH3ImageToVideo",
    "easy anythingIndexSwitch",
)

#: The official Comfy-Org files with their exact published sizes. Size is the
#: every-boot check; full SHA256 of 75 GB is a `deep=True` operation for
#: provisioning, not for a health poll.
REQUIRED_WEIGHTS: dict[str, int] = {
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20_970_379_616,
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20_970_379_616,
    "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors": 27_141_342_152,
    "vae/minimax_h3_video_vae_fp16.safetensors": 5_207_808_496,
    "vae/minimax_h3_audio_vae_fp32.safetensors": 605_254_808,
}

_MIN_FREE_DISK_BYTES = 50 * 2**30
_MIN_VRAM_BYTES = 40 * 2**30  # measured peaks: 52-63 GB on the 95.6 GB card


async def h3_comfy_health(client: ComfyClient | None = None) -> tuple[bool, str]:
    """The Phase-4 checklist. Any critical miss → unavailable, stated plainly."""
    problems: list[str] = []

    service = client or ComfyClient(
        settings.h3_comfy_base_url,
        request_timeout=settings.h3_comfy_request_timeout,
    )
    up, detail = await service.reachable()
    if not up:
        return False, detail

    try:
        stats = await service.system_stats()
        devices = stats.get("devices") or []
        vram = devices[0].get("vram_total", 0) if devices else 0
        if vram and vram < _MIN_VRAM_BYTES:
            problems.append(
                f"GPU too small for this runtime: {vram / 2**30:.0f} GB VRAM "
                f"(measured peaks reach 63 GB)"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"system_stats failed: {exc}")

    try:
        classes = await service.node_classes()
        missing = [name for name in REQUIRED_NODE_CLASSES if name not in classes]
        if missing:
            problems.append(f"missing node classes {missing} (pinned custom nodes not loaded)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"object_info failed: {exc}")

    for filename in _GRAPHS.values():
        if not (settings.h3_comfy_workflows_dir / filename).is_file():
            problems.append(f"frozen graph missing: {filename}")

    models_dir = settings.h3_comfy_models_dir
    if models_dir is None:
        problems.append("h3_comfy_models_dir not configured")
    else:
        for rel, expected_size in REQUIRED_WEIGHTS.items():
            path = models_dir / rel
            if not path.is_file():
                problems.append(f"weight missing: {rel}")
            elif path.stat().st_size != expected_size:
                problems.append(
                    f"weight size mismatch: {rel} ({path.stat().st_size} != {expected_size})"
                )
        if models_dir.exists():
            free = shutil.disk_usage(models_dir).free
            if free < _MIN_FREE_DISK_BYTES:
                problems.append(f"low disk: {free / 2**30:.0f} GB free")

    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg not on PATH")

    if problems:
        return False, "; ".join(problems)
    return True, detail
