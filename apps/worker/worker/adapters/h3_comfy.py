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

    **OFF by default since 28 Aug 2026** (`h3_comfy_video_to_video`). Not
    re-enacting the motion turned out to be the whole complaint: a customer
    who uploads footage and asks for it back better is not asking for a
    different performance in a similar room. `supports()` declines the
    workflow while the switch is off, and Video to Video runs on LTX
    transform at both quality levels — the engine that drives every frame
    from the source and replaces the person inside it.

  * `text-to-video` → the client T2V graph — added 25 Aug at the user's
    explicit request as a client-test experiment. LTX remains the measured
    default for T2V (faster at a larger canvas, and H3's identity edge does
    not apply without an input image); routing YAML decides, and the flip is
    one line either way.

Everything else — music video, extend — stays on LTX per the measured
routing. Turbo is not reachable from here at all: rejected on quality.

The ComfyUI service is treated exactly like ACE-Step: a long-lived local
process the worker connects to and never manages. Worker and service share a
filesystem (same GPU node), which is what lets the Final Decode node write the
assembled MP4 straight into the job's own workspace.
"""

from __future__ import annotations

import hashlib
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
from worker.longform.h3_prompts import (
    discipline_prompts,
    plan_cameras,
    plan_from_prompt,
)
from worker.longform.language import spoken_language_clause
from worker.longform.progress import GENERATE_FROM, GENERATE_TO, StageReporter
from worker.media.ffmpeg import ffmpeg
from worker.media.probe import probe_media

logger = logging.getLogger("zolexai.worker.h3_comfy")

#: workflow id → graph filename in the frozen client pack.
_GRAPHS = {
    "image-to-video": "minimax_h3_i2v_extender.json",
    "video-to-video": "minimax_h3_r2v_extender.json",
    "text-to-video": "minimax_h3_t2v_extender.json",
}

#: Client-test T2V canvases: the proven tier pixel budgets reshaped to the
#: product's aspect ratios, every side a multiple of 32 — the pack documents
#: the width/height primitives as authoritative under exactly that invariant.
_T2V_CANVAS = {
    "quality": {
        "16:9": (960, 544),
        "9:16": (544, 960),
        "1:1": (704, 704),
        "4:5": (640, 800),
    },
    "draft": {
        "16:9": (544, 320),
        "9:16": (320, 544),
        "1:1": (416, 416),
        "4:5": (384, 480),
    },
}

#: Measured wall-clock per second of output on the RTX PRO 6000, used ONLY to
#: pace the progress bar, never to declare completion. 25 Aug measurements:
#: R2V draft 11x, R2V 960x544 33x, I2V 1280x736 ~55x warm.
_EXPECTED_RATE = {
    ("video-to-video", "draft"): 12.0,
    ("video-to-video", "quality"): 34.0,
    ("image-to-video", "quality"): 60.0,
    ("image-to-video", "draft"): 60.0,  # I2V has one proven canvas; same pace
    ("text-to-video", "draft"): 14.0,  # measured 25 Aug: 70.3s cold for 5.17s
    ("text-to-video", "quality"): 34.0,  # assumed ≈ R2V quality; pacing only
}


class H3ComfyAdapter:
    name = "h3_comfy"

    def __init__(self, client: ComfyClient | None = None) -> None:
        self._client = client

    def supports(self, workflow_id: str) -> bool:
        if workflow_id == "video-to-video" and not settings.h3_comfy_video_to_video:
            # Off by default — see `h3_comfy_video_to_video`. The R2V graph
            # re-imagines the scene from two stills instead of following the
            # customer's footage, which is a different product from the one
            # Video to Video sells.
            return False
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
    def _steps(job: AdapterJob) -> int | None:
        """`execution.h3_steps` — the user-decided speed dial (26 Aug).

        None keeps the pack's pinned 20. 12 was measured 28% faster at the
        quality canvas and judged visually acceptable by the user; the bounds
        refuse configuration typos rather than render garbage from them.
        """
        raw = job.execution.get("h3_steps")
        if raw is None:
            return None
        try:
            steps = int(raw)
        except (TypeError, ValueError):
            steps = -1
        if not 4 <= steps <= 40:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"h3_steps={raw!r} is not an integer in 4..40",
                retriable=False,
            )
        return steps

    @staticmethod
    def _seed_base(job: AdapterJob) -> int:
        """Every job is its own video; every retry is its own job again.

        The pack's fixed seeds plus a deterministic model plus ComfyUI's
        cache meant "regenerate" returned the byte-identical file in seconds,
        forever (production, 26 Aug). A customer-supplied `seed` parameter
        wins when the product sends one; otherwise the job id — stable across
        retries of one attempt, different for every new job.
        """
        raw = job.parameters.get("seed")
        try:
            if raw is not None:
                return abs(int(raw)) % (2**48)
        except (TypeError, ValueError):
            pass
        digest = hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

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
    async def _duration_index(job: AdapterJob) -> tuple[int, float | None]:
        seconds = parse_duration_seconds(job.parameters.get("duration"))
        if seconds is None and job.workflow_id == "video-to-video":
            # `duration_mode: source`: the API sends no duration because the
            # source clip's length IS the request. H3 generates preset lengths,
            # so the nearest preset to the source is the honest reading.
            source = job.input_for("source_video")
            if source is not None and source.path is not None:
                info = await probe_media(source.require_path())
                if info.duration_seconds:
                    return nearest_duration_index(info.duration_seconds), None
        if seconds is None:
            raise AdapterError(
                "Please choose a video length.",
                internal_detail=f"no usable duration in {job.parameters.get('duration')!r}",
                retriable=False,
            )
        index = duration_index_for(seconds)
        # The Fast/Best product decision (client-approved 27 Aug): Best (H3)
        # offers up to 30s — the 60s preset exists in the pack but is not
        # sold on this engine. The cap is config so the decision stays a
        # YAML line, and the refusal lists only what is actually on offer.
        max_seconds = job.execution.get("h3_max_seconds")
        offered = {
            i: v
            for i, v in DURATION_PRESETS.items()
            if max_seconds is None or v <= float(max_seconds)
        }
        if index is None and seconds is not None:
            # A length the lattice cannot render exactly (the client sells
            # 20s; the pack's plans are 5/10/15/30/60): render the NEXT
            # preset up and trim the finished file to the promised length.
            # The render costs the larger preset — an engineering fact the
            # product accepted (27 Aug) in exchange for the exact duration.
            longer = [i for i, v in sorted(offered.items()) if v > seconds]
            if longer:
                return longer[0], float(seconds)
        if index is None or index not in offered:
            supported = ", ".join(f"{int(v)}s" for v in offered.values())
            raise AdapterError(
                f"This tool supports these lengths: {supported}.",
                internal_detail=(
                    f"{seconds}s not offered (h3_max_seconds={max_seconds!r})"
                ),
                retriable=False,
            )
        return index, None

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
        if not self.supports(job.workflow_id):
            # `supports()` guides the resolver's quality-level fallback, but a
            # workflow whose BASE runtime is this engine never consults it —
            # and that is not hypothetical. Production carried
            # `runtime: h3_comfy` for video-to-video on 28 Aug 2026, under a
            # `runtime_by_quality` map, so withdrawing the workflow from this
            # adapter would have changed nothing at all for Best.
            #
            # A misconfiguration must fail loudly rather than quietly ship the
            # product this engine was withdrawn from. The customer-facing text
            # says nothing about engines; the internal detail names the exact
            # YAML key to fix.
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"h3_comfy does not serve '{job.workflow_id}' "
                    f"(h3_comfy_video_to_video={settings.h3_comfy_video_to_video}); "
                    "check this deployment's execution.runtime / runtime_by_quality"
                ),
                retriable=False,
            )

        reporter = StageReporter(on_progress)
        await reporter.preparing("Setting up your video…")

        graph = load_graph(self._graph_path(job.workflow_id))
        index, trim_to = await self._duration_index(job)
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
        elif job.workflow_id == "video-to-video":
            reference = job.input_for("reference_image")
            source_video = job.input_for("source_video")
            if reference is None:
                # A plain restyle has no identity reference; the H3 R2V graph
                # would generate an unrelated scene. LTX transform owns that
                # job — refuse rather than half-serve.
                #
                # The message names the way out, which until 28 Aug 2026 did
                # not exist: this workflow ran on one engine, so a customer
                # restyling footage without a reference photo was told what
                # was missing and had no setting that would accept the job.
                # Two production jobs failed that way in one evening. Fast
                # is now the engine that restyles without a photo.
                raise AdapterError(
                    "Best quality replaces the person in your video with the "
                    "one in a reference photo, so it needs that photo. To "
                    "restyle your video without one, switch quality to Fast.",
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
        # A shot per segment instead of one camera repeated. Without this the
        # compiler forced "The camera holds one steady shot" onto every
        # generation, so a customer asking for a movie scene got a locked-off
        # frame for thirty seconds (client report, 27 Aug 2026). There is no
        # audio to read here — H3 writes its own — so the roles come from
        # position alone, which is what `plan_shots` falls back to anyway.
        # `plan_shots` used to supply these. It is the music-video director:
        # it reads roles from audio this path does not have, speaks LTX's
        # camera language rather than H3's, and — because it was written for
        # CUTS between sections — is free to change shot scale across a
        # boundary. On a 30s two-segment render that meant segment 1 was told
        # to push in "until the subject fills the frame" and segment 2 was
        # handed that face plus a low-angle tilt-up: it read the head as an
        # object on a desk and rebuilt the room around it (client frame-audit,
        # 28 Aug 2026). An H3 seam is a handoff, not a cut, so the shots come
        # from a plan that holds scale across the boundary.
        cameras = plan_cameras(segments) if segments > 1 else None
        # This engine writes its own audio from these prompts, and said
        # nothing about what language it should be in — the same silence LTX
        # had, reported by the client on 28 Aug 2026 as sound coming back "in
        # a language that is not english". Image to Video on Best routes here,
        # which is precisely the case they were describing.
        prompts = dict(
            enumerate(
                discipline_prompts(
                    plan,
                    segments,
                    total_seconds=nominal_seconds,
                    cameras=cameras,
                    spoken_language=spoken_language_clause(
                        job.parameters, job.execution
                    ),
                ),
                start=1,
            )
        )

        # ── Canvas tier (I2V's one proven canvas is the graph's own) ──
        width = height = None
        if job.workflow_id == "video-to-video":
            if tier == "draft":
                width, height = settings.h3_comfy_draft_canvas
            else:
                width, height = settings.h3_comfy_quality_canvas
        elif job.workflow_id == "text-to-video":
            # T2V has no input image to inherit a shape from, so the product's
            # aspect_ratio parameter picks the canvas within the tier's proven
            # pixel budget. An unknown value falls back to the proven 16:9.
            aspect = str(job.parameters.get("aspect_ratio") or "16:9").strip()
            width, height = _T2V_CANVAS[tier].get(aspect, _T2V_CANVAS[tier]["16:9"])

        audio_context = job.execution.get("h3_audio_context")
        context_length = job.execution.get("h3_context_length")
        steps = self._steps(job)
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
            context_length=(
                int(context_length) if context_length is not None else None
            ),
            steps=steps,
            seed_base=self._seed_base(job),
        )
        api_prompt = to_api_prompt(graph, edits)

        # ── Submit and wait ──────────────────────────────────────────
        service = self._service()
        expected = nominal_seconds * _EXPECTED_RATE.get((job.workflow_id, tier), 40.0)
        if steps is not None:
            # Pacing only. Measured shape at the quality canvas: sampling is
            # ~72% of the wall at 20 steps, the rest is fixed pipeline cost.
            expected *= 0.28 + 0.72 * (steps / 20.0)
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

        # The pack's decoder ends the track at full level — a hard cut the
        # client's 26 Aug frame-audit called out — and its peaks sit at
        # -0.1 dBFS, which clips audibly after any platform re-encode (same
        # audit, second video). One audio-only pass: tail fade plus a -1 dBTP
        # ceiling; the video stream is copied untouched.
        if trim_to is not None:
            # The promised length is not on the lattice: the render was the
            # next preset up, the delivery is an exact cut. Re-encoded (a
            # copy cut lands on the previous keyframe, seconds early) at the
            # pack's own quality settings.
            cut = output.with_name(f"{output.stem}_cut.mp4")
            await ffmpeg(
                [
                    "-i", str(output),
                    "-t", f"{trim_to:.3f}",
                    "-c:v", "libx264", "-crf", "17", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "192k",
                    str(cut), "-y",
                ]
            )
            output = cut
            info = await probe_media(output)

        wants_sound = str(
            job.parameters.get("sound", True)
        ).strip().lower() not in ("false", "no", "off", "0")
        if info.has_audio and not wants_sound:
            # The customer asked for a silent video (part of the Fast/Best
            # product round, 27 Aug): drop the audio stream outright — the
            # video stream is copied untouched, so this costs milliseconds.
            muted = output.with_name(f"{output.stem}_muted.mp4")
            await ffmpeg(["-i", str(output), "-an", "-c:v", "copy", str(muted), "-y"])
            output = muted
        elif info.has_audio and info.duration_seconds > 1.5:
            faded = output.with_name(f"{output.stem}_faded.mp4")
            fade_start = max(0.0, info.duration_seconds - 0.75)
            await ffmpeg(
                [
                    "-i", str(output),
                    "-c:v", "copy",
                    "-af",
                    f"afade=t=out:st={fade_start:.3f}:d=0.75,"
                    "alimiter=limit=0.891:level=disabled",
                    "-c:a", "aac", "-b:a", "192k",
                    str(faded), "-y",
                ]
            )
            output = faded
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
