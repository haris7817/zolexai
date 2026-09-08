from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .adapters import generate_anchors, make_anchor_adapter, make_render_adapter
from .analysis import analyze_song
from .assembler import assemble_working_master
from .config import WorkerConfig
from .director import expand_direction
from .errors import QualityError, ValidationError
from .identity import build_identity_package
from .job_state import JobJournal
from .lyrics import apply_lyric_directions, prepare_lyrics
from .media import normalize_clip, prepare_audio
from .models import MusicVideoRequest, Shot
from .planner import build_plan, compile_legacy_manifest
from .performance import build_latency_capacity_plan
from .prompts import compile_shot_prompts
from .qa import run_delivery_qa, run_shot_qa
from .reference_fetch import fetch_reference_video
from .reference_style import analyze_reference_video
from .upscaler import upscale_master
from .utils import (
    atomic_write_json,
    ensure_existing_file,
    exclusive_file_lock,
    format_template_argv,
    read_json,
    run_command,
    sha256_file,
)


def _format_slug(value: str) -> str:
    return value.replace(":", "x")


def load_request(path: Path) -> MusicVideoRequest:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValidationError("Request JSON must contain an object")
    return MusicVideoRequest.from_dict(payload)


def _persist_request(job_dir: Path, request: MusicVideoRequest) -> None:
    path = job_dir / "request.json"
    payload = request.to_dict()
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise ValidationError(
                f"Job {request.job_id} already exists with a different request; use a new job_id"
            )
    else:
        atomic_write_json(path, payload)


def _apply_lipsync(
    *,
    source: Path,
    shot: Shot,
    plan,
    audio,
    attempt_dir: Path,
    config: WorkerConfig,
) -> Path:
    if not config.lipsync_command or not shot.vocals_present:
        return source
    output = attempt_dir / "lipsynced-raw.mp4"
    normalized = attempt_dir / "lipsynced-normalized.mp4"
    request_json = attempt_dir / "lipsync-request.json"
    atomic_write_json(
        request_json,
        {
            "clip": str(source),
            "audio": audio.aligned_wav,
            "audio_start_seconds": shot.start_frame / plan.fps,
            "duration_seconds": shot.frame_count / plan.fps,
            "performer_ids": shot.performer_ids,
            "output": str(output),
        },
    )
    values = {
        "request_json": str(request_json),
        "clip": str(source),
        "audio": audio.aligned_wav,
        "start_seconds": f"{shot.start_frame / plan.fps:.12f}",
        "duration_seconds": f"{shot.frame_count / plan.fps:.12f}",
        "output": str(output),
    }
    run_command(
        format_template_argv(config.lipsync_command, values),
        timeout=config.command_timeout_seconds,
        log_path=attempt_dir / "lipsync-command.json",
    )
    ensure_existing_file(output, "Lip-sync output")
    normalize_clip(
        output,
        normalized,
        frame_count=shot.frame_count,
        width=plan.scene_width,
        height=plan.scene_height,
        config=config,
        log_path=attempt_dir / "lipsync-normalize-command.json",
    )
    return normalized


def _render_one_shot(
    *,
    shot: Shot,
    plan,
    audio,
    request: MusicVideoRequest,
    format_dir: Path,
    config: WorkerConfig,
    journal: JobJournal,
    worker_slot: int = 0,
    gpu_id: str | None = None,
) -> Path:
    shot_dir = format_dir / "shots" / shot.id
    accepted = shot_dir / "accepted.mp4"
    accepted_report = shot_dir / "accepted.qa.json"
    if accepted.exists():
        result = run_shot_qa(accepted, shot, plan, config, accepted_report)
        if result.passed:
            return accepted

    adapter = make_render_adapter(config, request.render_backend)
    failures: list[dict[str, Any]] = []
    for attempt in range(1, config.max_attempts + 1):
        journal.check_cancelled()
        attempt_dir = shot_dir / f"attempt-{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        raw = attempt_dir / "raw.mp4"
        normalized = attempt_dir / "normalized.mp4"
        if not raw.exists():
            journal.update(
                "rendering",
                format=plan.format,
                shot_id=shot.id,
                attempt=attempt,
                shot_index=int(shot.id.rsplit("-", 1)[-1]),
                shot_count=len(plan.shots),
            )
            adapter.render(
                shot=shot,
                plan=plan,
                audio=audio,
                output=raw,
                attempt=attempt,
                worker_slot=worker_slot,
                gpu_id=gpu_id,
                timeout=config.command_timeout_seconds,
                log_path=attempt_dir / "render-command.json",
            )
        if not normalized.exists():
            normalize_clip(
                raw,
                normalized,
                frame_count=shot.frame_count,
                width=plan.scene_width,
                height=plan.scene_height,
                config=config,
                log_path=attempt_dir / "normalize-command.json",
            )
        candidate = _apply_lipsync(
            source=normalized,
            shot=shot,
            plan=plan,
            audio=audio,
            attempt_dir=attempt_dir,
            config=config,
        ) if request.lip_sync else normalized
        qa_result = run_shot_qa(candidate, shot, plan, config, attempt_dir / "qa.json")
        if qa_result.passed:
            shot_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, accepted)
            shutil.copy2(attempt_dir / "qa.json", accepted_report)
            atomic_write_json(
                shot_dir / "accepted.json",
                {
                    "shot_id": shot.id,
                    "attempt": attempt,
                    "clip": str(accepted),
                    "sha256": sha256_file(accepted),
                    "lipsync_applied": bool(request.lip_sync and config.lipsync_command and shot.vocals_present),
                },
            )
            return accepted
        failures.append({"attempt": attempt, "failures": list(qa_result.failures)})
    raise QualityError(f"Shot {shot.id} failed {config.max_attempts} attempts: {failures}")


def _render_all_shots(
    *,
    plan,
    audio,
    request: MusicVideoRequest,
    format_dir: Path,
    config: WorkerConfig,
    journal: JobJournal,
) -> list[Path]:
    def render_index(index: int, shot: Shot, worker_slot: int) -> tuple[int, Path]:
        gpu_id = config.render_gpu_ids[worker_slot] if config.render_gpu_ids else None
        clip = _render_one_shot(
            shot=shot,
            plan=plan,
            audio=audio,
            request=request,
            format_dir=format_dir,
            config=config,
            journal=journal,
            worker_slot=worker_slot,
            gpu_id=gpu_id,
        )
        return index, clip

    if config.render_concurrency == 1:
        return [render_index(index, shot, 0)[1] for index, shot in enumerate(plan.shots)]

    ordered: list[Path | None] = [None] * len(plan.shots)
    lanes: list[list[tuple[int, Shot]]] = [[] for _ in range(config.render_concurrency)]
    for index, shot in enumerate(plan.shots):
        lanes[index % config.render_concurrency].append((index, shot))
    progress_lock = Lock()
    completed = 0

    def render_lane(worker_slot: int, items: list[tuple[int, Shot]]) -> list[tuple[int, Path]]:
        nonlocal completed
        results: list[tuple[int, Path]] = []
        for index, shot in items:
            results.append(render_index(index, shot, worker_slot))
            with progress_lock:
                completed += 1
                current_completed = completed
            journal.update(
                "rendering",
                format=plan.format,
                completed_shots=current_completed,
                shot_count=len(plan.shots),
                render_concurrency=config.render_concurrency,
            )
        return results

    with ThreadPoolExecutor(max_workers=config.render_concurrency, thread_name_prefix="zolex-render") as pool:
        futures = [pool.submit(render_lane, worker_slot, items) for worker_slot, items in enumerate(lanes) if items]
        for future in as_completed(futures):
            for index, clip in future.result():
                ordered[index] = clip
    if any(path is None for path in ordered):
        raise QualityError("Parallel renderer did not return every planned shot")
    return [path for path in ordered if path is not None]


def _run_job_locked(
    request: MusicVideoRequest,
    config: WorkerConfig,
    *,
    plan_only: bool = False,
) -> dict[str, Any]:
    config.validate()
    if request.reference_video and request.reference_video_url:
        raise ValidationError("Use either reference_video or reference_video_url, not both")
    selected_render_backend = request.render_backend or config.render_backend
    if selected_render_backend == "ltx" and config.render_concurrency > 1 and not config.render_gpu_ids:
        raise ValidationError("Parallel direct LTX rendering requires explicit ZOLEX_RENDER_GPU_IDS")
    source_audio = ensure_existing_file(request.audio, "Audio")
    source_reference: Path | None = (
        ensure_existing_file(request.reference_video, "Reference video")
        if request.reference_video
        else None
    )
    for performer in request.performers:
        for reference in performer.reference_images:
            ensure_existing_file(reference, f"Reference for {performer.id}")

    job_dir = (config.work_root / request.job_id).resolve()
    config.work_root.mkdir(parents=True, exist_ok=True)
    journal = JobJournal(job_dir, request.job_id)
    try:
        _persist_request(job_dir, request)
        journal.update("validating")
        journal.check_cancelled()

        if request.reference_video_url:
            journal.update("fetching_reference_video", source="url")
            source_reference = fetch_reference_video(
                request.reference_video_url,
                output_dir=job_dir / "inputs" / "reference-video",
                config=config,
            )

        journal.update("preparing_audio")
        audio = prepare_audio(source_audio, job_dir, config)
        atomic_write_json(job_dir / "audio" / "audio-info.json", audio.to_dict())
        latency_plan = build_latency_capacity_plan(
            audio_seconds=audio.aligned_duration,
            format_count=len(request.formats),
            render_backend=selected_render_backend,
            config=config,
        )
        atomic_write_json(job_dir / "latency-plan.json", latency_plan)
        if config.enforce_latency_capacity and not latency_plan["capacity_ready"] and not plan_only:
            raise ValidationError(str(latency_plan["note"]))

        journal.update("analyzing_song_lyrics_and_reference")
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="zolex-analysis") as pool:
            analysis_future = pool.submit(
                analyze_song,
                Path(audio.aligned_wav),
                fps=config.fps,
                total_frames=audio.total_frames,
            )
            transcript_future = pool.submit(
                prepare_lyrics,
                request,
                audio,
                config,
                job_dir / "analysis",
            )
            reference_future = (
                pool.submit(
                    analyze_reference_video,
                    source_reference,
                    config=config,
                    output_dir=job_dir / "analysis" / "reference-video",
                )
                if source_reference
                else None
            )
            analysis = analysis_future.result()
            transcript = transcript_future.result()
            reference_style = reference_future.result() if reference_future else None
        atomic_write_json(job_dir / "analysis" / "song-analysis.json", analysis.to_dict())

        journal.update("planning")
        treatment = expand_direction(request, analysis, transcript, reference_style)
        atomic_write_json(job_dir / "treatment.json", treatment)
        anchor_adapter = make_anchor_adapter(config, request.anchor_backend)
        journal.update("building_identity_package")
        identity_references = build_identity_package(
            request=request,
            treatment=treatment,
            output_dir=job_dir / "identities",
            adapter=anchor_adapter,
            timeout=config.command_timeout_seconds,
        )
        results: dict[str, Any] = {}

        for output_format in request.formats:
            journal.check_cancelled()
            slug = _format_slug(output_format)
            format_dir = job_dir / slug
            plan = build_plan(request, treatment, analysis, output_format, config, transcript)
            apply_lyric_directions(plan, transcript, treatment, config, format_dir / "lyric-direction")
            journal.update("generating_anchors", format=output_format)
            generate_anchors(
                plan=plan,
                request=request,
                treatment=treatment,
                output_dir=format_dir / "anchors",
                adapter=anchor_adapter,
                timeout=config.command_timeout_seconds,
                identity_references=identity_references,
                concurrency=config.anchor_concurrency,
            )
            compile_shot_prompts(plan, request, treatment)
            atomic_write_json(format_dir / "editorial-plan.json", plan.to_dict())
            atomic_write_json(format_dir / "shots.json", compile_legacy_manifest(plan, format_dir))

            if plan_only:
                results[output_format] = {
                    "status": "planned",
                    "shots": len(plan.shots),
                    "editorial_plan": str(format_dir / "editorial-plan.json"),
                    "renderer_manifest": str(format_dir / "shots.json"),
                }
                continue

            accepted_clips = _render_all_shots(
                plan=plan,
                audio=audio,
                request=request,
                format_dir=format_dir,
                config=config,
                journal=journal,
            )

            journal.update("assembling", format=output_format)
            final_dir = format_dir / "final"
            working_master = final_dir / "working-master.mp4"
            output = final_dir / "output.mp4"
            assemble_working_master(
                plan=plan,
                accepted_clips=accepted_clips,
                audio=audio,
                output=working_master,
                config=config,
                log_path=final_dir / "assembly-command.json",
            )
            journal.update(
                "upscaling",
                format=output_format,
                backend=config.upscale_backend,
                source_resolution=f"{plan.scene_width}x{plan.scene_height}",
                delivery_resolution=f"{plan.width}x{plan.height}",
            )
            upscale_master(
                input_path=working_master,
                output_path=output,
                plan=plan,
                config=config,
                log_path=final_dir / "upscale-command.json",
            )
            journal.update("final_quality_control", format=output_format)
            qa = run_delivery_qa(output, plan, config, final_dir / "delivery-qa.json")
            if not qa.passed:
                raise QualityError(f"Final {output_format} delivery failed QA: {list(qa.failures)}")
            results[output_format] = {
                "status": "complete",
                "output": str(output),
                "sha256": sha256_file(output),
                "shots": len(plan.shots),
                "frames": plan.total_frames,
                "seconds": plan.total_frames / plan.fps,
                "creative_review_required": True,
                "resolution": f"{plan.width}x{plan.height}",
                "render_resolution": f"{plan.scene_width}x{plan.scene_height}",
                "working_master": str(working_master),
                "upscale_backend": config.upscale_backend,
                "audio_during_upscale": "stream_copy",
                "lyric_driven": treatment["lyric_driven"],
                "singing_lipsync_mode": "external_correction" if config.lipsync_command else "ltx_audio_conditioning_only",
            }

        state = "planned" if plan_only else "complete"
        result = {
            "job_id": request.job_id,
            "status": state,
            "job_dir": str(job_dir),
            "audio": audio.to_dict(),
            "treatment": treatment,
            "lyrics": {
                "requested": transcript.requested,
                "source": transcript.source,
                "language": transcript.language,
                "segments": len(transcript.segments),
            },
            "reference_video": {
                "used": reference_style is not None,
                "input_type": "url" if request.reference_video_url else "file" if source_reference else None,
                "source_url": request.reference_video_url,
                "local_copy": str(source_reference) if source_reference else None,
                "analysis_backend": reference_style.get("analysis_backend") if reference_style else None,
                "style_profile": str(job_dir / "analysis" / "reference-video" / "style-profile.json")
                if reference_style
                else None,
                "originality_policy": reference_style.get("originality_policy") if reference_style else None,
            },
            "latency_target": latency_plan,
            "rendering": {
                "backend": selected_render_backend,
                "concurrency": config.render_concurrency,
                "gpu_ids": list(config.render_gpu_ids),
            },
            "upscaling": {
                "backend": config.upscale_backend,
                "gpu_id": config.upscale_gpu_id if config.upscale_backend != "cpu" else None,
                "audio_policy": "AAC encoded once in working master, then stream-copied unchanged to 4K delivery",
            },
            "formats": results,
        }
        atomic_write_json(job_dir / "result.json", result)
        journal.update(state, result=str(job_dir / "result.json"))
        return result
    except Exception as exc:
        if not journal.cancel_path.exists():
            journal.update("failed", error_type=type(exc).__name__, error=str(exc))
        raise


def run_job(
    request: MusicVideoRequest,
    config: WorkerConfig,
    *,
    plan_only: bool = False,
) -> dict[str, Any]:
    job_dir = (config.work_root / request.job_id).resolve()
    with exclusive_file_lock(job_dir / ".worker.lock"):
        return _run_job_locked(request, config, plan_only=plan_only)
