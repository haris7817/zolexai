from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import __version__
from .adapters import make_render_adapter
from .config import WorkerConfig
from .errors import WorkerError
from .utils import ensure_executable, read_json, validate_job_id
from .worker import load_request, run_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zolex-music-video")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Plan and render a music-video job")
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--work-root", type=Path)
    run.add_argument("--plan-only", action="store_true")

    validate = sub.add_parser("validate", help="Validate a request without processing media")
    validate.add_argument("--request", required=True, type=Path)

    status = sub.add_parser("status", help="Print a job's current status")
    status.add_argument("--job-id", required=True)
    status.add_argument("--work-root", type=Path)

    cancel = sub.add_parser("cancel", help="Request cooperative cancellation")
    cancel.add_argument("--job-id", required=True)
    cancel.add_argument("--work-root", type=Path)

    doctor = sub.add_parser("doctor", help="Check worker configuration and executables")
    doctor.add_argument("--work-root", type=Path)

    models = sub.add_parser(
        "install-models",
        help="Download and verify the required LTX, Whisper and Qwen weights",
    )
    models.add_argument("--models-root", type=Path, default=Path("/models"))
    models.add_argument("--max-workers", type=int, default=4)
    models.add_argument("--skip-ltx", action="store_true")
    models.add_argument("--skip-whisper", action="store_true")
    models.add_argument("--skip-qwen", action="store_true")
    models.add_argument("--size-only", action="store_true")
    models.add_argument("--dry-run", action="store_true")
    models.add_argument("--env-output", type=Path)
    return parser


def _config(work_root: Path | None) -> WorkerConfig:
    return WorkerConfig.from_env(work_root=work_root)


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "install-models":
            from .model_installer import install_required_models

            skip = [
                key
                for key, enabled in (
                    ("ltx", args.skip_ltx),
                    ("whisper", args.skip_whisper),
                    ("qwen", args.skip_qwen),
                )
                if enabled
            ]
            result = install_required_models(
                models_root=args.models_root,
                skip=skip,
                max_workers=args.max_workers,
                verify_sha256=not args.size_only,
                dry_run=args.dry_run,
                env_output=args.env_output,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.command == "validate":
            request = load_request(args.request.resolve())
            print(json.dumps({"valid": True, "request": request.to_dict()}, indent=2))
            return
        if args.command == "doctor":
            config = _config(args.work_root)
            config.validate()
            checks = {
                "ffmpeg": ensure_executable(config.ffmpeg),
                "ffprobe": ensure_executable(config.ffprobe),
                "anchor_backend": config.anchor_backend,
                "render_backend": config.render_backend,
                "render_concurrency": config.render_concurrency,
                "render_gpu_ids": list(config.render_gpu_ids),
                "upscale_backend": config.upscale_backend,
                "upscale_gpu_id": config.upscale_gpu_id if config.upscale_backend != "cpu" else None,
                "upscale_audio_policy": "stream_copy",
                "anchor_concurrency": config.anchor_concurrency,
                "target_job_seconds": config.target_job_seconds,
                "baseline_job_seconds": config.baseline_job_seconds,
                "baseline_audio_seconds": config.baseline_audio_seconds,
                "baseline_render_concurrency": config.baseline_render_concurrency,
                "required_capacity_multiplier_for_baseline": round(
                    config.baseline_job_seconds / config.target_job_seconds,
                    3,
                ),
                "enforce_latency_capacity": config.enforce_latency_capacity,
                "transcription_backend": config.transcription_backend,
                "transcription_model": config.transcription_model,
                "reference_video_analysis": "built_in_metrics_plus_semantic_vision",
                "reference_analyzer_command_configured": bool(config.reference_analyzer_command),
                "reference_vision_backend": config.reference_vision_backend,
                "reference_vision_model": config.reference_vision_model,
                "reference_vision_device": config.reference_vision_device,
                "reference_ai_dependencies_installed": all(
                    importlib.util.find_spec(module) is not None
                    for module in ("torch", "transformers", "qwen_vl_utils")
                ),
                "reference_sample_frames": config.reference_sample_frames,
                "reference_max_seconds": config.reference_max_seconds,
                "reference_video_url_input": True,
                "reference_fetch_backend": config.reference_fetch_backend,
                "reference_allowed_hosts": list(config.reference_allowed_hosts),
                "reference_max_bytes": config.reference_max_bytes,
                "faster_whisper_installed": importlib.util.find_spec("faster_whisper") is not None,
                "delivery_resolutions": {
                    "16:9": "3840x2160",
                    "9:16": "2160x3840",
                    "1:1": "2160x2160",
                },
                "ltx_scene_resolutions": {
                    "16:9": "1280x704",
                    "9:16": "704x1280",
                    "1:1": "1024x1024",
                },
                "work_root": str(config.work_root),
            }
            if config.render_backend == "ltx":
                make_render_adapter(config)
                checks["ltx_python"] = ensure_executable(config.ltx_python)
                checks["ltx_model_paths"] = "validated"
                checks["ltx_num_inference_steps"] = config.ltx_num_inference_steps
            print(json.dumps({"ok": True, "checks": checks}, indent=2))
            return
        config = _config(args.work_root)
        if args.command == "status":
            job_id = validate_job_id(args.job_id)
            print(json.dumps(read_json(config.work_root / job_id / "status.json"), indent=2))
            return
        if args.command == "cancel":
            job_id = validate_job_id(args.job_id)
            cancel_path = config.work_root / job_id / "CANCEL"
            cancel_path.parent.mkdir(parents=True, exist_ok=True)
            cancel_path.write_text("cancel requested\n", encoding="utf-8")
            print(json.dumps({"job_id": job_id, "cancel_requested": True}, indent=2))
            return
        request = load_request(args.request.resolve())
        result = run_job(request, config, plan_only=args.plan_only)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (WorkerError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
