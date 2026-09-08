from __future__ import annotations

from pathlib import Path

from .config import WorkerConfig
from .models import EditorialPlan
from .utils import atomic_write_json, ensure_existing_file, format_template_argv, run_command


def _center_crop(plan: EditorialPlan) -> str | None:
    """Return an even center crop that removes the LTX alignment padding."""
    source_width, source_height = plan.scene_width, plan.scene_height
    target_ratio = plan.width / plan.height
    source_ratio = source_width / source_height
    if abs(source_ratio - target_ratio) < 1e-9:
        return None
    if source_ratio > target_ratio:
        crop_width = min(source_width, round(source_height * target_ratio / 2) * 2)
        x = (source_width - crop_width) // 2
        return f"crop={crop_width}:{source_height}:{x}:0"
    crop_height = min(source_height, round(source_width / target_ratio / 2) * 2)
    y = (source_height - crop_height) // 2
    return f"crop={source_width}:{crop_height}:0:{y}"


def build_cpu_upscale_filter(plan: EditorialPlan) -> str:
    filters = [_center_crop(plan)] if _center_crop(plan) else []
    filters.extend(
        [
            f"scale={plan.width}:{plan.height}:flags=lanczos",
            "setsar=1",
            "format=yuv420p",
        ]
    )
    return ",".join(filters)


def build_cuda_upscale_filter(plan: EditorialPlan) -> str:
    filters = [_center_crop(plan)] if _center_crop(plan) else []
    filters.extend(
        [
            "format=nv12",
            "hwupload_cuda=device=0",
            f"scale_cuda=w={plan.width}:h={plan.height}:interp_algo=lanczos:format=nv12",
        ]
    )
    return ",".join(filters)


def _builtin_upscale(
    *,
    input_path: Path,
    output_path: Path,
    plan: EditorialPlan,
    config: WorkerConfig,
    log_path: Path,
) -> None:
    cuda = config.upscale_backend == "cuda"
    argv = [
        config.ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        build_cuda_upscale_filter(plan) if cuda else build_cpu_upscale_filter(plan),
        "-frames:v",
        str(plan.total_frames),
        "-r",
        str(plan.fps),
    ]
    if cuda:
        argv.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(config.upscale_cq),
                "-b:v",
                "0",
            ]
        )
    else:
        argv.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                str(config.upscale_cq),
                "-pix_fmt",
                "yuv420p",
            ]
        )
    argv.extend(
        [
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    environment = {"CUDA_VISIBLE_DEVICES": config.upscale_gpu_id} if cuda else None
    run_command(
        argv,
        timeout=config.upscale_timeout_seconds,
        log_path=log_path,
        env_overrides=environment,
    )


def _command_upscale(
    *,
    input_path: Path,
    output_path: Path,
    plan: EditorialPlan,
    config: WorkerConfig,
    log_path: Path,
) -> None:
    raw_output = output_path.with_name("upscaled-video-raw.mp4")
    request_path = output_path.with_name("upscale-request.json")
    atomic_write_json(
        request_path,
        {
            "schema_version": 1,
            "input": str(input_path),
            "output": str(raw_output),
            "source_width": plan.scene_width,
            "source_height": plan.scene_height,
            "delivery_width": plan.width,
            "delivery_height": plan.height,
            "fps": plan.fps,
            "frames": plan.total_frames,
            "gpu_id": config.upscale_gpu_id,
            "audio_policy": "video_only_output; worker remuxes unchanged audio",
        },
    )
    values = {
        "request_json": str(request_path),
        "input": str(input_path),
        "output": str(raw_output),
        "width": plan.scene_width,
        "height": plan.scene_height,
        "delivery_width": plan.width,
        "delivery_height": plan.height,
        "fps": plan.fps,
        "frames": plan.total_frames,
        "gpu_id": config.upscale_gpu_id,
    }
    run_command(
        format_template_argv(config.upscale_command or [], values),
        timeout=config.upscale_timeout_seconds,
        log_path=log_path,
        env_overrides={"CUDA_VISIBLE_DEVICES": config.upscale_gpu_id},
    )
    ensure_existing_file(raw_output, "Upscale command output")
    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(raw_output),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(plan.total_frames),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        timeout=config.upscale_timeout_seconds,
        log_path=log_path.with_name("upscale-audio-remux-command.json"),
    )


def upscale_master(
    *,
    input_path: Path,
    output_path: Path,
    plan: EditorialPlan,
    config: WorkerConfig,
    log_path: Path,
) -> None:
    input_path = ensure_existing_file(input_path, "Working master")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.upscale_backend == "command":
        _command_upscale(
            input_path=input_path,
            output_path=output_path,
            plan=plan,
            config=config,
            log_path=log_path,
        )
    else:
        _builtin_upscale(
            input_path=input_path,
            output_path=output_path,
            plan=plan,
            config=config,
            log_path=log_path,
        )
    ensure_existing_file(output_path, "4K delivery")
