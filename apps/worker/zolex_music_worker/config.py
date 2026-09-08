from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import ValidationError


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _json_argv(name: str) -> list[str] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{name} must contain a JSON string array") from exc
    if not isinstance(parsed, list) or not parsed or not all(isinstance(x, str) and x for x in parsed):
        raise ValidationError(f"{name} must contain a nonempty JSON string array")
    return parsed


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValidationError(f"{name} must be true or false")


@dataclass(frozen=True)
class WorkerConfig:
    work_root: Path
    fps: int = 24
    max_source_seconds: int = 300
    shot_target_seconds: float = 4.5
    shot_min_seconds: float = 2.0
    shot_max_seconds: float = 7.0
    max_attempts: int = 3
    command_timeout_seconds: int = 7200
    render_concurrency: int = 1
    render_gpu_ids: tuple[str, ...] = ()
    anchor_concurrency: int = 4
    target_job_seconds: int = 300
    baseline_job_seconds: int = 900
    baseline_audio_seconds: int = 180
    baseline_render_concurrency: int = 1
    enforce_latency_capacity: bool = False
    anchor_backend: str = "reference"
    render_backend: str = "ltx"
    upscale_backend: str = "cpu"
    upscale_command: list[str] | None = None
    upscale_gpu_id: str = "0"
    upscale_timeout_seconds: int = 1800
    upscale_cq: int = 18
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    anchor_command: list[str] | None = None
    render_command: list[str] | None = None
    transcription_backend: str = "faster_whisper"
    transcription_command: list[str] | None = None
    transcription_model: str = "large-v3"
    transcription_device: str = "cuda"
    transcription_compute_type: str = "float16"
    transcription_download_root: Path | None = None
    transcription_beam_size: int = 5
    transcription_timeout_seconds: int = 1800
    lyric_director_command: list[str] | None = None
    reference_analyzer_command: list[str] | None = None
    reference_vision_backend: str = "disabled"
    reference_vision_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    reference_vision_device: str = "cuda:0"
    reference_vision_max_new_tokens: int = 768
    reference_vision_min_pixels: int = 200_704
    reference_vision_max_pixels: int = 1_003_520
    reference_vision_local_files_only: bool = False
    reference_sample_frames: int = 12
    reference_max_seconds: int = 600
    reference_scene_threshold: float = 0.32
    reference_analysis_timeout_seconds: int = 900
    reference_fetch_backend: str = "yt_dlp"
    reference_fetch_command: list[str] | None = None
    reference_fetch_python: str = "python"
    reference_allowed_hosts: tuple[str, ...] = ("youtube.com", "youtu.be", "vimeo.com")
    reference_max_bytes: int = 1_073_741_824
    reference_fetch_timeout_seconds: int = 1800
    external_qa_command: list[str] | None = None
    lipsync_command: list[str] | None = None
    ltx_python: str = "python"
    ltx_module: str = "ltx_pipelines.a2vid_two_stage"
    ltx_transformer_path: Path | None = None
    ltx_text_encoder_path: Path | None = None
    ltx_video_vae_path: Path | None = None
    ltx_audio_vae_path: Path | None = None
    ltx_spatial_upsampler_path: Path | None = None
    ltx_distilled_lora_path: Path | None = None
    ltx_distilled_lora_strength: float = 1.0
    ltx_num_inference_steps: int = 24
    ltx_extra_args: list[str] | None = None
    ltx_negative_prompt: str = (
        "identity drift, face change, deformed hands, extra fingers, extra limbs, duplicate person, "
        "cropped head, cropped feet, flicker, exposure pulsing, sudden darkness, captions, text, logo, watermark"
    )

    @classmethod
    def from_env(cls, work_root: str | Path | None = None) -> "WorkerConfig":
        root = Path(work_root or os.getenv("ZOLEX_WORK_ROOT", "./runs")).expanduser().resolve()
        return cls(
            work_root=root,
            fps=int(os.getenv("ZOLEX_FPS", "24")),
            max_source_seconds=int(os.getenv("ZOLEX_MAX_SOURCE_SECONDS", "300")),
            shot_target_seconds=float(os.getenv("ZOLEX_SHOT_TARGET_SECONDS", "4.5")),
            shot_min_seconds=float(os.getenv("ZOLEX_SHOT_MIN_SECONDS", "2.0")),
            shot_max_seconds=float(os.getenv("ZOLEX_SHOT_MAX_SECONDS", "7.0")),
            max_attempts=int(os.getenv("ZOLEX_MAX_ATTEMPTS", "3")),
            command_timeout_seconds=int(os.getenv("ZOLEX_COMMAND_TIMEOUT", "7200")),
            render_concurrency=int(os.getenv("ZOLEX_RENDER_CONCURRENCY", "1")),
            render_gpu_ids=tuple(
                value.strip() for value in os.getenv("ZOLEX_RENDER_GPU_IDS", "").split(",") if value.strip()
            ),
            anchor_concurrency=int(os.getenv("ZOLEX_ANCHOR_CONCURRENCY", "4")),
            target_job_seconds=int(os.getenv("ZOLEX_TARGET_JOB_SECONDS", "300")),
            baseline_job_seconds=int(os.getenv("ZOLEX_BASELINE_JOB_SECONDS", "900")),
            baseline_audio_seconds=int(os.getenv("ZOLEX_BASELINE_AUDIO_SECONDS", "180")),
            baseline_render_concurrency=int(os.getenv("ZOLEX_BASELINE_RENDER_CONCURRENCY", "1")),
            enforce_latency_capacity=_env_bool("ZOLEX_ENFORCE_LATENCY_CAPACITY", False),
            anchor_backend=os.getenv("ZOLEX_ANCHOR_BACKEND", "reference"),
            render_backend=os.getenv("ZOLEX_RENDER_BACKEND", "ltx"),
            upscale_backend=os.getenv("ZOLEX_UPSCALE_BACKEND", "cuda").strip().casefold(),
            upscale_command=_json_argv("ZOLEX_UPSCALE_COMMAND_JSON"),
            upscale_gpu_id=os.getenv("ZOLEX_UPSCALE_GPU_ID", "0").strip(),
            upscale_timeout_seconds=int(os.getenv("ZOLEX_UPSCALE_TIMEOUT", "1800")),
            upscale_cq=int(os.getenv("ZOLEX_UPSCALE_CQ", "18")),
            ffmpeg=os.getenv("FFMPEG", "ffmpeg"),
            ffprobe=os.getenv("FFPROBE", "ffprobe"),
            anchor_command=_json_argv("ZOLEX_ANCHOR_COMMAND_JSON"),
            render_command=_json_argv("ZOLEX_RENDER_COMMAND_JSON"),
            transcription_backend=os.getenv("ZOLEX_TRANSCRIPTION_BACKEND", "faster_whisper").strip().casefold(),
            transcription_command=_json_argv("ZOLEX_TRANSCRIPTION_COMMAND_JSON"),
            transcription_model=os.getenv("ZOLEX_TRANSCRIPTION_MODEL", "large-v3").strip(),
            transcription_device=os.getenv("ZOLEX_TRANSCRIPTION_DEVICE", "cuda").strip(),
            transcription_compute_type=os.getenv("ZOLEX_TRANSCRIPTION_COMPUTE_TYPE", "float16").strip(),
            transcription_download_root=_optional_path("ZOLEX_TRANSCRIPTION_DOWNLOAD_ROOT"),
            transcription_beam_size=int(os.getenv("ZOLEX_TRANSCRIPTION_BEAM_SIZE", "5")),
            transcription_timeout_seconds=int(os.getenv("ZOLEX_TRANSCRIPTION_TIMEOUT", "1800")),
            lyric_director_command=_json_argv("ZOLEX_LYRIC_DIRECTOR_COMMAND_JSON"),
            reference_analyzer_command=_json_argv("ZOLEX_REFERENCE_ANALYZER_COMMAND_JSON"),
            reference_vision_backend=os.getenv("ZOLEX_REFERENCE_VISION_BACKEND", "qwen").strip().casefold(),
            reference_vision_model=os.getenv(
                "ZOLEX_REFERENCE_VISION_MODEL",
                "Qwen/Qwen2.5-VL-3B-Instruct",
            ).strip(),
            reference_vision_device=os.getenv("ZOLEX_REFERENCE_VISION_DEVICE", "cuda:0").strip(),
            reference_vision_max_new_tokens=int(
                os.getenv("ZOLEX_REFERENCE_VISION_MAX_NEW_TOKENS", "768")
            ),
            reference_vision_min_pixels=int(
                os.getenv("ZOLEX_REFERENCE_VISION_MIN_PIXELS", "200704")
            ),
            reference_vision_max_pixels=int(
                os.getenv("ZOLEX_REFERENCE_VISION_MAX_PIXELS", "1003520")
            ),
            reference_vision_local_files_only=_env_bool(
                "ZOLEX_REFERENCE_VISION_LOCAL_FILES_ONLY",
                False,
            ),
            reference_sample_frames=int(os.getenv("ZOLEX_REFERENCE_SAMPLE_FRAMES", "12")),
            reference_max_seconds=int(os.getenv("ZOLEX_REFERENCE_MAX_SECONDS", "600")),
            reference_scene_threshold=float(os.getenv("ZOLEX_REFERENCE_SCENE_THRESHOLD", "0.32")),
            reference_analysis_timeout_seconds=int(
                os.getenv("ZOLEX_REFERENCE_ANALYSIS_TIMEOUT", "900")
            ),
            reference_fetch_backend=os.getenv("ZOLEX_REFERENCE_FETCH_BACKEND", "yt_dlp").strip().casefold(),
            reference_fetch_command=_json_argv("ZOLEX_REFERENCE_FETCH_COMMAND_JSON"),
            reference_fetch_python=os.getenv("ZOLEX_REFERENCE_FETCH_PYTHON", "python").strip(),
            reference_allowed_hosts=tuple(
                value.strip().casefold().rstrip(".")
                for value in os.getenv(
                    "ZOLEX_REFERENCE_ALLOWED_HOSTS",
                    "youtube.com,youtu.be,vimeo.com",
                ).split(",")
                if value.strip()
            ),
            reference_max_bytes=int(os.getenv("ZOLEX_REFERENCE_MAX_BYTES", "1073741824")),
            reference_fetch_timeout_seconds=int(os.getenv("ZOLEX_REFERENCE_FETCH_TIMEOUT", "1800")),
            external_qa_command=_json_argv("ZOLEX_QA_COMMAND_JSON"),
            lipsync_command=_json_argv("ZOLEX_LIPSYNC_COMMAND_JSON"),
            ltx_python=os.getenv("LTX_PYTHON", "python"),
            ltx_module=os.getenv("LTX_MODULE", "ltx_pipelines.a2vid_two_stage"),
            ltx_transformer_path=_optional_path("LTX_TRANSFORMER_PATH"),
            ltx_text_encoder_path=_optional_path("LTX_TEXT_ENCODER_PATH"),
            ltx_video_vae_path=_optional_path("LTX_VIDEO_VAE_PATH"),
            ltx_audio_vae_path=_optional_path("LTX_AUDIO_VAE_PATH"),
            ltx_spatial_upsampler_path=_optional_path("LTX_SPATIAL_UPSAMPLER_PATH"),
            ltx_distilled_lora_path=_optional_path("LTX_DISTILLED_LORA_PATH"),
            ltx_distilled_lora_strength=float(os.getenv("LTX_DISTILLED_LORA_STRENGTH", "1.0")),
            ltx_num_inference_steps=int(os.getenv("LTX_NUM_INFERENCE_STEPS", "24")),
            ltx_extra_args=_json_argv("LTX_EXTRA_ARGS_JSON"),
            ltx_negative_prompt=os.getenv("LTX_NEGATIVE_PROMPT", cls.ltx_negative_prompt),
        )

    def validate(self) -> None:
        if self.fps != 24:
            raise ValidationError("Explicit music-video mode currently requires exactly 24 FPS")
        if not (1 <= self.max_source_seconds <= 300):
            raise ValidationError("max_source_seconds must be between 1 and 300")
        if not (1.0 <= self.shot_min_seconds <= self.shot_target_seconds <= self.shot_max_seconds <= 20.0):
            raise ValidationError("Shot timing must satisfy 1 <= min <= target <= max <= 20")
        if not (1 <= self.max_attempts <= 10):
            raise ValidationError("max_attempts must be between 1 and 10")
        if not (1 <= self.render_concurrency <= 32):
            raise ValidationError("ZOLEX_RENDER_CONCURRENCY must be between 1 and 32")
        if not (1 <= self.anchor_concurrency <= 32):
            raise ValidationError("ZOLEX_ANCHOR_CONCURRENCY must be between 1 and 32")
        if not (60 <= self.target_job_seconds <= 7200):
            raise ValidationError("ZOLEX_TARGET_JOB_SECONDS must be between 60 and 7200")
        if not (60 <= self.baseline_job_seconds <= 14400):
            raise ValidationError("ZOLEX_BASELINE_JOB_SECONDS must be between 60 and 14400")
        if not (10 <= self.baseline_audio_seconds <= 300):
            raise ValidationError("ZOLEX_BASELINE_AUDIO_SECONDS must be between 10 and 300")
        if not (1 <= self.baseline_render_concurrency <= 32):
            raise ValidationError("ZOLEX_BASELINE_RENDER_CONCURRENCY must be between 1 and 32")
        if self.render_gpu_ids and len(self.render_gpu_ids) < self.render_concurrency:
            raise ValidationError("ZOLEX_RENDER_GPU_IDS must provide at least one GPU id per render worker")
        if self.render_backend == "ltx" and self.render_concurrency > 1 and not self.render_gpu_ids:
            raise ValidationError("Parallel direct LTX rendering requires explicit ZOLEX_RENDER_GPU_IDS")
        if not (1 <= self.ltx_num_inference_steps <= 100):
            raise ValidationError("LTX_NUM_INFERENCE_STEPS must be between 1 and 100")
        if self.anchor_backend not in {"reference", "command", "mock"}:
            raise ValidationError("ZOLEX_ANCHOR_BACKEND must be reference, command, or mock")
        if self.render_backend not in {"ltx", "command", "mock"}:
            raise ValidationError("ZOLEX_RENDER_BACKEND must be ltx, command, or mock")
        if self.upscale_backend not in {"cuda", "cpu", "command"}:
            raise ValidationError("ZOLEX_UPSCALE_BACKEND must be cuda, cpu, or command")
        if self.upscale_backend == "command" and not self.upscale_command:
            raise ValidationError("command upscale backend requires ZOLEX_UPSCALE_COMMAND_JSON")
        if self.upscale_backend == "cuda" and not self.upscale_gpu_id:
            raise ValidationError("ZOLEX_UPSCALE_GPU_ID cannot be empty for the CUDA backend")
        if not (30 <= self.upscale_timeout_seconds <= 7200):
            raise ValidationError("ZOLEX_UPSCALE_TIMEOUT must be between 30 and 7200")
        if not (0 <= self.upscale_cq <= 51):
            raise ValidationError("ZOLEX_UPSCALE_CQ must be between 0 and 51")
        if self.transcription_backend not in {"faster_whisper", "command", "disabled"}:
            raise ValidationError(
                "ZOLEX_TRANSCRIPTION_BACKEND must be faster_whisper, command, or disabled"
            )
        if self.transcription_backend == "command" and not self.transcription_command:
            raise ValidationError(
                "command transcription backend requires ZOLEX_TRANSCRIPTION_COMMAND_JSON"
            )
        if not self.transcription_model.strip():
            raise ValidationError("ZOLEX_TRANSCRIPTION_MODEL cannot be empty")
        if not (1 <= self.transcription_beam_size <= 20):
            raise ValidationError("ZOLEX_TRANSCRIPTION_BEAM_SIZE must be between 1 and 20")
        if not (30 <= self.transcription_timeout_seconds <= 7200):
            raise ValidationError("ZOLEX_TRANSCRIPTION_TIMEOUT must be between 30 and 7200")
        if not (4 <= self.reference_sample_frames <= 36):
            raise ValidationError("ZOLEX_REFERENCE_SAMPLE_FRAMES must be between 4 and 36")
        if not (1 <= self.reference_max_seconds <= 3600):
            raise ValidationError("ZOLEX_REFERENCE_MAX_SECONDS must be between 1 and 3600")
        if not (0.05 <= self.reference_scene_threshold <= 0.95):
            raise ValidationError("ZOLEX_REFERENCE_SCENE_THRESHOLD must be between 0.05 and 0.95")
        if not (30 <= self.reference_analysis_timeout_seconds <= 3600):
            raise ValidationError("ZOLEX_REFERENCE_ANALYSIS_TIMEOUT must be between 30 and 3600")
        if self.reference_vision_backend not in {"qwen", "disabled"}:
            raise ValidationError("ZOLEX_REFERENCE_VISION_BACKEND must be qwen or disabled")
        if not self.reference_vision_model:
            raise ValidationError("ZOLEX_REFERENCE_VISION_MODEL cannot be empty")
        if self.reference_vision_device != "auto" and not (
            self.reference_vision_device == "cpu"
            or re.fullmatch(r"cuda(?::[0-9]+)?", self.reference_vision_device)
        ):
            raise ValidationError(
                "ZOLEX_REFERENCE_VISION_DEVICE must be auto, cpu, cuda, or cuda:N"
            )
        if not (128 <= self.reference_vision_max_new_tokens <= 2048):
            raise ValidationError(
                "ZOLEX_REFERENCE_VISION_MAX_NEW_TOKENS must be between 128 and 2048"
            )
        if not (
            50_176 <= self.reference_vision_min_pixels
            <= self.reference_vision_max_pixels
            <= 8_028_160
        ):
            raise ValidationError(
                "Reference vision pixels must satisfy 50176 <= min <= max <= 8028160"
            )
        if self.reference_fetch_backend not in {"yt_dlp", "command"}:
            raise ValidationError("ZOLEX_REFERENCE_FETCH_BACKEND must be yt_dlp or command")
        if self.reference_fetch_backend == "command" and not self.reference_fetch_command:
            raise ValidationError("command reference fetch backend requires ZOLEX_REFERENCE_FETCH_COMMAND_JSON")
        if not self.reference_fetch_python:
            raise ValidationError("ZOLEX_REFERENCE_FETCH_PYTHON cannot be empty")
        if not self.reference_allowed_hosts:
            raise ValidationError("ZOLEX_REFERENCE_ALLOWED_HOSTS must contain at least one hostname")
        if any(
            "." not in host
            or host.startswith(".")
            or host.endswith(".")
            or any(character in host for character in ("/", ":", "*", "@"))
            for host in self.reference_allowed_hosts
        ):
            raise ValidationError("ZOLEX_REFERENCE_ALLOWED_HOSTS entries must be exact domain names")
        if not (1_048_576 <= self.reference_max_bytes <= 10_737_418_240):
            raise ValidationError("ZOLEX_REFERENCE_MAX_BYTES must be between 1 MiB and 10 GiB")
        if not (30 <= self.reference_fetch_timeout_seconds <= 7200):
            raise ValidationError("ZOLEX_REFERENCE_FETCH_TIMEOUT must be between 30 and 7200")
        if self.anchor_backend == "command" and not self.anchor_command:
            raise ValidationError("command anchor backend requires ZOLEX_ANCHOR_COMMAND_JSON")
        if self.render_backend == "command" and not self.render_command:
            raise ValidationError("command render backend requires ZOLEX_RENDER_COMMAND_JSON")
