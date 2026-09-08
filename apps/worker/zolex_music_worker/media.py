from __future__ import annotations

import json
import math
import subprocess
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .config import WorkerConfig
from .errors import ExternalCommandError, ValidationError
from .models import AudioInfo
from .utils import ensure_executable, run_command, sha256_file


def _probe_json(config: WorkerConfig, path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    argv = [config.ffprobe, "-v", "error"]
    if count_frames:
        argv.append("-count_frames")
    argv.extend(
        [
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except OSError as exc:
        raise ExternalCommandError(f"Unable to run ffprobe: {exc}") from exc
    if result.returncode != 0:
        raise ValidationError(f"ffprobe rejected {path}: {result.stderr.strip()[-2000:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"ffprobe returned invalid JSON for {path}") from exc


def prepare_audio(source: Path, job_dir: Path, config: WorkerConfig) -> AudioInfo:
    ensure_executable(config.ffmpeg)
    ensure_executable(config.ffprobe)
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    decoded = audio_dir / "decoded-48k-stereo.wav"
    aligned = audio_dir / "delivery-aligned.wav"
    conditioning = audio_dir / "conditioning-master-plus-2s.wav"

    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ],
        timeout=600,
        log_path=audio_dir / "decode-command.json",
    )

    with wave.open(str(decoded), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        decoded_samples = handle.getnframes()
    decoded_duration = decoded_samples / sample_rate
    if decoded_duration < 1.0:
        raise ValidationError("Audio must be at least one second")
    if decoded_duration > config.max_source_seconds + (1 / config.fps):
        raise ValidationError(
            f"Decoded audio is {decoded_duration:.3f}s; limit is {config.max_source_seconds}s"
        )

    total_frames = max(1, round(decoded_duration * config.fps))
    aligned_duration = total_frames / config.fps
    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(decoded),
            "-af",
            f"apad,atrim=duration={aligned_duration:.12f}",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(aligned),
        ],
        timeout=600,
        log_path=audio_dir / "align-command.json",
    )
    conditioning_duration = aligned_duration + 2.0
    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(aligned),
            "-af",
            f"apad,atrim=duration={conditioning_duration:.12f}",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(conditioning),
        ],
        timeout=600,
        log_path=audio_dir / "conditioning-command.json",
    )

    return AudioInfo(
        source_path=str(source),
        decoded_wav=str(decoded),
        aligned_wav=str(aligned),
        conditioning_wav=str(conditioning),
        sample_rate=sample_rate,
        channels=channels,
        decoded_samples=decoded_samples,
        total_frames=total_frames,
        fps=config.fps,
        decoded_duration=decoded_duration,
        aligned_duration=aligned_duration,
        source_sha256=sha256_file(source),
    )


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValidationError("Analysis WAV must use signed 16-bit PCM")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return rate, pcm


def video_probe(path: Path, config: WorkerConfig, *, count_frames: bool = True) -> dict[str, Any]:
    raw = _probe_json(config, path, count_frames=count_frames)
    streams = raw.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video:
        raise ValidationError(f"No video stream in {path}")
    fps_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_text))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(raw.get("format", {}).get("duration") or 0.0)
    frames_text = video.get("nb_read_frames") or video.get("nb_frames")
    frame_count = int(frames_text) if frames_text and str(frames_text).isdigit() else round(duration * fps)
    return {
        "path": str(path),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "video_codec": video.get("codec_name"),
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def normalize_clip(
    raw_clip: Path,
    output_clip: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    config: WorkerConfig,
    log_path: Path,
) -> None:
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    filter_text = (
        f"trim=start_frame=0:end_frame={frame_count},setpts=PTS-STARTPTS,"
        f"fps={config.fps},scale={width}:{height}:flags=lanczos,format=yuv420p"
    )
    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(raw_clip),
            "-an",
            "-vf",
            filter_text,
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(config.fps),
            str(output_clip),
        ],
        timeout=1200,
        log_path=log_path,
    )


def brightness_statistics(path: Path, config: WorkerConfig) -> dict[str, float]:
    width, height = 64, 64
    argv = [
        config.ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-an",
        "-vf",
        f"fps={config.fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    result = subprocess.run(argv, capture_output=True, timeout=600, check=False)
    if result.returncode != 0:
        raise ExternalCommandError(f"Brightness analysis failed for {path}")
    frame_bytes = width * height
    usable = len(result.stdout) - (len(result.stdout) % frame_bytes)
    if usable == 0:
        raise ValidationError(f"No frames decoded for brightness analysis: {path}")
    pixels = np.frombuffer(result.stdout[:usable], dtype=np.uint8).reshape(-1, frame_bytes)
    means = pixels.mean(axis=1)
    median = float(np.median(means))
    minimum = float(means.min())
    maximum = float(means.max())
    if len(means) > 1:
        max_step = float(np.abs(np.diff(means)).max())
    else:
        max_step = 0.0
    return {
        "mean": float(means.mean()),
        "median": median,
        "min": minimum,
        "max": maximum,
        "min_to_median_ratio": minimum / max(median, 1e-6),
        "max_frame_step": max_step,
    }


def ltx_window_frames(delivered_frames: int, minimum: int = 121, maximum: int = 481) -> int:
    if delivered_frames <= 0:
        raise ValidationError("delivered_frames must be positive")
    raw = max(minimum, delivered_frames)
    raw = int(math.ceil((raw - 1) / 8) * 8 + 1)
    if raw > maximum:
        raise ValidationError(f"Required raw LTX window {raw} exceeds supported maximum {maximum}")
    return raw

