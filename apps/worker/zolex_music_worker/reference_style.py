from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import WorkerConfig
from .errors import ValidationError
from .media import video_probe
from .utils import (
    atomic_write_json,
    ensure_existing_file,
    format_template_argv,
    read_json,
    run_command,
    sha256_file,
)


STYLE_FIELDS = (
    "style_summary",
    "camera_language",
    "shot_scale_and_angles",
    "scene_archetypes",
    "lighting_style",
    "color_palette",
    "editing_rhythm",
    "composition_style",
    "motion_style",
)


def _merge_style_response(
    profile: dict[str, Any],
    response: Any,
    *,
    backend: str,
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValidationError("Reference analyzer output must be a JSON object")
    recognized = 0
    for field in STYLE_FIELDS:
        if field not in response:
            continue
        value = response[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValidationError(
                f"Reference analyzer field {field} must be a nonempty string up to 2000 characters"
            )
        profile[field] = value.strip()
        recognized += 1
    if recognized == 0:
        raise ValidationError("Reference analyzer returned no recognized style fields")
    profile["analysis_backend"] = backend
    return profile


def _describe_level(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def _extract_samples(
    reference_video: Path,
    output_dir: Path,
    *,
    duration: float,
    config: WorkerConfig,
) -> list[Path]:
    frames_dir = output_dir / "samples"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = config.reference_sample_frames / max(duration, 0.01)
    output_pattern = frames_dir / "sample-%03d.jpg"
    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(reference_video),
            "-an",
            "-vf",
            f"fps={sample_rate:.12f},scale=384:-2:flags=lanczos",
            "-frames:v",
            str(config.reference_sample_frames),
            "-q:v",
            "3",
            str(output_pattern),
        ],
        timeout=config.reference_analysis_timeout_seconds,
        log_path=output_dir / "sample-command.json",
    )
    samples = sorted(frames_dir.glob("sample-*.jpg"))
    if not samples:
        raise ValidationError("Reference-video analysis could not extract any frames")
    return samples


def _detect_cuts(
    reference_video: Path,
    output_dir: Path,
    *,
    config: WorkerConfig,
) -> list[float]:
    result = run_command(
        [
            config.ffmpeg,
            "-v",
            "info",
            "-nostdin",
            "-i",
            str(reference_video),
            "-an",
            "-vf",
            f"select=gt(scene\\,{config.reference_scene_threshold:.6f}),showinfo",
            "-f",
            "null",
            "-",
        ],
        timeout=config.reference_analysis_timeout_seconds,
        log_path=output_dir / "cut-detection-command.json",
    )
    values = [float(item) for item in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]
    return sorted({round(value, 6) for value in values if value > 0.05})


def _contact_sheet(samples: list[Path], output: Path) -> None:
    columns = 4
    cell_width, cell_height = 320, 200
    rows = (len(samples) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(samples):
        with Image.open(path) as image:
            fitted = ImageOps.fit(image.convert("RGB"), (cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        canvas.paste(fitted, (x, y))
        draw.rectangle((x + 5, y + 5, x + 42, y + 25), fill=(0, 0, 0))
        draw.text((x + 10, y + 8), f"{index + 1:02d}", fill=(255, 255, 255))
    canvas.save(output, format="JPEG", quality=88)


def _technical_profile(
    samples: list[Path],
    *,
    probe: dict[str, Any],
    cut_times: list[float],
) -> dict[str, Any]:
    arrays: list[np.ndarray] = []
    for path in samples:
        with Image.open(path) as image:
            arrays.append(np.asarray(ImageOps.fit(image.convert("RGB"), (128, 72)), dtype=np.float32) / 255.0)
    stack = np.stack(arrays)
    luma = stack[..., 0] * 0.2126 + stack[..., 1] * 0.7152 + stack[..., 2] * 0.0722
    maxima = stack.max(axis=-1)
    minima = stack.min(axis=-1)
    saturation = np.where(maxima > 1e-5, (maxima - minima) / maxima, 0.0)
    brightness = float(luma.mean())
    contrast = float(luma.std())
    saturation_mean = float(saturation.mean())
    warmth = float((stack[..., 0] - stack[..., 2]).mean())
    motion = (
        float(np.mean([np.abs(right - left).mean() for left, right in zip(arrays, arrays[1:])]))
        if len(arrays) > 1
        else 0.0
    )
    duration = float(probe["duration"])
    average_shot = duration / (len(cut_times) + 1)
    brightness_text = _describe_level(brightness, 0.35, 0.67, ("low-key", "balanced", "high-key"))
    contrast_text = _describe_level(contrast, 0.13, 0.24, ("soft-contrast", "moderate-contrast", "high-contrast"))
    saturation_text = _describe_level(saturation_mean, 0.23, 0.48, ("muted", "natural-color", "vibrant"))
    temperature_text = "warm" if warmth > 0.035 else "cool" if warmth < -0.035 else "neutral"
    motion_text = _describe_level(motion, 0.035, 0.10, ("restrained", "moderate", "energetic"))
    rhythm_text = _describe_level(average_shot, 2.0, 5.0, ("rapid-cut", "measured", "long-take"))
    mean_rgb = tuple(round(float(value) * 255) for value in stack.mean(axis=(0, 1, 2)))
    orientation = "landscape" if probe["width"] > probe["height"] else "portrait" if probe["height"] > probe["width"] else "square"
    return {
        "style_summary": (
            f"{brightness_text}, {contrast_text}, {saturation_text} {temperature_text} imagery with "
            f"{motion_text} visual movement and a {rhythm_text} editorial rhythm"
        ),
        "camera_language": f"Use {motion_text} camera or subject movement with stable, intentional framing",
        "lighting_style": f"{brightness_text} {temperature_text} lighting with {contrast_text} tonal separation",
        "color_palette": f"{saturation_text} {temperature_text} color centered near RGB {mean_rgb}",
        "editing_rhythm": f"{rhythm_text}; detected average shot length about {average_shot:.2f} seconds",
        "composition_style": f"Reference is {orientation} at {probe['width']}x{probe['height']}; preserve intentional subject separation and safe framing",
        "motion_style": f"{motion_text} frame-to-frame visual change",
        "average_shot_seconds": round(average_shot, 3),
        "detected_cut_seconds": cut_times,
        "technical_metrics": {
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "saturation": round(saturation_mean, 4),
            "warmth": round(warmth, 4),
            "sample_change": round(motion, 4),
            "mean_rgb": mean_rgb,
        },
    }


def _apply_ai_analysis(
    profile: dict[str, Any],
    *,
    reference_video: Path,
    contact_sheet: Path,
    output_dir: Path,
    config: WorkerConfig,
) -> dict[str, Any]:
    if not config.reference_analyzer_command and config.reference_vision_backend == "disabled":
        profile["analysis_backend"] = "built_in_visual_metrics"
        return profile
    if not config.reference_analyzer_command:
        from .vision_reference_analyzer import analyze_contact_sheet_with_qwen

        response = analyze_contact_sheet_with_qwen(
            contact_sheet,
            technical_profile=profile,
            output_dir=output_dir,
            config=config,
        )
        return _merge_style_response(
            profile,
            response,
            backend="built_in_metrics_plus_qwen2_5_vl_3b",
        )
    request_path = output_dir / "reference-analyzer.request.json"
    response_path = output_dir / "reference-analyzer.output.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": 1,
            "task": "Describe transferable filmmaking style only. Do not identify or reproduce people, logos, dialogue, lyrics, copyrighted characters, or an exact shot sequence.",
            "reference_video": str(reference_video),
            "contact_sheet": str(contact_sheet),
            "technical_profile": profile,
            "allowed_output_fields": list(STYLE_FIELDS),
            "output": str(response_path),
        },
    )
    values = {
        "request_json": str(request_path),
        "video": str(reference_video),
        "contact_sheet": str(contact_sheet),
        "output": str(response_path),
    }
    run_command(
        format_template_argv(config.reference_analyzer_command, values),
        timeout=config.reference_analysis_timeout_seconds,
        log_path=output_dir / "reference-analyzer.command.json",
    )
    response_path = ensure_existing_file(response_path, "Reference analyzer output")
    response = read_json(response_path)
    return _merge_style_response(
        profile,
        response,
        backend="built_in_metrics_plus_ai_vision_command",
    )


def analyze_reference_video(
    reference_video: Path,
    *,
    config: WorkerConfig,
    output_dir: Path,
) -> dict[str, Any]:
    reference_video = ensure_existing_file(reference_video, "Reference video")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = video_probe(reference_video, config, count_frames=False)
    duration = float(probe["duration"])
    if duration <= 0:
        raise ValidationError("Reference video has no measurable duration")
    if duration > config.reference_max_seconds + 0.05:
        raise ValidationError(
            f"Reference video is {duration:.3f}s; limit is {config.reference_max_seconds}s"
        )
    samples = _extract_samples(reference_video, output_dir, duration=duration, config=config)
    cut_times = _detect_cuts(reference_video, output_dir, config=config)
    contact_sheet = output_dir / "contact-sheet.jpg"
    _contact_sheet(samples, contact_sheet)
    profile = _technical_profile(samples, probe=probe, cut_times=cut_times)
    profile.update(
        {
            "schema_version": 1,
            "reference_video": str(reference_video),
            "reference_sha256": sha256_file(reference_video),
            "duration": duration,
            "sample_count": len(samples),
            "contact_sheet": str(contact_sheet),
            "originality_policy": (
                "Transfer only general filmmaking grammar—pacing, motion, lighting, color and composition. "
                "Create new scenes for the uploaded song; do not copy people, dialogue, lyrics, logos, locations or the exact shot sequence."
            ),
        }
    )
    profile = _apply_ai_analysis(
        profile,
        reference_video=reference_video,
        contact_sheet=contact_sheet,
        output_dir=output_dir,
        config=config,
    )
    atomic_write_json(output_dir / "style-profile.json", profile)
    return profile
