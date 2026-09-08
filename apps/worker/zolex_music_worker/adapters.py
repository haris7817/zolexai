from __future__ import annotations

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .config import WorkerConfig
from .errors import ValidationError
from .media import ltx_window_frames
from .models import AudioInfo, EditorialPlan, MusicVideoRequest, Shot
from .prompts import anchor_prompt
from .utils import atomic_write_json, ensure_executable, ensure_existing_file, format_template_argv, run_command


class AnchorAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        request_payload: dict[str, Any],
        output: Path,
        width: int,
        height: int,
        references: list[Path],
        timeout: int,
        log_path: Path,
    ) -> None:
        raise NotImplementedError


class ReferenceAnchorAdapter(AnchorAdapter):
    """Development fallback: fit the first reference to the requested canvas."""

    def generate(
        self,
        *,
        request_payload: dict[str, Any],
        output: Path,
        width: int,
        height: int,
        references: list[Path],
        timeout: int,
        log_path: Path,
    ) -> None:
        if references:
            with Image.open(references[0]) as image:
                converted = image.convert("RGB")
                fitted = ImageOps.fit(converted, (width, height), method=Image.Resampling.LANCZOS)
                fitted.save(output, format="PNG")
        else:
            _draw_mock_anchor(output, width, height, request_payload.get("anchor_role", "environment"))
        atomic_write_json(
            log_path,
            {
                "backend": "reference",
                "warning": "Reference backend does not create new cinematic compositions; configure command backend for production.",
                "references": [str(item) for item in references],
                "output": str(output),
            },
        )


def _draw_mock_anchor(output: Path, width: int, height: int, role: str) -> None:
    image = Image.new("RGB", (width, height), (20, 58, 76))
    pixels = image.load()
    for y in range(height):
        factor = y / max(1, height - 1)
        color = (
            int(22 + 160 * factor),
            int(62 + 72 * factor),
            int(82 + 30 * factor),
        )
        for x in range(width):
            pixels[x, y] = color
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, int(height * 0.67), width, height), fill=(22, 86, 104))
    draw.text((24, 24), f"MOCK ANCHOR — {role}", fill=(255, 240, 205))
    image.save(output, format="PNG")


class MockAnchorAdapter(AnchorAdapter):
    def generate(
        self,
        *,
        request_payload: dict[str, Any],
        output: Path,
        width: int,
        height: int,
        references: list[Path],
        timeout: int,
        log_path: Path,
    ) -> None:
        _draw_mock_anchor(output, width, height, request_payload.get("anchor_role", "unknown"))
        atomic_write_json(log_path, {"backend": "mock", "output": str(output)})


class CommandAnchorAdapter(AnchorAdapter):
    def __init__(self, template: list[str], config: WorkerConfig) -> None:
        self.template = template
        self.config = config

    def generate(
        self,
        *,
        request_payload: dict[str, Any],
        output: Path,
        width: int,
        height: int,
        references: list[Path],
        timeout: int,
        log_path: Path,
    ) -> None:
        request_path = output.with_suffix(".request.json")
        atomic_write_json(request_path, request_payload)
        values = {
            "request_json": str(request_path),
            "output": str(output),
            "width": width,
            "height": height,
            "references_json": json.dumps([str(item) for item in references]),
        }
        argv = format_template_argv(self.template, values)
        run_command(argv, timeout=timeout, log_path=log_path)
        ensure_existing_file(output, "Anchor adapter output")


def make_anchor_adapter(config: WorkerConfig, backend: str | None = None) -> AnchorAdapter:
    selected = backend or config.anchor_backend
    if selected == "reference":
        return ReferenceAnchorAdapter()
    if selected == "mock":
        return MockAnchorAdapter()
    if selected == "command" and config.anchor_command:
        return CommandAnchorAdapter(config.anchor_command, config)
    raise ValidationError(f"Anchor backend is not configured: {selected}")


class RenderAdapter(ABC):
    @abstractmethod
    def render(
        self,
        *,
        shot: Shot,
        plan: EditorialPlan,
        audio: AudioInfo,
        output: Path,
        attempt: int,
        worker_slot: int,
        gpu_id: str | None,
        timeout: int,
        log_path: Path,
    ) -> int:
        """Render and return the raw frame-window length."""
        raise NotImplementedError


class LtxRenderAdapter(RenderAdapter):
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        ensure_executable(config.ltx_python)
        required = {
            "LTX_TRANSFORMER_PATH": config.ltx_transformer_path,
            "LTX_TEXT_ENCODER_PATH": config.ltx_text_encoder_path,
            "LTX_VIDEO_VAE_PATH": config.ltx_video_vae_path,
            "LTX_AUDIO_VAE_PATH": config.ltx_audio_vae_path,
            "LTX_SPATIAL_UPSAMPLER_PATH": config.ltx_spatial_upsampler_path,
            "LTX_DISTILLED_LORA_PATH": config.ltx_distilled_lora_path,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise ValidationError(f"LTX backend is missing configuration: {', '.join(missing)}")
        for name, path in required.items():
            ensure_existing_file(path, name)

    def render(
        self,
        *,
        shot: Shot,
        plan: EditorialPlan,
        audio: AudioInfo,
        output: Path,
        attempt: int,
        worker_slot: int,
        gpu_id: str | None,
        timeout: int,
        log_path: Path,
    ) -> int:
        raw_frames = ltx_window_frames(shot.frame_count)
        seed = (shot.seed + attempt - 1) & 0x7FFFFFFF
        argv = [
            self.config.ltx_python,
            "-u",
            "-m",
            self.config.ltx_module,
            "--transformer-path",
            str(self.config.ltx_transformer_path),
            "--text-encoder-path",
            str(self.config.ltx_text_encoder_path),
            "--video-vae-path",
            str(self.config.ltx_video_vae_path),
            "--audio-vae-path",
            str(self.config.ltx_audio_vae_path),
            "--spatial-upsampler-path",
            str(self.config.ltx_spatial_upsampler_path),
            "--distilled-lora",
            str(self.config.ltx_distilled_lora_path),
            str(self.config.ltx_distilled_lora_strength),
            "--prompt",
            shot.prompt,
            "--negative-prompt",
            self.config.ltx_negative_prompt,
            "--image",
            shot.anchor_image,
            "0",
            "1.0",
            "--audio-path",
            audio.conditioning_wav,
            "--audio-start-time",
            f"{shot.start_frame / plan.fps:.12f}",
            "--num-frames",
            str(raw_frames),
            "--frame-rate",
            str(plan.fps),
            "--width",
            str(plan.scene_width),
            "--height",
            str(plan.scene_height),
            "--seed",
            str(seed),
            "--num-inference-steps",
            str(self.config.ltx_num_inference_steps),
            "--output-path",
            str(output),
        ]
        extra_args = self.config.ltx_extra_args or []
        forbidden = {
            "--output-path",
            "--audio-path",
            "--audio-start-time",
            "--audio-max-duration",
            "--num-frames",
            "--frame-rate",
            "--width",
            "--height",
            "--seed",
            "--image",
            "--prompt",
            "--negative-prompt",
        }
        conflict = next((token for token in extra_args if token in forbidden), None)
        if conflict:
            raise ValidationError(f"LTX_EXTRA_ARGS_JSON may not override worker-owned option {conflict}")
        argv.extend(extra_args)
        environment = {"CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id is not None else None
        run_command(argv, timeout=timeout, log_path=log_path, env_overrides=environment)
        ensure_existing_file(output, "LTX output")
        return raw_frames


class CommandRenderAdapter(RenderAdapter):
    def __init__(self, template: list[str], config: WorkerConfig) -> None:
        self.template = template
        self.config = config

    def render(
        self,
        *,
        shot: Shot,
        plan: EditorialPlan,
        audio: AudioInfo,
        output: Path,
        attempt: int,
        worker_slot: int,
        gpu_id: str | None,
        timeout: int,
        log_path: Path,
    ) -> int:
        raw_frames = ltx_window_frames(shot.frame_count)
        request_path = output.with_suffix(".request.json")
        payload = {
            "shot": shot.to_dict(),
            "format": plan.format,
            "width": plan.scene_width,
            "height": plan.scene_height,
            "delivery_width": plan.width,
            "delivery_height": plan.height,
            "fps": plan.fps,
            "raw_frames": raw_frames,
            "delivered_frames": shot.frame_count,
            "audio": audio.conditioning_wav,
            "audio_start_seconds": shot.start_frame / plan.fps,
            "attempt": attempt,
            "worker_slot": worker_slot,
            "gpu_id": gpu_id,
            "job_latency_target_seconds": self.config.target_job_seconds,
            "latency_priority": "five_minute_target",
            "quality_policy": "preserve_configured_quality",
            "output": str(output),
        }
        atomic_write_json(request_path, payload)
        values = {
            "request_json": str(request_path),
            "output": str(output),
            "anchor": shot.anchor_image,
            "audio": audio.conditioning_wav,
            "prompt": shot.prompt,
            "start_seconds": f"{shot.start_frame / plan.fps:.12f}",
            "raw_frames": raw_frames,
            "delivered_frames": shot.frame_count,
            "fps": plan.fps,
            "width": plan.scene_width,
            "height": plan.scene_height,
            "delivery_width": plan.width,
            "delivery_height": plan.height,
            "seed": (shot.seed + attempt - 1) & 0x7FFFFFFF,
            "worker_slot": worker_slot,
            "gpu_id": gpu_id or "",
            "target_job_seconds": self.config.target_job_seconds,
        }
        argv = format_template_argv(self.template, values)
        environment = {"CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id is not None else None
        run_command(argv, timeout=timeout, log_path=log_path, env_overrides=environment)
        ensure_existing_file(output, "Render adapter output")
        return raw_frames


class MockRenderAdapter(RenderAdapter):
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config

    def render(
        self,
        *,
        shot: Shot,
        plan: EditorialPlan,
        audio: AudioInfo,
        output: Path,
        attempt: int,
        worker_slot: int,
        gpu_id: str | None,
        timeout: int,
        log_path: Path,
    ) -> int:
        raw_frames = ltx_window_frames(shot.frame_count)
        vf = (
            f"scale={plan.scene_width}:{plan.scene_height}:force_original_aspect_ratio=increase,"
            f"crop={plan.scene_width}:{plan.scene_height},zoompan=z='min(zoom+0.0004,1.03)':"
            f"d={raw_frames}:s={plan.scene_width}x{plan.scene_height}:fps={plan.fps},format=yuv420p"
        )
        run_command(
            [
                self.config.ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-loop",
                "1",
                "-i",
                shot.anchor_image,
                "-vf",
                vf,
                "-frames:v",
                str(raw_frames),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            timeout=timeout,
            log_path=log_path,
        )
        return raw_frames


def make_render_adapter(config: WorkerConfig, backend: str | None = None) -> RenderAdapter:
    selected = backend or config.render_backend
    if selected == "ltx":
        return LtxRenderAdapter(config)
    if selected == "mock":
        return MockRenderAdapter(config)
    if selected == "command" and config.render_command:
        return CommandRenderAdapter(config.render_command, config)
    raise ValidationError(f"Render backend is not configured: {selected}")


def generate_anchors(
    *,
    plan: EditorialPlan,
    request: MusicVideoRequest,
    treatment: dict,
    output_dir: Path,
    adapter: AnchorAdapter,
    timeout: int,
    identity_references: dict[str, list[Path]] | None = None,
    concurrency: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    references_by_performer: dict[str, list[Path]] = {}
    for performer in request.performers:
        references_by_performer[performer.id] = [
            ensure_existing_file(item, f"Reference for {performer.id}") for item in performer.reference_images
        ]
    for performer_id, paths in (identity_references or {}).items():
        references_by_performer[performer_id] = [
            ensure_existing_file(item, f"Identity reference for {performer_id}") for item in paths
        ]

    shot_keys: dict[str, tuple[str, str, tuple[str, ...], str]] = {}
    jobs: dict[tuple[str, str, tuple[str, ...], str], tuple[Shot, Path, list[Path], dict[str, Any]]] = {}
    for shot in plan.shots:
        performer_key = tuple(shot.performer_ids)
        key = (shot.section, shot.anchor_role, performer_key, shot.lyric_visual_direction)
        shot_keys[shot.id] = key
        if key in jobs:
            continue
        suffix = f"-{'-'.join(performer_key)}" if performer_key else ""
        output = output_dir / f"{shot.id}-{shot.section}-{shot.anchor_role}{suffix}.png"
        refs: list[Path] = []
        for performer_id in performer_key:
            refs.extend(references_by_performer.get(performer_id, []))
        payload = {
            "schema_version": 1,
            "anchor_role": shot.anchor_role,
            "format": plan.format,
            "width": plan.scene_width,
            "height": plan.scene_height,
            "delivery_width": plan.width,
            "delivery_height": plan.height,
            "performer_ids": list(performer_key),
            "references": [str(item) for item in refs],
            "prompt": anchor_prompt(shot, request, treatment, plan.format),
            "output": str(output),
        }
        jobs[key] = (shot, output, refs, payload)

    def generate_one(
        key: tuple[str, str, tuple[str, ...], str],
        job: tuple[Shot, Path, list[Path], dict[str, Any]],
    ) -> tuple[tuple[str, str, tuple[str, ...], str], Path]:
        _, output, refs, payload = job
        if not output.exists():
            adapter.generate(
                request_payload=payload,
                output=output,
                width=plan.scene_width,
                height=plan.scene_height,
                references=refs,
                timeout=timeout,
                log_path=output.with_suffix(".command.json"),
            )
        with Image.open(output) as image:
            if image.size != (plan.scene_width, plan.scene_height):
                raise ValidationError(
                    f"Anchor {output} is {image.size[0]}x{image.size[1]}, expected {plan.scene_width}x{plan.scene_height}"
                )
            image.verify()
        return key, output.resolve()

    cache: dict[tuple[str, str, tuple[str, ...], str], Path] = {}
    if concurrency == 1:
        for key, job in jobs.items():
            resolved_key, output = generate_one(key, job)
            cache[resolved_key] = output
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="zolex-anchor") as pool:
            futures = [pool.submit(generate_one, key, job) for key, job in jobs.items()]
            for future in as_completed(futures):
                key, output = future.result()
                cache[key] = output
    for shot in plan.shots:
        shot.anchor_image = str(cache[shot_keys[shot.id]])
