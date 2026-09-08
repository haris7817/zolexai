"""The bridge to the client's music-video worker (v1.8.0, 8 Sep 2026).

The client delivered a complete orchestrator — `zolexai-music-video-worker`,
vendored byte-for-byte as the top-level package `zolex_music_worker` beside
this one — that turns a song plus a short direction into a planned, shot-by-
shot, resumable render: song analysis, lyric transcription, up to five
performers with their own reference pictures and roles, anchor stills per
shot, an audio-conditioned LTX pass per shot, assembly, one 4K finishing
pass and delivery QA. Their package is never edited; everything platform-
specific lives here and in `worker.adapters.music_video`.

What this module does is translate:

  * a claimed ZolexAI job (prompt, `aspect_ratio`, the staged audio, the
    `performer_N` picture inputs and the `performers` parameter) into the
    request JSON their worker accepts (`build_request`);
  * this node's settings into their `WorkerConfig` (`build_config`), pointing
    every backend they leave to the deployer at the services this node has —
    the LTX environment for rendering, ComfyUI for anchor stills, faster-
    whisper for lyrics, FFmpeg CUDA for the 4K finish;
  * their `status.json` states into the platform's progress vocabulary
    (`progress_for`), so the customer's bar moves per shot.

The vendored package is imported lazily and only here, so a node that does
not serve this runtime never needs numpy, Pillow or faster-whisper.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from worker.adapters.base import AdapterJob
from worker.longform.progress import GENERATE_FROM, GENERATE_TO

#: Input roles that carry a performer's reference picture: `performer_1` …
#: `performer_5`. The number is the performer's slot, and it pairs the picture
#: with the same slot in the `performers` parameter (role, description).
PERFORMER_ROLE = re.compile(r"^performer_([1-9])$")

#: The band size the client's package supports.
MAX_PERFORMERS = 5

#: The lyric-language names the platform's picker offers, as the ISO codes
#: faster-whisper wants. An unknown name is passed through unchanged — the
#: transcriber then treats it as a code, and a bad code is refused with the
#: language named rather than silently detected.
LANGUAGE_CODES: dict[str, str] = {
    "english": "en",
    "urdu": "ur",
    "hindi": "hi",
    "arabic": "ar",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "turkish": "tr",
    "russian": "ru",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
}


def ensure_vendored_package() -> None:
    """Makes `zolex_music_worker` importable from a plain file copy.

    The GPU node's checkout is a `git archive | tar` copy, and its editable
    install was made before the vendored package existed — setuptools'
    editable finder only knows the top-level packages present at install
    time. The package sits beside `worker/` in the same tree, so the tree's
    root goes on the path when the import would otherwise fail. A proper
    reinstall (`uv pip install -e ".[music-video]"`) makes this a no-op.
    """
    try:
        import zolex_music_worker  # noqa: F401
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if (root / "zolex_music_worker" / "__init__.py").is_file():
            sys.path.insert(0, str(root))


def language_code(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = str(name).strip()
    return LANGUAGE_CODES.get(cleaned.casefold(), cleaned) or None


# ── Request ──────────────────────────────────────────────────────────────


def performers_from(job: AdapterJob) -> list[dict[str, Any]]:
    """The band, as the client's `performers` array.

    A performer exists when the job carries EITHER a picture in its slot
    (`performer_N` input) OR an entry for that slot in the `performers`
    parameter — a described member with no photo is allowed by the package
    (it generates an identity portrait from the description). Slots are
    ordered, so `performer-1` is always the first member the customer named.
    """
    described: dict[int, dict[str, Any]] = {}
    for item in job.parameters.get("performers") or []:
        if not isinstance(item, dict):
            continue
        try:
            slot = int(item.get("slot"))
        except (TypeError, ValueError):
            continue
        if 1 <= slot <= MAX_PERFORMERS:
            described[slot] = item

    pictures: dict[int, Path] = {}
    for item in job.inputs:
        match = PERFORMER_ROLE.match(item.role)
        if match and item.path is not None:
            slot = int(match.group(1))
            if 1 <= slot <= MAX_PERFORMERS:
                pictures[slot] = Path(item.path)

    performers: list[dict[str, Any]] = []
    for slot in sorted(set(described) | set(pictures)):
        entry = described.get(slot, {})
        role = str(entry.get("role") or "performer").strip() or "performer"
        description = str(entry.get("description") or "").strip()
        performers.append(
            {
                "id": f"performer-{slot}",
                "role": role,
                "description": description,
                "reference_images": [str(pictures[slot])] if slot in pictures else [],
            }
        )
    return performers[:MAX_PERFORMERS]


def build_request(job: AdapterJob, audio_path: Path, *, lyric_mode: str) -> dict[str, Any]:
    """The request JSON the client's `MusicVideoRequest.from_dict` accepts."""
    aspect = str(job.parameters.get("aspect_ratio") or "16:9")
    lyrics = job.parameters.get("lyrics")
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "audio": str(audio_path),
        "prompt": job.prompt.strip(),
        "formats": [aspect],
        "performers": performers_from(job),
        "lyric_mode": str(job.execution.get("lyric_mode") or lyric_mode),
        # No external lip-sync command exists on this platform, so the flag
        # only records intent; the package reports `ltx_audio_conditioning_only`.
        "lip_sync": True,
    }
    if isinstance(lyrics, str) and lyrics.strip():
        payload["lyrics"] = lyrics.strip()
    code = language_code(job.parameters.get("lyrics_language"))
    if code:
        payload["lyrics_language"] = code
    return payload


# ── Config ───────────────────────────────────────────────────────────────


def _json_list(raw: str) -> list[str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, list) or not all(isinstance(x, str) and x for x in parsed):
        raise ValueError("expected a JSON array of non-empty strings")
    return parsed


def build_config(
    job: AdapterJob,
    *,
    work_root: Path,
    settings: Any,
    anchor_command: list[str],
    render_command: list[str],
    upscale_command: list[str],
    ltx_python: str,
    ltx_models_root: Path,
) -> Any:
    """This node's settings as the client's `WorkerConfig`.

    Everything the package leaves to the deployer is decided here:

      * **render**: `scripts/mv_render.py`, which evicts ComfyUI and then
        runs the official `ltx_pipelines.a2vid_two_stage` (dev transformer
        + distilled LoRA, the same audio tier the platform already runs)
        with this node's six model files and its measured offload setting;
        their own built-in direct adapter does the same without the
        eviction; or a command renderer the deployment names (a warm
        service, their recommended path at scale).
      * **anchors**: a command that renders each shot's starting still on
        this node's ComfyUI (`scripts/mv_anchor.py`).
      * **lyrics**: faster-whisper in-process, the model this deployment
        caches; `disabled` makes a lyric-driven prompt fail early with the
        package's own message rather than render an unrelated mood video.
      * **finish**: `scripts/mv_upscale.py`, the package's own FFmpeg CUDA
        Lanczos and NVENC recipe with the frame count stated rather than
        `-shortest`, which is what makes a long job land on its exact
        length.

    Per-job `execution` keys override the shot ladder and the step count,
    the same shape the other runtimes use.
    """
    ensure_vendored_package()
    from zolex_music_worker.config import WorkerConfig

    steps = job.execution_int("inference_steps", int(settings.music_video_inference_steps))
    extra = _json_list(settings.music_video_ltx_extra_args) or []
    offload = str(settings.ltx_unquantized_offload or "cpu").strip()
    if offload != "none" and "--offload" not in extra:
        # The unquantized 22B transformer fits this card the way the
        # platform's own audio tier fits it (adapters/ltx.py): streamed from
        # host RAM unless the node says it has the room.
        extra = ["--offload", offload, *extra]

    render_backend = str(job.execution.get("render_backend") or settings.music_video_render_backend)
    render_command = _json_list(settings.music_video_render_command) or render_command
    anchor_backend = str(job.execution.get("anchor_backend") or settings.music_video_anchor_backend)
    upscale_backend = str(settings.music_video_upscale_backend or "cuda").strip().casefold()
    download_root = settings.music_video_whisper_download_root

    return WorkerConfig(
        work_root=work_root,
        fps=24,
        max_source_seconds=int(settings.music_video_max_source_seconds),
        shot_target_seconds=job.execution_float(
            "shot_target_seconds", float(settings.music_video_shot_target_seconds)
        ),
        shot_min_seconds=job.execution_float(
            "shot_min_seconds", float(settings.music_video_shot_min_seconds)
        ),
        shot_max_seconds=job.execution_float(
            "shot_max_seconds", float(settings.music_video_shot_max_seconds)
        ),
        max_attempts=int(settings.music_video_max_attempts),
        command_timeout_seconds=int(settings.music_video_command_timeout),
        render_concurrency=1,
        # One lane, named, so the render command is handed a GPU id and the
        # package's own validator sees a lane per worker.
        render_gpu_ids=("0",),
        anchor_concurrency=1,
        enforce_latency_capacity=False,
        anchor_backend=anchor_backend,
        anchor_command=anchor_command if anchor_backend == "command" else None,
        render_backend=render_backend,
        render_command=render_command if render_backend == "command" else None,
        upscale_backend=upscale_backend,
        upscale_command=upscale_command if upscale_backend == "command" else None,
        upscale_gpu_id="0",
        upscale_timeout_seconds=int(settings.music_video_upscale_timeout),
        upscale_cq=int(settings.music_video_upscale_cq),
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        transcription_backend=str(settings.music_video_transcription_backend),
        transcription_model=str(settings.music_video_whisper_model),
        transcription_device=str(settings.music_video_whisper_device),
        transcription_compute_type=str(settings.music_video_whisper_compute_type),
        transcription_download_root=Path(download_root) if download_root else None,
        transcription_beam_size=5,
        transcription_timeout_seconds=1800,
        # Reference-video style matching is part of the package but not of
        # this product's Music Video form; the vision model it wants is not
        # installed and nothing sends a reference, so it stays off.
        reference_vision_backend="disabled",
        reference_fetch_backend="command",
        reference_fetch_command=["false"],
        ltx_python=ltx_python,
        ltx_module="ltx_pipelines.a2vid_two_stage",
        ltx_transformer_path=ltx_models_root
        / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        ltx_text_encoder_path=ltx_models_root
        / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        ltx_video_vae_path=ltx_models_root / "vae/ltx-2.5-video-vae-bf16.safetensors",
        ltx_audio_vae_path=ltx_models_root / "vae/ltx-2.5-audio-vae-bf16.safetensors",
        ltx_spatial_upsampler_path=ltx_models_root
        / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        ltx_distilled_lora_path=ltx_models_root
        / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        ltx_distilled_lora_strength=1.0,
        ltx_num_inference_steps=steps,
        ltx_extra_args=extra or None,
    )


# ── Progress ─────────────────────────────────────────────────────────────

#: Where each of the package's job states sits on the customer's bar. The
#: generation band is spent on anchors (its first fifth) and shots (the
#: rest); everything after rendering is post-processing.
_PREPARING: dict[str, tuple[int, str]] = {
    "validating": (8, "Setting up your generation…"),
    "fetching_reference_video": (9, "Setting up your generation…"),
    "preparing_audio": (10, "Listening to your track…"),
    "analyzing_song_lyrics_and_reference": (12, "Listening to your track…"),
    "planning": (13, "Planning your video…"),
}
_ANCHOR_BAND = (GENERATE_FROM, GENERATE_FROM + (GENERATE_TO - GENERATE_FROM) // 5)
_POST: dict[str, tuple[int, str]] = {
    "assembling": (88, "Assembling your video…"),
    "upscaling": (91, "Finishing your video in 4K…"),
    "final_quality_control": (94, "Checking the finished video…"),
}


def progress_for(status: dict[str, Any]) -> tuple[str, int, str, dict[str, Any]] | None:
    """(status, progress, message, details) for one `status.json` reading,
    or None for a state that carries no customer-facing step (complete,
    failed, cancelled — the runner reports those itself)."""
    state = str(status.get("state") or "")
    if state in _PREPARING:
        progress, message = _PREPARING[state]
        return "preparing", progress, message, {"phase": "preparing"}
    if state == "building_identity_package":
        return "generating", _ANCHOR_BAND[0], "Creating your performers…", {"phase": "generating"}
    if state == "generating_anchors":
        return (
            "generating",
            _ANCHOR_BAND[0] + 1,
            "Composing your shots…",
            {"phase": "generating"},
        )
    if state == "rendering":
        total = int(status.get("shot_count") or 0)
        done = status.get("completed_shots")
        index = status.get("shot_index")
        if done is None and index is not None:
            done = max(0, int(index) - 1)
        done = int(done or 0)
        low, high = _ANCHOR_BAND[1], GENERATE_TO - 1
        fraction = (done / total) if total else 0.0
        progress = low + int((high - low) * min(1.0, fraction))
        current = int(index) if index is not None else min(total, done + 1)
        message = f"Rendering shot {current} of {total}…" if total else "Rendering your video…"
        details: dict[str, Any] = {"phase": "generating"}
        if total >= 1:
            # The API takes the four section fields TOGETHER or not at all
            # (`JobProgressRequest._coherent_section`), and rejects the whole
            # report with a 422 otherwise — which the worker reads as a lost
            # lease and fails the job on. Sending two of them failed every
            # music video the moment it started rendering (job 0c7bf9a4,
            # 8 Sep 2026: three attempts, three 422s, no video).
            #
            # A shot ladder has no song time range in `status.json` — only
            # counts — so the range is 0.0/0.0, which is exactly what
            # `StageReporter.section` sends for a caller that has none.
            # The index is clamped because the API also requires 1 <= index
            # <= total, and a status file mid-write can carry a shot number
            # ahead of the count.
            details |= {
                "section_index": min(max(1, current), total),
                "section_total": total,
                "section_start_seconds": 0.0,
                "section_end_seconds": 0.0,
            }
        return ("generating", progress, message, details)
    if state in _POST:
        progress, message = _POST[state]
        return "post_processing", progress, message, {"phase": "post_processing"}
    return None


# ── Whisper's CUDA libraries ─────────────────────────────────────────────


def prepare_whisper_libraries() -> list[str]:
    """Loads the pip-installed NVIDIA runtime libraries CTranslate2 needs.

    faster-whisper on CUDA loads cuBLAS and cuDNN through CTranslate2 by
    library name at first use. The `nvidia-*-cu12` wheels put them under
    `site-packages/nvidia/*/lib`, where the dynamic loader never looks, and
    this node's CUDA 13 system libraries do not satisfy a CUDA 12 build.
    `LD_LIBRARY_PATH` is read once, at process start, so setting it here is
    too late for this process (measured 8 Sep 2026: the model loaded, the
    first transcription died on `libcublas.so.12`). What works is opening
    each library into the process's global namespace first — a later
    `dlopen` by name then finds it already loaded. The path is deliberately
    NOT exported: it leaks into the LTX render subprocess, whose own torch
    carries a different cuDNN, and the text encoder then dies with
    `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` (measured the same day).
    Harmless when the wheels are absent; returns what was loaded.
    """
    if os.environ.get("ZOLEX_WHISPER_LIBS_READY"):
        return []
    import ctypes

    directories: list[str] = []
    for base in sys.path:
        if not base or "site-packages" not in base:
            continue
        directories.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib")))
    directories = list(dict.fromkeys(directories))

    # Dependencies first: cuBLAS needs cuBLASLt, cuDNN's sub-libraries need
    # the core one. Anything that fails to load is skipped, not fatal.
    order = ("libcublasLt", "libcublas", "libcudnn.so", "libcudnn_")
    loaded: list[str] = []
    for prefix in order:
        for directory in directories:
            for path in sorted(glob.glob(os.path.join(directory, f"{prefix}*.so*"))):
                if path in loaded:
                    continue
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                loaded.append(path)
    os.environ["ZOLEX_WHISPER_LIBS_READY"] = "1"
    return loaded


__all__ = [
    "LANGUAGE_CODES",
    "MAX_PERFORMERS",
    "PERFORMER_ROLE",
    "build_config",
    "build_request",
    "ensure_vendored_package",
    "language_code",
    "performers_from",
    "prepare_whisper_libraries",
    "progress_for",
]
