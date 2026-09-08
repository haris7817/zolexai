"""The music-video worker's render command: one shot on the audio tier.

The client's package (`zolex_music_worker`) calls this once per shot attempt
with a request JSON and expects a decodable video at `--output`; it then
trims and normalises the clip itself. The request is the package's
`CommandRenderAdapter` contract: the shot (prompt, anchor picture, seed,
delivered frame count), the working size, the conditioning master and the
shot's offset into it, and the raw 8k+1 frame window to render.

Why a command of our own rather than the package's built-in direct LTX
adapter, which launches the same CLI: the shot before this one may have
been an anchor still, and ComfyUI keeps that model warm. This command frees
ComfyUI's VRAM (cheap when it is already empty) before the 22B audio tier
takes the card, and it is the one seam where a warm render service can
replace the per-shot process later without touching the package.

The pipeline is the official `ltx_pipelines.a2vid_two_stage` — the dev
transformer with the distilled LoRA, unquantized, this node's six model
files — exactly the flags the platform's own audio tier sends
(`adapters/ltx.py`, `_A2VID`), plus the package's own image conditioning:
the anchor pinned at frame 0 at full strength.

Usage:  mv_render.py --request REQUEST.json --output OUT.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _log(message: str, **fields: object) -> None:
    print(json.dumps({"mv_render": message, **fields}), file=sys.stderr, flush=True)


def _free_comfy() -> None:
    """Best effort: an unreachable or empty ComfyUI is not this shot's problem."""
    try:
        from worker.comfy.client import ComfyClient
        from worker.core.config import settings

        client = ComfyClient(settings.ltx_comfy_base_url, request_timeout=10.0)
        asyncio.run(client.free_memory())
    except Exception as exc:  # noqa: BLE001
        _log("comfy_free_skipped", error=str(exc)[:200])


def build_argv(request: dict, output: Path) -> tuple[list[str], Path]:
    from worker.core.config import settings

    shot = request["shot"]
    root = settings.ltx_models_root
    python = settings.ltx_repo_dir / ".venv" / "bin" / "python"
    interpreter = str(python) if python.is_file() else "python"
    attempt = int(request.get("attempt") or 1)
    seed = (int(shot.get("seed") or 0) + attempt - 1) & 0x7FFFFFFF
    fps = int(request.get("fps") or 24)
    start = float(request.get("audio_start_seconds") or 0.0)

    argv = [
        interpreter,
        "-u",
        "-m",
        "ltx_pipelines.a2vid_two_stage",
        "--transformer-path",
        str(root / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"),
        "--text-encoder-path",
        str(root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "--video-vae-path",
        str(root / "vae/ltx-2.5-video-vae-bf16.safetensors"),
        "--audio-vae-path",
        str(root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        "--spatial-upsampler-path",
        str(root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    ]
    duration_head = root / "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    if duration_head.is_file():
        argv += ["--duration-head-path", str(duration_head)]
    offload = str(settings.ltx_unquantized_offload or "cpu").strip()
    extra = json.loads(settings.music_video_ltx_extra_args or "[]")
    if offload != "none" and "--offload" not in extra:
        argv += ["--offload", offload]
    argv += [
        "--distilled-lora",
        str(root / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
        "1.0",
        "--prompt",
        str(shot["prompt"]),
        "--negative-prompt",
        str(settings.music_video_negative_prompt),
        "--image",
        str(shot["anchor_image"]),
        "0",
        "1.0",
        "--audio-path",
        str(request["audio"]),
        "--audio-start-time",
        f"{start:.12f}",
        "--num-frames",
        str(int(request["raw_frames"])),
        "--frame-rate",
        str(fps),
        "--width",
        str(int(request["width"])),
        "--height",
        str(int(request["height"])),
        "--seed",
        str(seed),
        "--num-inference-steps",
        str(int(settings.music_video_inference_steps)),
        "--output-path",
        str(output),
    ]
    a2v = settings.music_video_a2v_guidance_scale
    if a2v is not None:
        argv += ["--a2v-guidance-scale", str(float(a2v))]
    argv += [str(item) for item in extra]
    return argv, settings.ltx_repo_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(str(request.get("output") or args.output)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    _free_comfy()
    argv, cwd = build_argv(request, output)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The worker's own CUDA-12 wheels (faster-whisper's cuBLAS/cuDNN) must
    # not shadow the LTX environment's torch libraries: with them on the
    # path the text encoder fails at its first attention call.
    scrubbed = [
        entry
        for entry in env.get("LD_LIBRARY_PATH", "").split(":")
        if entry and "site-packages" not in entry
    ]
    if scrubbed:
        env["LD_LIBRARY_PATH"] = ":".join(scrubbed)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    gpu = request.get("gpu_id")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path = output.with_suffix(".ltx.log")
    started = time.monotonic()
    _log("starting", shot=request["shot"].get("id"), frames=request.get("raw_frames"),
         size=f"{request.get('width')}x{request.get('height')}", attempt=request.get("attempt"))
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    _log(
        "finished",
        shot=request["shot"].get("id"),
        returncode=completed.returncode,
        seconds=round(time.monotonic() - started, 1),
        log=str(log_path),
    )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        print(tail, file=sys.stderr)
        raise SystemExit(completed.returncode)
    if not output.is_file():
        raise SystemExit(f"mv_render: the pipeline exited 0 but wrote no file at {output}")


if __name__ == "__main__":
    main()
