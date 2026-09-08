"""The music-video worker's render command: one shot, on either engine.

The client's package (`zolex_music_worker`) calls this once per shot attempt
with a request JSON and expects a decodable video at `--output`; it then
trims and normalises the clip itself. The request is the package's
`CommandRenderAdapter` contract: the shot (prompt, anchor picture, seed,
delivered frame count), the working size, the conditioning master and the
shot's offset into it, and the raw 8k+1 frame window to render.

Two engines answer that contract, chosen by `MUSIC_VIDEO_RENDER_ENGINE`.

**cli** is the package's own path, the official
`ltx_pipelines.a2vid_two_stage` process: the development transformer with
the distilled LoRA, unquantized, 24 steps, this node's six model files, and
the anchor pinned at frame 0. It frees ComfyUI's VRAM first, because the
anchor stage left an image model warm there. Measured 96 s per 121-frame
shot, about 35 s of which is starting the process and loading 51 GB.

**comfy** submits Lightricks' own audio-to-video graph to the ComfyUI this
node already runs (`worker/comfy/ltx_a2v.py`): the distilled transformer,
8 steps, then the 2x latent upscale and a 3-step refine. The models stay
resident between shots, so the per-shot loading disappears. Measured 24.2 s
for the same shot. The song is conditioned on identically — encoded,
frozen, denoised against — so what differs is speed and look.

Usage:  mv_render.py --request REQUEST.json --output OUT.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def _snap(frames: int) -> int:
    """Up to the model's 8k+1 lattice."""
    return int(math.ceil(max(1, frames - 1) / 8) * 8 + 1)


def _stage(client, path: Path, prefix: str) -> str:
    """Puts a file where ComfyUI can load it, once, under a content name.

    Named by digest so the conditioning master is transferred on the first
    shot and recognised on the other forty. When the worker shares a
    filesystem with the service a copy is enough; otherwise it goes over
    HTTP like every other input.
    """
    from worker.core.config import settings

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"{prefix}_{digest}{path.suffix.lower()}"
    input_dir = settings.ltx_comfy_input_dir
    if input_dir:
        target = Path(input_dir) / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        return name
    return asyncio.run(client.upload_input(path, name=name))


def render_on_comfy(request: dict, output: Path) -> None:
    """One shot through Lightricks' audio-to-video graph on the warm ComfyUI."""
    from worker.adapters.base import AdapterJob
    from worker.comfy.client import ComfyClient, ComfyError
    from worker.comfy.ltx_a2v import compile_a2v, missing_nodes
    from worker.core.config import settings

    shot = request["shot"]
    fps = int(request.get("fps") or 24)
    delivered = int(request["delivered_frames"])
    frames = _snap(delivered) if settings.music_video_tight_frames else int(request["raw_frames"])
    attempt = int(request.get("attempt") or 1)
    seed = (int(shot.get("seed") or 0) + attempt - 1) & 0x7FFFFFFF

    client = ComfyClient(
        settings.ltx_comfy_base_url,
        request_timeout=settings.ltx_comfy_request_timeout,
        poll_seconds=settings.ltx_comfy_poll_seconds,
    )
    absent = missing_nodes(asyncio.run(client.node_classes()))
    if absent:
        raise SystemExit(f"mv_render: this ComfyUI lacks the nodes {absent}")

    # The anchor stage left an image model resident. Free once per job, on
    # the first shot, rather than before every shot: the whole gain here is
    # that the video model stays loaded from one shot to the next.
    marker = output.parents[3] / ".comfy-freed"
    if not marker.exists():
        _free_comfy()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("freed before the first shot\n", encoding="utf-8")

    anchor = _stage(client, Path(shot["anchor_image"]), "zolex_mv_anchor")
    audio = _stage(client, Path(request["audio"]), "zolex_mv_audio")

    api = compile_a2v(
        positive=str(shot["prompt"]),
        negative=str(settings.music_video_negative_prompt),
        audio=audio,
        image=anchor,
        audio_start_seconds=float(request.get("audio_start_seconds") or 0.0),
        seconds=frames / fps,
        frames=frames,
        width=int(request["width"]),
        height=int(request["height"]),
        fps=float(fps),
        seed=seed,
        filename_prefix=f"zolexai/mv/{shot['id']}-{seed}",
        image_strength=float(settings.music_video_image_strength),
    )

    started = time.monotonic()
    _log("starting", engine="comfy", shot=shot.get("id"), frames=frames,
         delivered=delivered, size=f"{request['width']}x{request['height']}", attempt=attempt)
    job = AdapterJob(
        job_id=str(shot["id"]),
        workflow_id="music-video",
        workflow_version="1",
        prompt=str(shot["prompt"]),
        parameters={},
    )
    try:
        prompt_id = asyncio.run(client.submit(api, client_id=f"mv-{shot['id']}"))
        history = asyncio.run(
            client.wait(job, prompt_id, timeout_seconds=settings.ltx_comfy_generation_timeout)
        )
    except ComfyError as exc:
        raise SystemExit(f"mv_render: {exc.internal_detail}") from exc

    found = None
    for node_output in (history.get("outputs") or {}).values():
        for item in (
            (node_output.get("videos") or [])
            + (node_output.get("gifs") or [])
            + (node_output.get("images") or [])
        ):
            found = item
    if not found:
        raise SystemExit("mv_render: ComfyUI finished without a video output")
    asyncio.run(
        client.download_output(
            filename=str(found.get("filename")),
            subfolder=str(found.get("subfolder") or ""),
            output_type=str(found.get("type") or "output"),
            dest=output,
        )
    )
    _log("finished", engine="comfy", shot=shot.get("id"),
         seconds=round(time.monotonic() - started, 1), prompt_id=prompt_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(str(request.get("output") or args.output)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    from worker.core.config import settings

    if settings.music_video_render_engine == "comfy":
        render_on_comfy(request, output)
        if not output.is_file():
            raise SystemExit(f"mv_render: no file at {output}")
        return

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
