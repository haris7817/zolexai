#!/usr/bin/env python3
"""Drive one real Video to Video job through the shipped adapter, on the GPU.

Not a test and not a fixture: this runs `LtxAdapter.run` exactly as the job
runner does, against a real source file, so the proxy render, the section
stitch, the audio restore and the delivery encode are all the production code
paths. It exists because the 10 Sep rework had never produced a single clip.

  python v2v_smoke.py --source CLIP.mp4 --delivery 1080p [--reference PHOTO]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

from worker.adapters.base import AdapterInput, AdapterJob
from worker.adapters.ltx import LtxAdapter
from worker.media import probe_media


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prompt", default=(
        "a charcoal sketch of the same scene, heavy graphite shading, "
        "white paper, the same people in the same places"
    ))
    parser.add_argument("--delivery", default="1080p",
                        choices=("native", "1080p", "4k", "8k"))
    parser.add_argument("--proxy", default="480p", choices=("480p", "off"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="trim the source to this many seconds first")
    parser.add_argument("--workspace", type=Path,
                        default=Path("/workspace/v2vtest/smoke"))
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    workspace = args.workspace
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    source = workspace / "source.mp4"
    if args.seconds > 0:
        from worker.media import ffmpeg
        await ffmpeg([
            "-i", str(args.source), "-t", f"{args.seconds:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
        ])
    else:
        shutil.copy(args.source, source)

    info = await probe_media(source)
    print(f"source     : {info.width}x{info.height}  "
          f"{info.duration_seconds:.2f}s  audio={info.has_audio}", flush=True)

    inputs = [AdapterInput(
        role="source_video", kind="video", content_type="video/mp4",
        download_url="file://local", path=source,
    )]
    if args.reference is not None:
        reference = workspace / "reference.png"
        shutil.copy(args.reference, reference)
        inputs.append(AdapterInput(
            role="reference_image", kind="image", content_type="image/png",
            download_url="file://local", path=reference,
        ))

    execution = {
        "runtime": "ltx",
        "timeout_seconds": 7200,
        "v2v_engine": "transform",
        "transform_pass_seconds": 8,
        "v2v_control_strength": 1.0,
        "v2v_lora_strength": 1.0,
        "v2v_continuity_strength": 0.85,
        "v2v_reference_strength": 0.30,
        "v2v_reference_identity": True,
        "v2v_identity_describe_reference": True,
        "v2v_identity_composited_anchor": True,
        "v2v_identity_anchor_strength": 1.0,
        "v2v_identity_refresh_strength": 0.30,
        "v2v_identity_subject_attention": 0.50,
        "delivery": args.delivery,
    }
    if args.proxy == "480p":
        execution["render_proxy"] = "480p"

    job = AdapterJob(
        job_id="v2v-smoke-0000-0000-000000000001",
        workflow_id="video-to-video",
        workflow_version="1",
        prompt=args.prompt,
        parameters={"quality": args.delivery},
        inputs=inputs,
        execution=execution,
        output_content_type="video/mp4",
        workspace=workspace,
    )

    last = {"pct": -1}

    async def on_progress(status, progress, message, _details=None):
        if progress != last["pct"]:
            last["pct"] = progress
            print(f"  [{progress:3d}%] {status:16s} {message}", flush=True)

    started = time.time()
    result = await LtxAdapter().run(job, on_progress)
    wall = time.time() - started

    out = await probe_media(workspace / "output.mp4")
    print(json.dumps({
        "wall_seconds": round(wall, 1),
        "delivery": args.delivery,
        "proxy": args.proxy,
        "reference": bool(args.reference),
        "result_size": [result.width, result.height],
        "file_size": [out.width, out.height],
        "source_seconds": round(info.duration_seconds, 2),
        "output_seconds": round(out.duration_seconds, 2),
        "has_audio": out.has_audio,
        "bytes": (workspace / "output.mp4").stat().st_size,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
