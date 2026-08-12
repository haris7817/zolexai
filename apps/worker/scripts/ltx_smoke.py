"""Stage-1 GPU smoke test: the LtxAdapter against the real model, no platform.

Every seam around the model is already proven by tests/test_ltx.py with a stub
render. This script swaps the stub for the real thing and nothing else: it
builds an `AdapterJob` by hand and calls `LtxAdapter().run()` directly — no
API, no database, no object storage, no tunnel. What it demonstrates, on a GPU
node, is the exact contract the runner will rely on:

  * the generated command line actually launches the pipeline,
  * the progress markers parse real pipeline output in order,
  * the finished file verifies against the requested duration,
  * the result carries measured (not asserted) metadata.

Usage, from the worker checkout on the GPU node:

    python scripts/ltx_smoke.py                      # 10s text-to-video
    python scripts/ltx_smoke.py a koi pond at dawn   # custom prompt
    IMAGE=/path/to/still.png python scripts/ltx_smoke.py gentle camera push in
                                                     # image-to-video
    VIDEO=/path/to/clip.mp4 DURATION=5s python scripts/ltx_smoke.py the scene continues
                                                     # extend-video

Environment: LTX_REPO_DIR if the LTX checkout is not /workspace/ltx2-benchmark;
DURATION (e.g. "5s"), ASPECT_RATIO (e.g. "9:16") to vary the request; IMAGE to
condition on a still (image-to-video); VIDEO to extend a clip (DURATION is then
the EXTENSION length, and the output is source + continuation stitched).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from worker.adapters.base import AdapterInput, AdapterJob
from worker.adapters.ltx import LtxAdapter


async def main() -> int:
    prompt = " ".join(sys.argv[1:]) or (
        "A slow cinematic dolly shot through a neon-lit city at dusk, rain reflections"
    )
    image = os.getenv("IMAGE")
    video = os.getenv("VIDEO")
    if image and video:
        print("Set IMAGE or VIDEO, not both.")
        return 1

    inputs = []
    workflow_id = "text-to-video"
    if image:
        still = Path(image).expanduser().resolve()
        if not still.is_file():
            print(f"IMAGE not found: {still}")
            return 1
        workflow_id = "image-to-video"
        inputs = [
            AdapterInput(
                role="source_image",
                kind="image",
                content_type="image/png",
                download_url="file://smoke-test",
                path=still,
            )
        ]
    elif video:
        clip = Path(video).expanduser().resolve()
        if not clip.is_file():
            print(f"VIDEO not found: {clip}")
            return 1
        workflow_id = "extend-video"
        inputs = [
            AdapterInput(
                role="source_video",
                kind="video",
                content_type="video/mp4",
                download_url="file://smoke-test",
                path=clip,
            )
        ]

    workspace = Path(tempfile.mkdtemp(prefix="ltx-smoke-"))
    job = AdapterJob(
        # Varies per run so the seed derived from it varies too.
        job_id=f"smoke-{int(time.time())}",
        workflow_id=workflow_id,
        workflow_version="1",
        prompt=prompt,
        parameters={
            "duration": os.getenv("DURATION", "10s"),
            "aspect_ratio": os.getenv("ASPECT_RATIO", "16:9"),
            "quality": "High",
        },
        inputs=inputs,
        execution={"runtime": "ltx"},
        output_content_type="video/mp4",
        workspace=workspace,
    )

    print(f"mode:      {job.workflow_id}")
    print(f"prompt:    {prompt}")
    if inputs:
        print(f"input:     {inputs[0].path}")
    print(f"request:   {job.parameters['duration']} @ {job.parameters['aspect_ratio']}")
    print(f"workspace: {workspace}")
    print("-" * 60)

    async def on_progress(status: str, progress: int, message: str) -> None:
        print(
            f"[{time.strftime('%H:%M:%S')}] {status:>15} {progress:3d}%  {message}",
            flush=True,
        )

    started = time.monotonic()
    result = await LtxAdapter().run(job, on_progress)
    elapsed = time.monotonic() - started

    print("-" * 60)
    print(f"file:      {result.path}")
    print(f"type:      {result.content_type} ({result.kind})")
    print(f"duration:  {result.duration_seconds}s (measured by ffprobe)")
    print(f"frame:     {result.width}x{result.height}")
    print(f"size:      {result.size_bytes / 1024:.0f} KiB")
    print(f"wall time: {elapsed:.1f}s")
    print()
    print("SMOKE TEST PASSED — the adapter and the model agree.")
    print(f"Keep or inspect the output at: {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
