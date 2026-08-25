"""Stage-1 GPU smoke: the H3ComfyAdapter against the real pinned service.

The exact counterpart of `ltx_smoke.py`: every seam around the runtime is
already proven by the worker suite against a fake ComfyUI; this swaps the fake
for the real service and nothing else. It builds an `AdapterJob` by hand and
calls `H3ComfyAdapter().run()` directly — no API, no database, no storage —
demonstrating the contract the runner relies on:

  * the frozen client graph compiles and the service accepts it,
  * staged inputs reach ComfyUI's input directory and the LoadImage nodes,
  * the disciplined prompts reach the Extender,
  * the Final Decode writes into the job workspace,
  * the finished file passes the same probe a real job's would.

Usage, from the worker checkout on the GPU node:

    # image animation (I2V, FL2VA)
    MODE=i2v IMAGE=/path/face.png DURATION=5s python scripts/h3_smoke.py he speaks calmly

    # reference video (R2V): identity photo + optional source video
    MODE=refv2v REFERENCE=/path/person.png VIDEO=/path/perf.mp4 DURATION=5s \
        python scripts/h3_smoke.py he sings at the microphone

    # 60 s long-form with the generated prompt discipline
    MODE=refv2v REFERENCE=/path/person.png VIDEO=/path/perf.mp4 DURATION=60s \
        TIER=draft python scripts/h3_smoke.py ...

Environment: H3_COMFY_BASE_URL / H3_COMFY_INPUT_DIR / H3_COMFY_MODELS_DIR as
in worker/core/config.py; TIER=draft|quality; DURATION from the pack presets
(5s/10s/15s/30s/60s).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from worker.adapters.base import AdapterInput, AdapterJob
from worker.adapters.h3_comfy import H3ComfyAdapter, h3_comfy_health


def _staged(role: str, variable: str, *, kind: str, content_type: str, required: bool) -> AdapterInput | None:
    value = os.getenv(variable)
    if not value:
        if required:
            raise SystemExit(f"{variable} is required for this mode")
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{variable} not found: {path}")
    return AdapterInput(
        role=role, kind=kind, content_type=content_type,
        download_url="file://" + str(path), path=path,
    )


async def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        raise SystemExit("usage: MODE=i2v|refv2v ... python scripts/h3_smoke.py <prompt>")

    mode = os.getenv("MODE", "i2v")
    inputs: list[AdapterInput] = []
    if mode == "i2v":
        workflow = "image-to-video"
        inputs.append(_staged("source_image", "IMAGE", kind="image", content_type="image/png", required=True))
    elif mode == "refv2v":
        workflow = "video-to-video"
        inputs.append(_staged("reference_image", "REFERENCE", kind="image", content_type="image/png", required=True))
        video = _staged("source_video", "VIDEO", kind="video", content_type="video/mp4", required=False)
        if video is not None:
            inputs.append(video)
    else:
        raise SystemExit(f"unknown MODE {mode!r} (i2v | refv2v)")

    execution: dict[str, object] = {"runtime": "h3_comfy"}
    if os.getenv("STEPS"):
        execution["h3_steps"] = int(os.environ["STEPS"])
    if os.getenv("TIER"):
        execution["h3_tier"] = os.environ["TIER"]

    ok, detail = await h3_comfy_health()
    print(f"health: {'OK' if ok else 'UNAVAILABLE'} — {detail}")
    if not ok:
        raise SystemExit(2)

    workspace = Path(tempfile.mkdtemp(prefix="h3-smoke-"))
    job = AdapterJob(
        job_id=f"smoke{int(time.time())}",
        workflow_id=workflow,
        workflow_version="1",
        prompt=prompt,
        parameters={"duration": os.getenv("DURATION", "5s")},
        inputs=inputs,
        execution=execution,
        output_content_type="video/mp4",
        workspace=workspace,
    )

    async def on_progress(status: str, progress: int, message: str, details=None) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {status:>15} {progress:3d}%  {message}", flush=True)

    began = time.monotonic()
    result = await H3ComfyAdapter().run(job, on_progress)
    wall = time.monotonic() - began

    print("-" * 60)
    print(f"file:      {result.path}")
    print(f"type:      {result.content_type} ({result.kind})")
    print(f"duration:  {result.duration_seconds}s (measured by ffprobe)")
    print(f"frame:     {result.width}x{result.height}")
    print(f"size:      {result.size_bytes // 1024} KiB")
    print(f"wall time: {wall:.1f}s")
    print("SMOKE TEST PASSED — the adapter and the service agree.")
    print(f"Keep or inspect the output at: {result.path}")


asyncio.run(main())
