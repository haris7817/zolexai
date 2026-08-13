"""Stage-1 GPU smoke test: the LtxAdapter against the real model, no platform.

Every seam around the model is already proven by the worker suite with a stub
render. This script swaps the stub for the real thing and nothing else: it
builds an `AdapterJob` by hand and calls `LtxAdapter().run()` directly — no
API, no database, no object storage, no tunnel, no workflow routing. What it
demonstrates, on a GPU node, is the exact contract the runner relies on:

  * the generated command line actually launches the pipeline,
  * conditioning reaches the model the way the adapter believes it does,
  * the progress markers parse real pipeline output in order,
  * the finished file passes the same validation a real job's would,
  * the result carries measured (not asserted) metadata.

Because it bypasses routing entirely, **no workflow YAML has to be switched to
`runtime: ltx` to run any of this.**

Usage, from the worker checkout on the GPU node:

    # text to video
    python scripts/ltx_smoke.py a koi pond at dawn

    # image to video
    IMAGE=/path/still.png python scripts/ltx_smoke.py gentle camera push in

    # video extension — DURATION is the EXTENSION length
    MODE=extend VIDEO=/path/clip.mp4 DURATION=10s python scripts/ltx_smoke.py it continues

    # video to video — duration comes from the SOURCE, so DURATION is ignored
    MODE=restyle VIDEO=/path/clip.mp4 python scripts/ltx_smoke.py as a charcoal sketch
    MODE=restyle VIDEO=/path/clip.mp4 REFERENCE=/path/look.png python scripts/ltx_smoke.py …

    # music video — duration comes from the TRACK
    MODE=music-video AUDIO=/path/song.mp3 python scripts/ltx_smoke.py a dancer, hard side light

Environment: LTX_REPO_DIR if the checkout is not /workspace/ltx2-benchmark;
DURATION and ASPECT_RATIO to vary the request (both ignored by the
source-duration modes); MAX_SEGMENT_SECONDS to force chaining on a short input
so the multi-pass path can be exercised without a long upload.
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

#: MODE → (workflow id, the input role its source occupies, env var for it).
_MODES: dict[str, tuple[str, str | None, str | None]] = {
    "text": ("text-to-video", None, None),
    "image": ("image-to-video", "source_image", "IMAGE"),
    "extend": ("extend-video", "source_video", "VIDEO"),
    "restyle": ("video-to-video", "source_video", "VIDEO"),
    "music-video": ("music-video", "source_audio", "AUDIO"),
}

_CONTENT_TYPES = {
    "source_image": ("image", "image/png"),
    "source_video": ("video", "video/mp4"),
    "source_audio": ("audio", "audio/mpeg"),
    "reference_image": ("image", "image/png"),
}


def _resolve_mode() -> str:
    explicit = os.getenv("MODE")
    if explicit:
        if explicit not in _MODES:
            raise SystemExit(f"MODE must be one of {sorted(_MODES)}")
        return explicit
    # Convenience: the old IMAGE=/VIDEO= invocations keep working.
    if os.getenv("AUDIO"):
        return "music-video"
    if os.getenv("VIDEO"):
        return "extend"
    if os.getenv("IMAGE"):
        return "image"
    return "text"


def _staged(role: str, variable: str, *, required: bool) -> AdapterInput | None:
    raw = os.getenv(variable)
    if not raw:
        if required:
            raise SystemExit(f"{variable} is required for this mode")
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{variable} not found: {path}")
    kind, content_type = _CONTENT_TYPES[role]
    return AdapterInput(
        role=role,
        kind=kind,
        content_type=content_type,
        download_url="file://smoke-test",
        path=path,
    )


async def main() -> int:
    mode = _resolve_mode()
    workflow_id, role, variable = _MODES[mode]

    prompt = " ".join(sys.argv[1:]) or (
        "A slow cinematic dolly shot through a neon-lit city at dusk, rain reflections"
    )

    inputs: list[AdapterInput] = []
    if role and variable:
        staged = _staged(role, variable, required=True)
        assert staged is not None
        inputs.append(staged)
    if workflow_id == "video-to-video":
        reference = _staged("reference_image", "REFERENCE", required=False)
        if reference is not None:
            inputs.append(reference)

    execution: dict[str, object] = {"runtime": "ltx"}
    if os.getenv("MAX_SEGMENT_SECONDS"):
        # Forces the chaining path on a short input, so the multi-pass
        # behaviour can be checked without waiting on a two-minute source.
        execution["max_segment_seconds"] = int(os.environ["MAX_SEGMENT_SECONDS"])

    parameters: dict[str, object] = {"aspect_ratio": os.getenv("ASPECT_RATIO", "16:9")}
    # The source-duration workflows take no duration at all — passing one would
    # be testing a shape the API never sends.
    if workflow_id not in ("video-to-video", "music-video"):
        parameters["duration"] = os.getenv("DURATION", "10s")
        parameters["quality"] = "High"

    workspace = Path(tempfile.mkdtemp(prefix="ltx-smoke-"))
    job = AdapterJob(
        # Varies per run so the seed derived from it varies too.
        job_id=f"smoke-{int(time.time())}",
        workflow_id=workflow_id,
        workflow_version="1",
        prompt=prompt,
        parameters=parameters,
        inputs=inputs,
        execution=execution,
        output_content_type="video/mp4",
        workspace=workspace,
    )

    print(f"mode:      {mode} ({workflow_id})")
    print(f"prompt:    {prompt}")
    for item in inputs:
        print(f"input:     {item.role} = {item.path}")
    print(f"request:   {parameters}")
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
