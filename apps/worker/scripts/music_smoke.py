"""Stage-1 smoke test for the music runtime, against whatever model is wired.

The music adapter reaches its model through one seam — a
`MusicGenerationProvider`. This script drives the adapter directly, with no API,
no database, no storage and no workflow routing, so the only thing it exercises
that the test suite cannot is the real service.

Everything else — minute-based length, per-genre song structure, lyric density,
sectioning, crossfading, loudness matching, the refusal to ship a repeated
section, final validation — is already covered by `tests/test_music.py` against
a fake provider, and behaves identically here.

Prerequisite: the music service must already be running and holding its weights.
On the GPU node that is:

    cd /workspace/acestep-benchmark
    ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
    uv run acestep-api --host 127.0.0.1 --port 8001

Usage:

    python scripts/music_smoke.py an upbeat pop song about summer
    DURATION=3m python scripts/music_smoke.py a slow piano ballad about leaving
    LYRICS=/path/to/lyrics.txt python scripts/music_smoke.py …
    ACESTEP_BASE_URL=http://127.0.0.1:8001 python scripts/music_smoke.py …
    MAX_SEGMENT_SECONDS=30 python scripts/music_smoke.py …   # force sectioning

With the service down the script stops immediately with the same refusal a real
job would produce — which is correct behaviour, not a fault in the script.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from worker.adapters.base import AdapterError, AdapterJob
from worker.adapters.music import MusicAdapter
from worker.music import LyricBrief, plan_song


async def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "an upbeat pop song about summer in the city"
    duration = os.getenv("DURATION", "1m")

    execution: dict[str, object] = {"runtime": "music"}
    if os.getenv("MAX_SEGMENT_SECONDS"):
        execution["max_segment_seconds"] = int(os.environ["MAX_SEGMENT_SECONDS"])

    parameters: dict[str, object] = {"duration": duration}
    # The controls a customer can set. Passed through here so the smoke test
    # exercises the same parameter path a real job takes rather than a
    # simplified one.
    for name, key in (("BPM", "bpm"), ("KEY", "key")):
        if os.getenv(name):
            parameters[key] = os.environ[name]
    if os.getenv("INSTRUMENTAL"):
        parameters["instrumental"] = True

    lyrics_file = os.getenv("LYRICS")
    if lyrics_file:
        path = Path(lyrics_file).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"LYRICS not found: {path}")
        parameters["lyrics"] = path.read_text(encoding="utf-8")

    workspace = Path(tempfile.mkdtemp(prefix="music-smoke-"))
    job = AdapterJob(
        job_id=f"smoke-{int(time.time())}",
        workflow_id="music",
        workflow_version="1",
        prompt=prompt,
        parameters=parameters,
        inputs=[],
        execution=execution,
        output_content_type="audio/mpeg",
        workspace=workspace,
    )

    # Printed before anything runs, because the plan is the part that is
    # already finished and worth eyeballing even if the provider is missing.
    brief = LyricBrief.from_prompt(prompt)
    seconds = MusicAdapter()._requested_seconds(job)
    plan = plan_song(seconds, genre=brief.genre)

    print(f"prompt:    {prompt}")
    print(f"length:    {duration} ({seconds:.0f}s)")
    print(f"genre:     {plan.genre}")
    print(f"structure: {plan.outline}")
    # The constraint measured on the GPU: more lines than this and the model
    # drops some without saying which.
    print(f"lines:     at most {plan.line_budget} (~{plan.lines_per_section}/section)")
    print(f"keep:      {brief.must_keep or '—'}")
    print(f"workspace: {workspace}")
    print("-" * 60)

    async def on_progress(status: str, progress: int, message: str) -> None:
        print(
            f"[{time.strftime('%H:%M:%S')}] {status:>15} {progress:3d}%  {message}",
            flush=True,
        )

    started = time.monotonic()
    try:
        result = await MusicAdapter().run(job, on_progress)
    except AdapterError as exc:
        print("-" * 60)
        print(f"REFUSED: {exc.user_message}")
        print(f"reason:  {exc.internal_detail}")
        return 1
    elapsed = time.monotonic() - started

    print("-" * 60)
    print(f"file:      {result.path}")
    print(f"type:      {result.content_type} ({result.kind})")
    print(f"duration:  {result.duration_seconds}s (measured by ffprobe)")
    print(f"size:      {result.size_bytes / 1024:.0f} KiB")
    print(f"wall time: {elapsed:.1f}s")
    print()
    print("SMOKE TEST PASSED — the adapter and the provider agree.")
    print(f"Listen to it at: {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
