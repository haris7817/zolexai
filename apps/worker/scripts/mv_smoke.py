"""Run one Music Video job through the client's music-video worker, directly.

Bypasses the API, the queue and storage — the same shape as `ltx_smoke.py`:
build the job an adapter would receive, call the adapter, print what it
reported and what it produced. For measuring the pipeline on a node and for
looking at the result, not for production traffic.

    set -a; . /workspace/zolexai/.env.gpu-worker; set +a
    cd /workspace/zolexai/apps/worker
    .venv/bin/python scripts/mv_smoke.py --audio song.mp3 \
        --prompt "make me a cinematic video according to the lyrics of the song" \
        --performer 1:lead_vocalist:/path/singer.jpg \
        --performer 2:guitarist:/path/guitarist.jpg \
        --aspect 16:9 --out /workspace/mv-smoke/run1

`--performer SLOT:ROLE[:PICTURE[:DESCRIPTION]]`; a slot without a picture is
described only. `--steps`, `--render`/`--anchor` (command|ltx|mock /
command|reference|mock) and `--lyrics FILE` map onto the job's execution
block and parameters. The job directory the package writes — plan, anchors,
every shot, the working master and the 4K delivery — stays under `--out`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.adapters.base import AdapterInput, AdapterJob  # noqa: E402
from worker.adapters.music_video import MusicVideoAdapter  # noqa: E402


def _performer(spec: str) -> tuple[int, dict, Path | None]:
    parts = spec.split(":", 3)
    slot = int(parts[0])
    role = parts[1] if len(parts) > 1 and parts[1] else "performer"
    picture = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    description = parts[3] if len(parts) > 3 else ""
    return slot, {"slot": slot, "role": role, "description": description}, picture


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--performer", action="append", default=[])
    parser.add_argument("--lyrics", type=Path)
    parser.add_argument("--lyrics-language")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--render", choices=("command", "ltx", "mock"))
    parser.add_argument("--anchor", choices=("command", "reference", "mock"))
    parser.add_argument("--lyric-mode", choices=("automatic", "always", "off"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--job-id", default=f"mv-smoke-{int(time.time())}")
    args = parser.parse_args()

    workspace = args.out.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    staged_audio = workspace / f"inputs/source_audio{args.audio.suffix}"
    staged_audio.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.audio, staged_audio)
    inputs = [
        AdapterInput("source_audio", "audio", "audio/mpeg", "file://audio", path=staged_audio)
    ]

    performers: list[dict] = []
    for spec in args.performer:
        slot, entry, picture = _performer(spec)
        performers.append(entry)
        if picture is not None:
            staged = workspace / f"inputs/performer_{slot}{picture.suffix}"
            shutil.copy2(picture, staged)
            inputs.append(
                AdapterInput(f"performer_{slot}", "image", "image/png", "file://p", path=staged)
            )

    parameters: dict = {"aspect_ratio": args.aspect}
    if performers:
        parameters["performers"] = performers
    if args.lyrics:
        parameters["lyrics"] = args.lyrics.read_text(encoding="utf-8")
    if args.lyrics_language:
        parameters["lyrics_language"] = args.lyrics_language
    execution: dict = {"runtime": "music_video"}
    if args.steps:
        execution["inference_steps"] = args.steps
    if args.render:
        execution["render_backend"] = args.render
    if args.anchor:
        execution["anchor_backend"] = args.anchor
    if args.lyric_mode:
        execution["lyric_mode"] = args.lyric_mode

    job = AdapterJob(
        job_id=args.job_id,
        workflow_id="music-video",
        workflow_version="1",
        prompt=args.prompt,
        parameters=parameters,
        inputs=inputs,
        execution=execution,
        workspace=workspace,
    )

    started = time.monotonic()
    last = ""

    async def on_progress(status: str, progress: int, message: str, details=None) -> None:
        nonlocal last
        line = f"{status:16s} {progress:3d}%  {message}"
        if line != last:
            print(f"[{time.monotonic() - started:7.1f}s] {line}", flush=True)
            last = line

    result = await MusicVideoAdapter().run(job, on_progress)
    wall = time.monotonic() - started
    summary = {
        "job_id": args.job_id,
        "wall_seconds": round(wall, 1),
        "output": str(result.path),
        "duration_seconds": result.duration_seconds,
        "width": result.width,
        "height": result.height,
        "size_mb": round(result.size_bytes / 1e6, 1),
        "job_dir": str(workspace / "music-video" / args.job_id),
    }
    (workspace / "smoke-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
