"""Time one music-video shot on the ComfyUI audio-to-video graph.

The question this exists to answer: the client's package costs 96 seconds
per 121-frame shot on the command-line pipeline, and a 3-minute song is
about 41 shots, so 70 minutes. Two things could change that — running the
distilled 8-step graph on a warm ComfyUI instead of a fresh process per
shot, and making the shots longer so there are fewer of them.

The second only helps if a step costs about the same for 241 frames as for
121. That is plausible from what is already measured — quadrupling the
pixels cost only 24 percent more per step, which says the pass is moving
weights rather than crunching tokens — but plausible is not measured, so
this renders the same shot at several frame counts and prints the curve.

    set -a; . /workspace/zolexai/.env.gpu-worker; set +a
    cd /workspace/zolexai/apps/worker
    .venv/bin/python scripts/mv_a2v_bench.py \
        --anchor /path/anchor.png --audio /path/conditioning.wav \
        --frames 121,193,241 --out /workspace/mv-smoke/a2v

Every run is given its own seed, because ComfyUI returns a cached result
for an identical prompt in about a second and that reads as a miracle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.adapters.base import AdapterJob  # noqa: E402
from worker.comfy.client import ComfyClient  # noqa: E402
from worker.comfy.ltx_a2v import compile_a2v, missing_nodes  # noqa: E402
from worker.core.config import settings  # noqa: E402

PROMPT = (
    "Photorealistic cinematic medium performance shot on a rooftop over the city at blue hour. "
    "The performer sings the current vocal phrase toward the lens with natural mouth "
    "articulation, controlled breathing and believable eye focus. The camera slides gently "
    "sideways while maintaining stable framing. Natural skin, deep neutral shadows and warm "
    "practical light, stable exposure throughout. One continuous take with no internal cuts, "
    "no duplicate subjects, no captions, overlays, logos or watermarks."
)
NEGATIVE = (
    "identity drift, face change, deformed hands, extra fingers, extra limbs, duplicate person, "
    "cropped head, cropped feet, flicker, exposure pulsing, sudden darkness, captions, text, "
    "logo, watermark"
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--frames", default="121,193,241")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--single-stage", action="store_true",
                        help="one 8-step pass at the delivered size, no latent upscale")
    parser.add_argument("--transformer")
    parser.add_argument("--audio-start", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(settings.ltx_comfy_base_url, request_timeout=60.0, poll_seconds=2.0)

    absent = missing_nodes(await client.node_classes())
    if absent:
        raise SystemExit(f"this ComfyUI lacks {absent}")

    anchor_name = await client.upload_input(args.anchor, name=f"a2vbench{args.anchor.suffix}")
    audio_name = await client.upload_input(args.audio, name=f"a2vbench{args.audio.suffix}")
    print(f"uploaded anchor={anchor_name} audio={audio_name}", flush=True)

    job = AdapterJob(
        job_id="a2v-bench",
        workflow_id="music-video",
        workflow_version="1",
        prompt=PROMPT,
        parameters={},
    )

    rows: list[dict] = []
    seed = int(time.time()) % 100000
    for frames in [int(x) for x in args.frames.split(",") if x.strip()]:
        seconds = frames / args.fps
        for attempt in range(args.repeat):
            seed += 7
            # A distinct prompt per run, because ComfyUI caches a node's
            # output by its inputs: three runs of one prompt would encode
            # the text once and report two shots that never paid for it.
            # Every shot of a real job carries its own words.
            positive = f"{PROMPT} The light settles for a {seed} beat."
            api = compile_a2v(
                positive=positive,
                negative=NEGATIVE,
                audio=audio_name,
                image=anchor_name,
                audio_start_seconds=args.audio_start,
                seconds=seconds,
                frames=frames,
                width=args.width,
                height=args.height,
                fps=args.fps,
                seed=seed,
                filename_prefix=f"zolexai/a2v-bench/f{frames}-{seed}",
                two_stage=not args.single_stage,
                **({"transformer": args.transformer} if args.transformer else {}),
            )
            started = time.monotonic()
            prompt_id = await client.submit(api, client_id="a2v-bench")
            history = await client.wait(job, prompt_id, timeout_seconds=1800.0)
            wall = time.monotonic() - started

            exec_seconds = None
            status = history.get("status") or {}
            for message in status.get("messages") or []:
                if message and message[0] == "execution_success":
                    pass
            # SaveVideo reports its file under "images" with animated=true.
            found = None
            for node_output in (history.get("outputs") or {}).values():
                for item in (
                    (node_output.get("videos") or [])
                    + (node_output.get("gifs") or [])
                    + (node_output.get("images") or [])
                ):
                    found = item
            if found:
                dest = args.out / f"f{frames}-{seed}.mp4"
                await client.download_output(
                    filename=str(found.get("filename")),
                    subfolder=str(found.get("subfolder") or ""),
                    output_type=str(found.get("type") or "output"),
                    dest=dest,
                )
            row = {
                "frames": frames,
                "seconds_of_video": round(seconds, 2),
                "wall_seconds": round(wall, 1),
                "per_second_of_video": round(wall / seconds, 2),
                "attempt": attempt + 1,
                "two_stage": not args.single_stage,
                "canvas": f"{args.width}x{args.height}",
                "exec_seconds": exec_seconds,
                "output": str(dest) if found else None,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    (args.out / "bench.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n frames  video   wall   wall per second of video")
    for row in rows:
        print(f"  {row['frames']:5d} {row['seconds_of_video']:6.2f}s {row['wall_seconds']:7.1f}s"
              f"   {row['per_second_of_video']:6.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
