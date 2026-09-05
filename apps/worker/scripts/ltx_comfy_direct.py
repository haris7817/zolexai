"""Submit a frozen client graph to ComfyUI exactly as its frontend would — no adapter.

    python scripts/ltx_comfy_direct.py t2v --prompt "..." --seed 42 --seconds 5 \
        --aspect 16:9 --out /tmp/direct.mp4
    python scripts/ltx_comfy_direct.py flf --first a.png [--last b.png] ...
    python scripts/ltx_comfy_direct.py cr --video src.mp4 --image ref.png --seconds 8 ...

This is the "A" side of the Step 8 comparison: the same UI→API conversion the
ComfyUI browser client performs on Queue (subgraphs flattened, Set/Get links
resolved, widget values as inputs), with only the user inputs filled in, posted
straight to `/prompt`. The "B" side is the ZolexAI adapter (`ltx_comfy_bench.py`).
Same seed + same inputs on both sides is what "the result must match" tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from worker.adapters.base import AdapterJob
from worker.comfy.client import ComfyClient
from worker.comfy.ltx_graphs import (
    GenerationEdits,
    ReplacementEdits,
    aspect_label_for,
    compile_character_replacement,
    compile_first_last_frame,
    compile_text_to_video,
    oriented_canvas,
)
from worker.comfy.ltx_prompts import character_replacement_prompt, negative_for
from worker.core.config import settings
from worker.media import probe_media
from worker.providers.ltx_comfy import LtxComfyService


async def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("graph", choices=["t2v", "flf", "cr"])
    p.add_argument("--prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--aspect", default="16:9")
    p.add_argument("--first", type=Path)
    p.add_argument("--last", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dump-prompt", type=Path, help="write the compiled API prompt JSON here")
    args = p.parse_args()

    service = LtxComfyService()
    client: ComfyClient = service.client
    options = await service.aspect_options()
    label = aspect_label_for(args.aspect, options)
    stem = f"direct_{args.graph}_{int(time.time())}"
    prefix = f"zolexai/direct/{stem}"

    async def upload(path: Path, name: str) -> str:
        return await service.upload(path, name=name)

    if args.graph == "t2v":
        edits = GenerationEdits(
            positive=args.prompt,
            negative=negative_for("text-to-video", {}),
            seconds=args.seconds,
            aspect_label=label,
            seed_base=args.seed,
            filename_prefix=prefix,
        )
        api = compile_text_to_video(service.load("text_to_video"), edits)
    elif args.graph == "flf":
        first = await upload(args.first, f"{stem}_first.png")
        last = await upload(args.last, f"{stem}_last.png") if args.last else None
        edits = GenerationEdits(
            positive=args.prompt,
            negative=negative_for("image-to-video", {}),
            seconds=args.seconds,
            aspect_label=label,
            seed_base=args.seed,
            filename_prefix=prefix,
            first_image=first,
            last_image=last,
        )
        api = compile_first_last_frame(service.load("first_last_frame"), edits)
    else:
        info = await probe_media(args.video)
        width, height = oriented_canvas(
            tuple(settings.character_replacement_canvas),
            source_width=info.width,
            source_height=info.height,
        )
        video = await upload(args.video, f"{stem}_source.mp4")
        image = await upload(args.image, f"{stem}_reference.png")
        edits = ReplacementEdits(
            positive=character_replacement_prompt(args.prompt),
            negative=negative_for("character-replacement", {}),
            video=video,
            image=image,
            seconds=int(args.seconds),
            width=width,
            height=height,
            seed_base=args.seed,
            filename_prefix=prefix,
        )
        api = compile_character_replacement(service.load("character_replacement"), edits)

    if args.dump_prompt:
        args.dump_prompt.write_text(json.dumps(api, indent=1), encoding="utf-8")

    job = AdapterJob(
        job_id=stem, workflow_id=args.graph, workflow_version="1", prompt=args.prompt, parameters={}
    )
    started = time.monotonic()
    prompt_id = await client.submit(api, client_id=f"direct-{stem}")
    print(f"submitted {prompt_id} ({len(api)} nodes)", flush=True)
    history = await client.wait(job, prompt_id, timeout_seconds=3600, on_tick=None)
    wall = time.monotonic() - started
    out = await service.collect(history, args.out)
    info = await probe_media(out)
    print(
        json.dumps(
            {
                "output": str(out),
                "wall_seconds": round(wall, 1),
                "duration": info.duration_seconds,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frames": info.frame_count,
                "has_audio": info.has_audio,
                "seed": args.seed,
                "aspect_label": label,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
