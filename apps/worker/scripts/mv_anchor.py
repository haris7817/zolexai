"""The music-video worker's anchor command: one shot's starting still.

The client's package (`zolex_music_worker`) calls this once per distinct
shot with a request JSON and expects a PNG of exactly the requested size at
`--output`. The picture is rendered on this node's LTX ComfyUI service with
Qwen-Image-Edit-2509 (`worker.comfy.qwen_edit`): the performers' reference
photos go in, and a new composition of the same people in the shot's
location and light comes out. With no references (the package's fictional
identity portraits, and every empty environment shot) the same graph is a
text-to-image model.

Contract, from the package's README ("Anchor command contract"):

    {"schema_version": 1, "anchor_role": "...", "format": "16:9",
     "width": 1280, "height": 704, "performer_ids": [...],
     "references": ["/abs/ref.png", ...], "prompt": "...",
     "output": "/abs/required-output.png"}

The command must exit zero and write an RGB PNG at exactly width x height.
More than three references (a four- or five-piece band in one wide) are
tiled into one contact sheet, because the edit encoder has three picture
slots; the prompt tells the model who is who, left to right.

Usage:  mv_anchor.py --request REQUEST.json --output OUT.png
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path


def _log(message: str, **fields: object) -> None:
    print(json.dumps({"mv_anchor": message, **fields}), file=sys.stderr, flush=True)


def _contact_sheet(paths: list[Path], dest: Path) -> Path:
    """Every reference in one picture, left to right, one row."""
    from PIL import Image, ImageOps

    cell = 768
    canvas = Image.new("RGB", (cell * len(paths), cell), (18, 18, 18))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            fitted = ImageOps.fit(image.convert("RGB"), (cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (index * cell, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, format="PNG")
    return dest


def _reference_clause(performer_ids: list[str], count: int, sheet: bool) -> str:
    if count == 0:
        return ""
    if sheet:
        order = ", ".join(performer_ids) if performer_ids else "the members"
        return (
            f"The reference sheet shows {count} people side by side, left to right: {order}. "
            "Keep each person's exact face, hair, skin, build and clothing from the sheet. "
        )
    if count == 1:
        return (
            "Keep the exact face, hair, skin, build and clothing of the person in the "
            "reference picture. "
        )
    return (
        f"The {count} reference pictures show the assigned performers in order. Keep each "
        "person's exact face, hair, skin, build and clothing; never merge or swap them. "
    )


async def _render(request: dict, output: Path) -> None:
    # Imported here so the usage error above needs no worker install.
    from PIL import Image

    from worker.adapters.base import AdapterJob
    from worker.comfy.client import ComfyClient, ComfyError
    from worker.comfy.qwen_edit import MAX_REFERENCES, compile_anchor, missing_nodes
    from worker.core.config import settings

    width = int(request["width"])
    height = int(request["height"])
    prompt = str(request.get("prompt") or "").strip()
    references = [Path(item) for item in request.get("references") or []]
    performer_ids = [str(item) for item in request.get("performer_ids") or []]
    role = str(request.get("anchor_role") or "anchor")
    if not prompt:
        raise SystemExit("mv_anchor: the request carries no prompt")

    client = ComfyClient(
        settings.ltx_comfy_base_url,
        request_timeout=settings.ltx_comfy_request_timeout,
        poll_seconds=min(2.0, settings.ltx_comfy_poll_seconds),
    )
    catalogue = await client.node_classes()
    absent = missing_nodes(catalogue)
    if absent:
        raise SystemExit(f"mv_anchor: this ComfyUI lacks the nodes {absent}")

    # References: up to three go straight in; more become one sheet.
    uploads: list[str] = []
    sheet = False
    work = output.parent / f".{output.stem}-refs"
    if len(references) > MAX_REFERENCES:
        sheet = True
        references = [_contact_sheet(references, work / "sheet.png")]
    for path in references:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        uploads.append(
            await client.upload_input(path, name=f"{digest}{path.suffix.lower() or '.png'}",
                                      subfolder="mv-anchors")
        )

    full_prompt = _reference_clause(performer_ids, len(request.get("references") or []), sheet)
    full_prompt += prompt
    seed = zlib.crc32(f"{output}:{prompt}".encode()) & 0x7FFFFFFF
    # The still is conditioning, not delivered pixels, and the graph encodes
    # it down to the first stage's canvas anyway — so it may be generated
    # smaller and resized up to the size the package validates.
    scale = max(0.25, min(1.0, float(settings.music_video_anchor_scale)))
    gen_width = max(256, int(round(width * scale / 16)) * 16)
    gen_height = max(256, int(round(height * scale / 16)) * 16)
    api = compile_anchor(
        prompt=full_prompt,
        width=gen_width,
        height=gen_height,
        seed=seed,
        filename_prefix=f"zolexai/mv-anchors/{output.stem}",
        references=uploads,
        steps=int(settings.music_video_anchor_steps),
        lightning=bool(settings.music_video_anchor_lightning),
    )

    started = time.monotonic()
    job = AdapterJob(
        job_id=output.stem,
        workflow_id="music-video",
        workflow_version="1",
        prompt=full_prompt,
        parameters={},
    )
    try:
        prompt_id = await client.submit(api, client_id=f"mv-anchor-{output.stem}")
        history = await client.wait(job, prompt_id, timeout_seconds=900.0)
    except ComfyError as exc:
        raise SystemExit(f"mv_anchor: {exc.internal_detail}") from exc

    image_ref = None
    for node_output in (history.get("outputs") or {}).values():
        for item in node_output.get("images") or []:
            image_ref = item
            break
        if image_ref:
            break
    if not image_ref:
        raise SystemExit("mv_anchor: ComfyUI finished without an image output")

    raw = output.with_name(f"{output.stem}.comfy.png")
    await client.download_output(
        filename=str(image_ref.get("filename")),
        subfolder=str(image_ref.get("subfolder") or ""),
        output_type=str(image_ref.get("type") or "output"),
        dest=raw,
    )
    with Image.open(raw) as image:
        picture = image.convert("RGB")
        if picture.size != (width, height):
            picture = picture.resize((width, height), Image.Resampling.LANCZOS)
        output.parent.mkdir(parents=True, exist_ok=True)
        picture.save(output, format="PNG")
    raw.unlink(missing_ok=True)
    _log(
        "rendered",
        role=role,
        size=f"{width}x{height}",
        generated=f"{gen_width}x{gen_height}",
        references=len(uploads),
        sheet=sheet,
        seconds=round(time.monotonic() - started, 1),
        prompt_id=prompt_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(str(request.get("output") or args.output)).resolve()
    asyncio.run(_render(request, output))
    if not output.is_file():
        raise SystemExit(f"mv_anchor: no file at {output}")


if __name__ == "__main__":
    main()
