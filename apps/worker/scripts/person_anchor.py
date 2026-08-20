"""The composited identity anchor for `v2v_reference_identity`, GPU-side.

A reference photograph is the WRONG SHAPE to anchor a video pass: the photo
is a portrait, the pass's first frame is whatever the footage is — often a
full-body wide shot where the face occupies forty pixels. Anchored raw, the
model takes the photo's colours and mood and invents the person, which is
exactly what the first full-body production jobs delivered. The reference
engine's answer, shipped in their product at strength 1.0, is to build the
anchor instead of borrowing it:

  1. take the source video's opening frame, on the pass's own grid;
  2. matte the person in it (BiRefNet — the same model person lock uses)
     and REMOVE them (OpenCV TELEA inpaint) — any leftover pixels of the
     original person are a first-frame cue pulling the render back to them;
  3. matte the person in the reference photo and cut them out;
  4. scale the cutout to the source person's own box, feet to the same
     ground, and composite.

The result is a frame whose COMPOSITION is the footage's and whose PERSON is
the reference's — an anchor that can be applied at full strength without
hijacking the shot. Deliberately a CLI in the LTX environment, like
`person_matte.py`: the worker has no torch and no OpenCV, and must not grow
them to prepare a picture.

    uv run python scripts/person_anchor.py \\
        --source clip.mp4 --reference person.png --dest anchor.png \\
        --start-seconds 0 --width 576 --height 1024

Exit codes: 0 success; 3 no person found in the source frame (the caller
falls back to the raw-photo anchor); anything else is a real failure.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: Matting resolution — BiRefNet is trained square; see person_matte.py.
MATTE_SIDE = 1024

#: A matte below this fraction of the frame is noise, not a person, and
#: compositing a person onto noise puts them in a random corner.
MIN_PERSON_AREA = 0.02

#: The cutout fills this much of the source person's box. Slightly inside it
#: (the reference engine uses 0.90 x 0.96) so the new person never overhangs
#: the silhouette the control signal is about to enforce.
FIT_WIDTH, FIT_HEIGHT = 0.90, 0.96

#: How far the removal mask is grown before inpainting, in pixels at the
#: working grid. Leftover slivers of the original person are exactly the
#: first-frame cue this exists to remove; over-inpainting costs nothing
#: because the composite covers most of it again.
INPAINT_DILATION = 9

#: How close to the reference photo's bottom edge the matte must reach before
#: the subject counts as TRUNCATED — a crop of a person rather than a whole
#: one. As a fraction of the reference's height.
#:
#: This decides which edge the cutout is aligned by, and getting it wrong is
#: not a subtle quality loss. Bottom-aligning assumes the cutout's lowest
#: pixels are the person's FEET. For a head-and-shoulders photo they are a
#: crop edge across the chest, so the bust gets planted at the source person's
#: feet and scaled to their width — measured 20 Aug 2026 on a full-body dance
#: source, which anchored on a disembodied bust sitting on the road, and the
#: renders duly carried a woman standing in the street for the whole video
#: alongside the dancer. A headshot is the commonest thing a customer uploads,
#: so this was the common case, not an edge case.
REFERENCE_TRUNCATION_TOLERANCE = 0.02

#: The fraction of a matte's height treated as "the head", measured down from
#: its topmost pixel.
#:
#: A cropped reference is scaled by matching ITS head to the source person's
#: head, rather than by matching bounding-box widths. The box is not reliably a
#: person: BiRefNet mattes the salient OBJECT, and on a seam frame of someone
#: standing beside a car it returns the person and the car as one region. Width
#: matching against that box scaled a bust to half the frame and produced a
#: giant floating head — measured 20 Aug 2026. The top of the matte is the
#: person's head in every framing where a head is visible at all, so it is the
#: one landmark worth trusting here.
HEAD_BAND = 0.12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source video")
    parser.add_argument("--reference", type=Path, required=True, help="person photo")
    parser.add_argument("--dest", type=Path, required=True, help="output PNG")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--model", default="ZhengPeng7/BiRefNet")
    return parser.parse_args(argv)


def extract_frame(args: argparse.Namespace, dest: Path) -> Path:
    """The source's opening frame, on the render's exact grid."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{max(0.0, args.start_seconds):.3f}",
            "-i", str(args.source),
            "-frames:v", "1",
            "-vf", (
                f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,"
                f"crop={args.width}:{args.height}"
            ),
            str(dest),
        ],
        check=True,
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit("no opening frame could be extracted")
    return dest


def load_matter(model_id: str):
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True
    )
    model.to("cuda").eval().half()
    return model


def matte_of(model, image) -> object:
    """A float mask in [0,1] at the image's own size."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    prepare = transforms.Compose(
        [
            transforms.Resize((MATTE_SIDE, MATTE_SIDE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with torch.no_grad():
        tensor = prepare(image).unsqueeze(0).to("cuda").half()
        prediction = model(tensor)[-1].sigmoid().float().cpu()[0, 0].numpy()
    mask = Image.fromarray((prediction * 255).astype("uint8")).resize(image.size)
    return np.asarray(mask).astype("float32") / 255.0


def bbox_of(mask, min_area: float) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the matte, or None when there is no real person."""
    import numpy as np

    binary = mask > 0.5
    if binary.mean() < min_area:
        return None
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def head_band(mask, box: tuple[int, int, int, int]) -> tuple[int, int] | None:
    """(x0, x1) of the matte across the top `HEAD_BAND` of its height.

    None when the band holds nothing, which means the matte is not shaped like
    a person and the caller should not trust it.
    """
    import numpy as np

    x0, y0, x1, y1 = box
    band = max(1, round((y1 - y0) * HEAD_BAND))
    sub = mask[y0 : y0 + band, x0:x1] > 0.5
    columns = np.any(sub, axis=0)
    if not columns.any():
        return None
    left = x0 + int(np.argmax(columns))
    right = x0 + len(columns) - int(np.argmax(columns[::-1]))
    if right - left < 2:
        return None
    return left, right


def is_truncated(box: tuple[int, int, int, int], height: int) -> bool:
    """Does the reference's subject run off the bottom of their own photo?

    If it does, the cutout's lowest row is a crop edge and not a pair of feet,
    which is what decides how the cutout may be placed (see
    `REFERENCE_TRUNCATION_TOLERANCE`).
    """
    return box[3] >= height - max(1, round(REFERENCE_TRUNCATION_TOLERANCE * height))


def place_cutout(
    source_box: tuple[int, int, int, int],
    cutout_size: tuple[int, int],
    *,
    truncated: bool,
    source_head: tuple[int, int] | None = None,
    reference_head: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """How big the reference cutout should be, and where it goes.

    Two cases, and they differ in which part of the person is trustworthy:

    * A WHOLE reference figure shares its ground with the source person, so it
      is fitted inside their box and bottom-aligned — their feet meet the same
      floor.
    * A TRUNCATED one (a headshot, a bust) has no feet to align. It is scaled
      so its HEAD matches the source person's head and hung from the top of
      their box. Its body continues past the bottom; the control signal states
      the body's pose for every frame regardless.

    Head matching rather than box-width matching, because the box is not
    reliably a person — see `HEAD_BAND`. Both head bands are needed for it; if
    either is missing, the box width is the fallback, which is the previously
    shipped arithmetic.
    """
    sx0, sy0, sx1, sy1 = source_box
    box_w, box_h = sx1 - sx0, sy1 - sy0
    cut_w, cut_h = cutout_size

    if not truncated:
        scale = min(FIT_WIDTH * box_w / cut_w, FIT_HEIGHT * box_h / cut_h)
        size = (max(1, round(cut_w * scale)), max(1, round(cut_h * scale)))
        return size, (sx0 + (box_w - size[0]) // 2, sy1 - size[1])

    if source_head and reference_head:
        source_width = source_head[1] - source_head[0]
        reference_width = reference_head[1] - reference_head[0]
        scale = source_width / reference_width
        size = (max(1, round(cut_w * scale)), max(1, round(cut_h * scale)))
        # Line the two heads up horizontally, rather than the two boxes.
        reference_centre = (reference_head[0] + reference_head[1]) / 2 * scale
        paste_x = round((source_head[0] + source_head[1]) / 2 - reference_centre)
        return size, (paste_x, sy0)

    scale = FIT_WIDTH * box_w / cut_w
    size = (max(1, round(cut_w * scale)), max(1, round(cut_h * scale)))
    return size, (sx0 + (box_w - size[0]) // 2, sy0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.source, args.reference):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")

    workspace = Path(tempfile.mkdtemp(prefix="person-anchor-"))
    try:
        import cv2
        import numpy as np
        from PIL import Image

        frame_path = extract_frame(args, workspace / "frame.png")
        frame = Image.open(frame_path).convert("RGB")
        reference = Image.open(args.reference).convert("RGB")

        model = load_matter(args.model)
        print("matting the source frame", flush=True)
        source_mask = matte_of(model, frame)
        print("matting the reference", flush=True)
        reference_mask = matte_of(model, reference)

        source_box = bbox_of(source_mask, MIN_PERSON_AREA)
        reference_box = bbox_of(reference_mask, MIN_PERSON_AREA)
        if source_box is None or reference_box is None:
            where = "source frame" if source_box is None else "reference"
            print(f"no person found in the {where}", flush=True)
            return 3

        # Remove the original person entirely before compositing: any visible
        # sliver of them is a first-frame cue for their clothes and skin.
        frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        removal = (source_mask > 0.3).astype("uint8") * 255
        kernel = np.ones((INPAINT_DILATION, INPAINT_DILATION), np.uint8)
        removal = cv2.dilate(removal, kernel)
        inpainted = cv2.inpaint(frame_bgr, removal, 7, cv2.INPAINT_TELEA)
        canvas = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

        # The reference person, cut out and placed against the source person's
        # own box. WHICH EDGE they are aligned by depends on whether the photo
        # shows a whole person or a crop of one — see `place_cutout`.
        rx0, ry0, rx1, ry1 = reference_box
        cutout = reference.crop((rx0, ry0, rx1, ry1))
        alpha = Image.fromarray(
            (np.clip(reference_mask[ry0:ry1, rx0:rx1], 0, 1) * 255).astype("uint8")
        )
        truncated = is_truncated(reference_box, reference.height)
        # Head bands are measured in each image's own coordinates; the
        # reference's is rebased onto the cutout, which is what gets scaled.
        source_head = head_band(source_mask, source_box)
        reference_head = head_band(reference_mask, reference_box)
        if reference_head is not None:
            reference_head = (reference_head[0] - rx0, reference_head[1] - rx0)
        size, (paste_x, paste_y) = place_cutout(
            source_box,
            (cutout.width, cutout.height),
            truncated=truncated,
            source_head=source_head,
            reference_head=reference_head,
        )
        print(
            f"reference is {'a crop' if truncated else 'a whole figure'}; "
            f"anchored by the {'head' if truncated else 'feet'}",
            flush=True,
        )
        cutout = cutout.resize(size, Image.LANCZOS)
        alpha = alpha.resize(size, Image.LANCZOS)

        canvas.paste(cutout, (paste_x, paste_y), alpha)

        args.dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.dest)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if not args.dest.exists() or args.dest.stat().st_size == 0:
        raise SystemExit("no anchor was written")
    print(f"anchor written to {args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
