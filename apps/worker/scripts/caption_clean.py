"""Detect burned-in captions in a rendered clip and paint them out.

Runs in the LTX environment (`uv run python`), which already has `cv2`,
`numpy` and `torch`. Reads a clip, writes a cleaned clip, prints one JSON
report on stdout.

Client instruction, 10 Sep 2026: "OCR caption detection -> expand caption mask
-> temporal video inpainting -> upscaling", performed before the upscale
because "removing text after upscale is slower and leaves larger artifacts".
This is that pipeline with two substitutions, both measured on their own
failing clip (job 70a97bf1) the same day.

## Why not OCR

We need to know **where** the text is, never what it says, and what it says is
not words: "What a tasty carırtt you have there", "I'll nibile genntly,
thankıes for sharing". The model is not transcribing the dialogue — it is
drawing subtitle-SHAPED decoration, because its training data is full of
captioned clips. A recogniser's language model is dead weight against that and
would score its own confidence lowest on exactly the frames we most need. So
detection is classical, on `cv2`: no new dependency, no model load, no VRAM on
a node that has already been OOM-killed once.

## Why not `delogo`, and why not ProPainter yet

`delogo` was tried first, at the client's suggested starting point. It removes
the text and replaces it with a vertical-smear band across the whole
rectangle — on water it streaks, on the rabbit it destroys the carrot and the
paw. The cure was as visible as the disease, because `delogo` blanks the whole
box while the glyphs are only 4-20% of it.

Masking the GLYPHS instead leaves the background untouched, and that is what
this does. Measured on the failing clip: text gone, no dark ghost, grass,
water, carrot and fur all intact, mild softening confined to the stroke
neighbourhood. 6.6 s for a 15 s clip, on CPU.

ProPainter remains the answer if a clip appears where this is not enough — it
is a real temporal model and this is a spatial approximation. This runs first
because it costs one decode pass and no weights.

## What makes a caption a caption

Ordinary picture content trips every single-frame test, which is why there are
four and why one of them is temporal. Measured against a caption-free lunar
clip (job 5e7594d7) that the first version reported as a caption covering 41%
of the frame:

* **bright** — subtitles are near-white;
* **sparse** — thin strokes, so 1.5-40% of the box is bright. Sunlit regolith
  is bright over 60-90% of it and fails here;
* **centred and wide** — a subtitle line, not a bright corner;
* **still** — the same baseline in every frame. Scenery throws candidates at a
  different height each time; this is the test scenery cannot pass.
"""

from __future__ import annotations

import argparse
import json
import sys

#: A glyph core is near-white. Below this is picture, not text.
CORE_THRESHOLD = 185
#: Bright fraction inside a candidate box. Text is sparse; a sunlit surface is
#: not. The upper bound is what rejected the lunar clip.
MIN_BRIGHT_FRACTION = 0.015
MAX_BRIGHT_FRACTION = 0.40
#: How far a line's centre may sit from the frame's, as a fraction of width.
MAX_CENTRE_OFFSET = 0.18


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and remove burned-in captions")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="", help="cleaned clip; omit to detect only")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--band-top", type=float, default=0.60,
                        help="ignore text above this fraction of height: captions sit "
                             "low, and a shop sign higher up is the customer's picture")
    parser.add_argument("--min-frames", type=float, default=0.25)
    parser.add_argument("--pad", type=int, default=6)
    parser.add_argument("--grow", type=int, default=9, help="glyph mask dilation")
    return parser.parse_args(argv)


def _text_boxes(gray, band_top: int):
    """Candidate caption lines in one frame, as (x, y, w, h) in frame pixels."""
    import cv2
    import numpy as np

    height, width = gray.shape
    strip = gray[band_top:, :]
    if strip.size == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(strip, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 40), 3))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line_kernel)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 6 or h > height * 0.18:
            continue
        if w < width * 0.10 or w / max(1, h) < 3.0:
            continue
        patch = strip[y:y + h, x:x + w]
        if patch.size == 0 or np.percentile(patch, 92) < 170:
            continue
        bright = float(np.count_nonzero(patch >= 200)) / patch.size
        if not MIN_BRIGHT_FRACTION <= bright <= MAX_BRIGHT_FRACTION:
            continue
        if abs((x + w / 2) - width / 2) / width > MAX_CENTRE_OFFSET:
            continue
        boxes.append((x, y + band_top, w, h))
    return boxes


def detect(video: str, *, samples: int, band_top: float, min_frames: float, pad: int) -> dict:
    """Where the caption sits in this clip, or that there is none."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return {"error": f"cannot open {video}"}
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0 or width <= 0:
        capture.release()
        return {"error": "clip reports no frames"}

    top = int(height * band_top)
    indices = [int(i * (total - 1) / max(1, samples - 1)) for i in range(samples)]
    per_frame = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        boxes = _text_boxes(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), top)
        if boxes:
            per_frame.append({"frame": index, "boxes": boxes})
    capture.release()

    # The temporal test: keep only boxes sharing the clip's dominant baseline.
    baselines = [b[1] + b[3] for f in per_frame for b in f["boxes"]]
    if baselines:
        anchor = float(np.median(baselines))
        tolerance = max(8.0, height * 0.05)
        per_frame = [
            {"frame": f["frame"],
             "boxes": [b for b in f["boxes"] if abs((b[1] + b[3]) - anchor) <= tolerance]}
            for f in per_frame
        ]
        per_frame = [f for f in per_frame if f["boxes"]]

    inspected = len(indices)
    hits = len(per_frame)
    ratio = hits / max(1, inspected)
    report = {
        "width": width,
        "height": height,
        "frames_inspected": inspected,
        "frames_with_text": hits,
        "ratio": round(ratio, 3),
        "detected": ratio >= min_frames,
    }
    if report["detected"]:
        xs = [b[0] for f in per_frame for b in f["boxes"]]
        ys = [b[1] for f in per_frame for b in f["boxes"]]
        x2 = [b[0] + b[2] for f in per_frame for b in f["boxes"]]
        y2 = [b[1] + b[3] for f in per_frame for b in f["boxes"]]
        # Percentiles, not extremes: one stray contour in one frame must not
        # stretch the mask across the picture.
        x = max(0, int(np.percentile(xs, 5)) - pad)
        y = max(0, int(np.percentile(ys, 5)) - pad)
        right = min(width, int(np.percentile(x2, 95)) + pad)
        bottom = min(height, int(np.percentile(y2, 95)) + pad)
        report["box"] = {"x": x, "y": y, "w": max(2, right - x), "h": max(2, bottom - y)}
        report["coverage"] = round((right - x) * (bottom - y) / float(width * height), 4)
    return report


def _glyph_mask(roi, grow: int):
    """The glyph strokes AND their dark outline, dilated.

    Both halves matter. A mask of the bright core alone leaves a dark ghost
    where the outline was — measured, and visible. The gradient lights up on
    both sides of every stroke; intersecting it with the neighbourhood of a
    bright core keeps the glyph rim and discards the grass.
    """
    import cv2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, core = cv2.threshold(gray, CORE_THRESHOLD, 255, cv2.THRESH_BINARY)
    gradient = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    _, edges = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    near = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    mask = cv2.bitwise_or(core, cv2.bitwise_and(edges, near))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))


def clean(video: str, out: str, box: dict, *, grow: int) -> dict:
    """Every frame with the caption strokes painted out, written to `out`.

    Video only: audio is not read or written here. The caller remuxes the
    original soundtrack, which is what keeps this off the audio path
    entirely — the same reason the finishing pass copies rather than encodes.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return {"error": f"cannot open {video}"}
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    frames, painted = 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        roi = frame[y:y + h, x:x + w]
        if roi.size:
            mask = _glyph_mask(roi, grow)
            if np.count_nonzero(mask):
                frame[y:y + h, x:x + w] = cv2.inpaint(roi, mask, 6, cv2.INPAINT_NS)
                painted += 1
        writer.write(frame)
        frames += 1
    capture.release()
    writer.release()
    return {"frames": frames, "frames_painted": painted}


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    report = detect(
        args.video,
        samples=args.samples,
        band_top=args.band_top,
        min_frames=args.min_frames,
        pad=args.pad,
    )
    if "error" in report:
        print(json.dumps(report))
        return 2
    if args.out and report.get("detected"):
        report.update(clean(args.video, args.out, report["box"], grow=args.grow))
        report["cleaned"] = "error" not in report
    else:
        report["cleaned"] = False
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
