"""Find burned-in caption text in a rendered clip. Runs in the LTX venv.

Client instruction, 10 Sep 2026: "OCR caption detection -> expand caption mask
-> temporal video inpainting -> upscaling", performed before the upscale
because "removing text after upscale is slower and leaves larger artifacts".
This is the detection half.

## Why this does not read the text

Their spec names PaddleOCR or EasyOCR. We only need to know **where** the
text is, never what it says, and on the measured failure (job
70a97bf1, 10 Sep 2026) what it says is not words: "What a tasty carırtt you
have there", "I'll nibile genntly, thankıes for sharing". The model is not
transcribing the dialogue, it is drawing subtitle-SHAPED decoration, because
its training data is full of captioned clips. A recogniser's language model
is dead weight against that, and would score its own confidence low on
exactly the frames we most need to catch.

So detection is classical and runs on `cv2`, which the LTX venv already has:
no new dependency, no model load, no VRAM on a node that has been OOM-killed
once. A recogniser can be added behind `--engine easyocr` when one is
installed, and the mask it produces is consumed identically.

## What it looks for

Subtitle text has a signature that ordinary picture content does not:

* **High local contrast at small scale** — white glyphs with a dark outline.
  A morphological gradient lights up on the glyph edges and stays dark on
  grass, sky and fur.
* **Rows of small blobs on one baseline** — many connected components of
  similar height, side by side, far more regular than foliage.
* **The same place in every frame.** Captions do not drift; picture content
  does. Agreement ACROSS sampled frames is the strongest signal here and the
  one that cannot be faked by a bright horizon or a line of fence posts.

The band is returned in SOURCE pixels of the clip handed in, which is the
generation canvas (864x480), not the delivered frame.
"""

from __future__ import annotations

import argparse
import json
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--samples", type=int, default=24,
                        help="frames to inspect, spread across the clip")
    parser.add_argument("--band-top", type=float, default=0.60,
                        help="ignore text above this fraction of frame height; "
                             "captions sit low and a shop sign in the scene is "
                             "the customer's picture, not our defect")
    parser.add_argument("--min-frames", type=float, default=0.25,
                        help="fraction of sampled frames that must agree before "
                             "a band is called a caption")
    parser.add_argument("--pad", type=int, default=6, help="mask dilation, pixels")
    parser.add_argument("--engine", default="cv2", choices=("cv2", "easyocr"))
    parser.add_argument("--debug-dir", default="")
    return parser.parse_args(argv)


def _text_boxes(gray, band_top: int):
    """Candidate text rows in one frame, as (x, y, w, h) in frame pixels."""
    import cv2
    import numpy as np

    height, width = gray.shape
    strip = gray[band_top:, :]
    if strip.size == 0:
        return []

    # Glyph edges: a morphological gradient is bright where a thin light shape
    # meets a dark outline, and flat over grass, fur and sky.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(strip, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Join glyphs into words and words into a line: wide, short kernel, so
    # characters merge sideways and separate lines stay separate.
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 40), 3))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line_kernel)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 6 or h > height * 0.18:      # too thin to read, or a whole region
            continue
        if w < width * 0.10:                 # a word or two at least
            continue
        if w / max(1, h) < 3.0:              # text lines are wide, not blocky
            continue
        # Subtitles are bright. Check the actual pixels rather than the edges,
        # so a dark textured strip does not qualify.
        patch = strip[y:y + h, x:x + w]
        if patch.size == 0 or np.percentile(patch, 92) < 170:
            continue
        # **Glyphs are SPARSE.** Text is thin bright strokes over a darker
        # picture; a sunlit surface is bright nearly everywhere. Measured on
        # the two real clips (10 Sep 2026): the captioned meadow scores
        # 0.04-0.20 bright fraction inside the box, the caption-free lunar
        # regolith scores 0.6-0.9 and was detected as a 41%-of-frame caption
        # that `delogo` would have smeared across half the picture.
        bright = float(np.count_nonzero(patch >= 200)) / patch.size
        if not 0.015 <= bright <= 0.40:
            continue
        # Subtitles are centred. A bright feature hugging one edge is scenery.
        centre_offset = abs((x + w / 2) - width / 2) / width
        if centre_offset > 0.18:
            continue
        boxes.append((x, y + band_top, w, h))
    return boxes


def _easyocr_boxes(frame, band_top: int, reader):
    """The same list, from a recogniser, when one is installed."""
    boxes = []
    for points, _text, confidence in reader.readtext(frame):
        if confidence < 0.20:
            continue
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        y = min(ys)
        if y < band_top:
            continue
        boxes.append((min(xs), y, max(xs) - min(xs), max(ys) - y))
    return boxes


def main(argv: list[str]) -> int:
    import cv2
    import numpy as np

    args = _parse_args(argv)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(json.dumps({"error": f"cannot open {args.video}"}))
        return 2

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0 or width <= 0:
        print(json.dumps({"error": "clip reports no frames"}))
        return 2

    reader = None
    if args.engine == "easyocr":
        import easyocr

        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    band_top = int(height * args.band_top)
    indices = [int(i * (total - 1) / max(1, args.samples - 1)) for i in range(args.samples)]
    hits, per_frame = 0, []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = (
            _easyocr_boxes(frame, band_top, reader)
            if reader is not None
            else _text_boxes(gray, band_top)
        )
        if boxes:
            hits += 1
            per_frame.append({"frame": index, "boxes": boxes})
    capture.release()

    inspected = len(indices)

    # **A caption does not move.** This is the discriminator that scenery
    # cannot fake: bright textured ground throws candidate boxes at a
    # different height in every frame, while a subtitle sits on the same
    # baseline all clip. Keep only the boxes that agree on a baseline, and
    # judge the ratio on those rather than on every candidate.
    baselines = [b[1] + b[3] for f in per_frame for b in f["boxes"]]
    if baselines:
        anchor = float(np.median(baselines))
        tolerance = max(8.0, height * 0.05)
        per_frame = [
            {
                "frame": f["frame"],
                "boxes": [b for b in f["boxes"] if abs((b[1] + b[3]) - anchor) <= tolerance],
            }
            for f in per_frame
        ]
        per_frame = [f for f in per_frame if f["boxes"]]
    hits = len(per_frame)
    ratio = hits / max(1, inspected)
    # One box covering every surviving hit: the caption does not move, so the
    # union is tight, and a single static rectangle is what `delogo` acts on.
    detected = ratio >= args.min_frames
    result = {
        "video": args.video,
        "width": width,
        "height": height,
        "frames_inspected": inspected,
        "frames_with_text": hits,
        "ratio": round(ratio, 3),
        "detected": detected,
        "engine": args.engine,
    }
    if detected:
        xs = [b[0] for f in per_frame for b in f["boxes"]]
        ys = [b[1] for f in per_frame for b in f["boxes"]]
        x2 = [b[0] + b[2] for f in per_frame for b in f["boxes"]]
        y2 = [b[1] + b[3] for f in per_frame for b in f["boxes"]]
        # 5th/95th percentile rather than min/max: one stray contour in one
        # frame should not stretch the mask across the whole picture.
        x = max(0, int(np.percentile(xs, 5)) - args.pad)
        y = max(0, int(np.percentile(ys, 5)) - args.pad)
        right = min(width, int(np.percentile(x2, 95)) + args.pad)
        bottom = min(height, int(np.percentile(y2, 95)) + args.pad)
        result["box"] = {"x": x, "y": y, "w": max(2, right - x), "h": max(2, bottom - y)}
        result["coverage"] = round(
            (right - x) * (bottom - y) / float(width * height), 4
        )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
