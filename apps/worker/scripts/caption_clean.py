"""Detect burned-in captions with OCR and remove them by temporal inpainting.

Runs in the LTX environment (`uv run python`). Reads a clip, writes a cleaned
clip, prints one JSON report on stdout.

This is the client's specification of 10 Sep 2026, implemented as written:

    Generated frames
    -> OCR caption detection
    -> Expand caption mask
    -> Temporal video inpainting
    -> 1080p/4K/8K upscaling
    -> Final encoding

with EasyOCR for the detection, ProPainter for the inpainting, and the whole
thing placed before the upscale because "removing text after upscale is slower
and leaves larger artifacts".

## Why the hand-written detector this replaces was not good enough

A first version used classical `cv2` heuristics — a morphological gradient,
a brightness floor, a sparseness band, a centredness test. It worked on the
clip it was written against and **missed the very next one**: job 6bdf08bf
scored `ratio 0.0`, not one candidate frame, on a clip whose captions are
plainly visible ("Nicch weatther a hop, lIttle friend"). Its text was fainter
and lower than the clip the thresholds were tuned on, and every one of those
constants was a guess about what a caption looks like.

EasyOCR finds the same clip at 6 of 12 sampled frames with confidences up to
1.00. A detector trained on text does not need to be told what text looks
like, which is the entire argument for the client's choice over ours.

## Where the numbers come from

Only two thresholds remain and neither describes appearance:

* `--min-confidence` — how sure the recogniser must be before a box counts.
* `--min-frames` — how much of the clip must carry text before it is called a
  caption rather than a passing object.

Everything else is measured from the clip in hand. In particular the vertical
band used for the per-frame masks is the median of the boxes THIS clip's
detection pass actually found, not a constant: a caption sits where it sits,
and asking the data is what the previous version failed to do.

## Two passes over the text, on purpose

`readtext` (detection + recognition, with a confidence) decides whether the
clip is captioned at all, on sampled frames. `detect` (boxes only, no
recogniser) then runs on every frame to build the masks, because it is much
cheaper and a mask does not need to know what the letters say.

## The quality check

The client asked for one: "if text is detected and captions are disabled, it
should automatically inpaint the affected frames or regenerate the video."
After inpainting, the output is sampled and read again. `residual_ratio` in
the report is what survived, so the caller can decide — and so a silent
half-fix is impossible to mistake for a clean one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

#: Frames whose OCR confidence is below this do not count toward the decision.
#: EasyOCR reported 0.42-1.00 on real captions and this keeps the weak tail
#: out of the vote without discarding the boxes it draws.
DEFAULT_MIN_CONFIDENCE = 0.30
#: Fraction of sampled frames that must carry text before the clip is called
#: captioned. A passing road sign in three frames of a hundred is the
#: customer's picture; a caption is present for most of the clip.
DEFAULT_MIN_FRAMES = 0.20


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and remove burned-in captions")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="", help="cleaned clip; omit to detect only")
    parser.add_argument("--samples", type=int, default=16,
                        help="frames read by the recogniser for the decision")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-frames", type=float, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--mask-dilation", type=int, default=8,
                        help='"expand caption mask" — pixels grown around each box')
    parser.add_argument("--band-pad", type=float, default=0.08,
                        help="how far outside the measured caption band a per-frame "
                             "box may sit, as a fraction of height")
    parser.add_argument("--resize-ratio", type=float, default=1.0,
                        help="ProPainter working scale; 1.0 is the generation canvas")
    parser.add_argument("--verify-samples", type=int, default=12,
                        help="frames re-read after inpainting for the quality check")
    parser.add_argument("--work-dir", default="")
    return parser.parse_args(argv)


def _reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=True, verbose=False)


def _sample_indices(total: int, samples: int) -> list[int]:
    if total <= 1:
        return [0]
    return [int(i * (total - 1) / max(1, samples - 1)) for i in range(samples)]


def _box_bounds(points) -> tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def detect(video: str, reader, *, samples: int, min_confidence: float,
           min_frames: float) -> dict:
    """Whether this clip carries text, and where the recogniser found it."""
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

    indices = _sample_indices(total, samples)
    hits, boxes, texts, confidences = 0, [], [], []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        found = [
            (points, text, conf)
            for points, text, conf in reader.readtext(frame)
            if conf >= min_confidence
        ]
        if not found:
            continue
        hits += 1
        for points, text, conf in found:
            boxes.append(_box_bounds(points))
            texts.append(str(text))
            confidences.append(float(conf))
    capture.release()

    ratio = hits / max(1, len(indices))
    report = {
        "width": width,
        "height": height,
        "total_frames": total,
        "frames_inspected": len(indices),
        "frames_with_text": hits,
        "ratio": round(ratio, 3),
        "detected": ratio >= min_frames,
        "engine": "easyocr",
        # A sample of what was read, so a human reading the log can see this
        # is caption-shaped nonsense rather than a sign in the scene.
        "sample_text": texts[:6],
        "max_confidence": round(max(confidences), 2) if confidences else 0.0,
    }
    if report["detected"] and boxes:
        tops = [b[1] for b in boxes]
        bottoms = [b[3] for b in boxes]
        # The band is MEASURED from this clip, never assumed. It is what the
        # per-frame mask pass uses to reject boxes the recogniser draws
        # somewhere else in the picture.
        report["band"] = {
            "top": int(np.percentile(tops, 5)),
            "bottom": int(np.percentile(bottoms, 95)),
        }
    return report


def _mask_frames(video: str, reader, mask_dir: str, frame_dir: str, band: dict,
                 *, band_pad: float, dilation: int) -> dict:
    """Write every frame and its caption mask as PNGs, for ProPainter.

    Uses `reader.detect` — boxes only, no recogniser — because a mask does not
    need to know what the letters say and the recogniser is the expensive half.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(video)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    pad = int(height * band_pad)
    low, high = band["top"] - pad, band["bottom"] + pad
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1,) * 2)

    index, masked = 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = np.zeros(frame.shape[:2], np.uint8)
        result = reader.detect(frame)
        # EasyOCR's detect returns (horizontal_boxes, free_form_boxes) nested
        # one level deeper than readtext; both halves are taken.
        horizontal = result[0][0] if result and result[0] else []
        freeform = result[1][0] if len(result) > 1 and result[1] else []
        for box in horizontal:
            x1, x2, y1, y2 = (int(v) for v in box)
            if y2 < low or y1 > high:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        for points in freeform:
            x1, y1, x2, y2 = _box_bounds(points)
            if y2 < low or y1 > high:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        if np.count_nonzero(mask):
            mask = cv2.dilate(mask, kernel)
            masked += 1
        cv2.imwrite(os.path.join(frame_dir, f"{index:06d}.png"), frame)
        cv2.imwrite(os.path.join(mask_dir, f"{index:06d}.png"), mask)
        index += 1
    capture.release()
    return {"frames": index, "frames_masked": masked, "fps": fps}


def _inpaint(frame_dir: str, mask_dir: str, out_path: str, fps: float, *,
             resize_ratio: float, dilation: int) -> dict:
    """ProPainter over the whole clip, written back out as video only."""
    import cv2
    from propainter.propainter_video import (
        FilePathDirSequencer,
        RawFrameSequencer,
        RawMaskSequencer,
        ScaledProPainterIterator,
        run_streaming_propainter,
    )

    frames = RawFrameSequencer(data=FilePathDirSequencer(frame_dir))
    masks = RawMaskSequencer(data=FilePathDirSequencer(mask_dir))
    painted = run_streaming_propainter(
        ScaledProPainterIterator(
            raw_frames=frames,
            raw_masks=masks,
            image_resize_ratio=resize_ratio,
            mask_dilation=dilation,
        )
    )
    if painted is None or not len(painted):
        return {"error": "inpainting produced no frames"}
    height, width = painted[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in painted:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return {"frames_painted": int(len(painted))}


def verify(video: str, reader, *, samples: int, min_confidence: float) -> float:
    """Fraction of sampled frames that still read as text after the repair.

    The client's quality check. A silent half-fix is the failure this exists
    to make impossible.
    """
    import cv2

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return 1.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _sample_indices(total, samples)
    hits = 0
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        if any(conf >= min_confidence for _, _, conf in reader.readtext(frame)):
            hits += 1
    capture.release()
    return round(hits / max(1, len(indices)), 3)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    reader = _reader()
    report = detect(
        args.video,
        reader,
        samples=args.samples,
        min_confidence=args.min_confidence,
        min_frames=args.min_frames,
    )
    if "error" in report:
        print(json.dumps(report))
        return 2
    report["cleaned"] = False
    if not (args.out and report.get("detected") and report.get("band")):
        print(json.dumps(report))
        return 0

    work = args.work_dir or tempfile.mkdtemp(prefix="captions-")
    frame_dir, mask_dir = os.path.join(work, "frames"), os.path.join(work, "masks")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    try:
        masking = _mask_frames(
            args.video, reader, mask_dir, frame_dir, report["band"],
            band_pad=args.band_pad, dilation=args.mask_dilation,
        )
        report.update(masking)
        if not masking["frames_masked"]:
            report["detail"] = "no per-frame boxes fell inside the measured band"
            print(json.dumps(report))
            return 0
        painting = _inpaint(
            frame_dir, mask_dir, args.out, masking["fps"],
            resize_ratio=args.resize_ratio, dilation=args.mask_dilation,
        )
        report.update(painting)
        if "error" in painting:
            print(json.dumps(report))
            return 0
        report["cleaned"] = True
        report["residual_ratio"] = verify(
            args.out, reader,
            samples=args.verify_samples, min_confidence=args.min_confidence,
        )
    finally:
        if not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
