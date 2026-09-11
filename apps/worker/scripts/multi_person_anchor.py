#!/usr/bin/env python3
"""Build the strict first-frame identity/background anchor used by V2V v2."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from _person_segmentation import PersonSegmenter, components, composite_person, cover, run_checked
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--start-seconds", required=True, type=float)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= len(args.reference) <= 4:
        raise RuntimeError("provide between one and four --reference images")

    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="zolexai-anchor-") as directory:
        frame_path = Path(directory) / "source.png"
        run_checked(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{max(0.0, args.start_seconds):.3f}",
                "-i",
                str(args.source),
                "-frames:v",
                "1",
                "-vf",
                (
                    f"scale={args.width}:{args.height}:"
                    "force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={args.width}:{args.height}"
                ),
                str(frame_path),
            ]
        )

        source = Image.open(frame_path).convert("RGB")
        segmenter = PersonSegmenter(args.model)
        source_mask = segmenter.mask(source)
        source_people = components(source_mask)
        if len(source_people) < len(args.reference):
            raise RuntimeError(
                f"found {len(source_people)} clear source people but received "
                f"{len(args.reference)} reference images; use a frame where every mapped "
                "person is clearly visible"
            )

        # Ignore tiny/background detections when there are more people than
        # references, then make the public slot mapping screen-left to right.
        targets = sorted(source_people[: len(args.reference)], key=lambda item: item.center_x)

        if args.background:
            base = cover(Image.open(args.background), args.width, args.height)
        else:
            import cv2
            import numpy as np

            pixels = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
            remove = np.zeros((args.height, args.width), dtype=np.uint8)
            for target in targets:
                remove = cv2.bitwise_or(remove, target.mask)
            remove = cv2.dilate(remove, np.ones((11, 11), np.uint8), iterations=2)
            cleaned = cv2.inpaint(pixels, remove, 7, cv2.INPAINT_TELEA)
            base = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))

        for index, (path, target) in enumerate(zip(args.reference, targets, strict=True), 1):
            reference = Image.open(path).convert("RGB")
            mask = segmenter.mask(reference)
            people = components(mask, minimum_area_ratio=0.015)
            if len(people) != 1:
                raise RuntimeError(
                    f"reference {index} must contain one clear person; found {len(people)}"
                )
            composite_person(base, reference, people[0].mask, target)

        args.dest.parent.mkdir(parents=True, exist_ok=True)
        base.save(args.dest, "PNG")
        print(f"mapped {len(targets)} reference people left-to-right")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
