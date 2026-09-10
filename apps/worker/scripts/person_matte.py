#!/usr/bin/env python3
"""Generate an aligned soft person matte clip with local BiRefNet weights."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from _person_segmentation import PersonSegmenter, run_checked
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--start-seconds", required=True, type=float)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--dilation-passes", required=True, type=int)
    parser.add_argument("--feather-radius", required=True, type=int)
    parser.add_argument("--model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import cv2
    import numpy as np

    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="zolexai-matte-") as directory:
        root = Path(directory)
        frames_pattern = root / "frame-%06d.png"
        masks_pattern = root / "mask-%06d.png"
        pad_seconds = args.frames / max(args.fps, 1e-6)
        run_checked(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{max(0.0, args.start_seconds):.3f}",
                "-t",
                f"{max(0.0, args.duration_seconds):.3f}",
                "-i",
                str(args.source),
                "-vf",
                (
                    f"scale={args.width}:{args.height}:"
                    "force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={args.width}:{args.height},fps={args.fps:g},"
                    f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}"
                ),
                "-frames:v",
                str(args.frames),
                str(frames_pattern),
            ]
        )

        segmenter = PersonSegmenter(args.model)
        kernel = np.ones((5, 5), np.uint8)
        blur = max(1, args.feather_radius * 2 + 1)
        if blur % 2 == 0:
            blur += 1
        for index in range(1, args.frames + 1):
            frame_path = root / f"frame-{index:06d}.png"
            if not frame_path.exists():
                raise RuntimeError(f"source extraction stopped at frame {index}")
            mask = segmenter.mask(Image.open(frame_path).convert("RGB"))
            if args.dilation_passes > 0:
                mask = cv2.dilate(mask, kernel, iterations=args.dilation_passes)
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
            Image.fromarray(mask).save(root / f"mask-{index:06d}.png")

        args.dest.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                f"{args.fps:g}",
                "-i",
                str(masks_pattern),
                "-frames:v",
                str(args.frames),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(args.dest),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
