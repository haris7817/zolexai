"""Person matting for `execution.v2v_person_lock`, run on a GPU node.

Deliberately a CLI rather than an import. The worker has no torch and should
not grow one to obtain a mask, so this runs in the LTX environment — the same
place the pipelines run — and speaks a small, stable argument list. Swapping
the segmentation model behind it is a change to this file alone.

Contract, and every part of it matters:

  * frames are matted at the render's own grid, frame rate, window and FRAME
    COUNT. A matte of a different length protects the wrong part of the picture
    on every frame, which is worse than no matte at all;
  * the output is a greyscale video — white where the people are, black
    elsewhere, soft in between;
  * the matte is smoothed across time, then grown, then feathered. Each of
    those is load-bearing and the reasons are at the constants below.

Model: BiRefNet (MIT licence), which mattes a foreground subject without
needing a prompt, a box or a click. Measured 2026-08-18 at 1024x576: a clean
silhouette of a man and the bird on his outstretched glove, no per-frame
flicker after smoothing.

Usage (the worker builds this line; it is documented here for hand-runs):

    uv run python -m worker_tools.person_matte \\
        --source clip.mp4 --dest matte.mp4 \\
        --start-seconds 0 --duration-seconds 8.04 \\
        --width 1024 --height 576 --fps 24 --frames 193
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: How strongly each frame's matte is pulled toward the previous frame's.
#:
#: Segmentation is per frame and independent, so a single frame where the model
#: loses a shoulder becomes a one-frame hole in the conditioning — which the
#: render turns into a visible flicker exactly on the subject. Leaning on the
#: previous matte costs a little edge crispness on fast motion and removes that
#: failure entirely. Measured at 0.4: stable, with no visible lag on a walking
#: subject.
TEMPORAL_BLEND = 0.4

#: Matting resolution. BiRefNet is trained at 1024x1024; feeding it the render's
#: aspect directly loses accuracy at the edges, so mattes are computed square
#: and resized back.
MATTE_SIDE = 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--dilation-passes", type=int, default=2)
    parser.add_argument("--feather-radius", type=int, default=12)
    parser.add_argument(
        "--model", default="ZhengPeng7/BiRefNet",
        help="HuggingFace id of the matting model.",
    )
    return parser.parse_args(argv)


def extract_frames(args: argparse.Namespace, into: Path) -> list[Path]:
    """The render's exact window, on the render's exact grid.

    The scale/crop/fps/pad recipe mirrors `worker.media.control` so the matte
    registers pixel for pixel with the edge map it will be merged against.
    """
    pad_seconds = args.frames / max(args.fps, 1e-6)
    filters = (
        f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,"
        f"crop={args.width}:{args.height},"
        f"fps={args.fps:g},"
        f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{max(0.0, args.start_seconds):.3f}",
            "-t", f"{max(0.0, args.duration_seconds):.3f}",
            "-i", str(args.source),
            "-filter:v", filters,
            "-frames:v", str(args.frames),
            "-fps_mode", "cfr",
            str(into / "f%05d.png"),
        ],
        check=True,
    )
    frames = sorted(into.glob("f*.png"))
    if len(frames) != args.frames:
        raise SystemExit(
            f"expected {args.frames} frames from the window, extracted {len(frames)}"
        )
    return frames


def matte_frames(frames: list[Path], into: Path, model_id: str) -> None:
    """Writes one greyscale matte per frame, smoothed against its predecessor."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True
    )
    model.to("cuda").eval().half()

    prepare = transforms.Compose(
        [
            transforms.Resize((MATTE_SIDE, MATTE_SIDE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    previous: np.ndarray | None = None
    with torch.no_grad():
        for index, frame in enumerate(frames):
            image = Image.open(frame).convert("RGB")
            tensor = prepare(image).unsqueeze(0).to("cuda").half()
            prediction = model(tensor)[-1].sigmoid().float().cpu()[0, 0].numpy()
            matte = np.asarray(
                Image.fromarray((prediction * 255).astype("uint8")).resize(image.size)
            ).astype("float32")
            if previous is not None:
                matte = (1.0 - TEMPORAL_BLEND) * matte + TEMPORAL_BLEND * previous
            previous = matte
            Image.fromarray(matte.astype("uint8")).save(into / frame.name)
            if index % 48 == 0:
                print(f"matted {index}/{len(frames)}", flush=True)


def encode_matte(args: argparse.Namespace, mattes: Path) -> None:
    """Grows, feathers and encodes the per-frame mattes into one clip."""
    grow = ",".join(["dilation=coordinates=255"] * max(0, args.dilation_passes))
    chain = [f for f in (grow, f"boxblur={max(0, args.feather_radius)}") if f]
    chain.append("format=yuv420p")

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-framerate", f"{args.fps:g}",
            "-i", str(mattes / "f%05d.png"),
            "-filter:v", ",".join(chain),
            "-frames:v", str(args.frames),
            "-fps_mode", "cfr",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            str(args.dest),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")

    workspace = Path(tempfile.mkdtemp(prefix="person-matte-"))
    try:
        source_frames = workspace / "frames"
        mattes = workspace / "mattes"
        source_frames.mkdir()
        mattes.mkdir()

        frames = extract_frames(args, source_frames)
        matte_frames(frames, mattes, args.model)
        encode_matte(args, mattes)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if not args.dest.exists() or args.dest.stat().st_size == 0:
        raise SystemExit("matting produced no output")
    print(f"matte written to {args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
