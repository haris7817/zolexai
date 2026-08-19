"""Measure a V2V output's TIMING against its source — offset, not similarity.

`drift_check.sh` answers "does the output still resemble the source"; this
answers the lip-sync question, which is about WHEN: at several points of the
delivered file, how far is the picture's content from the moment of the
source timeline the soundtrack is currently playing? A file can match its
source's length to the millisecond and still be out of sync everywhere in
between — duration is one number, sync is a curve.

Two measurements, both against the same source file:

**Video content offset.** At each probe point t, the output's frame is
compared against a window of source frames around t, and the best-matching
source time t' is reported as `offset = t' − t`. Positive means the picture
shows a LATER source moment than the audio is playing (video leads); negative
means the picture lags — the mouth moves after the words, the direction the
stitched-section arithmetic predicts and the one viewers forgive least.
Frames are compared as tiny grayscale thumbnails: the restyle changes every
pixel's value but not the composition's motion, and at 64x36 the motion is
what survives.

**Audio offset.** The output's soundtrack is cross-correlated against the
source's. For video-to-video these are the same recording, so anything
nonzero here is a mux/encoder-delay artefact — worth separating from the
video measurement, because the fix for each lives in a different place.

Pure Python + ffmpeg on purpose: this runs on the GPU box AND on a dev
machine, and the worker environment carries no numpy.

    python scripts/av_offset_probe.py SOURCE.mp4 OUTPUT.mp4
    python scripts/av_offset_probe.py SOURCE.mp4 OUTPUT.mp4 --points 9 --window 0.75
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

#: Thumbnail grid for frame comparison. Small enough that a ±0.75s window of
#: candidates is a few hundred kilobytes, big enough that a walking figure or
#: a turning head lands in different cells.
THUMB_W, THUMB_H = 64, 36

#: Audio correlation rate. Speech energy structure survives 4 kHz easily and
#: keeps the pure-Python correlation loop under a few seconds.
AUDIO_RATE = 4000


def _run(cmd: list[str]) -> bytes:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(
            f"{cmd[0]} failed: {result.stderr.decode('utf-8', 'replace')[-400:]}"
        )
    return result.stdout


def probe_duration(path: Path) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    value = json.loads(out or b"{}").get("format", {}).get("duration")
    if not value:
        raise SystemExit(f"{path} reports no duration")
    return float(value)


def gray_frames(path: Path, start: float, duration: float, fps: float) -> list[bytes]:
    """Tiny grayscale frames of one window, one `bytes` per frame."""
    raw = _run([
        "ffmpeg", "-v", "error", "-nostdin",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}", "-i", str(path),
        "-vf", f"fps={fps:g},scale={THUMB_W}:{THUMB_H},format=gray",
        "-f", "rawvideo", "-",
    ])
    size = THUMB_W * THUMB_H
    return [raw[i : i + size] for i in range(0, len(raw) - size + 1, size)]


def frame_distance(a: bytes, b: bytes) -> float:
    """Mean absolute difference of two grayscale thumbnails, mean-equalized.

    Each frame is shifted to zero mean before comparing, so a restyle's
    global brightness change does not swamp the motion signal.
    """
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    return sum(
        abs((pa - mean_a) - (pb - mean_b)) for pa, pb in zip(a, b, strict=True)
    ) / len(a)


#: Consecutive frames matched as one block. A single talking-head frame has
#: near-duplicates on both sides of it and a lone-frame match snaps to any of
#: them; half a second of MOTION is unambiguous.
BLOCK_FRAMES = 12


def video_offset_at(
    source: Path, output: Path, t: float, window: float, fps: float
) -> float | None:
    """Best-matching source time minus t, at probe point t of the output.

    Matches a BLOCK of consecutive frames, not one frame: the score of a
    candidate alignment is the summed distance across the whole block, so the
    match locks onto the motion trajectory rather than onto whichever single
    near-duplicate frame happens to score best.
    """
    target = gray_frames(output, t, (BLOCK_FRAMES + 0.5) / fps, fps)[:BLOCK_FRAMES]
    if len(target) < 2:
        return None
    candidates = gray_frames(
        source, t - window, 2 * window + (BLOCK_FRAMES + 0.5) / fps, fps
    )
    slides = len(candidates) - len(target) + 1
    if slides < 1:
        return None
    scores = [
        sum(
            frame_distance(frame, candidates[i + k])
            for k, frame in enumerate(target)
        )
        for i in range(slides)
    ]
    best = min(range(slides), key=scores.__getitem__)
    return (t - window + best / fps) - t


def mono_audio(path: Path, seconds: float) -> list[float]:
    raw = _run([
        "ffmpeg", "-v", "error", "-nostdin", "-t", f"{seconds:.3f}", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ])
    count = len(raw) // 2
    return [v / 32768.0 for v in struct.unpack(f"<{count}h", raw[: count * 2])]


def audio_offset(source: Path, output: Path, *, span: float = 3.0, max_lag: float = 0.5) -> float | None:
    """Lag (seconds) of the output's track behind the source's, by peak
    cross-correlation. Positive = the output's audio starts LATE."""
    a = mono_audio(source, span + max_lag)
    b = mono_audio(output, span + max_lag)
    if not a or not b:
        return None
    n = int(span * AUDIO_RATE)
    lags = range(-int(max_lag * AUDIO_RATE), int(max_lag * AUDIO_RATE) + 1, 4)
    best_lag, best_score = 0, float("-inf")
    for lag in lags:
        score = 0.0
        for i in range(0, n, 2):  # every other sample is plenty at 4 kHz
            j = i + lag
            if 0 <= j < len(b) and i < len(a):
                score += a[i] * b[j]
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag / AUDIO_RATE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--points", type=int, default=5,
                        help="probe points, spread from start to end")
    parser.add_argument("--window", type=float, default=0.75,
                        help="± seconds of source searched around each point")
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args(argv)

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is required")
    for path in (args.source, args.output):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")

    duration = min(probe_duration(args.source), probe_duration(args.output))
    margin = args.window + 0.5
    span = duration - 2 * margin
    if span <= 0 or args.points < 2:
        raise SystemExit("clip too short for the requested probe layout")

    print(f"source {args.source.name}  output {args.output.name}  ({duration:.2f}s)")
    print(f"{'point':>8} {'t':>8} {'video offset':>14}")
    offsets: list[tuple[float, float]] = []
    for index in range(args.points):
        t = margin + span * index / (args.points - 1)
        offset = video_offset_at(args.source, args.output, t, args.window, args.fps)
        shown = f"{offset * 1000:+8.0f} ms" if offset is not None else "unmeasurable"
        print(f"{index + 1:>8} {t:>7.2f}s {shown:>14}")
        if offset is not None:
            offsets.append((t, offset))

    mux = audio_offset(args.source, args.output)
    if mux is not None:
        print(f"\naudio track offset vs source: {mux * 1000:+.0f} ms "
              "(nonzero = mux/encoder delay, not a model problem)")

    if len(offsets) >= 2:
        (t0, o0), (t1, o1) = offsets[0], offsets[-1]
        slope = (o1 - o0) / (t1 - t0) * 1000
        print(
            f"\nstart {o0 * 1000:+.0f} ms → end {o1 * 1000:+.0f} ms "
            f"({slope:+.1f} ms per second of runtime)\n"
            "  flat & near zero: in sync. flat & nonzero: constant offset "
            "(mux/start). growing: cumulative drift (stitching)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
