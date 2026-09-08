"""The music-video worker's 4K finishing command: exact frames, on the GPU.

The client's package finishes a job by scaling the assembled working master
once to 4K. Its built-in CUDA path is `scale_cuda` plus NVENC with
`-shortest`, and on a long job that arrives a few frames short: measured
8 Sep 2026 on a 3-minute song, a working master of 4,319 frames produced a
delivery of 4,315, and the package's own delivery QA correctly refused the
job. A 40-second job through the same code was exact, so the fault only
shows at length. It is nothing to do with which engine rendered the shots.

Their README documents a command adapter for exactly this
("Automatic 4K upscale"), so rather than edit their code the deployment
supplies this. Two differences from the built-in path, and both are the
reason it lands exactly:

  * **No `-shortest`.** The frame count is stated instead, and the stream
    that decides the length is the video.
  * **`-fps_mode passthrough`.** The concatenated master carries exact
    1/12288 timestamps; asking ffmpeg to re-derive a constant rate is what
    lets it decide a frame is surplus. The rate is not forced beside it —
    ffmpeg rejects the pair, and the master is already exactly 24 fps.

The picture treatment is the client's own: scale to cover the delivery
frame, centre-crop the overhang, NVENC. This writes VIDEO ONLY, which is
their contract — the worker then remuxes the untouched audio from the
working master.

Usage:  mv_upscale.py --request REQUEST.json --output OUT.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _log(message: str, **fields: object) -> None:
    print(json.dumps({"mv_upscale": message, **fields}), file=sys.stderr, flush=True)


def _stream_seconds(path: Path, kind: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind[0] + ":0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    try:
        return float(out[0])
    except (IndexError, ValueError):
        return 0.0


def repair_short_audio(master: Path) -> bool:
    """Make the working master's audio last at least as long as its picture.

    A defect in the package, found 8 Sep 2026 and reported: it assembles the
    master with `-c:a aac … -shortest`, so the AAC track is truncated to the
    last whole 1024-sample frame that fits inside the video. On a 3-minute
    job that left the audio 16 samples short of the picture. Its 4K stage
    then remuxes with `-shortest` again, which drops every video frame whose
    end falls past the audio — four of them, since the encoder's B-frame
    group goes with it — and its own delivery QA refuses the job for being
    four frames short. A 40-second job happened to land on a boundary and
    passed, which is why this only shows at length.

    The repair is one AAC encode from the SAME lossless source the package
    used, `audio/delivery-aligned.wav`, with a quarter-second of silence
    appended. The song is still a single generation from the master WAV, and
    the package's own remux trims the tail back off, so the delivered audio
    is unchanged. Does nothing when the audio already outlasts the picture.
    """
    video, audio = _stream_seconds(master, "video"), _stream_seconds(master, "audio")
    if not video or audio >= video - 1e-9:
        return False
    aligned = master.parents[2] / "audio" / "delivery-aligned.wav"
    source = aligned if aligned.is_file() else master
    patched = master.with_name(f"{master.stem}.audio-repaired.mp4")
    argv = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", str(master),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", "apad=pad_dur=0.25",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(patched),
    ]
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0 or not patched.is_file():
        _log("audio_repair_failed", error=done.stderr.strip()[-300:])
        patched.unlink(missing_ok=True)
        return False
    os.replace(patched, master)
    _log("audio_repaired", was=round(audio, 6), video=round(video, 6),
         now=round(_stream_seconds(master, "audio"), 6), source=source.name)
    return True


def build_argv(request: dict, output: Path, *, cuda: bool, cq: int) -> list[str]:
    width = int(request["delivery_width"])
    height = int(request["delivery_height"])
    frames = int(request["frames"])

    # Cover the delivery frame, then trim the overhang from the middle. The
    # source carries the model's alignment padding, so a plain scale would
    # stretch it: 1280x704 is 1.818:1 against 16:9's 1.778.
    cover = (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    argv = [
        "ffmpeg",
        "-v", "error",
        "-nostdin",
        "-y",
        "-i", str(request["input"]),
        "-map", "0:v:0",
        "-an",
        "-vf", cover,
        # Every decoded frame reaches the encoder, in order, once. No `-r`
        # beside it: ffmpeg refuses a forced rate together with a non-CFR
        # mode, and the master this reads is already exactly 24 fps.
        "-fps_mode", "passthrough",
        "-frames:v", str(frames),
    ]
    if cuda:
        argv += [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
            # No B-frames. The package remuxes this file with `-shortest`,
            # and a reorder group left in flight when that fires takes three
            # more frames with it (measured: 4 lost with B-frames, 1 without).
            "-bf", "0",
        ]
    else:
        argv += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cq), "-pix_fmt", "yuv420p"]
    argv += ["-movflags", "+faststart", str(output)]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(str(request.get("output") or args.output)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from worker.core.config import settings

    # Before anything is encoded: the package's remux runs `-shortest` over
    # this same master, and a master whose audio stops first loses the tail
    # of the picture.
    repair_short_audio(Path(str(request["input"])))

    cuda = settings.music_video_upscale_backend != "cpu"
    argv = build_argv(request, output, cuda=cuda, cq=int(settings.music_video_upscale_cq))
    env = dict(os.environ)
    if cuda and request.get("gpu_id"):
        env["CUDA_VISIBLE_DEVICES"] = str(request["gpu_id"])

    started = time.monotonic()
    _log("starting", frames=request.get("frames"),
         size=f"{request.get('delivery_width')}x{request.get('delivery_height')}", cuda=cuda)
    completed = subprocess.run(argv, env=env, capture_output=True, text=True)
    if completed.returncode != 0 and cuda:
        # A node without NVENC is a deployment fact, not a job failure.
        _log("nvenc_failed_retrying_on_cpu", error=completed.stderr.strip()[-300:])
        argv = build_argv(request, output, cuda=False, cq=int(settings.music_video_upscale_cq))
        completed = subprocess.run(argv, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr.strip()[-2000:], file=sys.stderr)
        raise SystemExit(completed.returncode)
    if not output.is_file():
        raise SystemExit(f"mv_upscale: no file at {output}")
    _log("finished", seconds=round(time.monotonic() - started, 1), output=str(output))


if __name__ == "__main__":
    main()
