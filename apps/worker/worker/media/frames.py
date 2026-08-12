"""Frame extraction and clip normalization — the extension/stitching toolkit.

Video extension is "generate a continuation that starts where the source
stopped, then join them". Both halves of that need primitives no adapter
should own privately:

  * `extract_final_frame` — the continuation's conditioning image comes from
    the source's last picture, and later segments chain off the previous
    segment's last picture. Any provider that supports image conditioning
    extends video this way, so the primitive is provider-agnostic.
  * `normalize_clip` — a user's upload and a model's render never share
    resolution, frame rate or encoder parameters, and ffmpeg's concat demuxer
    (which `concat_segments` drives) breaks on mixed parameters. Normalizing
    every part to one explicit target first makes the final concat a
    deterministic stream-copy instead of a lucky one.
"""

from __future__ import annotations

from pathlib import Path

from worker.media.ffmpeg import FfmpegError, ffmpeg
from worker.media.probe import probe_media

#: Everything normalize_clip emits: H.264 + AAC in yuv420p, the one
#: combination every browser, phone and <video> tag plays.
_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]


async def extract_final_frame(source: Path, dest: Path, *, timeout: float = 120.0) -> Path:
    """Writes the last decodable frame of `source` to `dest` (a PNG).

    `-sseof -1` starts decoding one second before the end and `-update 1`
    keeps overwriting `dest` with each decoded frame, so whatever frame the
    stream genuinely ends on wins — including streams whose declared duration
    overshoots the last real packet, where seeking *to* the end would produce
    nothing. Sources shorter than a second decode from their beginning, which
    degrades to the same result.
    """
    try:
        await ffmpeg(
            ["-sseof", "-1", "-i", str(source), "-update", "1", str(dest)],
            timeout=timeout,
        )
    except FfmpegError:
        dest.unlink(missing_ok=True)
        raise
    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FfmpegError(f"no decodable final frame in {source.name}")
    return dest


async def normalize_clip(
    source: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: float,
    audio: bool,
    timeout: float = 900.0,
) -> Path:
    """Re-encodes `source` to exactly the given parameters.

    Aspect mismatches fill and centre-crop rather than letterbox: an extension
    seam where the picture suddenly grows black bars reads as a glitch, while
    a slight crop of one part is invisible.

    `audio=False` strips sound entirely; `audio=True` guarantees exactly one
    sound track, synthesizing silence when the source has none. The caller
    decides once per assembly, so no part of a stitched file can disagree
    about having audio — mixed presence is exactly what breaks the concat
    demuxer.
    """
    info = await probe_media(source)
    if not info.has_video:
        raise FfmpegError(f"{source.name} has no video stream to normalize")

    filters = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},fps={fps:g},format=yuv420p"
    )

    args = ["-i", str(source)]
    synthesize_silence = audio and not info.has_audio
    if synthesize_silence:
        # anullsrc is endless; `-shortest` below cuts it at the video's end.
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]

    args += ["-filter:v", filters, "-map", "0:v:0"]
    if synthesize_silence:
        args += ["-map", "1:a:0", "-shortest", *_AUDIO_ARGS]
    elif audio:
        args += ["-map", "0:a:0", *_AUDIO_ARGS]
    else:
        args += ["-an"]
    args += [*_VIDEO_ARGS, str(dest)]

    await ffmpeg(args, timeout=timeout)
    return dest
