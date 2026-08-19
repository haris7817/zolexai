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


async def extract_frames_at(
    source: Path,
    timestamps: list[float],
    destination_dir: Path,
    *,
    prefix: str = "keyframe",
    timeout: float = 120.0,
) -> list[Path]:
    """Grabs one still per timestamp. Returns them in the order asked for.

    This is how a restyle keeps the source's composition: the model is shown
    what the shot actually looks like at several moments inside the window it
    is about to generate, so the subject stays where the subject was and the
    camera keeps moving the way it was moving.

    Seeking happens BEFORE `-i`, which makes ffmpeg jump to the nearest
    keyframe and decode forward from there rather than decoding the whole file
    per still — the difference between one and several minutes on a long
    source. Frames are extracted one at a time on purpose: a single-pass filter
    graph would fail the whole set when one timestamp lands past the last
    packet, which is exactly what the final timestamp of a segment tends to do.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    for index, when in enumerate(timestamps):
        dest = destination_dir / f"{prefix}-{index:03d}.png"
        try:
            await ffmpeg(
                [
                    "-ss", f"{max(0.0, when):.3f}",
                    "-i", str(source),
                    "-frames:v", "1",
                    "-update", "1",
                    str(dest),
                ],
                timeout=timeout,
            )
        except FfmpegError:
            dest.unlink(missing_ok=True)
            raise
        if not dest.exists() or dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            raise FfmpegError(
                f"no decodable frame at {when:.2f}s in {source.name}"
            )
        extracted.append(dest)

    return extracted


async def normalize_clip(
    source: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: float,
    audio: bool,
    frames: int | None = None,
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

    `frames` pins the output to exactly that many frames. A stitched timeline
    needs this: every re-time can only produce WHOLE frames, so a section
    whose planned length is not a whole number of frames comes back a fraction
    long, and butt-joining such sections against one continuous soundtrack
    accumulates that fraction at every seam. `None` keeps the historical
    behaviour: the clip's own length, quantized.

    The pad that backs the pin is deliberately TWO FRAMES, not open-ended.
    Rounding is the only legitimate shortfall and it is under one frame; a
    clip materially shorter than its pin is a faulty render, and cloning its
    last picture up to length would hide exactly the wrong-length fault the
    output verification exists to catch. Such a clip comes back short here,
    and fails honestly there.
    """
    info = await probe_media(source)
    if not info.has_video:
        raise FfmpegError(f"{source.name} has no video stream to normalize")

    filters = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},fps={fps:g},format=yuv420p"
    )
    if frames is not None:
        if frames < 1:
            raise ValueError(f"a normalized clip needs at least one frame, got {frames}")
        # The clone-pad + hard count recipe the control extractor uses —
        # `-frames:v` alone cannot lengthen a short clip — but bounded to the
        # rounding it exists to absorb (see the docstring).
        filters += f",tpad=stop_mode=clone:stop_duration={2 / max(fps, 1e-6):.3f}"

    args = ["-i", str(source)]
    synthesize_silence = audio and not info.has_audio
    if synthesize_silence:
        # anullsrc is endless; `-shortest` below cuts it at the video's end.
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]

    args += ["-filter:v", filters, "-map", "0:v:0"]
    if frames is not None:
        args += ["-frames:v", str(frames), "-fps_mode", "cfr"]
    if synthesize_silence:
        args += ["-map", "1:a:0", "-shortest", *_AUDIO_ARGS]
    elif audio:
        args += ["-map", "0:a:0", *_AUDIO_ARGS]
    else:
        args += ["-an"]
    args += [*_VIDEO_ARGS, str(dest)]

    await ffmpeg(args, timeout=timeout)
    return dest
