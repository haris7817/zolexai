"""Audio: muxing a finished track onto picture, joining sections, measuring time.

Two client requirements land here, and they pull in opposite directions.

**A music video must carry the whole uploaded song, continuous.** The visuals
are produced in sections; the audio must not be. So the assembly order is
"stitch every visual section first, then lay the single original track over the
result once" — `mux_audio` is the only place a soundtrack is attached, and it
attaches the user's file, whole, exactly once.

**A generated song longer than one model pass is several passes.** Those really
are separate audio files and joining them with a hard cut is audible, so
`crossfade_concat` overlaps them and `loudness_normalize` stops the seam from
being a volume step instead.

The timing layer at the bottom is deliberately small: an amplitude envelope and
a peak picker, no dependencies beyond ffmpeg. It is enough to put a visual cut
on a musical event rather than on an arbitrary multiple of the pass ceiling. It
is NOT beat tracking, downbeat detection or structural segmentation, and
nothing in the product claims it is.
"""

from __future__ import annotations

import math
import struct
from enum import StrEnum
from pathlib import Path

from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg, ffmpeg_stdout
from worker.media.probe import probe_media

logger = get_logger(__name__)


class AudioMode(StrEnum):
    """The four supported ownership models for a finished soundtrack."""

    SOURCE_AUDIO = "SOURCE_AUDIO"
    GENERATED_MASTER_AUDIO = "GENERATED_MASTER_AUDIO"
    GENERATED_PER_SECTION_AUDIO = "GENERATED_PER_SECTION_AUDIO"
    NO_AUDIO = "NO_AUDIO"

#: One delivery encoding for every soundtrack we attach or produce.
_AAC_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

#: Below this the video is treated as already covering the track and the mux is
#: a stream copy. Roughly one frame at 24fps — inaudible, and not worth a
#: full re-encode of a four-minute video to recover.
_PAD_TOLERANCE_SECONDS = 0.05

#: Ceiling on how much still frame may be cloned to cover a short video. A
#: larger gap than this is a planning bug, not a rounding artefact, and padding
#: it would hide the bug behind a frozen picture.
_MAX_PAD_SECONDS = 3.0


async def mux_audio(
    video: Path,
    audio: Path,
    dest: Path,
    *,
    timeout: float = 1800.0,
) -> Path:
    """Lays one continuous audio track over `video`, and returns `dest`.

    The audio decides the length. Any sound the video already carries is
    dropped rather than mixed — for a music video that would be the model's own
    invented audio playing underneath the user's song.

    Video that falls a fraction short of the track (frame-count rounding makes
    this normal) has its final frame held rather than letting `-shortest` cut
    the last moment of the song off. That path re-encodes; the common path,
    where the video already covers the track, is a stream copy.
    """
    picture = await probe_media(video)
    track = await probe_media(audio)
    if not picture.has_video:
        raise FfmpegError(f"{video.name} has no video stream to mux onto")
    if not track.has_audio:
        raise FfmpegError(f"{audio.name} has no audio stream to mux")

    shortfall = 0.0
    if picture.duration_seconds and track.duration_seconds:
        shortfall = track.duration_seconds - picture.duration_seconds

    args = ["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
    if shortfall > _PAD_TOLERANCE_SECONDS:
        if shortfall > _MAX_PAD_SECONDS:
            raise FfmpegError(
                f"{video.name} is {shortfall:.2f}s shorter than {audio.name}; "
                "that is a planning failure, not frame rounding"
            )
        logger.info("mux_padding_video", extra={"shortfall_seconds": round(shortfall, 3)})
        args += [
            "-filter:v",
            f"tpad=stop_mode=clone:stop_duration={shortfall + 0.5:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        ]
    else:
        args += ["-c:v", "copy"]

    # `-shortest` against a video that now covers the track means the TRACK is
    # what ends the file — which is the promise: the result is the song's
    # length and the song plays to its end.
    args += [*_AAC_ARGS, "-shortest", "-movflags", "+faststart", str(dest)]
    await ffmpeg(args, timeout=timeout)
    return dest


async def crossfade_concat(
    paths: list[Path],
    dest: Path,
    *,
    fade_seconds: float = 1.0,
    timeout: float = 1800.0,
) -> Path:
    """Joins audio sections with an overlap instead of a cut.

    A butt-join between two independently generated sections is audible as a
    click or a sudden change of room. `acrossfade` overlaps them, which also
    means the result is SHORTER than the sum of its parts by one fade per
    join — the caller's plan has to account for that, and
    `overlap_cost_seconds` below is that arithmetic.
    """
    if not paths:
        raise ValueError("nothing to join")
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FfmpegError(f"missing audio sections: {', '.join(missing)}")
    if len(paths) == 1:
        paths[0].replace(dest)
        return dest

    fade = max(0.05, fade_seconds)
    inputs: list[str] = []
    for path in paths:
        inputs += ["-i", str(path)]

    # Chained pairwise crossfades: [0][1]->a1, [a1][2]->a2, ...
    steps = []
    previous = "0:a"
    for index in range(1, len(paths)):
        label = f"a{index}"
        steps.append(
            f"[{previous}][{index}:a]acrossfade=d={fade:g}:c1=tri:c2=tri[{label}]"
        )
        previous = label
    graph = ";".join(steps)

    await ffmpeg(
        [*inputs, "-filter_complex", graph, "-map", f"[{previous}]", str(dest)],
        timeout=timeout,
    )
    return dest


def overlap_cost_seconds(section_count: int, fade_seconds: float) -> float:
    """How much total length `crossfade_concat` will consume in its joins.

    Each join overlaps two sections by one fade, so N sections lose N-1 fades.
    A planner that ignores this produces a song noticeably shorter than the
    length the user picked.
    """
    return max(0, section_count - 1) * max(0.0, fade_seconds)


async def loudness_normalize(
    source: Path, dest: Path, *, timeout: float = 1800.0
) -> Path:
    """One consistent playback level across the whole track.

    Independently generated sections arrive at independent levels, and a
    crossfade between two of them turns that difference into an audible step
    mid-song. EBU R128 at the streaming-typical -14 LUFS.
    """
    await ffmpeg(
        [
            "-i", str(source),
            "-filter:a", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100",
            str(dest),
        ],
        timeout=timeout,
    )
    return dest


# ── Timing: where a cut can land without fighting the music ──────────────

#: Envelope resolution. 10ms is finer than any cut a viewer perceives as
#: mistimed and coarse enough that a five-minute track is 30k floats.
_HOP_SECONDS = 0.01

#: The envelope is measured from a mono, heavily downsampled decode — this is
#: amplitude over time, not audio anyone listens to.
_ENVELOPE_RATE = 8000


async def audio_envelope(
    path: Path, *, hop_seconds: float = _HOP_SECONDS, timeout: float = 600.0
) -> list[float]:
    """Short-time RMS of a track, one value per hop.

    Decoded to mono 8 kHz through a pipe: enough resolution to see where the
    energy jumps, small enough that a five-minute song is a few megabytes and
    a fraction of a second of arithmetic.
    """
    raw = await ffmpeg_stdout(
        [
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(_ENVELOPE_RATE),
            "-f", "s16le", "-acodec", "pcm_s16le", "-",
        ],
        timeout=timeout,
    )
    if not raw:
        raise FfmpegError(f"{path.name} decoded to no audio samples")

    samples = struct.unpack(f"<{len(raw) // 2}h", raw[: len(raw) // 2 * 2])
    window = max(1, int(hop_seconds * _ENVELOPE_RATE))

    envelope: list[float] = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            break
        total = sum(value * value for value in chunk)
        envelope.append(math.sqrt(total / len(chunk)) / 32768.0)
    return envelope


def detect_onsets(
    envelope: list[float],
    *,
    hop_seconds: float = _HOP_SECONDS,
    min_gap_seconds: float = 0.15,
    sensitivity: float = 1.5,
) -> list[float]:
    """Times (seconds) where the track's energy rises sharply.

    A rise in log-energy is the cheapest usable onset signal, and an adaptive
    threshold — local mean plus `sensitivity` standard deviations — keeps a
    quiet passage from being ignored and a loud one from firing on every
    sample. `min_gap_seconds` stops one drum hit registering as four.

    This finds events, which is all a cut point needs. It does not identify
    beats, downbeats, bars or sections, and callers must not describe it as
    though it does.
    """
    if len(envelope) < 3:
        return []

    # Positive difference of log energy: loudness is multiplicative, so a
    # linear difference would make quiet onsets invisible next to loud ones.
    flux = [0.0]
    for previous, current in zip(envelope, envelope[1:], strict=False):
        rise = math.log(current + 1e-6) - math.log(previous + 1e-6)
        flux.append(max(0.0, rise))

    mean = sum(flux) / len(flux)
    variance = sum((value - mean) ** 2 for value in flux) / len(flux)
    threshold = mean + sensitivity * math.sqrt(variance)

    min_gap = max(1, int(min_gap_seconds / hop_seconds))
    onsets: list[float] = []
    last = -min_gap
    for index in range(1, len(flux) - 1):
        value = flux[index]
        if value < threshold or value <= flux[index - 1] or value < flux[index + 1]:
            continue
        if index - last < min_gap:
            continue
        onsets.append(index * hop_seconds)
        last = index
    return onsets


async def audio_onsets(path: Path, *, timeout: float = 600.0) -> list[float]:
    """`audio_envelope` then `detect_onsets` — the whole timing layer."""
    envelope = await audio_envelope(path, timeout=timeout)
    return detect_onsets(envelope)
