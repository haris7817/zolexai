"""Measuring media instead of assuming it.

M1 reported an output's dimensions from a lookup table keyed on the requested
aspect ratio, and its duration by parsing the string the user asked for. Both
were fine for a placeholder and both are assertions, not measurements — a real
model returns what it returns, and the asset row should record that.

This is also what makes the client's automatic-duration requirements possible:
"final video matches the source" needs the source's actual length, not a value
the user typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker.media.ffmpeg import FfmpegError, ffprobe_json


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float | None
    width: int | None
    height: int | None
    has_video: bool
    has_audio: bool

    fps: float | None = None
    """Average video frame rate. Needed when another clip must be normalized
    to match this one — retiming to a guessed rate produces judder."""

    @property
    def aspect_ratio(self) -> float | None:
        if not self.width or not self.height:
            return None
        return self.width / self.height


async def probe_media(path: Path) -> MediaInfo:
    """Reads real duration, dimensions and stream presence from a file.

    Raises `FfmpegError` if the file is unreadable — which is the useful answer
    for an input a user uploaded that is corrupt, or truncated, or not actually
    the media type its extension claims.
    """
    if not path.exists():
        raise FfmpegError(f"cannot probe missing file {path.name}")

    payload = await ffprobe_json(path)
    streams: list[dict[str, Any]] = payload.get("streams") or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    return MediaInfo(
        duration_seconds=_duration_of(payload, video, audio),
        width=_int_or_none(video.get("width")) if video else None,
        height=_int_or_none(video.get("height")) if video else None,
        has_video=video is not None,
        has_audio=audio is not None,
        fps=_fps_of(video) if video else None,
    )


def _fps_of(video: dict[str, Any]) -> float | None:
    """Parses ffprobe's fractional rate ("30000/1001") into a float.

    `avg_frame_rate` reflects what the stream actually contains;
    `r_frame_rate` is the container's declared tick rate and lies for
    variable-rate sources, so it is only the fallback.
    """
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(video.get(key) or "")
        numerator, _, denominator = raw.partition("/")
        try:
            value = float(numerator) / float(denominator or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if value > 0:
            return value
    return None


def _duration_of(
    payload: dict[str, Any],
    video: dict[str, Any] | None,
    audio: dict[str, Any] | None,
) -> float | None:
    """Container duration first, falling back to a stream's own.

    Some encoders leave the container duration off entirely — notably raw
    streams and a few of the fragmented MP4 variants — so the streams are the
    backstop rather than the primary source.
    """
    for candidate in (payload.get("format") or {}, video or {}, audio or {}):
        value = _float_or_none(candidate.get("duration"))
        if value is not None and value > 0:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
