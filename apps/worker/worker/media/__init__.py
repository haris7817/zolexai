"""Media tooling shared by every real adapter.

Deliberately provider-agnostic and adapter-independent. Probing a duration,
planning segments, concatenating them and verifying the result are the same
operations whether the frames came from LTX, another model, or the ffmpeg test
harness — so they live here rather than inside whichever adapter needed them
first (directive §7: "do NOT tightly couple this logic to one workflow").

Nothing in this package knows what a job is.
"""

from worker.media.ffmpeg import FfmpegError, ffmpeg, ffprobe_json, tools_available
from worker.media.frames import extract_final_frame, normalize_clip
from worker.media.probe import MediaInfo, probe_media
from worker.media.segments import (
    Segment,
    concat_segments,
    plan_segments,
    verify_duration,
)

__all__ = [
    "FfmpegError",
    "MediaInfo",
    "Segment",
    "concat_segments",
    "extract_final_frame",
    "ffmpeg",
    "ffprobe_json",
    "normalize_clip",
    "plan_segments",
    "probe_media",
    "tools_available",
    "verify_duration",
]
