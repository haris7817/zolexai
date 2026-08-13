"""Media tooling shared by every real adapter.

Deliberately provider-agnostic and adapter-independent. Probing a duration,
planning segments, concatenating them, laying a soundtrack over the result and
verifying what came out are the same operations whether the frames came from
LTX, another model, or the ffmpeg test harness — so they live here rather than
inside whichever adapter needed them first (directive §7: "do NOT tightly
couple this logic to one workflow").

Nothing in this package knows what a job is.
"""

from worker.media.audio import (
    audio_envelope,
    audio_onsets,
    crossfade_concat,
    detect_onsets,
    loudness_normalize,
    mux_audio,
    overlap_cost_seconds,
)
from worker.media.ffmpeg import (
    FfmpegError,
    ffmpeg,
    ffmpeg_stdout,
    ffprobe_json,
    tools_available,
)
from worker.media.frames import extract_final_frame, extract_frames_at, normalize_clip
from worker.media.probe import MediaInfo, probe_media
from worker.media.segments import (
    Segment,
    concat_segments,
    plan_segments,
    verify_duration,
)
from worker.media.validate import (
    OutputExpectation,
    duration_tolerance,
    verify_output,
)

__all__ = [
    "FfmpegError",
    "MediaInfo",
    "OutputExpectation",
    "Segment",
    "audio_envelope",
    "audio_onsets",
    "concat_segments",
    "crossfade_concat",
    "detect_onsets",
    "duration_tolerance",
    "extract_final_frame",
    "extract_frames_at",
    "ffmpeg",
    "ffmpeg_stdout",
    "ffprobe_json",
    "loudness_normalize",
    "mux_audio",
    "normalize_clip",
    "overlap_cost_seconds",
    "plan_segments",
    "probe_media",
    "tools_available",
    "verify_duration",
    "verify_output",
]
