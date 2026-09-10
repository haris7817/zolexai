"""The last gate before a job is called complete.

A generation that fails loudly costs one retry. A generation that succeeds
while producing a file with no audio stream, or half the requested length, or
a container nothing can play, costs the customer's trust and is usually found
by the customer. So every adapter runs its finished artifact through here
before returning it, and the checks are deliberately the ones a *player* would
care about rather than the ones the pipeline happens to expose.

`probe_media` measures. This decides whether the measurement is acceptable, and
says precisely what was wrong when it is not — the message goes into the job's
internal detail, where it is the first thing anyone reads at 2am.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg
from worker.media.probe import MediaInfo, probe_media

logger = get_logger(__name__)

#: A container can hold a valid header, a valid stream table and no frames.
#: Anything this small did not come back from a real encode.
_MIN_PLAUSIBLE_BYTES = 1024


def duration_tolerance(expected_seconds: float, *, floor: float = 0.75) -> float:
    """How far a finished file may sit from its plan and still be right.

    Frame counts are integers and encoders round; every join and re-time adds
    a fraction more. Scaling with length rather than using one constant is
    what keeps a four-minute music video from failing on the same absolute
    drift that a five-second clip would rightly be failed for.
    """
    return max(floor, 0.03 * expected_seconds)


@dataclass(frozen=True)
class OutputExpectation:
    """What a finished artifact has to be, stated by the workflow that made it."""

    expect_video: bool = False
    expect_audio: bool = False
    expected_seconds: float | None = None
    tolerance_seconds: float | None = None

    expected_width: int | None = None
    expected_height: int | None = None

    expected_frame_count: int | None = None
    """Exact frames the delivered video must contain.

    Duration alone cannot state this. A file can carry the right number of
    seconds and the wrong number of frames — the container rounds, and a
    workflow that promises "the same length as your video" is promising the
    frames, not a rounded float. Video to Video pins it because its sections
    are planned from a duration and delivered against the source's own count
    (client measurement, 11 Sep 2026: 347 frames in, 345 out).

    `None` means the caller has no exact expectation, which is every workflow
    whose length it chose itself.
    """

    max_av_drift_seconds: float | None = None
    """How far the audio may sit from the video, when both are present.

    Separate from `tolerance_seconds`, which asks whether the FILE is the
    right length. This asks whether its two streams agree with each other, and
    the answer wants to be much tighter: a soundtrack laid once over a stitched
    picture is either aligned or it is visibly late.
    """

    require_decodable: bool = True
    """Decode the file rather than only reading its metadata.

    ffprobe is happy to describe a container it cannot actually play — a
    truncated MP4 reports a duration from its header long after the frames
    stopped. A real decode pass is a few seconds and it is the difference
    between "we measured the file" and "we know it plays".
    """


async def verify_output(path: Path, expectation: OutputExpectation) -> MediaInfo:
    """Raises `FfmpegError` unless `path` is a deliverable artifact.

    Returns the measured info so a caller does not have to probe twice.
    """
    if not path.exists():
        raise FfmpegError(f"output {path.name} was never written")
    size = path.stat().st_size
    if size < _MIN_PLAUSIBLE_BYTES:
        raise FfmpegError(f"output {path.name} is {size} bytes — no media was produced")

    info = await probe_media(path)

    problems: list[str] = []
    if expectation.expect_video and not info.has_video:
        problems.append("no video stream")
    if expectation.expect_audio and not info.has_audio:
        problems.append("no audio stream")
    if expectation.expect_audio and info.audio_stream_count != 1:
        problems.append(
            f"expected exactly one audio stream, found {info.audio_stream_count}"
        )
    if expectation.expect_audio and (
        info.audio_duration_seconds is None or info.audio_duration_seconds <= 0
    ):
        problems.append("audio stream reports no usable duration")

    if expectation.expected_seconds is not None:
        tolerance = expectation.tolerance_seconds
        if tolerance is None:
            tolerance = duration_tolerance(expectation.expected_seconds)
        if info.duration_seconds is None:
            problems.append("reports no duration")
        else:
            drift = abs(info.duration_seconds - expectation.expected_seconds)
            if drift > tolerance:
                problems.append(
                    f"duration {info.duration_seconds:.2f}s differs from planned "
                    f"{expectation.expected_seconds:.2f}s by {drift:.2f}s "
                    f"(tolerance {tolerance:.2f}s)"
                )
        if expectation.expect_audio and info.audio_duration_seconds is not None:
            audio_drift = abs(info.audio_duration_seconds - expectation.expected_seconds)
            if audio_drift > tolerance:
                problems.append(
                    f"audio duration {info.audio_duration_seconds:.2f}s differs from planned "
                    f"{expectation.expected_seconds:.2f}s by {audio_drift:.2f}s "
                    f"(tolerance {tolerance:.2f}s)"
                )

    if expectation.expected_frame_count is not None:
        if info.frame_count is None:
            problems.append(
                f"reports no frame count, but {expectation.expected_frame_count} "
                "were required"
            )
        elif info.frame_count != expectation.expected_frame_count:
            problems.append(
                f"has {info.frame_count} frames, not the required "
                f"{expectation.expected_frame_count}"
            )

    if (
        expectation.max_av_drift_seconds is not None
        and info.duration_seconds is not None
        and info.audio_duration_seconds is not None
    ):
        drift = abs(info.audio_duration_seconds - info.duration_seconds)
        if drift > expectation.max_av_drift_seconds:
            problems.append(
                f"audio and video differ by {drift:.3f}s, over the "
                f"{expectation.max_av_drift_seconds:.3f}s allowed"
            )

    if expectation.expected_width and info.width != expectation.expected_width:
        problems.append(f"width {info.width} is not the requested {expectation.expected_width}")
    if expectation.expected_height and info.height != expectation.expected_height:
        problems.append(
            f"height {info.height} is not the requested {expectation.expected_height}"
        )

    if problems:
        raise FfmpegError(f"output {path.name} failed validation: " + "; ".join(problems))

    if expectation.require_decodable:
        await _assert_decodable(path, expectation)

    logger.info(
        "output_validated",
        extra={
            "size_bytes": size,
            "duration_seconds": info.duration_seconds,
            "width": info.width,
            "height": info.height,
            "has_audio": info.has_audio,
            "audio_stream_count": info.audio_stream_count,
            "audio_duration_seconds": info.audio_duration_seconds,
        },
    )
    return info


async def _assert_decodable(path: Path, expectation: OutputExpectation) -> None:
    """Decodes every stream we promised, discarding the output.

    `-f null` means nothing is written; the only thing being tested is whether
    the frames come back out. A truncated or mis-muxed file fails here after
    passing every metadata check above.
    """
    maps: list[str] = []
    if expectation.expect_video:
        maps += ["-map", "0:v:0"]
    if expectation.expect_audio:
        maps += ["-map", "0:a:0"]

    try:
        await ffmpeg(["-i", str(path), *maps, "-f", "null", "-"], timeout=900.0)
    except FfmpegError as exc:
        raise FfmpegError(
            f"output {path.name} does not decode cleanly: {exc}", stderr=exc.stderr
        ) from exc
