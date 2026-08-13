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
