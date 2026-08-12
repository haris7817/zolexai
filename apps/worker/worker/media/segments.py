"""Long-form orchestration primitives: plan, stitch, verify.

The client's requirements assume outputs longer than any current model produces
in one pass — video-to-video matching a 60-second source, a music video matching
a four-minute song. Those are produced by generating sections and assembling
them, and the customer must never see the seam or the mechanism.

The rule this encodes: **the plan is authoritative about the final timeline.**
Each segment declares where it sits and how much finished material it
contributes, so the assembled duration is arithmetic rather than hope — and
`verify_duration` checks the arithmetic against the actual file at the end.

Nothing here generates anything. It decides what to ask for and what to do with
the answers, which is why it works identically for the ffmpeg harness and for
whichever model M2 selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg
from worker.media.probe import probe_media

logger = get_logger(__name__)


@dataclass(frozen=True)
class Segment:
    index: int
    start_seconds: float
    """Where this segment's contribution begins in the final timeline."""

    duration_seconds: float
    """How much *new* material it contributes."""

    overlap_seconds: float = 0.0
    """
    Extra material generated before `start_seconds`, discarded when stitching.

    Continuity insurance: models drift at a cold start, so a segment that begins
    by re-generating the tail of its predecessor has something to match against.
    Set to zero when the provider supports a real continuation signal instead.
    """

    @property
    def source_start_seconds(self) -> float:
        return max(0.0, self.start_seconds - self.overlap_seconds)

    @property
    def generate_seconds(self) -> float:
        """Total length to ask the provider for, including the overlap."""
        return self.duration_seconds + (self.start_seconds - self.source_start_seconds)

    @property
    def trim_start_seconds(self) -> float:
        """How much of the generated segment to drop when assembling."""
        return self.start_seconds - self.source_start_seconds


def plan_segments(
    total_seconds: float,
    *,
    max_segment_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[Segment]:
    """Splits a target duration into generation windows.

    Guarantees, all of them relied on by `verify_duration`:

      * contributions sum to `total_seconds` exactly;
      * segments are contiguous and ordered;
      * a duration within one pass yields exactly one segment with no overlap,
        so short jobs never pay for machinery they do not need.
    """
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    if max_segment_seconds <= 0:
        raise ValueError("max_segment_seconds must be positive")

    if total_seconds <= max_segment_seconds:
        return [Segment(index=0, start_seconds=0.0, duration_seconds=total_seconds)]

    # Overlap cannot exceed the window it is borrowing from, or a segment would
    # be entirely re-generated material and contribute nothing.
    overlap = max(0.0, min(overlap_seconds, max_segment_seconds / 2))

    segments: list[Segment] = []
    start = 0.0
    index = 0
    while start < total_seconds - 1e-9:
        duration = min(max_segment_seconds, total_seconds - start)
        segments.append(
            Segment(
                index=index,
                start_seconds=start,
                duration_seconds=duration,
                overlap_seconds=0.0 if index == 0 else overlap,
            )
        )
        start += duration
        index += 1
    return segments


async def concat_segments(paths: list[Path], dest: Path, *, timeout: float = 900.0) -> Path:
    """Concatenates rendered segments into one file.

    Stream copy first: it is fast and lossless, and re-encoding every assembly
    would be a second quality loss on top of generation. It only works when the
    parts share codec parameters, which they do when one adapter produced them
    all — so the re-encode is the fallback for when that assumption breaks
    rather than the default.
    """
    if not paths:
        raise ValueError("nothing to concatenate")
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FfmpegError(f"missing segment files: {', '.join(missing)}")

    if len(paths) == 1:
        paths[0].replace(dest)
        return dest

    listing = dest.parent / f"{dest.stem}-concat.txt"
    listing.write_text(
        "".join(f"file '{_escape(path)}'\n" for path in paths), encoding="utf-8"
    )

    common = ["-f", "concat", "-safe", "0", "-i", str(listing)]
    try:
        await ffmpeg([*common, "-c", "copy", str(dest)], timeout=timeout)
    except FfmpegError as exc:
        logger.info("concat_copy_failed_reencoding", extra={"detail": str(exc)})
        await ffmpeg(
            [*common, "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(dest)],
            timeout=timeout,
        )
    finally:
        listing.unlink(missing_ok=True)

    return dest


def _escape(path: Path) -> str:
    """Quoting for ffmpeg's concat demuxer, which uses shell-ish single quotes."""
    return str(path).replace("\\", "/").replace("'", r"'\''")


async def verify_duration(
    path: Path, *, expected_seconds: float, tolerance_seconds: float = 0.75
) -> float:
    """Checks an assembled file against its plan. Returns the measured duration.

    This is the honesty check on the whole long-form mechanism. "The output
    matches the source" is a promise made to the client, and the only way to
    keep it is to measure the finished file and refuse to call a job complete
    when it does not hold.
    """
    info = await probe_media(path)
    actual = info.duration_seconds
    if actual is None:
        raise FfmpegError(f"assembled output {path.name} reports no duration")

    drift = abs(actual - expected_seconds)
    if drift > tolerance_seconds:
        raise FfmpegError(
            f"assembled duration {actual:.2f}s differs from planned "
            f"{expected_seconds:.2f}s by {drift:.2f}s"
        )
    return actual
