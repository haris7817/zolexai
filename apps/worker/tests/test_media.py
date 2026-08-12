"""The long-form media layer: planning, assembly, measurement.

Split deliberately into two halves.

The planning half is pure arithmetic and runs everywhere. It is also where the
client's promises live — "the final video matches the source duration" is a
property of `plan_segments`, and it either holds for every input or it does not.

The ffmpeg half needs the binaries and skips without them. It is the half that
proves the arithmetic survives contact with a real encoder, so it must run
somewhere real: the worker image ships ffmpeg, and that is where it counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.media import (
    FfmpegError,
    concat_segments,
    extract_final_frame,
    ffmpeg,
    normalize_clip,
    plan_segments,
    probe_media,
    tools_available,
    verify_duration,
)

needs_ffmpeg = pytest.mark.skipif(
    not tools_available(), reason="ffmpeg/ffprobe not installed"
)


# ── Segment planning (no ffmpeg required) ────────────────────────────────


@pytest.mark.parametrize(
    ("total", "window"),
    [(5, 10), (10, 10), (11, 10), (45, 10), (60, 5), (240, 8), (3.5, 1.0)],
)
def test_contributions_always_sum_to_the_target(total: float, window: float) -> None:
    """The single guarantee everything else rests on.

    Automatic V2V duration and full-length music videos are both this property:
    if the parts do not add up to the source length, the output is the wrong
    length and the feature is broken however good the frames look.
    """
    segments = plan_segments(total, max_segment_seconds=window)
    assert sum(s.duration_seconds for s in segments) == pytest.approx(total)


def test_a_short_job_is_a_single_pass_with_no_overlap() -> None:
    """Short generations must not pay for machinery they do not need — no
    overlap to re-render, no concat step, no seam to hide."""
    segments = plan_segments(8, max_segment_seconds=10, overlap_seconds=2)
    assert len(segments) == 1
    assert segments[0].overlap_seconds == 0
    assert segments[0].generate_seconds == 8


def test_segments_are_contiguous_and_ordered() -> None:
    segments = plan_segments(45, max_segment_seconds=10)
    assert [s.index for s in segments] == list(range(len(segments)))
    for previous, current in zip(segments, segments[1:], strict=False):
        assert current.start_seconds == pytest.approx(
            previous.start_seconds + previous.duration_seconds
        )


def test_overlap_adds_regenerated_lead_in_without_changing_the_timeline() -> None:
    """Continuity insurance: a segment re-renders the tail of its predecessor so
    the model has something to match, and the extra is trimmed on assembly. The
    final duration must be unaffected."""
    segments = plan_segments(30, max_segment_seconds=10, overlap_seconds=2)

    assert segments[0].overlap_seconds == 0
    assert all(s.overlap_seconds == 2 for s in segments[1:])
    # Asked for more than it contributes...
    assert segments[1].generate_seconds == 12
    assert segments[1].trim_start_seconds == 2
    # ...but the timeline is still exactly the target.
    assert sum(s.duration_seconds for s in segments) == pytest.approx(30)


def test_overlap_cannot_swallow_a_whole_window() -> None:
    """An overlap larger than the window would make a segment entirely
    re-generated material contributing nothing."""
    segments = plan_segments(30, max_segment_seconds=10, overlap_seconds=999)
    assert all(s.overlap_seconds <= 5 for s in segments)
    assert sum(s.duration_seconds for s in segments) == pytest.approx(30)


@pytest.mark.parametrize(("total", "window"), [(0, 10), (-5, 10), (10, 0), (10, -1)])
def test_impossible_plans_are_rejected(total: float, window: float) -> None:
    with pytest.raises(ValueError):
        plan_segments(total, max_segment_seconds=window)


# ── Real media (ffmpeg required) ─────────────────────────────────────────


async def _make_clip(path: Path, seconds: float) -> Path:
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24",
            "-t", f"{seconds:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", "24",
            str(path),
        ]
    )
    return path


@needs_ffmpeg
async def test_probe_measures_a_file_rather_than_trusting_a_request(tmp_path: Path) -> None:
    """M1 reported dimensions from a lookup table and duration from the string
    the user typed. Both were assertions; this is a measurement."""
    clip = await _make_clip(tmp_path / "clip.mp4", 2.0)
    info = await probe_media(clip)

    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)
    assert (info.width, info.height) == (160, 120)
    assert info.has_video is True


@needs_ffmpeg
async def test_probing_an_unreadable_file_fails_clearly(tmp_path: Path) -> None:
    """A truncated or mislabelled upload is a user-facing case, not a crash."""
    junk = tmp_path / "not-really.mp4"
    junk.write_bytes(b"this is not a video")

    with pytest.raises(FfmpegError):
        await probe_media(junk)

    with pytest.raises(FfmpegError):
        await probe_media(tmp_path / "absent.mp4")


@needs_ffmpeg
async def test_segments_assemble_into_one_file_of_the_planned_length(tmp_path: Path) -> None:
    """The end-to-end long-form claim, in miniature: plan, render, stitch, and
    the result is the length that was asked for."""
    target = 6.0
    segments = plan_segments(target, max_segment_seconds=2.0)
    assert len(segments) == 3

    parts = [
        await _make_clip(tmp_path / f"part-{s.index}.mp4", s.generate_seconds)
        for s in segments
    ]
    output = await concat_segments(parts, tmp_path / "final.mp4")

    measured = await verify_duration(output, expected_seconds=target)
    assert measured == pytest.approx(target, abs=0.75)


@needs_ffmpeg
async def test_a_single_segment_needs_no_concatenation(tmp_path: Path) -> None:
    clip = await _make_clip(tmp_path / "only.mp4", 1.5)
    output = await concat_segments([clip], tmp_path / "final.mp4")

    assert output.exists()
    assert not clip.exists(), "the single part should be moved, not copied"


@needs_ffmpeg
async def test_a_wrong_length_assembly_is_caught(tmp_path: Path) -> None:
    """Without this check a stitching bug ships as "the video is a bit short",
    which is exactly the kind of thing nobody notices until the client does."""
    clip = await _make_clip(tmp_path / "short.mp4", 1.0)

    with pytest.raises(FfmpegError, match="differs from planned"):
        await verify_duration(clip, expected_seconds=10.0)


@needs_ffmpeg
async def test_missing_segment_files_are_reported_not_silently_skipped(tmp_path: Path) -> None:
    """Skipping a failed segment would produce a short video that looks fine."""
    clip = await _make_clip(tmp_path / "one.mp4", 1.0)

    with pytest.raises(FfmpegError, match="missing segment"):
        await concat_segments([clip, tmp_path / "never-rendered.mp4"], tmp_path / "out.mp4")


def test_concatenating_nothing_is_a_programming_error() -> None:
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(concat_segments([], Path("out.mp4")))


# ── Extension primitives: frame extraction and normalization ─────────────


@needs_ffmpeg
async def test_probe_reports_the_frame_rate(tmp_path: Path) -> None:
    """Normalization retimes clips to the source's rate — a guessed rate
    means judder at the seam."""
    clip = await _make_clip(tmp_path / "clip.mp4", 2.0)
    info = await probe_media(clip)
    assert info.fps == pytest.approx(24.0, abs=0.1)


@needs_ffmpeg
async def test_the_final_frame_of_a_clip_can_be_extracted(tmp_path: Path) -> None:
    """The continuation's conditioning image. It must be a genuinely decodable
    picture with the source's dimensions — a corrupt or empty frame here
    conditions the whole extension on garbage."""
    clip = await _make_clip(tmp_path / "clip.mp4", 2.0)
    frame = await extract_final_frame(clip, tmp_path / "last.png")

    info = await probe_media(frame)
    assert (info.width, info.height) == (160, 120)
    assert frame.stat().st_size > 0


@needs_ffmpeg
async def test_final_frame_extraction_fails_clearly_on_junk(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")

    with pytest.raises(FfmpegError):
        await extract_final_frame(junk, tmp_path / "last.png")
    assert not (tmp_path / "last.png").exists(), "a failed extraction must not leave artifacts"


@needs_ffmpeg
async def test_normalization_makes_mismatched_clips_concatenable(tmp_path: Path) -> None:
    """The whole reason normalize_clip exists: a user upload and a model
    render never share parameters, and the concat demuxer breaks on mixed
    input. Normalized to one target, assembly is deterministic."""
    small = await _make_clip(tmp_path / "small.mp4", 1.0)  # 160x120@24
    big = tmp_path / "big.mp4"
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=12",
            "-t", "1.0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(big),
        ]
    )

    parts = []
    for index, clip in enumerate((small, big)):
        parts.append(
            await normalize_clip(
                clip, tmp_path / f"norm-{index}.mp4",
                width=160, height=120, fps=24, audio=False,
            )
        )
    output = await concat_segments(parts, tmp_path / "joined.mp4")

    info = await probe_media(output)
    assert (info.width, info.height) == (160, 120)
    assert info.fps == pytest.approx(24.0, abs=0.1)
    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)


@needs_ffmpeg
async def test_normalization_synthesizes_silence_when_audio_is_required(tmp_path: Path) -> None:
    """Mixed audio presence breaks the concat demuxer just like mixed video
    parameters: every part of an assembly must agree about having sound."""
    silent = await _make_clip(tmp_path / "silent.mp4", 1.0)
    normalized = await normalize_clip(
        silent, tmp_path / "with-audio.mp4", width=160, height=120, fps=24, audio=True
    )

    info = await probe_media(normalized)
    assert info.has_audio is True
    assert info.duration_seconds == pytest.approx(1.0, abs=0.3)


@needs_ffmpeg
async def test_normalization_strips_audio_when_told_to(tmp_path: Path) -> None:
    noisy = tmp_path / "noisy.mp4"
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", "1.0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(noisy),
        ]
    )
    normalized = await normalize_clip(
        noisy, tmp_path / "muted.mp4", width=160, height=120, fps=24, audio=False
    )
    info = await probe_media(normalized)
    assert info.has_audio is False
