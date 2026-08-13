"""Audio muxing, section joining, and the output gate.

These are the pieces of the media layer that only exist because of M2's audio
workflows, and each one guards a specific way a job could succeed while
delivering something wrong: a music video whose song is cut off a frame early,
a generated song with an audible step at every join, a truncated file that
ffprobe cheerfully describes as forty seconds long.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_clip, make_track, needs_ffmpeg
from worker.media import (
    FfmpegError,
    OutputExpectation,
    crossfade_concat,
    duration_tolerance,
    ffprobe_json,
    loudness_normalize,
    mux_audio,
    overlap_cost_seconds,
    probe_media,
    verify_output,
)

pytestmark = needs_ffmpeg


# ── Muxing ───────────────────────────────────────────────────────────────


async def test_the_track_decides_the_length_and_plays_to_its_end(tmp_path: Path) -> None:
    """The music-video promise, at the level of one ffmpeg call."""
    picture = await make_clip(tmp_path / "picture.mp4", 3.0)
    track = await make_track(tmp_path / "song.mp3", 3.0)

    result = await mux_audio(picture, track, tmp_path / "out.mp4")
    info = await probe_media(result)

    assert info.has_video and info.has_audio
    assert info.duration_seconds == pytest.approx(3.0, abs=0.3)


async def test_a_picture_a_fraction_short_does_not_cut_the_song_off(
    tmp_path: Path,
) -> None:
    """Frame counts are integers, so the picture routinely lands a few
    hundredths under the track. Letting `-shortest` resolve that would clip the
    last moment of every song — instead the final frame is held."""
    picture = await make_clip(tmp_path / "picture.mp4", 2.6)
    track = await make_track(tmp_path / "song.mp3", 3.0)

    result = await mux_audio(picture, track, tmp_path / "out.mp4")
    info = await probe_media(result)

    assert info.duration_seconds == pytest.approx(3.0, abs=0.2)


async def test_a_picture_far_shorter_than_the_song_is_a_planning_failure(
    tmp_path: Path,
) -> None:
    """Padding a real shortfall would hide a broken plan behind ten seconds of
    frozen frame, which is exactly the kind of thing that ships."""
    picture = await make_clip(tmp_path / "picture.mp4", 1.0)
    track = await make_track(tmp_path / "song.mp3", 8.0)

    with pytest.raises(FfmpegError, match="planning failure"):
        await mux_audio(picture, track, tmp_path / "out.mp4")


async def test_the_pictures_own_audio_is_replaced_not_mixed(tmp_path: Path) -> None:
    """A generated soundtrack playing underneath the user's song is noise."""
    picture = await make_clip(tmp_path / "picture.mp4", 2.0, audio=True)
    track = await make_track(tmp_path / "song.mp3", 2.0)

    result = await mux_audio(picture, track, tmp_path / "out.mp4")

    payload = await ffprobe_json(result)
    audio = [s for s in payload["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == 1


async def test_muxing_refuses_inputs_that_are_not_what_they_claim(
    tmp_path: Path,
) -> None:
    picture = await make_clip(tmp_path / "picture.mp4", 1.0)
    silent = await make_clip(tmp_path / "silent.mp4", 1.0)

    with pytest.raises(FfmpegError, match="no audio stream"):
        await mux_audio(picture, silent, tmp_path / "out.mp4")

    track = await make_track(tmp_path / "song.mp3", 1.0)
    with pytest.raises(FfmpegError, match="no video stream"):
        await mux_audio(track, track, tmp_path / "out.mp4")


# ── Joining generated sections ───────────────────────────────────────────


def test_the_cost_of_the_joins_is_arithmetic_the_planner_can_use() -> None:
    assert overlap_cost_seconds(1, 1.5) == 0.0
    assert overlap_cost_seconds(4, 1.5) == 4.5


async def test_sections_overlap_rather_than_butt_together(tmp_path: Path) -> None:
    """A hard cut between two independently generated sections is audible. The
    overlap is why, and the shortening it causes is why the planner pads."""
    parts = [
        await make_track(tmp_path / f"part-{index}.mp3", 3.0) for index in range(3)
    ]
    joined = await crossfade_concat(parts, tmp_path / "joined.mp3", fade_seconds=1.0)
    info = await probe_media(joined)

    # 9 seconds of material, two joins, one second each.
    assert info.duration_seconds == pytest.approx(7.0, abs=0.4)


async def test_a_single_section_needs_no_join(tmp_path: Path) -> None:
    only = await make_track(tmp_path / "only.mp3", 2.0)
    joined = await crossfade_concat([only], tmp_path / "joined.mp3")

    assert joined.exists()
    assert not only.exists(), "the single part should be moved, not copied"


async def test_loudness_normalisation_produces_a_playable_track(tmp_path: Path) -> None:
    """Independently generated sections arrive at independent levels, and a
    crossfade turns that into an audible step mid-song."""
    track = await make_track(tmp_path / "song.mp3", 3.0)
    levelled = await loudness_normalize(track, tmp_path / "levelled.mp3")

    info = await probe_media(levelled)
    assert info.has_audio
    assert info.duration_seconds == pytest.approx(3.0, abs=0.4)


# ── The output gate ──────────────────────────────────────────────────────


def test_tolerance_scales_with_length() -> None:
    """One absolute figure cannot be right for both a five-second clip and a
    four-minute music video: it either fails honest rounding on the short one
    or accepts a missing section on the long one."""
    assert duration_tolerance(5) == pytest.approx(0.75)
    assert duration_tolerance(240) == pytest.approx(7.2)


async def test_a_finished_video_passes_every_check_it_should(tmp_path: Path) -> None:
    clip = await make_clip(tmp_path / "clip.mp4", 2.0, audio=True, size="320x180")

    info = await verify_output(
        clip,
        OutputExpectation(
            expect_video=True,
            expect_audio=True,
            expected_seconds=2.0,
            expected_width=320,
            expected_height=180,
        ),
    )
    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)


async def test_a_missing_audio_stream_fails_the_job(tmp_path: Path) -> None:
    """The music-video check. A video with no song is not a music video, and
    it must not reach a customer as one."""
    silent = await make_clip(tmp_path / "silent.mp4", 2.0)

    with pytest.raises(FfmpegError, match="no audio stream"):
        await verify_output(silent, OutputExpectation(expect_video=True, expect_audio=True))


async def test_the_wrong_length_fails_the_job(tmp_path: Path) -> None:
    clip = await make_clip(tmp_path / "clip.mp4", 2.0)

    with pytest.raises(FfmpegError, match="differs from planned"):
        await verify_output(
            clip, OutputExpectation(expect_video=True, expected_seconds=10.0)
        )


async def test_the_wrong_resolution_fails_the_job(tmp_path: Path) -> None:
    """Delivery is at the source's resolution. A restyle returning the 512p
    generation grid instead of the user's 1080p is a visible regression."""
    clip = await make_clip(tmp_path / "clip.mp4", 2.0, size="320x180")

    with pytest.raises(FfmpegError, match="width"):
        await verify_output(
            clip, OutputExpectation(expect_video=True, expected_width=1920)
        )


async def test_an_empty_or_absent_file_is_caught_before_anything_else(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")

    with pytest.raises(FfmpegError, match="no media was produced"):
        await verify_output(empty, OutputExpectation(expect_video=True))

    with pytest.raises(FfmpegError, match="never written"):
        await verify_output(tmp_path / "absent.mp4", OutputExpectation(expect_video=True))


async def test_a_file_that_describes_itself_but_does_not_decode_is_rejected(
    tmp_path: Path,
) -> None:
    """ffprobe reads headers, and a truncated MP4's header still claims the
    full duration. Only a decode pass tells the difference between "we measured
    the file" and "we know it plays"."""
    clip = await make_clip(tmp_path / "clip.mp4", 3.0)
    data = clip.read_bytes()

    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(data[: len(data) // 2])

    with pytest.raises(FfmpegError):
        await verify_output(
            truncated, OutputExpectation(expect_video=True, require_decodable=True)
        )
