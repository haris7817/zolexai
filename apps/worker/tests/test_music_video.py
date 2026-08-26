"""Music Video: visuals for the whole song, with the whole song over them.

The client's requirement here is unusually specific, and it is about the audio
rather than the picture:

    "The final file should contain the full uploaded audio track, not restart
    audio on each visual segment. Audio should remain continuous across the
    entire final video."

That is a promise about a property of the finished file, so it is tested as
one. `test_the_audio_runs_continuously_from_start_to_finish` decodes the
delivered result and measures its loudness envelope against a track that ramps
from silence to full — audio restarted per section would saw-tooth, and the
assertion is that it climbs.

The rest is the automatic-duration promise: a 32-second track produces a
32-second video, a four-minute track a four-minute video, and the sectioning
that makes the second one possible is invisible to the customer.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    invocations,
    make_clip,
    make_job,
    make_track,
    needs_ffmpeg,
    render_stub,
    staged_input,
    value_of,
)
from worker.adapters.base import AdapterError, JobCancelled
from worker.adapters.ltx import LtxAdapter
from worker.core.config import settings
from worker.media import audio_envelope, ffmpeg, plan_segments, probe_media


def music_video_job(workspace: Path, track: Path | None, **overrides):
    defaults = dict(
        workflow_id="music-video",
        prompt="a lone dancer in an empty warehouse, hard side light, slow push in",
        # `duration_mode: source` — the API sends no duration for this workflow.
        parameters={"aspect_ratio": "16:9", "quality": "Standard"},
        inputs=[staged_input("source_audio", "audio", "audio/mpeg", track)],
    )
    return make_job(workspace, **{**defaults, **overrides})


async def make_ramp_track(path: Path, seconds: float) -> Path:
    """A track that grows steadily louder from silence to full.

    Its shape is the test instrument: any restart, loop or per-section
    re-attachment of the audio shows up as the envelope falling back.
    """
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", f"{seconds:.3f}",
            "-af", f"volume='min(1,t/{seconds:.3f})':eval=frame",
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(path),
        ]
    )
    return path


# ── Duration follows the song ────────────────────────────────────────────


@needs_ffmpeg
async def test_a_short_track_produces_a_video_of_the_same_length(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = await make_track(workspace / "song.mp3", 3.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 3.0))

    result, _ = await collect(music_video_job(workspace, track))

    measured = await probe_media(track)
    assert result.duration_seconds == pytest.approx(measured.duration_seconds, abs=1.0)


@needs_ffmpeg
async def test_a_track_longer_than_one_pass_becomes_several_scenes(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multi-minute case in miniature. A four-minute song against the real
    30-second ceiling runs exactly this arithmetic — eight passes, one file,
    one continuous track — and the customer sees one progress bar."""
    track = await make_track(workspace / "song.mp3", 4.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = music_video_job(
        workspace,
        track,
        execution={"runtime": "ltx", "max_segment_seconds": 1, "align_cuts_to_audio": False},
    )
    result, reported = await collect(job)

    # Five, not four: the MP3 probes a little over 4.0s because the encoder
    # pads, so a 1s ceiling needs five windows. They are EVEN — five of ~0.81s
    # — rather than four full ones and a 0.03s sliver that would have cost a
    # whole model invocation to contribute a single frame.
    assert len(invocations(log)) == 5
    assert all(s.duration_seconds > 0.1 for s in plan_segments(4.03, max_segment_seconds=1.0))
    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)

    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)
    messages = [message for _, _, message in reported]
    assert "Generating section 1 of 5…" in messages
    assert "Generating section 5 of 5…" in messages
    # The mechanism stays invisible: no model, no GPU, no file format.
    assert all(
        not any(word in message.lower() for word in ("ltx", "gpu", "mp4", "ffmpeg"))
        for message in messages
    )


@needs_ffmpeg
async def test_the_scene_plan_covers_the_track_exactly(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the sectioning decides, the frames asked for must add up to the
    song. A plan that is one section short produces a video that ends before
    the music does."""
    track = await make_track(workspace / "song.mp3", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = music_video_job(
        workspace, track, execution={"runtime": "ltx", "max_segment_seconds": 1}
    )
    await collect(job)

    measured = await probe_media(track)
    frames = [int(value_of(argv, "--num-frames")) for argv in invocations(log)]

    # Each pass is snapped up to the model's 8k+1 lattice and trimmed back
    # afterwards, so the counts sum to the song plus at most 7 frames per pass
    # rather than to it exactly. What must still hold is that nothing was left
    # uncovered: a plan one section short ends before the music does.
    planned = measured.duration_seconds * 24
    assert all(f % 8 == 1 for f in frames), f"not on the model's lattice: {frames}"
    assert sum(frames) >= planned - 1, "the plan does not reach the end of the track"
    assert sum(frames) <= planned + 7 * len(frames) + 1, (
        f"overshoot beyond the lattice: {sum(frames)} for {planned:.0f} planned"
    )


# ── The audio promise ────────────────────────────────────────────────────


@needs_ffmpeg
async def test_the_audio_runs_continuously_from_start_to_finish(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's actual requirement, measured on the delivered file.

    The source ramps from silence to full over its whole length. If the audio
    were attached per visual section — the obvious way to build this, and the
    wrong one — the delivered envelope would reset four times. It climbs
    instead, which is only possible if one continuous track was laid over the
    finished picture once.
    """
    track = await make_ramp_track(workspace / "song.mp3", 6.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.5))

    job = music_video_job(
        workspace, track, execution={"runtime": "ltx", "max_segment_seconds": 1.5}
    )
    result, _ = await collect(job)

    envelope = await audio_envelope(result.path)
    assert envelope, "the delivered file has no audio to measure"

    # Coarse bins: the point is the trend across the whole song, not the
    # sample-level detail that encoding would blur anyway.
    size = len(envelope) // 6
    bins = [
        sum(envelope[index * size : (index + 1) * size]) / size for index in range(6)
    ]
    assert bins == sorted(bins), f"the track restarted somewhere: {bins}"
    assert bins[-1] > bins[0] * 2, "the ramp did not survive to the delivered file"


@needs_ffmpeg
async def test_the_result_carries_both_a_picture_and_the_song(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = await make_track(workspace / "song.mp3", 3.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 3.0))

    result, reported = await collect(music_video_job(workspace, track))

    info = await probe_media(result.path)
    assert info.has_video is True
    assert info.has_audio is True
    assert "Adding your track…" in [message for _, _, message in reported]


@needs_ffmpeg
async def test_the_models_own_generated_audio_never_reaches_the_result(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The video model generates a soundtrack of its own. On a music video that
    is noise under the user's song, so the picture is stripped before the track
    goes on — exactly one audio stream reaches the customer."""
    track = await make_track(workspace / "song.mp3", 3.0)
    # A render that loudly carries its own audio.
    render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 3.0, audio=True)
    )

    result, _ = await collect(music_video_job(workspace, track))

    from worker.media import ffprobe_json

    payload = await ffprobe_json(result.path)
    audio_streams = [
        stream for stream in payload["streams"] if stream["codec_type"] == "audio"
    ]
    assert len(audio_streams) == 1, "the model's own audio was left in the mix"


# ── Failure handling ─────────────────────────────────────────────────────


async def test_a_missing_track_is_an_internal_error(
    workspace: Path, fake_models: Path
) -> None:
    with pytest.raises(AdapterError) as raised:
        await collect(music_video_job(workspace, track=None))

    assert raised.value.retriable is False
    assert "not staged" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_corrupt_track_fails_before_any_gpu_time(
    workspace: Path, fake_models: Path
) -> None:
    junk = workspace / "song.mp3"
    junk.write_bytes(b"not an audio file")

    with pytest.raises(AdapterError) as raised:
        await collect(music_video_job(workspace, junk))

    assert raised.value.retriable is False
    assert "audio" in raised.value.user_message.lower()


@needs_ffmpeg
async def test_a_silent_video_uploaded_as_the_track_is_refused(
    workspace: Path, fake_models: Path
) -> None:
    """Readable media with no audio stream. There is no song to match, and
    generating four minutes of visuals for nothing would be the expensive way
    to discover that."""
    silent = await make_clip(workspace / "song.mp4", 1.0)

    with pytest.raises(AdapterError) as raised:
        await collect(music_video_job(workspace, silent))

    assert raised.value.retriable is False
    assert "no audio stream" in raised.value.internal_detail
    # The customer uploaded a real, readable file; telling them it "could not
    # be read" sends them to check a file that is fine. Name the actual fault.
    assert "no sound" in raised.value.user_message.lower()


@needs_ffmpeg
async def test_a_track_longer_than_the_ceiling_is_refused_before_any_render(
    workspace: Path, fake_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing bounded the uploaded length, and the length decides the work.

    The upload cap is 64 MB — over an hour of ordinary MP3 — and an hour of
    audio is roughly 120 render passes. Such a job cannot finish inside its own
    timeout: it ran for hours, held the card, and failed having produced
    nothing, which from the outside is indistinguishable from a hang.

    So the refusal has to come from the PROBE, before a model is ever
    launched, and it has to say how long the file actually is.
    """
    monkeypatch.setattr(settings, "ltx_max_source_seconds", 2.0)
    track = await make_track(workspace / "song.mp3", 5.0)

    with pytest.raises(AdapterError) as raised:
        await collect(music_video_job(workspace, track))

    assert raised.value.retriable is False, "a long file is long on every attempt"
    # Both numbers, in minutes and seconds rather than raw seconds: the message
    # is asking someone to go and trim a track.
    assert "5 seconds" in raised.value.user_message
    assert "2 seconds" in raised.value.user_message
    assert "trim" in raised.value.user_message.lower()
    assert "source ceiling" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_failed_scene_fails_the_job(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = await make_track(workspace / "song.mp3", 3.0)
    render_stub(
        tmp_path, monkeypatch,
        await make_clip(tmp_path / "render.mp4", 1.0),
        fail_on_pass=1,
    )

    job = music_video_job(
        workspace, track, execution={"runtime": "ltx", "max_segment_seconds": 1}
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert "exited 3" in raised.value.internal_detail


@needs_ffmpeg
async def test_cancellation_stops_a_long_music_video_between_scenes(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A four-minute music video is the longest job the product has. Cancelling
    one must release the GPU now, not eight passes from now."""
    track = await make_track(workspace / "song.mp3", 4.0)
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0), sleep=15.0
    )

    cancelled = asyncio.Event()

    async def cancel_once_generating(
        status: str, progress: int, message: str, _details=None
    ) -> None:
        if status == "generating":
            cancelled.set()

    job = music_video_job(
        workspace, track,
        execution={"runtime": "ltx", "max_segment_seconds": 1},
        _cancelled=cancelled,
    )

    began = time.monotonic()
    with pytest.raises(JobCancelled):
        await LtxAdapter().run(job, cancel_once_generating)

    assert time.monotonic() - began < 10
    assert len(invocations(log)) == 1


@needs_ffmpeg
async def test_a_video_that_does_not_cover_the_song_fails_validation(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half a video under a whole song would ship as "it cuts out early". The
    finished file is measured against the track before anything is uploaded."""
    track = await make_track(workspace / "song.mp3", 6.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = music_video_job(
        workspace, track, execution={"runtime": "ltx", "max_segment_seconds": 3}
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert "planning failure" in raised.value.internal_detail or (
        "failed validation" in raised.value.internal_detail
    )


@needs_ffmpeg
async def test_later_sections_carry_the_first_seam_as_identity_anchor(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three-women problem (27 Aug audit: the specified singer held 15 of
    180 seconds): sections that inherit identity only from their
    predecessor's frame drift copy-by-copy. From section 3 onward every pass
    must ALSO pin section 1's final frame — when the frame count is measured
    safe for a second conditioning image."""
    import worker.adapters.ltx as ltx_module

    monkeypatch.setattr(
        ltx_module, "_TWO_IMAGE_SAFE_FRAMES", frozenset(range(1, 200))
    )
    track = await make_track(workspace / "song.mp3", 4.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = music_video_job(
        workspace,
        track,
        execution={"runtime": "ltx", "max_segment_seconds": 1, "align_cuts_to_audio": False},
    )
    await collect(job)

    calls = invocations(log)
    def images(argv: list[str]) -> list[str]:
        return [argv[i + 1] for i, a in enumerate(argv) if a == "--image"]

    assert images(calls[0]) == []            # section 1: nothing to continue
    assert len(images(calls[1])) == 1        # section 2: seam only (anchor IS the seam)
    for argv in calls[2:]:
        found = images(argv)
        assert len(found) == 2               # seam + section-1 anchor
        assert found[1] == images(calls[1])[0]  # the anchor never changes
