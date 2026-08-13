"""Video to Video: restyle footage, match its length, keep its structure.

Two client requirements meet in this workflow and they are easy to break in
opposite directions.

**"Same as source video."** The customer picks no duration — the API rejects
one — so the length has to come from measuring the upload. A 42-second source
produces a 42-second result whether that is one GPU pass or two, and a source
longer than the measured single-pass ceiling is segmented rather than refused
or truncated.

**"The prompt must change it; the source must still be recognisable."** The
failure that looks like success is a restyle that quietly ignored its source
and returned unrelated text-to-video footage. What prevents it is conditioning:
stills lifted from the same window of the source that each pass is generating,
so subject placement and the direction of movement carry over. These tests
assert the actual conditioning arguments, because "it produced a video" is
exactly the evidence that would not catch that failure.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    conditioning_of,
    invocations,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
    value_of,
)
from worker.adapters.base import AdapterError, JobCancelled
from worker.adapters.ltx import LtxAdapter, grid_for_source
from worker.media import probe_media


def restyle_job(workspace: Path, source: Path | None, reference: Path | None = None, **overrides):
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    if reference is not None:
        inputs.append(staged_input("reference_image", "image", "image/png", reference))

    defaults = dict(
        workflow_id="video-to-video",
        prompt="repaint it as a charcoal sketch, same two cars, same road",
        # `duration_mode: source` means the API never sends one. Nothing here
        # may start depending on it.
        parameters={"aspect_ratio": "16:9", "quality": "High"},
        inputs=inputs,
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── Duration comes from the file, not from the request ───────────────────


@needs_ffmpeg
async def test_the_result_is_the_length_of_the_source(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline promise. Nothing about it is configurable."""
    source = await make_clip(workspace / "source.mp4", 3.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 3.0))

    result, _ = await collect(restyle_job(workspace, source))

    assert result.duration_seconds == pytest.approx(3.0, abs=1.0)
    assert result.kind == "video"


@needs_ffmpeg
async def test_a_stray_duration_parameter_cannot_override_the_source(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API rejects a duration on this workflow, so one arriving here is a
    client bug. Honouring it would silently contradict "Same as source video"
    — the customer would get 60 seconds of a 3-second clip."""
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 3.0))

    job = restyle_job(
        workspace, source, parameters={"aspect_ratio": "16:9", "duration": "60s"}
    )
    result, _ = await collect(job)

    assert result.duration_seconds == pytest.approx(3.0, abs=1.0)
    assert len(invocations(log)) == 1, "60s would have been two passes"


@needs_ffmpeg
async def test_a_source_beyond_one_pass_is_segmented_not_refused(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The >30s case in miniature: the ceiling is lowered so a 4-second source
    needs four passes, which is the same arithmetic a 120-second source runs
    against the real 30-second ceiling."""
    source = await make_clip(workspace / "source.mp4", 4.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = restyle_job(
        workspace, source, execution={"runtime": "ltx", "max_segment_seconds": 1}
    )
    result, reported = await collect(job)

    assert len(invocations(log)) == 4
    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)

    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress), "four passes must not restart the bar"
    messages = [message for _, _, message in reported]
    assert "Generating section 1 of 4…" in messages
    assert "Generating section 4 of 4…" in messages


@needs_ffmpeg
async def test_an_awkward_length_keeps_its_remainder(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2.5 seconds against a 1-second ceiling is 1 + 1 + 0.5, and the last
    half second is the difference between "matches the source" and "close
    enough". The frame counts asked of the model prove where it went."""
    source = await make_clip(workspace / "source.mp4", 2.5)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = restyle_job(
        workspace, source, execution={"runtime": "ltx", "max_segment_seconds": 1}
    )
    await collect(job)

    # What is under test is the plan the model was given: the tail is asked
    # for as a real half-second pass, not dropped and not rounded up to a
    # fourth full one.
    frames = [int(value_of(argv, "--num-frames")) for argv in invocations(log)]
    assert frames == [24, 24, 12], "the remainder was rounded away"


# ── Conditioning: the source must actually reach the model ───────────────


@needs_ffmpeg
async def test_each_pass_is_shown_stills_from_its_own_window_of_the_source(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is what makes it a restyle rather than a fresh generation.

    Pass two must be conditioned on the SECOND second of the source, not on
    the first — otherwise the second half of the result reproduces the first
    half's composition while the prompt supplies the look, which reads as the
    video looping.
    """
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = restyle_job(
        workspace, source, execution={"runtime": "ltx", "max_segment_seconds": 1}
    )
    await collect(job)

    first, second = (conditioning_of(argv) for argv in invocations(log))

    # Distinct stills per pass, all extracted from the source.
    first_stills = {path for path, _, _ in first}
    second_stills = {path for path, _, _ in second}
    assert first_stills.isdisjoint(second_stills)
    assert all("keyframes" in path for path, _, _ in first)

    # Pass two opens on pass one's final frame, at high strength: that frame
    # IS the seam, and both sides of it must be the same video.
    seam_path, seam_frame, seam_strength = second[0]
    assert seam_frame == 0
    assert seam_path.endswith("restyled-condition-0001.png")
    assert seam_strength >= 0.8

    # …and the rest of pass two is still the source's own structure.
    assert len(second) > 1
    assert all(frame > 0 for _, frame, _ in second[1:])


@needs_ffmpeg
async def test_conditioning_strength_leaves_room_for_the_prompt(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full strength on a source still means the source frame IS the output
    frame and the prompt does nothing — a "restyle" that returns the input.
    The structure dial has to sit below 1.0 and stay tunable per workflow."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(restyle_job(workspace, source))
    strengths = [strength for _, _, strength in conditioning_of(invocations(log)[0])]
    assert strengths and all(0 < value < 1.0 for value in strengths)

    log.unlink()
    await collect(
        restyle_job(
            workspace, source,
            execution={"runtime": "ltx", "v2v_structure_strength": 0.4},
        )
    )
    assert all(
        strength == pytest.approx(0.4)
        for _, _, strength in conditioning_of(invocations(log)[0])
    )


@needs_ffmpeg
async def test_the_number_of_source_stills_is_tunable_and_bounded(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One still locks the opening and lets everything after it drift; too
    many turn the restyle into a slideshow of the original. It is a quality
    judgement to be made against real footage, so it is configuration."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(
        restyle_job(workspace, source, execution={"runtime": "ltx", "v2v_keyframes": 1})
    )
    # One still, plus the frame-0 anchor that has nothing else to occupy it.
    assert len(conditioning_of(invocations(log)[0])) == 2

    log.unlink()
    await collect(
        restyle_job(workspace, source, execution={"runtime": "ltx", "v2v_keyframes": 99})
    )
    assert len(conditioning_of(invocations(log)[0])) <= 9, "an unbounded count would OOM"


# ── The optional reference image ─────────────────────────────────────────


@needs_ffmpeg
async def test_a_reference_image_guides_without_displacing_the_source(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract the customer reads says the reference "guides the look".

    So it appears once, at the opening, at a strength well below the source's
    — and the source's own stills are still there, still stronger. A reference
    that outweighed them would be replacing the video the user uploaded with
    the picture they attached to it.
    """
    from worker.media import extract_final_frame

    source = await make_clip(workspace / "source.mp4", 2.0)
    # Any genuinely decodable still will do; one lifted from the clip itself
    # saves synthesising a second fixture.
    reference = await extract_final_frame(source, workspace / "reference.png")

    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))
    job = restyle_job(
        workspace, source, reference,
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    )
    await collect(job)

    first, second = (conditioning_of(argv) for argv in invocations(log))

    reference_items = [item for item in first if item[0] == str(reference)]
    assert len(reference_items) == 1, "the reference conditions the opening, once"
    _, frame, strength = reference_items[0]
    assert frame == 0
    assert strength <= 0.5, "a reference that strong would be the first frame"

    source_strengths = [s for path, _, s in first if path != str(reference)]
    assert source_strengths and min(source_strengths) > strength

    # Later passes continue from the render, not from the reference again —
    # otherwise every seam would snap back toward the attached picture.
    assert str(reference) not in [path for path, _, _ in second]


@needs_ffmpeg
async def test_an_unreadable_reference_image_fails_before_any_gpu_time(
    workspace: Path, fake_models: Path,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0)
    junk = workspace / "reference.png"
    junk.write_bytes(b"definitely not an image")

    with pytest.raises(AdapterError) as raised:
        await collect(restyle_job(workspace, source, junk))

    assert raised.value.retriable is False
    assert "image" in raised.value.user_message.lower()


# ── Shape, resolution, audio ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("size", "expected"),
    [("320x180", (320, 180)), ("180x320", (180, 320)), ("240x240", (240, 240))],
)
@needs_ffmpeg
async def test_landscape_portrait_and_square_all_come_back_as_themselves(
    size: str, expected: tuple[int, int],
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delivery is at the source's own resolution. A user's portrait phone clip
    coming back letterboxed into 16:9 is the single most visible way to get
    this wrong."""
    source = await make_clip(workspace / "source.mp4", 2.0, size=size)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    result, _ = await collect(restyle_job(workspace, source))

    assert (result.width, result.height) == expected

    # And the generation grid followed the SOURCE's aspect, not the request's
    # 16:9 — a mismatched grid makes the model keep the style and replace the
    # subject, which for a restyle is the whole failure.
    argv = invocations(log)[0]
    grid = (int(value_of(argv, "--width")), int(value_of(argv, "--height")))
    assert grid == grid_for_source(*expected)
    assert grid[0] % 64 == 0 and grid[1] % 64 == 0


@needs_ffmpeg
async def test_the_sources_own_audio_survives_the_restyle(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model generates its own soundtrack. Shipping that over a restyle
    would replace the user's audio with an invented one — so the picture is
    stripped and the original track is laid back over it."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0, audio=True))

    result, reported = await collect(restyle_job(workspace, source))

    info = await probe_media(result.path)
    assert info.has_audio is True
    assert "Restoring your audio…" in [message for _, _, message in reported]


@needs_ffmpeg
async def test_a_silent_source_stays_silent(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No invented audio on a clip that had none."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0, audio=True))

    result, _ = await collect(restyle_job(workspace, source))

    info = await probe_media(result.path)
    assert info.has_audio is False


# ── Failure handling ─────────────────────────────────────────────────────


async def test_a_missing_source_is_an_internal_error_not_a_render(
    workspace: Path, fake_models: Path
) -> None:
    with pytest.raises(AdapterError) as raised:
        await collect(restyle_job(workspace, source=None))

    assert raised.value.retriable is False
    assert "not staged" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_corrupt_source_fails_immediately_without_burning_retries(
    workspace: Path, fake_models: Path
) -> None:
    """A corrupt upload is corrupt on all three attempts. Probing first turns
    a wasted render into an answer the user can act on."""
    junk = workspace / "source.mp4"
    junk.write_bytes(b"not a video")

    with pytest.raises(AdapterError) as raised:
        await collect(restyle_job(workspace, junk))

    assert raised.value.retriable is False
    assert "video" in raised.value.user_message.lower()
    assert "ffprobe" not in raised.value.user_message.lower()


@needs_ffmpeg
async def test_an_audio_only_upload_is_refused(
    workspace: Path, fake_models: Path, tmp_path: Path
) -> None:
    """Unsupported media that happens to be readable — the probe finds no video
    stream, and the job must say so rather than generating from nothing."""
    from tests.conftest import make_track

    track = await make_track(workspace / "source.mp4", 1.0)

    with pytest.raises(AdapterError) as raised:
        await collect(restyle_job(workspace, track))

    assert raised.value.retriable is False
    assert "not usable video" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_failed_pass_stops_the_job_rather_than_shipping_a_short_video(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping a failed section would produce a video of the wrong length that
    otherwise looks fine — the failure nobody notices until the client does."""
    source = await make_clip(workspace / "source.mp4", 3.0)
    render_stub(
        tmp_path, monkeypatch,
        await make_clip(tmp_path / "render.mp4", 1.0),
        fail_on_pass=1,
    )

    job = restyle_job(
        workspace, source, execution={"runtime": "ltx", "max_segment_seconds": 1}
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert "exited 3" in raised.value.internal_detail
    assert "cuda" not in raised.value.user_message.lower()


@needs_ffmpeg
async def test_cancellation_stops_the_chain_before_the_next_pass(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled multi-pass restyle must release the GPU in seconds, not at
    the end of a render nobody is waiting for."""
    source = await make_clip(workspace / "source.mp4", 4.0)
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0), sleep=15.0
    )

    cancelled = asyncio.Event()

    async def cancel_once_generating(status: str, progress: int, message: str) -> None:
        if status == "generating":
            cancelled.set()

    job = restyle_job(
        workspace, source,
        execution={"runtime": "ltx", "max_segment_seconds": 1},
        _cancelled=cancelled,
    )

    began = time.monotonic()
    with pytest.raises(JobCancelled):
        await LtxAdapter().run(job, cancel_once_generating)

    assert time.monotonic() - began < 10, "cancellation waited out the render"
    assert len(invocations(log)) == 1, "a later pass started on a cancelled job"


@needs_ffmpeg
async def test_a_wrong_length_result_fails_validation(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"The output matches the source" is measured on the finished file. A
    render that quietly produced the wrong length fails the job rather than
    shipping as "the restyle feels short"."""
    source = await make_clip(workspace / "source.mp4", 4.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    with pytest.raises(AdapterError) as raised:
        await collect(restyle_job(workspace, source))

    assert "failed validation" in raised.value.internal_detail
