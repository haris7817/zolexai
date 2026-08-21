"""Music Video, audio-conditioned: the model hears the song it is drawing for.

The default music video draws its picture from the text prompt alone. The song
decides how long the video is and where the cuts fall, and then it is laid over
the finished picture. Nothing the model did was influenced by the music, which
is why a singer in that output does not — and structurally cannot — move their
mouth in time with a vocal.

Under `execution.audio_conditioning` the audio tier renders instead, and each
pass is handed the master track seeked to its own moment in the song. That is
the mechanism the client's own reference engine uses, and it is what makes
audio-driven performance possible at all.

Two things are easy to break here and both are tested directly:

**The master must never be sliced.** Seeking into one file is not the same as
cutting it into N files: slicing re-encodes the track once per section, puts a
codec boundary at every seam of the thing the visuals are synchronising to, and
is how "the audio restarts every section" bugs get built. The pipeline takes
`--audio-start-time`, so there is no reason to ever cut.

**Conditioning audio is not the soundtrack.** The generated parts still assemble
silent and the user's original file is still muxed over the result exactly once.
A2Vid returns audio in its output; if any of it reached the delivered file the
customer would hear the song stitched out of re-encoded fragments.
"""

from __future__ import annotations

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
from worker.adapters.ltx import (
    _AUDIO_PASS_SECONDS,
    _AUDIO_WINDOW_PAD_SECONDS,
    LtxAdapter,
)
from worker.core.config import settings
from worker.media import probe_media


def conditioned_job(workspace: Path, track: Path | None, **overrides):
    execution = {"runtime": "ltx", "audio_conditioning": True}
    execution.update(overrides.pop("execution", {}))
    defaults = dict(
        workflow_id="music-video",
        prompt="a singer performs into a vintage microphone on a dim club stage",
        parameters={"aspect_ratio": "16:9", "quality": "Standard"},
        inputs=[staged_input("source_audio", "audio", "audio/mpeg", track)],
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


def audio_of(argv: list[str]) -> tuple[str, float, float] | None:
    """The `--audio-path/-start-time/-max-duration` triple, if the pass had one."""
    if "--audio-path" not in argv:
        return None
    return (
        value_of(argv, "--audio-path"),
        float(value_of(argv, "--audio-start-time")),
        float(value_of(argv, "--audio-max-duration")),
    )


# ── Off by default; the existing product is unchanged ────────────────────


@needs_ffmpeg
async def test_a_plain_music_video_still_never_shows_the_model_the_song(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path is byte-for-byte the one that has been serving
    customers: distilled entry point, no audio flags, no LoRA.

    Audio conditioning costs roughly four times the compute, so it is a
    capability a workflow opts into rather than a silent change to every music
    video anyone has ever run.
    """
    track = await make_track(workspace / "song.mp3", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(
        make_job(
            workspace,
            workflow_id="music-video",
            prompt="a lone dancer",
            parameters={"aspect_ratio": "16:9"},
            inputs=[staged_input("source_audio", "audio", "audio/mpeg", track)],
        )
    )

    argv = invocations(log)[0]
    assert audio_of(argv) is None
    assert "--distilled-lora" not in argv
    assert value_of(argv, "--quantization") == settings.ltx_quantization


# ── The model is given the song ──────────────────────────────────────────


@needs_ffmpeg
async def test_every_pass_is_conditioned_on_the_track(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not just the first. A later section rendered without audio would be a
    silent-to-the-model stretch in the middle of a synchronised video."""
    track = await make_track(workspace / "song.mp3", 4.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 1.0))

    await collect(
        conditioned_job(
            workspace,
            track,
            execution={"audio_pass_seconds": 1, "align_cuts_to_audio": False},
        )
    )

    passes = invocations(log)
    assert len(passes) >= 3
    assert all(audio_of(argv) is not None for argv in passes)


@needs_ffmpeg
async def test_the_audio_tier_runs_unquantized_with_cpu_offload(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distilled LoRA is a LoRA, and the LoRA/quantization clash applies.

    This is the configuration that turned a supposed resolution ceiling into a
    clean 1024x576 render on 17 Aug 2026 — every earlier failure had been
    quantized. A `--quantization` flag reappearing here is a crash waiting to
    happen, not a speed-up.
    """
    track = await make_track(workspace / "song.mp3", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(conditioned_job(workspace, track))

    argv = invocations(log)[0]
    assert "--quantization" not in argv
    assert value_of(argv, "--offload") == "cpu"
    assert "distilled-lora" in value_of(argv, "--distilled-lora")
    assert "dev-transformer" in value_of(argv, "--transformer-path")


# ── The master is seeked, never cut ──────────────────────────────────────


@needs_ffmpeg
async def test_every_pass_points_at_the_one_uploaded_file(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One path, shared by every pass — the customer's own upload.

    If this ever becomes N different paths, someone has started slicing the
    master, and the song the model hears is no longer bit-for-bit the song the
    finished video plays.
    """
    track = await make_track(workspace / "song.mp3", 4.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 1.0))

    await collect(
        conditioned_job(
            workspace,
            track,
            execution={"audio_pass_seconds": 1, "align_cuts_to_audio": False},
        )
    )

    paths = {audio_of(argv)[0] for argv in invocations(log)}
    assert paths == {str(track)}


@needs_ffmpeg
async def test_each_pass_hears_its_own_moment_in_the_song(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start times advance, in order, matching where each section sits.

    A video whose every section was conditioned on the opening bars would be
    synchronised to the wrong music everywhere except the beginning — and would
    still look, to any check that only counts flags, exactly like a correct one.
    """
    track = await make_track(workspace / "song.mp3", 4.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 1.0))

    await collect(
        conditioned_job(
            workspace,
            track,
            execution={"audio_pass_seconds": 1, "align_cuts_to_audio": False},
        )
    )

    starts = [audio_of(argv)[1] for argv in invocations(log)]
    assert starts[0] == 0.0
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    # Consecutive sections are adjacent in the song, not overlapping copies.
    for earlier, later in zip(starts, starts[1:], strict=False):
        assert later - earlier == pytest.approx(1.0, abs=0.2)


@needs_ffmpeg
async def test_the_window_covers_the_frames_actually_rendered(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slightly more track than picture, never less.

    The audio latent grid is coarser than the frame grid, so a window cut to
    exactly the video's length can land one latent short and the pipeline reads
    past what it was handed. The pad is small and deliberate.
    """
    track = await make_track(workspace / "song.mp3", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(conditioned_job(workspace, track))

    argv = invocations(log)[0]
    _, _, window = audio_of(argv)
    frames = int(value_of(argv, "--num-frames"))
    covered = frames / float(settings.ltx_frame_rate)
    assert window == pytest.approx(covered + _AUDIO_WINDOW_PAD_SECONDS, abs=1e-3)
    assert window > covered


# ── Conditioning audio is not the soundtrack ─────────────────────────────


@needs_ffmpeg
async def test_the_delivered_file_still_carries_the_original_track_once(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's actual requirement, unchanged by any of this.

    The audio tier returns audio in every part it renders. Exactly one audio
    stream may survive to the customer, and it must be the file they uploaded —
    not the model's copy of it, reassembled from re-encoded sections.
    """
    track = await make_track(workspace / "song.mp3", 3.0)
    render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 1.0, audio=True)
    )

    result, _ = await collect(
        conditioned_job(
            workspace,
            track,
            execution={"audio_pass_seconds": 1, "align_cuts_to_audio": False},
        )
    )

    measured = await probe_media(result.path)
    assert measured.has_audio
    assert measured.audio_stream_count == 1
    assert measured.duration_seconds == pytest.approx(3.0, abs=1.0)


@needs_ffmpeg
async def test_the_result_is_still_the_length_of_the_song(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic duration is a property of the workflow, not of the engine."""
    track = await make_track(workspace / "song.mp3", 2.5)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.5))

    result, _ = await collect(conditioned_job(workspace, track))

    measured = await probe_media(result.path)
    assert measured.duration_seconds == pytest.approx(2.5, abs=1.0)


# ── The pass ceiling is the audio tier's own measurement ─────────────────


def test_the_audio_pass_ceiling_is_not_the_distilled_tiers(workspace: Path) -> None:
    """60s is what the DISTILLED decoder was measured at. The audio tier runs a
    different checkpoint, carries a LoRA, runs unquantized and streams weights
    from host RAM — none of that ceiling transfers, and assuming it does is how
    a 60s pass gets issued against a tier proven only to 20s."""
    adapter = LtxAdapter()
    job = make_job(workspace, workflow_id="music-video", execution={"runtime": "ltx"})

    # 481 frames at 24fps. Stated as the landing's own duration rather than a
    # round 20.0 so the planner's nominal window IS a measured count.
    assert adapter._audio_pass_seconds(job) == pytest.approx(481 / 24.0)
    assert adapter._audio_pass_seconds(job) < adapter._per_pass_seconds(job, (1024, 576))


def test_no_pass_can_reach_the_audio_decoder_at_an_unmeasured_count() -> None:
    """The defect that made this tier unshippable, as arithmetic.

    `_A2VID` had no landing table, so any 8k+1 count could reach its decoder.
    A real 60-second job planned 474/477/430/59 frames and its third section
    died in the video VAE. The sweep behind the table found the failing set is
    non-monotonic — 289, 337, 361, 409 and 457 all crash while 241, 385, 433,
    481 do not — so conforming to the lattice proves nothing on this path.
    """
    from worker.adapters.ltx import _A2VID, conforming_frames

    measured_bad = (289, 337, 361, 409, 457)
    assert _A2VID.measured_landings, "an empty table lets every count through"
    assert not set(_A2VID.measured_landings) & set(measured_bad)

    # Every count the planner can ask for, at every pass length it can plan.
    for requested in range(1, int(_AUDIO_PASS_SECONDS * 24) + 1):
        conforming = conforming_frames(requested)
        landing = next(
            (count for count in _A2VID.measured_landings if count >= conforming),
            None,
        )
        assert landing is not None, requested
        assert landing not in measured_bad
        assert landing >= requested, "a pass may never be shortened to fit a table"


def test_a_workflow_can_lower_the_audio_ceiling_but_not_raise_it_past_the_brake(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`execution.audio_pass_seconds` is where a future measurement lands, and
    the operational brake still clamps it — the one environment variable that
    can pull every shape down mid-incident must keep working here too."""
    adapter = LtxAdapter()
    lowered = make_job(
        workspace,
        workflow_id="music-video",
        execution={"runtime": "ltx", "audio_pass_seconds": 5},
    )
    assert adapter._audio_pass_seconds(lowered) == 5.0

    monkeypatch.setattr(settings, "ltx_max_seconds", 6)
    raised = make_job(
        workspace,
        workflow_id="music-video",
        execution={"runtime": "ltx", "audio_pass_seconds": 300},
    )
    assert adapter._audio_pass_seconds(raised) == 6.0


@needs_ffmpeg
async def test_a_missing_dev_checkpoint_fails_before_any_gpu_time(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node without the audio tier's weights refuses this job and still
    serves every other workflow — the check is per path, not per process."""
    from worker.adapters.base import AdapterError
    from worker.adapters.ltx import _OPTIONAL_MODEL_FILES

    track = await make_track(workspace / "song.mp3", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))
    (fake_models / _OPTIONAL_MODEL_FILES["transformer_dev"]).unlink()

    with pytest.raises(AdapterError) as raised:
        await collect(conditioned_job(workspace, track))

    assert raised.value.retriable is False
    assert "transformer_dev" in raised.value.internal_detail
    assert invocations(log) == []
