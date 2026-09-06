"""Character Replacement over the WHOLE source — a chain of windows.

Client request, 6 Sep 2026: the result should follow the length of the
source video, as Video to Video does, without changing the graph that
already works for 10 s. So a longer source is cut into whole-second
windows on the graph's 24 fps grid, each window is one unchanged run of the
graph, the next window's reference picture is the last frame the previous
one produced, the seam frame is dropped, and the source's own soundtrack is
laid over the joined result.

Everything below runs against the fake ComfyUI with real ffmpeg files, so
the cuts, the uploads, the per-window graph edits, the seam arithmetic, the
join, the soundtrack and the metadata are all measured. What the MODEL does
at a seam is GPU validation, still pending.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_character_replacement import _input, _job, _still
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _service
from worker.adapters.base import AdapterError, AdapterJob
from worker.adapters.character_replacement import (
    SEAM_OVERLAP_FRAMES,
    CharacterReplacementAdapter,
    Window,
    delivered_frames,
    plan_windows,
)
from worker.comfy.ltx_graphs import character_frames_for_seconds
from worker.core.config import settings
from worker.media import probe_media

FPS = 24


# ── The plan ────────────────────────────────────────────────────────────────


def test_windows_are_whole_seconds_even_and_chained_on_one_frame() -> None:
    plan = plan_windows(25, 10)
    assert [w.seconds for w in plan] == [9, 8, 8]  # even, never 10 + 10 + 5
    assert [w.frames for w in plan] == [217, 193, 193]  # the graph's own formula
    # Each window starts on the previous one's LAST frame (its reference).
    assert [w.start_frame for w in plan] == [0, 216, 408]
    assert [w.kept_frames for w in plan] == [217, 192, 192]
    assert delivered_frames(plan) == 601  # 25.04 s — the source's own timeline
    assert plan_windows(20, 10) == [
        Window(index=0, seconds=10, frames=241, start_frame=0),
        Window(index=1, seconds=10, frames=241, start_frame=240),
    ]
    assert [w.seconds for w in plan_windows(11, 10)] == [6, 5]
    assert [w.seconds for w in plan_windows(120, 10)] == [10] * 12
    assert plan_windows(10, 10) == [Window(index=0, seconds=10, frames=241, start_frame=0)]
    assert plan_windows(7, 10) == [Window(index=0, seconds=7, frames=169, start_frame=0)]
    with pytest.raises(ValueError):
        plan_windows(0, 10)


def test_the_chain_follows_the_whole_source_up_to_the_total_ceiling(tmp_path: Path) -> None:
    from worker.media.probe import MediaInfo

    info = MediaInfo(duration_seconds=25.4, width=576, height=1024, has_video=True, has_audio=True)
    plan = CharacterReplacementAdapter.windows_for(info, _job(tmp_path, []))
    assert [w.seconds for w in plan] == [9, 8, 8]
    # The total ceiling caps the chain; the per-window ceiling is a floor for it.
    capped = AdapterJob(
        job_id="j",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement", "max_total_seconds": 12},
    )
    assert [w.seconds for w in CharacterReplacementAdapter.windows_for(info, capped)] == [6, 6]
    short = MediaInfo(duration_seconds=8.6, width=576, height=1024, has_video=True, has_audio=True)
    assert [w.seconds for w in CharacterReplacementAdapter.windows_for(short, _job(tmp_path, []))] == [8]


# ── The chain, end to end against the fake ─────────────────────────────────


async def _render(path: Path, frames: int) -> Path:
    """What the graph writes for one window: exactly `frames` frames, with sound."""
    return await make_clip(path, frames / FPS, audio=True, size="144x256")


def _submission_facts(prompt: dict) -> dict[str, object]:
    [video] = [e for e in prompt.values() if e["class_type"] == "VHS_LoadVideoFFmpeg"]
    [image] = [e for e in prompt.values() if e["class_type"] == "LoadImage"]
    constants = {
        e["_meta"]["title"]: e["inputs"]["value"]
        for e in prompt.values()
        if e["class_type"] == "INTConstant"
    }
    [combine] = [e for e in prompt.values() if e["class_type"] == "VHS_VideoCombine"]
    seeds = [e["inputs"]["noise_seed"] for e in prompt.values() if e["class_type"] == "RandomNoise"]
    return {
        "video": video["inputs"]["video"],
        "image": image["inputs"]["image"],
        "length": constants["Set Length (seconds)"],
        "canvas": (constants["Set Width"], constants["Set Height"]),
        "prefix": combine["inputs"]["filename_prefix"],
        "seeds": seeds,
    }


@needs_ffmpeg
async def test_a_twenty_second_source_runs_two_windows_chained_on_the_last_frame(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.3, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "unused.mp4", 241))
    fake.output_queue = [
        await _render(tmp_path / "render0.mp4", 241),
        await _render(tmp_path / "render1.mp4", 241),
    ]
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, reports = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
            prompt="a woman with silver hair in a red coat",
        ),
        on_progress,
    )

    # The photo, then per window: the cut clip; after window 0 its last
    # frame becomes window 1's reference picture.
    assert [name for name, _ in fake.uploads] == [
        "zolex_job-cr-1_reference.png",
        "zolex_job-cr-1_window00.mp4",
        "zolex_job-cr-1_reference01.png",
        "zolex_job-cr-1_window01.mp4",
    ]
    assert len(fake.submissions) == 2
    first, second = (_submission_facts(s["prompt"]) for s in fake.submissions)
    assert first == {
        "video": "zolex_job-cr-1_window00.mp4",
        "image": "zolex_job-cr-1_reference.png",
        "length": 10,
        "canvas": (736, 1280),
        "prefix": "zolexai/job-cr-1/window00",
        "seeds": [123463, 42],  # the graph's own fixed seeds, every window
    }
    assert second == {
        "video": "zolex_job-cr-1_window01.mp4",
        "image": "zolex_job-cr-1_reference01.png",
        "length": 10,
        "canvas": (736, 1280),
        "prefix": "zolexai/job-cr-1/window01",
        "seeds": [123463, 42],
    }
    # The cut windows are on the graph's grid: exactly the formula's frames
    # at 24 fps, with sound, window 1 starting on window 0's last frame.
    for index in (0, 1):
        clip = await probe_media(workspace / f"zolex_job-cr-1_window{index:02d}.mp4")
        assert clip.frame_count == 241 and clip.fps == 24 and clip.has_audio
    # Delivered: 241 + (241 − 1) frames = 20.04 s, the source's own timeline,
    # with one audio track.
    assert result.kind == "video" and result.path == workspace / "output.mp4"
    assert result.duration_seconds and abs(result.duration_seconds - 481 / FPS) < 0.2
    out = await probe_media(result.path)
    assert out.has_audio and out.frame_count == 481
    metadata = json.loads((workspace / "character-replacement.json").read_text(encoding="utf-8"))
    assert metadata["reference_mode"] == "previous_frame"
    assert [w["seconds"] for w in metadata["windows"]] == [10, 10]
    assert [w["reference"] for w in metadata["windows"]] == [
        "zolex_job-cr-1_reference.png",
        "zolex_job-cr-1_reference01.png",
    ]
    assert metadata["seams"] == [10.0]
    assert metadata["overlap_frames_per_seam"] == SEAM_OVERLAP_FRAMES
    assert abs(metadata["measured_seconds"] - 481 / FPS) < 0.2
    # Progress announced both sections and ended uploading.
    statuses = [status for status, _, _ in reports]
    assert statuses[0] == "preparing" and statuses[-1] == "uploading"
    assert any("section 1 of 2" in message for _, _, message in reports)
    assert any("section 2 of 2" in message for _, _, message in reports)


@needs_ffmpeg
async def test_an_uneven_source_splits_into_even_whole_second_windows(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 25.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "unused.mp4", 193))
    fake.output_queue = [
        await _render(tmp_path / "r0.mp4", 217),
        await _render(tmp_path / "r1.mp4", 193),
        await _render(tmp_path / "r2.mp4", 193),
    ]
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
        ),
        on_progress,
    )

    assert [_submission_facts(s["prompt"])["length"] for s in fake.submissions] == [9, 8, 8]
    assert result.duration_seconds and abs(result.duration_seconds - 601 / FPS) < 0.2
    metadata = json.loads((workspace / "character-replacement.json").read_text(encoding="utf-8"))
    assert [w["start_frame"] for w in metadata["windows"]] == [0, 216, 408]
    assert metadata["seams"] == [round(216 / FPS, 4), round(408 / FPS, 4)]


@needs_ffmpeg
async def test_photo_mode_hands_every_window_the_customers_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "character_replacement_chain_reference", "photo")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", 241))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
        ),
        on_progress,
    )

    assert [name for name, _ in fake.uploads] == [
        "zolex_job-cr-1_reference.png",
        "zolex_job-cr-1_window00.mp4",
        "zolex_job-cr-1_window01.mp4",
    ]
    assert [_submission_facts(s["prompt"])["image"] for s in fake.submissions] == [
        "zolex_job-cr-1_reference.png",
        "zolex_job-cr-1_reference.png",
    ]


@needs_ffmpeg
async def test_the_total_ceiling_still_cuts_a_very_long_source(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 40.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", 241))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    job = AdapterJob(
        job_id="job-cr-1",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        inputs=[_input("source_video", source, "video"), _input("reference_image", reference, "image")],
        execution={"runtime": "character_replacement", "max_total_seconds": 20},
        output_content_type="video/mp4",
        workspace=workspace,
    )

    result = await adapter.run(job, on_progress)

    assert len(fake.submissions) == 2
    assert result.duration_seconds and abs(result.duration_seconds - 481 / FPS) < 0.2


@needs_ffmpeg
async def test_a_silent_source_gets_silent_windows_and_a_silent_track(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.0, audio=False, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", 241))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
        ),
        on_progress,
    )

    clip = await probe_media(workspace / "zolex_job-cr-1_window01.mp4")
    assert clip.has_audio  # the graph's combiner needs a track; silence is it
    out = await probe_media(result.path)
    assert out.has_audio and out.frame_count == 481


@needs_ffmpeg
async def test_a_window_that_fails_fails_the_job_with_no_output(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", 241))
    fake.fail_history = True
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    with pytest.raises(AdapterError) as raised:
        await adapter.run(
            _job(
                workspace,
                [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
            ),
            on_progress,
        )
    assert raised.value.retriable is True
    assert not (workspace / "output.mp4").exists()


@needs_ffmpeg
async def test_a_source_within_one_window_runs_the_path_that_ran_before(tmp_path: Path) -> None:
    """The guarantee: a short source is still ONE run with the clip uploaded
    as-is — no cutting, no chain files, the same two uploads as always."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 8.6, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    frames = min(character_frames_for_seconds(8, FPS), int(8.6 * FPS))
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", frames))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
        ),
        on_progress,
    )

    assert [name for name, _ in fake.uploads] == [
        "zolex_job-cr-1_source.mp4",
        "zolex_job-cr-1_reference.png",
    ]
    assert len(fake.submissions) == 1
    assert _submission_facts(fake.submitted["prompt"])["prefix"] == "zolexai/job-cr-1/output"
    assert not list(workspace.glob("window-*.mp4"))
    assert not (workspace / "character-replacement.json").exists()
    assert result.duration_seconds and abs(result.duration_seconds - frames / FPS) < 0.2
