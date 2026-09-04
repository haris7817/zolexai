"""The extension engine — chained continuation, proven with real files (Phase 4).

`continue_video` is exercised two ways: with a fake render pass that writes
real clips (so the trim, the fades, the normalisation, the concat and the
verification are all genuine ffmpeg work), and through the ltx_comfy adapter
against the fake ComfyUI, where every section is one submission of the
First/Last Frame graph with the previous part's last frame as the first
frame.

What the model does at a seam is WAITING FOR GPU VALIDATION; what the engine
does with the model's output is settled here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _rendered, _service
from worker.adapters.base import AdapterError, AdapterInput, AdapterJob
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.longform.chain import ChainStep
from worker.longform.continuation import (
    AUDIO_EDGE_FADE_SECONDS,
    SEAM_OVERLAP_FRAMES,
    ContinuationMetadata,
    continue_video,
    frames_for,
    passes_needed,
)
from worker.longform.progress import StageReporter
from worker.media import probe_media

FPS = 24


def _job(workspace: Path, **execution) -> AdapterJob:
    return AdapterJob(
        job_id="job-ext-1",
        workflow_id="extend-video",
        workflow_version="1",
        prompt="the walk continues along the pier",
        parameters={"duration": "10s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx_comfy", **execution},
        output_content_type="video/mp4",
        workspace=workspace,
    )


# ── The engine on its own ───────────────────────────────────────────────────


@needs_ffmpeg
async def test_a_chained_continuation_lands_on_the_promised_length(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 3.0, audio=True, size="160x96")
    rendered: list[tuple[int, float, str | None]] = []

    async def render_pass(step: ChainStep, frame: Path | None):
        rendered.append((step.index, step.seconds, frame.name if frame else None))
        # What the graph writes: fps·s + 1 frames, the first being the
        # conditioning frame rendered again.
        await make_clip(step.output, step.seconds + 1 / FPS, audio=True, size="128x72")
        return await probe_media(step.output)

    on_progress, reports = _recorder()
    output, metadata = await continue_video(
        _job(workspace),
        source=source,
        seconds=10.0,
        per_pass_seconds=5.0,
        fps=FPS,
        render_pass=render_pass,
        reporter=StageReporter(on_progress),
    )

    # Two passes of five seconds, each conditioned on a frame the timeline
    # already has: the source's last, then pass 0's last.
    assert [(i, s) for i, s, _ in rendered] == [(0, 5.0), (1, 5.0)]
    assert rendered[0][2] == "continuation-seed.png"
    assert rendered[1][2] is not None and rendered[1][2] != rendered[0][2]

    info = await probe_media(output)
    assert abs(info.duration_seconds - 13.0) < 0.2
    assert info.width == 160 and info.height == 96  # the SOURCE's frame, not the graph's
    assert info.has_audio
    # Every generated part contributes exactly s·fps frames: the overlap
    # frame is gone, so 3 s + 10 s of 24 fps is 312 frames.
    assert info.frame_count == 3 * FPS + 10 * FPS

    assert metadata.seams == [3.0, 8.0]
    assert metadata.overlap_frames_per_seam == SEAM_OVERLAP_FRAMES
    assert [p.frames_kept for p in metadata.passes] == [120, 120]
    assert metadata.promised_seconds == pytest.approx(13.0)
    assert metadata.measured_seconds == pytest.approx(13.0, abs=0.2)
    sidecar = json.loads((workspace / "continuation.json").read_text())
    assert sidecar["status"] == "WAITING FOR GPU VALIDATION"
    assert sidecar["source"]["seconds"] == pytest.approx(3.0, abs=0.1)
    assert len(sidecar["passes"]) == 2

    statuses = [s for s, _, _ in reports]
    assert "post_processing" in statuses
    progresses = [p for _, p, _ in reports]
    assert progresses == sorted(progresses)


@needs_ffmpeg
async def test_a_single_pass_extension_needs_no_chain_machinery(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=False, size="160x96")

    async def render_pass(step: ChainStep, frame: Path | None):
        await make_clip(step.output, step.seconds + 1 / FPS, audio=True, size="160x96")
        return await probe_media(step.output)

    on_progress, _ = _recorder()
    output, metadata = await continue_video(
        _job(workspace),
        source=source,
        seconds=5.0,
        per_pass_seconds=30.0,
        fps=FPS,
        render_pass=render_pass,
        reporter=StageReporter(on_progress),
    )
    info = await probe_media(output)
    assert abs(info.duration_seconds - 7.0) < 0.2
    assert info.has_audio  # a silent source is given silence so the join has one audio layout
    assert len(metadata.passes) == 1 and metadata.seams == [2.0]


@needs_ffmpeg
async def test_a_short_render_fails_the_length_check(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True)

    async def render_pass(step: ChainStep, frame: Path | None):
        await make_clip(step.output, 1.0, audio=True)  # 1 s for a 5 s ask
        return await probe_media(step.output)

    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await continue_video(
            _job(workspace),
            source=source,
            seconds=5.0,
            per_pass_seconds=30.0,
            fps=FPS,
            render_pass=render_pass,
            reporter=StageReporter(on_progress),
        )
    assert "differs from planned" in raised.value.internal_detail


async def test_an_unreadable_source_is_refused_before_rendering(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    calls = 0

    async def render_pass(step: ChainStep, frame: Path | None):
        nonlocal calls
        calls += 1
        raise AssertionError("must not render")

    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await continue_video(
            _job(tmp_path),
            source=bad,
            seconds=5.0,
            per_pass_seconds=30.0,
            fps=FPS,
            render_pass=render_pass,
            reporter=StageReporter(on_progress),
        )
    assert raised.value.retriable is False and calls == 0


def test_arithmetic_helpers() -> None:
    assert frames_for(5.0, 24) == 120
    assert frames_for(0.01, 24) == 1
    assert passes_needed(30, 30) == 1
    assert passes_needed(31, 30) == 2
    assert passes_needed(90, 30) == 3
    assert 0 < AUDIO_EDGE_FADE_SECONDS < 0.5
    meta = ContinuationMetadata("j", "extend-video", "ltx_comfy", 24.0, 30.0, 10.0, None)
    assert meta.status == "WAITING FOR GPU VALIDATION"


# ── Through the adapter ─────────────────────────────────────────────────────


def _video(path: Path) -> AdapterInput:
    return AdapterInput(
        role="source_video",
        kind="video",
        content_type="video/mp4",
        download_url="http://unused",
        path=path,
    )


@needs_ffmpeg
async def test_extend_video_runs_the_first_frame_graph_once_per_section(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 3.0, audio=True, size="160x90")
    # The fake serves the same 5 s render for every submission.
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    job = AdapterJob(
        job_id="job-ext-1",
        workflow_id="extend-video",
        workflow_version="1",
        prompt="the walk continues along the pier",
        parameters={"duration": "10s", "aspect_ratio": "16:9"},
        inputs=[_video(source)],
        execution={"runtime": "ltx_comfy", "max_segment_seconds": 5},
        output_content_type="video/mp4",
        workspace=workspace,
    )
    result = await adapter.run(job, on_progress)

    assert result.kind == "video"
    assert result.duration_seconds and abs(result.duration_seconds - 13.0) < 0.2
    assert result.width == 160 and result.height == 90
    # Two sections → two conditioning stills uploaded → two submissions of the
    # first-frame graph, each with its own still and seed.
    assert [name for name, _ in fake.uploads] == [
        "zolex_job-ext-1_continue00.png",
        "zolex_job-ext-1_continue01.png",
    ]
    prompt = fake.submitted["prompt"]
    types = {e["class_type"] for e in prompt.values()}
    assert "LTXVImgToVideoInplace" in types and "LTXVImgToVideoInplaceKJ" not in types
    [loader] = [e for e in prompt.values() if e["class_type"] == "LoadImage"]
    assert loader["inputs"]["image"] == "zolex_job-ext-1_continue01.png"
    [selector] = [e for e in prompt.values() if e["class_type"] == "ResolutionSelector"]
    assert selector["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"  # closest to a 160x90 source
    assert (workspace / "continuation.json").is_file()


@needs_ffmpeg
async def test_a_portrait_source_picks_the_portrait_canvas(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True, size="90x160")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    job = AdapterJob(
        job_id="job-ext-2",
        workflow_id="extend-video",
        workflow_version="1",
        prompt="x",
        parameters={"duration": "5s"},
        inputs=[_video(source)],
        execution={"runtime": "ltx_comfy"},
        output_content_type="video/mp4",
        workspace=workspace,
    )
    result = await adapter.run(job, on_progress)
    [selector] = [
        e for e in fake.submitted["prompt"].values() if e["class_type"] == "ResolutionSelector"
    ]
    assert selector["inputs"]["aspect_ratio"] == "9:16 (Portrait Widescreen)"
    assert result.width == 90 and result.height == 160


async def test_extend_without_a_source_is_refused(tmp_path: Path) -> None:
    adapter = LtxComfyAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path), on_progress)
    assert raised.value.retriable is False
    assert "video" in raised.value.user_message.lower()


def test_the_runtime_now_serves_extend_video() -> None:
    assert LtxComfyAdapter().supports("extend-video")
