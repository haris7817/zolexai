"""Character Replacement on the client's graph — a separate module (Phase 3).

Same fake service as `test_ltx_comfy.py`. What is pinned here: the two
uploads (the clip repacked with a guaranteed audio track, the reference
PNG-normalised), the graph 03 edits (video, image, whole-second window,
oriented canvas, prompt lead + description, seeds, output prefix), the
window rule, the refusals, and the separation from Video to Video.

STATUS for the model: WAITING FOR GPU VALIDATION (a side-by-side with the
delivered sample is the acceptance test).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _service
from worker.adapters.base import AdapterError, AdapterInput, AdapterJob
from worker.adapters.character_replacement import CharacterReplacementAdapter
from worker.adapters.registry import get_adapter
from worker.comfy.ltx_prompts import CHARACTER_REPLACEMENT_LEAD, CHARACTER_REPLACEMENT_NEGATIVE
from worker.core.config import settings
from worker.media import probe_media
from worker.media.ffmpeg import ffmpeg
from worker.workflows.resolver import resolve_adapter

FPS = 24


async def _still(path: Path) -> Path:
    await ffmpeg(
        ["-f", "lavfi", "-i", "color=c=green:s=96x64:d=0.1", "-frames:v", "1", str(path), "-y"]
    )
    return path


def _input(role: str, path: Path, kind: str) -> AdapterInput:
    return AdapterInput(
        role=role,
        kind=kind,
        content_type="video/mp4" if kind == "video" else "image/png",
        download_url="http://unused",
        path=path,
    )


def _job(workspace: Path, inputs: list[AdapterInput], prompt: str = "", **params) -> AdapterJob:
    return AdapterJob(
        job_id="job-cr-1",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt=prompt,
        parameters=params,
        inputs=inputs,
        execution={"runtime": "character_replacement"},
        output_content_type="video/mp4",
        workspace=workspace,
    )


async def _rendered_for(path: Path, seconds: int, source_seconds: float) -> Path:
    """What graph 03 writes: round((fps·s−1)/8)·8+1 frames, capped by the source."""
    from worker.comfy.ltx_graphs import character_frames_for_seconds

    frames = min(character_frames_for_seconds(seconds, FPS), int(source_seconds * FPS))
    return await make_clip(path, frames / FPS, audio=True)


# ── The flow ────────────────────────────────────────────────────────────────


@needs_ffmpeg
async def test_replacement_runs_the_client_graph_end_to_end(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 8.6, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _rendered_for(tmp_path / "render.mp4", 8, 8.6))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, reports = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
            prompt="a man with short black curls and a charcoal suit",
        ),
        on_progress,
    )

    assert result.kind == "video" and result.path.exists()
    assert [name for name, _ in fake.uploads] == [
        "zolex_job-cr-1_source.mp4",
        "zolex_job-cr-1_reference.png",
    ]
    prompt = fake.submitted["prompt"]
    [video] = [e for e in prompt.values() if e["class_type"] == "VHS_LoadVideoFFmpeg"]
    assert video["inputs"]["video"] == "zolex_job-cr-1_source.mp4"
    [image] = [e for e in prompt.values() if e["class_type"] == "LoadImage"]
    assert image["inputs"]["image"] == "zolex_job-cr-1_reference.png"
    constants = {
        e["_meta"]["title"]: e["inputs"]["value"]
        for e in prompt.values()
        if e["class_type"] == "INTConstant"
    }
    # 8.6 s source → an 8 s window (whole seconds); portrait source keeps the
    # pack's portrait canvas.
    assert constants == {"Set Length (seconds)": 8, "Set Width": 736, "Set Height": 1280}
    [conditioning] = [e for e in prompt.values() if e["class_type"] == "LTXVConditioning"]
    positive = prompt[conditioning["inputs"]["positive"][0]]["inputs"]["text"]
    negative = prompt[conditioning["inputs"]["negative"][0]]["inputs"]["text"]
    assert positive.startswith(CHARACTER_REPLACEMENT_LEAD)
    assert positive.endswith("a man with short black curls and a charcoal suit")
    assert negative == CHARACTER_REPLACEMENT_NEGATIVE
    [combine] = [e for e in prompt.values() if e["class_type"] == "VHS_VideoCombine"]
    assert combine["inputs"]["filename_prefix"] == "zolexai/job-cr-1/output"
    seeds = [e["inputs"]["noise_seed"] for e in prompt.values() if e["class_type"] == "RandomNoise"]
    assert len(seeds) == 2 and len(set(seeds)) == 2
    statuses = [status for status, _, _ in reports]
    assert statuses[0] == "preparing" and statuses[-1] == "uploading"


@needs_ffmpeg
async def test_a_landscape_source_flips_the_packs_canvas(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 5.0, audio=True, size="256x144")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _rendered_for(tmp_path / "render.mp4", 5, 5.0))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
        ),
        on_progress,
    )
    constants = {
        e["_meta"]["title"]: e["inputs"]["value"]
        for e in fake.submitted["prompt"].values()
        if e["class_type"] == "INTConstant"
    }
    assert constants["Set Width"] == 1280 and constants["Set Height"] == 736
    assert constants["Set Length (seconds)"] == 5


@needs_ffmpeg
async def test_a_silent_source_is_given_a_silent_track_before_upload(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 3.0, audio=False)
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _rendered_for(tmp_path / "render.mp4", 3, 3.0))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
        ),
        on_progress,
    )
    repacked = await probe_media(workspace / "zolex_job-cr-1_source.mp4")
    assert repacked.has_audio and repacked.has_video


@needs_ffmpeg
async def test_a_long_source_is_windowed_to_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "character_replacement_max_seconds", 6)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 12.0, audio=True)
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _rendered_for(tmp_path / "render.mp4", 6, 12.0))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    result = await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
        ),
        on_progress,
    )
    constants = {
        e["_meta"]["title"]: e["inputs"]["value"]
        for e in fake.submitted["prompt"].values()
        if e["class_type"] == "INTConstant"
    }
    assert constants["Set Length (seconds)"] == 6
    assert result.duration_seconds and result.duration_seconds < 7


def test_the_execution_block_can_lower_the_ceiling(tmp_path: Path) -> None:
    from worker.media.probe import MediaInfo

    info = MediaInfo(duration_seconds=15.0, width=576, height=1024, has_video=True, has_audio=True)
    job = _job(tmp_path, [])
    assert CharacterReplacementAdapter.window_seconds(info, job) == 15
    lowered = AdapterJob(
        job_id="j",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement", "max_seconds": 10},
    )
    assert CharacterReplacementAdapter.window_seconds(info, lowered) == 10


# ── Refusals ────────────────────────────────────────────────────────────────


async def test_missing_inputs_are_refused_before_any_upload(tmp_path: Path) -> None:
    fake = FakeLtxComfy(tmp_path / "none.mp4")
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path, []), on_progress)
    assert raised.value.retriable is False and "video" in raised.value.user_message.lower()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(
            _job(tmp_path, [_input("source_video", tmp_path / "x.mp4", "video")]), on_progress
        )
    assert raised.value.retriable is False and "character" in raised.value.user_message.lower()
    assert fake.uploads == []


@needs_ffmpeg
async def test_a_source_under_a_second_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 0.5, audio=True)
    reference = await _still(tmp_path / "reference.png")
    adapter = CharacterReplacementAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(
            _job(
                workspace,
                [
                    _input("source_video", source, "video"),
                    _input("reference_image", reference, "image"),
                ],
            ),
            on_progress,
        )
    assert "one second" in raised.value.user_message


async def test_other_workflows_are_refused(tmp_path: Path) -> None:
    adapter = CharacterReplacementAdapter(service=_service(FakeLtxComfy(tmp_path / "none.mp4")))
    assert adapter.supports("character-replacement")
    assert not adapter.supports("video-to-video")
    on_progress, _ = _recorder()
    job = AdapterJob(
        job_id="j",
        workflow_id="video-to-video",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement"},
        workspace=tmp_path,
    )
    with pytest.raises(AdapterError) as raised:
        await adapter.run(job, on_progress)
    assert raised.value.retriable is False


# ── Routing: a separate module ──────────────────────────────────────────────


def test_the_runtime_is_registered_and_nothing_committed_routes_to_it() -> None:
    assert get_adapter("character_replacement").name == "character_replacement"
    definitions = Path(__file__).resolve().parents[3] / "workflow-definitions"
    for yaml_path in definitions.glob("*.yaml"):
        assert "runtime: character_replacement" not in yaml_path.read_text(encoding="utf-8"), (
            yaml_path.name
        )


def test_character_replacement_resolves_to_its_own_adapter_and_never_to_v2v_paths() -> None:
    job = AdapterJob(
        job_id="j",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement"},
    )
    assert resolve_adapter(job).name == "character_replacement"
    v2v = AdapterJob(
        job_id="j",
        workflow_id="video-to-video",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "ltx", "runtime_by_quality": {"best": "character_replacement"}},
    )
    # The safety net keeps Video to Video on its own runtime even if a
    # deployment line pointed its Best level here.
    assert resolve_adapter(v2v).name == "ltx"
