"""First and last frame on an extension (client request, 6 Sep 2026).

The same client First/Last Frame graph, the same extension engine; only
what the integration hands the graph changes. With no stills the run is
what it was before this feature — pinned here, next to the new cases.

Everything runs against the fake ComfyUI with real ffmpeg files, so the
uploads, the compiled graph per pass, the seam arithmetic and the metadata
are all measured. What the MODEL does with a customer's first or last frame
at an extension seam is GPU validation, still pending.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_continuation import _video
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _rendered, _service
from worker.adapters.base import AdapterError, AdapterInput, AdapterJob
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.longform.continuation import SEAM_OVERLAP_FRAMES
from worker.media.ffmpeg import ffmpeg


async def _still(path: Path, colour: str = "red") -> Path:
    await ffmpeg(
        ["-f", "lavfi", "-i", f"color=c={colour}:s=96x64:d=0.1", "-frames:v", "1", str(path), "-y"]
    )
    return path


def _image(role: str, path: Path) -> AdapterInput:
    return AdapterInput(
        role=role, kind="image", content_type="image/png", download_url="http://unused", path=path
    )


def _job(workspace: Path, inputs: list[AdapterInput], **execution) -> AdapterJob:
    return AdapterJob(
        job_id="job-ext-f",
        workflow_id="extend-video",
        workflow_version="1",
        prompt="she turns and walks toward the lighthouse",
        parameters={"duration": "10s", "aspect_ratio": "16:9"},
        inputs=inputs,
        execution={"runtime": "ltx_comfy", **execution},
        output_content_type="video/mp4",
        workspace=workspace,
    )


def _loaders(prompt: dict) -> dict[str, str]:
    return {
        e["_meta"]["title"]: e["inputs"]["image"]
        for e in prompt.values()
        if e["class_type"] == "LoadImage"
    }


def _metadata(workspace: Path) -> dict:
    return json.loads((workspace / "continuation.json").read_text(encoding="utf-8"))


@needs_ffmpeg
async def test_a_customer_first_frame_replaces_the_source_final_frame(tmp_path: Path) -> None:
    """With a first frame, pass 0 is conditioned on the customer's picture —
    the source's own final frame is never extracted — and the graph still runs
    with ONE image (no last-frame group)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 3.0, audio=True, size="160x90")
    first = await _still(tmp_path / "first_frame.png", "green")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(workspace, [_video(source), _image("first_frame", first)]), on_progress
    )

    assert result.duration_seconds and abs(result.duration_seconds - 13.0) < 0.2
    assert [name for name, _ in fake.uploads] == ["zolex_job-ext-f_continue00.png"]
    assert not (workspace / "continuation-seed.png").exists()
    prompt = fake.submitted["prompt"]
    assert _loaders(prompt) == {"Load Image1": "zolex_job-ext-f_continue00.png"}
    [kj] = [e for e in prompt.values() if e["class_type"] == "LTXVImgToVideoInplaceKJ"]
    assert kj["inputs"]["num_images"] == "1"
    metadata = _metadata(workspace)
    assert metadata["first_frame"] == "first_frame.png"
    assert metadata["last_frame"] is None
    assert metadata["passes"][0]["conditioning_frame"] == "first_frame.png"
    # The overlap policy is unchanged: the rendered index-0 frame (the still
    # itself) is dropped, so the delivered length is exactly source + 10 s.
    assert metadata["overlap_frames_per_seam"] == SEAM_OVERLAP_FRAMES
    assert abs(metadata["measured_seconds"] - 13.0) < 0.2


@needs_ffmpeg
async def test_a_customer_last_frame_goes_to_the_final_pass_only(tmp_path: Path) -> None:
    """Two sections (10 s at a 5 s ceiling): the first pass runs first-frame
    only, the FINAL pass gets the customer's last frame as the graph's second
    image. Without a first frame the source's final frame still seeds pass 0."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True, size="160x90")
    last = await _still(tmp_path / "last_frame.png", "blue")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(workspace, [_video(source), _image("last_frame", last)], max_segment_seconds=5),
        on_progress,
    )

    assert result.duration_seconds and abs(result.duration_seconds - 12.0) < 0.2
    # The last frame is uploaded up front (a bad picture fails before any GPU
    # time), then one conditioning still per pass.
    assert [name for name, _ in fake.uploads] == [
        "zolex_job-ext-f_last.png",
        "zolex_job-ext-f_continue00.png",
        "zolex_job-ext-f_continue01.png",
    ]
    assert (workspace / "continuation-seed.png").exists()
    assert len(fake.submissions) == 2
    first_pass, final_pass = (s["prompt"] for s in fake.submissions)
    assert _loaders(first_pass) == {"Load Image1": "zolex_job-ext-f_continue00.png"}
    [kj] = [e for e in first_pass.values() if e["class_type"] == "LTXVImgToVideoInplaceKJ"]
    assert kj["inputs"]["num_images"] == "1"
    assert _loaders(final_pass) == {
        "Load Image1": "zolex_job-ext-f_continue01.png",
        "Load Image2": "zolex_job-ext-f_last.png",
    }
    [kj] = [e for e in final_pass.values() if e["class_type"] == "LTXVImgToVideoInplaceKJ"]
    assert kj["inputs"]["num_images.index_2"] == -1  # the graph's own two-image wiring
    metadata = _metadata(workspace)
    assert metadata["first_frame"] is None
    assert metadata["last_frame"] == "last_frame.png"
    assert metadata["passes"][0]["conditioning_frame"] == "continuation-seed.png"


@needs_ffmpeg
async def test_first_and_last_frame_together_bracket_a_single_pass(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True, size="160x90")
    first = await _still(tmp_path / "first_frame.png", "green")
    last = await _still(tmp_path / "last_frame.png", "blue")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _job(workspace, [_video(source), _image("first_frame", first), _image("last_frame", last)]),
        on_progress,
    )

    assert len(fake.submissions) == 1
    assert _loaders(fake.submitted["prompt"]) == {
        "Load Image1": "zolex_job-ext-f_continue00.png",
        "Load Image2": "zolex_job-ext-f_last.png",
    }
    metadata = _metadata(workspace)
    assert (metadata["first_frame"], metadata["last_frame"]) == ("first_frame.png", "last_frame.png")


@needs_ffmpeg
async def test_no_stills_means_the_run_before_this_feature(tmp_path: Path) -> None:
    """The guarantee the client asked for: with neither picture, the graph is
    handed exactly what it was handed before — the source's final frame as the
    only image — and the metadata says so."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True, size="160x90")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(_job(workspace, [_video(source)]), on_progress)

    assert [name for name, _ in fake.uploads] == ["zolex_job-ext-f_continue00.png"]
    assert _loaders(fake.submitted["prompt"]) == {"Load Image1": "zolex_job-ext-f_continue00.png"}
    metadata = _metadata(workspace)
    assert metadata["first_frame"] is None and metadata["last_frame"] is None
    assert metadata["passes"][0]["conditioning_frame"] == "continuation-seed.png"


@needs_ffmpeg
async def test_an_unreadable_last_frame_fails_before_any_render(tmp_path: Path) -> None:
    """A broken picture is refused up front, without a submission — the job
    fails cleanly, nothing is rendered, the source is untouched, and the
    customer can fix the picture and extend again."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True, size="160x90")
    broken = tmp_path / "last_frame.png"
    broken.write_bytes(b"not an image")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(workspace, [_video(source), _image("last_frame", broken)]), on_progress)
    assert raised.value.retriable is False
    assert "image" in raised.value.user_message.lower()
    assert fake.submitted is None and fake.uploads == []


@needs_ffmpeg
async def test_a_failed_pass_with_stills_leaves_no_output_and_is_retriable(tmp_path: Path) -> None:
    """A render that dies mid-pass surfaces as the same retriable failure as
    before; the workspace holds no assembled output for the runner to upload."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 2.0, audio=True, size="160x90")
    first = await _still(tmp_path / "first_frame.png", "green")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    fake.fail_history = True
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(workspace, [_video(source), _image("first_frame", first)]), on_progress)
    assert raised.value.retriable is True
    assert not (workspace / "output.mp4").exists()
