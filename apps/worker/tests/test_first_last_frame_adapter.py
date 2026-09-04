"""First/Last Frame Video on the client's graph — the adapter flow (Phase 2).

Same fake service as `test_ltx_comfy.py`; what is new here is the still
handling: the first frame is required and uploaded PNG-normalised, the last
frame is optional and, when absent, the graph runs with its last-frame
conditioning bypassed the way a ComfyUI user would bypass it.

STATUS for the model: WAITING FOR GPU VALIDATION.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import needs_ffmpeg
from tests.test_ltx_comfy import FakeLtxComfy, _job, _recorder, _rendered, _service
from worker.adapters.base import AdapterError, AdapterInput
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.comfy.ltx_prompts import FIRST_LAST_FRAME_NEGATIVE
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


def test_the_runtime_serves_first_last_frame_video() -> None:
    adapter = LtxComfyAdapter()
    assert adapter.supports("image-to-video")
    assert adapter.supports("text-to-video")
    assert not adapter.supports("video-to-video")


@needs_ffmpeg
async def test_first_frame_only_runs_the_graph_with_the_last_frame_bypassed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    first = await _still(tmp_path / "first.png")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 10.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(workspace, "image-to-video", [_image("source_image", first)], duration="10s"),
        on_progress,
    )

    assert result.kind == "video"
    assert [name for name, _ in fake.uploads] == ["zolex_job-ltx-1_first.png"]
    prompt = fake.submitted["prompt"]
    types = {e["class_type"] for e in prompt.values()}
    assert "LTXVImgToVideoInplace" in types  # the first-frame conditioning
    assert "LTXVImgToVideoInplaceKJ" not in types  # the last-frame node, bypassed
    loaders = [e["inputs"]["image"] for e in prompt.values() if e["class_type"] == "LoadImage"]
    assert loaders == ["zolex_job-ltx-1_first.png"]
    texts = {
        e["_meta"]["title"]: e["inputs"]["text"]
        for e in prompt.values()
        if e["class_type"] == "CLIPTextEncode"
    }
    assert texts["CLIP Text Encode (Prompt) negative"] == FIRST_LAST_FRAME_NEGATIVE
    assert texts["CLIP Text Encode (Prompt) positive"].startswith(
        "A koi pond at dawn, mist over the water."
    )
    [slider] = [e for e in prompt.values() if e["class_type"] == "mxSlider"]
    assert slider["inputs"]["Xi"] == 10


@needs_ffmpeg
async def test_first_and_last_frames_run_the_graph_unmodified(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    first = await _still(tmp_path / "first.png", "red")
    last = await _still(tmp_path / "last.png", "blue")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _job(
            workspace,
            "image-to-video",
            [_image("source_image", first), _image("last_frame", last)],
            aspect_ratio="1:1",
        ),
        on_progress,
    )

    assert [name for name, _ in fake.uploads] == [
        "zolex_job-ltx-1_first.png",
        "zolex_job-ltx-1_last.png",
    ]
    prompt = fake.submitted["prompt"]
    loaders = {
        e["_meta"]["title"]: e["inputs"]["image"]
        for e in prompt.values()
        if e["class_type"] == "LoadImage"
    }
    assert loaders == {
        "Load Image1": "zolex_job-ltx-1_first.png",
        "Load Image2": "zolex_job-ltx-1_last.png",
    }
    [kj] = [e for e in prompt.values() if e["class_type"] == "LTXVImgToVideoInplaceKJ"]
    assert kj["inputs"]["num_images.index_2"] == -1
    [selector] = [e for e in prompt.values() if e["class_type"] == "ResolutionSelector"]
    assert selector["inputs"]["aspect_ratio"] == "1:1 (Square)"


async def test_a_missing_first_frame_is_refused_before_anything_is_uploaded(tmp_path: Path) -> None:
    fake = FakeLtxComfy(tmp_path / "none.mp4")
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(_job(tmp_path, "image-to-video", []), on_progress)
    assert raised.value.retriable is False
    assert "first frame" in raised.value.user_message.lower()
    assert fake.uploads == [] and fake.submitted is None


@needs_ffmpeg
async def test_an_unreadable_still_is_refused_as_the_customers_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    fake = FakeLtxComfy(tmp_path / "none.mp4")
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    with pytest.raises(AdapterError) as raised:
        await adapter.run(
            _job(tmp_path, "image-to-video", [_image("source_image", bad)]), on_progress
        )
    assert raised.value.retriable is False
    assert "image" in raised.value.user_message.lower()


@needs_ffmpeg
async def test_text_to_video_ignores_stray_stills(tmp_path: Path) -> None:
    """A job cannot smuggle an image into Text to Video: the graph choice is
    the workflow's, never the inputs'."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    first = await _still(tmp_path / "first.png")
    fake = FakeLtxComfy(await _rendered(tmp_path / "render.mp4", 5.0))
    adapter = LtxComfyAdapter(service=_service(fake))
    on_progress, _ = _recorder()
    await adapter.run(
        _job(workspace, "text-to-video", [_image("source_image", first)]), on_progress
    )
    assert fake.uploads == []
    assert not any(e["class_type"] == "LoadImage" for e in fake.submitted["prompt"].values())
