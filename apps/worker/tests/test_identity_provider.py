"""Selecting a different engine for reference-identity video-to-video.

`execution.v2v_identity_provider` chooses WHO performs the replacement. The
default is the LTX path this adapter has always used; `wan_animate` hands the
whole render to Wan2.2-Animate in replacement mode.

What these pin is the seam rather than the model: that the default path is
untouched, that a job asking for an engine the node does not have is REFUSED
rather than quietly served by the other one, and that whichever engine draws
the pictures, the source's audio is still attached exactly once and the result
is still the source's length.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
)
from tests.test_reference_identity import stub_matte
from worker.adapters.base import AdapterError
from worker.media import extract_final_frame, probe_media


def provider_job(workspace: Path, source: Path, reference: Path | None, **execution):
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    if reference is not None:
        inputs.append(staged_input("reference_image", "image", "image/png", reference))
    return make_job(
        workspace,
        workflow_id="video-to-video",
        prompt="the person from the reference, same performance",
        parameters={"aspect_ratio": "16:9"},
        inputs=inputs,
        execution={
            "runtime": "ltx",
            "v2v_engine": "transform",
            "v2v_reference_identity": True,
            "v2v_identity_describe_reference": False,
            "v2v_identity_composited_anchor": False,
            **execution,
        },
    )


def stub_provider(monkeypatch: pytest.MonkeyPatch, clip: Path) -> list[dict]:
    """Stands in for the 68 GB model; records what it was asked to do."""
    calls: list[dict] = []

    async def fake(source, reference, dest, *, width, height, prompt="", timeout=0.0):
        calls.append(
            {"source": source, "reference": reference, "width": width,
             "height": height, "prompt": prompt, "timeout": timeout}
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(clip.read_bytes())
        return dest

    monkeypatch.setattr("worker.adapters.ltx.replace_person", fake)
    return calls


# ── the default path is untouched ─────────────────────────────────────────


@needs_ffmpeg
async def test_no_provider_key_never_reaches_the_second_engine(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every job shipping today omits this key, and must be unaffected."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    calls = stub_provider(monkeypatch, source)
    stub_matte(monkeypatch)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    await collect(provider_job(workspace, source, reference))

    assert calls == [], "the LTX path renders it, as it always has"


# ── an engine the node does not have is refused, not substituted ──────────


@needs_ffmpeg
async def test_an_unconfigured_provider_is_refused(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently serving the LTX render instead would be a lie about which
    engine produced the customer's video."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    with pytest.raises(AdapterError) as raised:
        await collect(provider_job(
            workspace, source, reference, v2v_identity_provider="wan_animate"
        ))
    assert "WAN_ANIMATE_COMMAND" in str(raised.value.internal_detail)
    assert raised.value.retriable is False


@needs_ffmpeg
async def test_an_unknown_provider_is_refused(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    with pytest.raises(AdapterError) as raised:
        await collect(provider_job(
            workspace, source, reference, v2v_identity_provider="something_else"
        ))
    assert "unknown v2v_identity_provider" in str(raised.value.internal_detail)


# ── when it does run, the delivery promises still hold ────────────────────


@needs_ffmpeg
async def test_the_provider_renders_and_keeps_the_delivery_promises(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whichever engine draws the pictures, the source's own audio goes on
    once and the result is the source's length — both live in
    `_deliver_restyle`, which this path shares."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    reference = await extract_final_frame(source, workspace / "reference.png")
    generated = await make_clip(tmp_path / "wan.mp4", 2.0)
    calls = stub_provider(monkeypatch, generated)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    await collect(provider_job(
        workspace, source, reference, v2v_identity_provider="wan_animate"
    ))

    assert len(calls) == 1, "one call for the whole video, not one per pass"
    assert calls[0]["source"].name == source.name
    assert calls[0]["reference"].name == reference.name

    info = await probe_media(workspace / "output.mp4")
    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)
    assert info.has_audio is True
    assert info.audio_stream_count == 1, "the source's track, exactly once"


@needs_ffmpeg
async def test_the_provider_is_inert_without_a_reference(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No reference, nothing to replace anyone with — an ordinary transform,
    not a refusal and not a second engine."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    calls = stub_provider(monkeypatch, source)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    await collect(provider_job(
        workspace, source, None, v2v_identity_provider="wan_animate"
    ))

    assert calls == []
