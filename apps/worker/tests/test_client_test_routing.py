"""Client-test routing invariants.

The client-test environment routes by flipping `execution.runtime` lines in
workflow YAML — the M2 mechanism — so what these tests pin is the machinery
that makes those flips safe: the new runtime is registered, the music-video
guard refuses the unconditioned route when the client-test key demands
conditioning, and the H3 provider's runtime modes stay internal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_job, make_track, needs_ffmpeg, staged_input
from worker.adapters.base import AdapterError
from worker.adapters.ltx import LtxAdapter
from worker.adapters.registry import available_runtimes, get_adapter
from worker.providers.h3 import H3Provider


def test_h3_comfy_runtime_is_registered_but_nothing_ships_routed_to_it() -> None:
    assert "h3_comfy" in available_runtimes()
    assert get_adapter("h3_comfy").name == "h3_comfy"
    # No shipped workflow definition points at it — routing stays a
    # client-test YAML edit, never a default.
    definitions = Path(__file__).resolve().parents[3] / "workflow-definitions"
    for yaml_path in definitions.glob("*.yaml"):
        assert "h3_comfy" not in yaml_path.read_text(encoding="utf-8"), (
            f"{yaml_path.name} routes to h3_comfy in the shipped definitions"
        )


def test_unknown_runtime_still_fails_hard() -> None:
    with pytest.raises(AdapterError):
        get_adapter("h3_turbo")


@needs_ffmpeg
async def test_music_video_guard_refuses_the_unconditioned_route(
    tmp_path: Path, fake_models: Path, stub_repo: Path
) -> None:
    """`require_audio_conditioning` without `audio_conditioning` must refuse.

    The unconditioned default generates from the prompt and muxes the track
    afterwards — measured 24 Aug: mouth motion uncorrelated with the vocal.
    A client-test build promising lip response must be unable to run it.
    """
    track = await make_track(tmp_path / "song.mp3", 4.0)
    job = make_job(
        tmp_path,
        workflow_id="music-video",
        prompt="a singer at a microphone",
        parameters={"aspect_ratio": "16:9"},
        inputs=[staged_input("source_audio", "audio", "audio/mpeg", track)],
        execution={"runtime": "ltx", "require_audio_conditioning": True},
    )

    async def progress(status, progress_value, message, details=None) -> None:
        pass

    with pytest.raises(AdapterError) as raised:
        await LtxAdapter().run(job, progress)
    assert "post-mux" in raised.value.internal_detail
    assert raised.value.retriable is False


def test_h3_provider_runtimes_are_internal_and_bounded() -> None:
    # Externally always `h3`; the runtime never leaks into the name.
    assert H3Provider().name == "h3"
    assert H3Provider("comfyui_int8").name == "h3"
    assert H3Provider("local_diffusers").name == "h3"
    with pytest.raises(ValueError):
        H3Provider("turbo")  # rejected on quality; not even a valid mode


def test_default_h3_provider_still_refuses() -> None:
    ok, reason = H3Provider().health()
    assert not ok
    assert "Licence" in reason or "licence" in reason


def test_quality_toggle_routes_between_engines(tmp_path: Path) -> None:
    """The client-approved Fast/Best toggle: `runtime_by_quality` maps the
    public quality parameter to an engine; anything unmapped or absent falls
    back to plain `runtime` — the toggle can never strand a request."""
    from worker.adapters.h3_comfy import H3ComfyAdapter
    from worker.workflows.resolver import resolve_adapter

    execution = {
        "runtime": "ltx",
        "runtime_by_quality": {"fast": "ltx", "best": "h3_comfy"},
    }
    fast = make_job(tmp_path, execution=execution, parameters={"quality": "fast", "duration": "5s"})
    best = make_job(tmp_path, execution=execution, parameters={"quality": "best", "duration": "5s"})
    unset = make_job(tmp_path, execution=execution, parameters={"duration": "5s"})
    weird = make_job(
        tmp_path, execution=execution, parameters={"quality": "ultra", "duration": "5s"}
    )

    assert isinstance(resolve_adapter(fast), LtxAdapter)
    assert isinstance(resolve_adapter(best), H3ComfyAdapter)
    assert isinstance(resolve_adapter(unset), LtxAdapter)
    assert isinstance(resolve_adapter(weird), LtxAdapter)


def test_video_to_video_never_routes_to_the_stills_engine(tmp_path: Path) -> None:
    """Best on Video to Video must return the customer's own footage.

    H3's R2V graph reads a reference photo and the source's FIRST FRAME and
    generates from there, which is what the client saw on 28 Aug 2026 —
    "I put a video and press better and give me a whole different video". The
    engine declines the workflow (`h3_comfy_video_to_video`, default off) and
    the resolver serves the job on the base runtime rather than failing it.
    """
    from worker.adapters.h3_comfy import H3ComfyAdapter
    from worker.workflows.resolver import resolve_adapter

    assert not H3ComfyAdapter().supports("video-to-video")

    stale = {
        "runtime": "ltx",
        "runtime_by_quality": {"fast": "ltx", "best": "h3_comfy"},
    }
    best = make_job(
        tmp_path,
        workflow_id="video-to-video",
        execution=stale,
        parameters={"quality": "best"},
    )
    assert isinstance(resolve_adapter(best), LtxAdapter)


def test_execution_by_quality_overlays_the_execution_block() -> None:
    """Fast and Best differ by what the engine is ASKED to do.

    Video to Video runs one engine at both levels; Best adds the matting pass
    that replaces the person. Keys in the overlay replace keys in the block,
    everything else is inherited, and an unnamed quality reads the block
    exactly as written.
    """
    from worker.workflows.resolver import _execution_for

    claim = {
        "execution": {
            "runtime": "ltx",
            "v2v_engine": "transform",
            "v2v_reference_identity": False,
            "execution_by_quality": {"best": {"v2v_reference_identity": True}},
        }
    }

    best = _execution_for({**claim, "parameters": {"quality": "Best"}})
    fast = _execution_for({**claim, "parameters": {"quality": "fast"}})
    unset = _execution_for(claim)

    assert best["v2v_reference_identity"] is True
    assert best["v2v_engine"] == "transform"  # inherited, not dropped
    assert fast["v2v_reference_identity"] is False
    assert unset["v2v_reference_identity"] is False


@needs_ffmpeg
async def test_the_stills_engine_refuses_video_to_video_even_as_the_base_runtime(
    tmp_path: Path,
) -> None:
    """`supports()` alone could not have fixed this, and production proved it.

    The resolver's fallback compares a quality level against the BASE runtime,
    so it never fires for a workflow whose base IS the withdrawn engine — and
    on 28 Aug 2026 production carried exactly that: `runtime: h3_comfy` for
    video-to-video, under a `runtime_by_quality` map. Withdrawing the workflow
    would have changed nothing for Best. A misconfiguration has to fail
    loudly, not quietly ship the product the engine was withdrawn from.
    """
    from worker.adapters.h3_comfy import H3ComfyAdapter

    job = make_job(
        tmp_path,
        workflow_id="video-to-video",
        execution={"runtime": "h3_comfy"},
        parameters={"quality": "best"},
    )

    async def progress(status, progress_value, message, details=None) -> None:
        pass

    with pytest.raises(AdapterError) as raised:
        await H3ComfyAdapter().run(job, progress)
    assert "does not serve 'video-to-video'" in raised.value.internal_detail
    assert "runtime_by_quality" in raised.value.internal_detail
    assert raised.value.retriable is False
    # The customer is never told which engine anything runs on.
    assert "h3" not in raised.value.user_message.lower()
