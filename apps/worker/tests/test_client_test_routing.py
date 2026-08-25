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
