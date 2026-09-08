"""Music Video on the client's music-video worker (v1.8.0, 8 Sep 2026).

The package is vendored verbatim and never edited; this adapter is the
platform's side of the seam. These pin three things: that the vendored
package is byte-identical to what was delivered, that a ZolexAI job becomes
the request and config their worker expects, and that the new runtime
touches nothing else — the CLI adapter still serves `music-video` as the
rollback, and no other adapter answers for it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.adapters.base import AdapterError, AdapterJob
from worker.adapters.ltx import LtxAdapter
from worker.adapters.music_video import SUPPORTED, WORKFLOW_ID, MusicVideoAdapter
from worker.adapters.registry import get_adapter
from worker.comfy.qwen_edit import MAX_REFERENCES, compile_anchor, missing_nodes
from worker.longform.progress import GENERATE_FROM, GENERATE_TO
from worker.musicvideo import (
    build_config,
    build_request,
    ensure_vendored_package,
    language_code,
    performers_from,
    progress_for,
)
from worker.workflows.resolver import resolve_adapter

REPO = Path(__file__).resolve().parents[3]
VENDORED = REPO / "apps/worker/zolex_music_worker"
DEFINITION = REPO / "workflow-definitions/music-video.yaml"

#: sha256 over the sorted per-file sha256 lines of the vendored package —
#: the delivered ZIP's `src/zolex_music_worker/*.py`
#: (ZIP sha256 5c512aecd3b28bb440f26ad853928fb38592666a4faf6d70d70d2d4f4dc69ead).
VENDORED_FINGERPRINT = "51b292d6d6fd6728f417e1104ea0a6a45c11c022b27806873143cb869c58d522"


def _fingerprint() -> str:
    lines = []
    for path in sorted(VENDORED.glob("*.py")):
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        lines.append(f"{digest}  {path.name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _job(workspace: Path, *, inputs=(), **params) -> AdapterJob:
    return AdapterJob(
        job_id="mv-job",
        workflow_id=WORKFLOW_ID,
        workflow_version="1",
        prompt="make me a cinematic video according to the lyrics of the song",
        parameters={"aspect_ratio": "16:9", **params},
        inputs=list(inputs),
        execution={"runtime": "music_video"},
        workspace=workspace,
    )


def _picture(workspace: Path, role: str) -> SimpleNamespace:
    path = workspace / f"{role}.png"
    path.write_bytes(b"png")
    return SimpleNamespace(role=role, kind="image", path=path)


def _settings(**overrides):
    base = dict(
        music_video_render_backend="command",
        music_video_render_command="",
        music_video_anchor_backend="command",
        music_video_ltx_extra_args="",
        music_video_inference_steps=24,
        music_video_shot_target_seconds=4.5,
        music_video_shot_min_seconds=2.0,
        music_video_shot_max_seconds=5.0,
        music_video_max_source_seconds=300,
        music_video_max_attempts=3,
        music_video_command_timeout=7200,
        music_video_transcription_backend="faster_whisper",
        music_video_whisper_model="large-v3",
        music_video_whisper_device="cuda",
        music_video_whisper_compute_type="float16",
        music_video_whisper_download_root=None,
        music_video_upscale_backend="cuda",
        music_video_upscale_cq=18,
        music_video_upscale_timeout=1800,
        ltx_unquantized_offload="cpu",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── The vendored package ────────────────────────────────────────────────────


def test_the_vendored_package_is_byte_identical_to_the_delivery() -> None:
    """The client's code is a delivery, like their graphs: run, never edited.
    Changing it is a decision that needs their word — and a new pin."""
    assert _fingerprint() == VENDORED_FINGERPRINT


def test_the_vendored_package_imports_without_its_heavy_dependencies() -> None:
    """Request and config construction need only the standard library, so
    a job can be validated — and refused with a message — before numpy,
    Pillow or faster-whisper are ever imported."""
    ensure_vendored_package()
    from zolex_music_worker import __version__
    from zolex_music_worker.config import WorkerConfig
    from zolex_music_worker.models import MusicVideoRequest

    assert __version__ == "1.8.0"
    assert WorkerConfig and MusicVideoRequest


# ── Isolation ───────────────────────────────────────────────────────────────


def test_the_runtime_answers_for_music_video_only() -> None:
    adapter = get_adapter("music_video")
    assert adapter.name == "music_video"
    assert SUPPORTED == {WORKFLOW_ID}
    for other in ("text-to-video", "image-to-video", "extend-video",
                  "video-to-video", "character-replacement", "music", "text-to-video-hd"):
        assert not adapter.supports(other), other


def test_the_cli_adapter_still_serves_music_video_as_the_rollback() -> None:
    assert LtxAdapter().supports(WORKFLOW_ID)
    for runtime in ("ltx_comfy", "character_replacement", "ltx_hd"):
        assert not get_adapter(runtime).supports(WORKFLOW_ID), runtime


def test_the_resolver_routes_by_the_execution_block(tmp_path: Path) -> None:
    routed = _job(tmp_path)
    assert resolve_adapter(routed).name == "music_video"
    rollback = AdapterJob(
        job_id="j", workflow_id=WORKFLOW_ID, workflow_version="1", prompt="x",
        parameters={}, execution={"runtime": "ltx"},
    )
    assert resolve_adapter(rollback).name == "ltx"


# ── Job → request ───────────────────────────────────────────────────────────


def test_performers_pair_pictures_with_their_slots(tmp_path: Path) -> None:
    job = _job(
        tmp_path,
        inputs=[_picture(tmp_path, "performer_2"), _picture(tmp_path, "performer_1")],
        performers=[
            {"slot": 2, "role": "drummer", "description": "red bandana"},
            {"slot": 4, "role": "bassist"},
        ],
    )
    performers = performers_from(job)
    assert [p["id"] for p in performers] == ["performer-1", "performer-2", "performer-4"]
    assert performers[0]["role"] == "performer" and performers[0]["reference_images"]
    assert performers[1]["role"] == "drummer"
    assert performers[1]["description"] == "red bandana"
    assert performers[1]["reference_images"] == [str(tmp_path / "performer_2.png")]
    # Described without a picture: the package generates the identity.
    assert performers[2]["reference_images"] == []


def test_the_request_carries_aspect_lyrics_and_language(tmp_path: Path) -> None:
    ensure_vendored_package()
    from zolex_music_worker.models import MusicVideoRequest

    job = _job(
        tmp_path,
        aspect_ratio="9:16",
        lyrics="[00:01.00] first line\n[00:04.00] second line",
        lyrics_language="Spanish",
    )
    payload = build_request(job, tmp_path / "song.wav", lyric_mode="automatic")
    assert payload["formats"] == ["9:16"]
    assert payload["lyrics"].startswith("[00:01.00]")
    assert payload["lyrics_language"] == "es"
    assert payload["lyric_mode"] == "automatic"
    request = MusicVideoRequest.from_dict(payload)
    assert request.job_id == "mv-job"
    assert request.max_people_visible_per_shot == 1


def test_language_names_become_codes_and_codes_pass_through() -> None:
    assert language_code("English") == "en"
    assert language_code("URDU") == "ur"
    assert language_code("pt") == "pt"
    assert language_code("") is None


def test_the_execution_block_picks_the_lyric_mode(tmp_path: Path) -> None:
    job = AdapterJob(
        job_id="j", workflow_id=WORKFLOW_ID, workflow_version="1", prompt="a love video",
        parameters={"aspect_ratio": "1:1"},
        execution={"runtime": "music_video", "lyric_mode": "always"},
        workspace=tmp_path,
    )
    assert build_request(job, tmp_path / "s.wav", lyric_mode="automatic")["lyric_mode"] == "always"


# ── Settings → config ───────────────────────────────────────────────────────


def test_the_config_points_every_backend_at_this_node(tmp_path: Path) -> None:
    job = _job(tmp_path)
    config = build_config(
        job,
        work_root=tmp_path / "mv",
        settings=_settings(),
        anchor_command=["py", "anchor", "{request_json}", "{output}"],
        render_command=["py", "render", "{request_json}", "{output}"],
        ltx_python="/ltx/.venv/bin/python",
        ltx_models_root=Path("/models/ltx-2.5"),
    )
    config.validate()
    assert config.render_backend == "command"
    assert config.render_command == ["py", "render", "{request_json}", "{output}"]
    assert config.anchor_backend == "command"
    assert config.anchor_command == ["py", "anchor", "{request_json}", "{output}"]
    assert config.upscale_backend == "cuda"
    assert config.transcription_backend == "faster_whisper"
    assert config.reference_vision_backend == "disabled"
    assert config.ltx_num_inference_steps == 24
    assert config.shot_max_seconds == 5.0
    assert config.ltx_extra_args == ["--offload", "cpu"]
    assert config.ltx_distilled_lora_path == Path(
        "/models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    )
    assert config.ltx_transformer_path.name == "ltx-2.5-22b-dev-transformer-bf16.safetensors"


def test_the_execution_block_overrides_steps_and_the_shot_ladder(tmp_path: Path) -> None:
    job = AdapterJob(
        job_id="j", workflow_id=WORKFLOW_ID, workflow_version="1", prompt="x",
        parameters={}, workspace=tmp_path,
        execution={
            "runtime": "music_video", "inference_steps": 15,
            "shot_target_seconds": 3.5, "shot_max_seconds": 4.0,
            "render_backend": "mock", "anchor_backend": "reference",
        },
    )
    config = build_config(
        job, work_root=tmp_path, settings=_settings(ltx_unquantized_offload="none"),
        anchor_command=["a"], render_command=["r"],
        ltx_python="python", ltx_models_root=Path("/m"),
    )
    config.validate()
    assert config.ltx_num_inference_steps == 15
    assert config.shot_target_seconds == 3.5 and config.shot_max_seconds == 4.0
    assert config.render_backend == "mock" and config.render_command is None
    assert config.anchor_backend == "reference" and config.anchor_command is None
    assert config.ltx_extra_args is None


def test_a_deployment_render_command_wins_over_the_shipped_script(tmp_path: Path) -> None:
    config = build_config(
        _job(tmp_path), work_root=tmp_path,
        settings=_settings(
            music_video_render_command=json.dumps(["/srv/warm", "{request_json}", "{output}"])
        ),
        anchor_command=["a"], render_command=["shipped"],
        ltx_python="python", ltx_models_root=Path("/m"),
    )
    assert config.render_command == ["/srv/warm", "{request_json}", "{output}"]


# ── Progress ────────────────────────────────────────────────────────────────


def test_progress_moves_forward_through_the_packages_states() -> None:
    sequence = [
        {"state": "validating"},
        {"state": "preparing_audio"},
        {"state": "analyzing_song_lyrics_and_reference"},
        {"state": "planning"},
        {"state": "building_identity_package"},
        {"state": "generating_anchors"},
        {"state": "rendering", "shot_index": 1, "shot_count": 10},
        {"state": "rendering", "shot_index": 6, "shot_count": 10},
        {"state": "rendering", "completed_shots": 10, "shot_count": 10},
        {"state": "assembling"},
        {"state": "upscaling"},
        {"state": "final_quality_control"},
    ]
    readings = [progress_for(item) for item in sequence]
    assert all(reading is not None for reading in readings)
    statuses = [reading[0] for reading in readings]
    progress = [reading[1] for reading in readings]
    assert statuses[:4] == ["preparing"] * 4
    assert statuses[4:9] == ["generating"] * 5
    assert statuses[9:] == ["post_processing"] * 3
    assert progress == sorted(progress)
    assert GENERATE_FROM <= progress[4] and progress[8] < GENERATE_TO
    assert "shot 6 of 10" in readings[7][2]
    assert readings[7][3]["section_index"] == 6 and readings[7][3]["section_total"] == 10
    assert progress_for({"state": "complete"}) is None
    assert progress_for({"state": "failed"}) is None


# ── The adapter's refusals ──────────────────────────────────────────────────


async def test_a_job_without_audio_is_refused_before_any_work(tmp_path: Path) -> None:
    async def on_progress(*_args, **_kwargs) -> None:
        pass

    with pytest.raises(AdapterError) as raised:
        await MusicVideoAdapter().run(_job(tmp_path), on_progress)
    assert raised.value.retriable is False
    assert "audio" in raised.value.user_message.lower()


def test_customer_messages_for_the_packages_refusals() -> None:
    message = MusicVideoAdapter._customer_message
    assert "lyrics" in message(ValueError("No usable lyrics were found; supply lyrics")).lower()
    assert "5 minutes" in message(ValueError("Decoded audio is 400.0s; limit is 300s"))
    assert "too short" in message(ValueError("Audio must be at least one second"))


# ── The anchor graph ────────────────────────────────────────────────────────


def test_the_anchor_graph_is_text_to_image_without_references() -> None:
    api = compile_anchor(prompt="a portrait", width=768, height=768, seed=1,
                         filename_prefix="zolexai/mv-anchors/x")
    kinds = {node["class_type"] for node in api.values()}
    assert "LoadImage" not in kinds
    assert "LoraLoaderModelOnly" in kinds
    encoder = next(n for n in api.values() if n["class_type"] == "TextEncodeQwenImageEditPlus")
    assert "image1" not in encoder["inputs"]
    sampler = next(n for n in api.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["steps"] == 4 and sampler["inputs"]["cfg"] == 1.0


def test_the_anchor_graph_feeds_each_reference_to_both_conditionings() -> None:
    api = compile_anchor(prompt="p", width=1280, height=704, seed=2, filename_prefix="x",
                         references=["a.png", "b.png"], lightning=False, steps=20, cfg=2.5)
    loaders = [n for n in api.values() if n["class_type"] == "LoadImage"]
    assert [n["inputs"]["image"] for n in loaders] == ["a.png", "b.png"]
    encoders = [n for n in api.values() if n["class_type"] == "TextEncodeQwenImageEditPlus"]
    assert len(encoders) == 2
    for encoder in encoders:
        assert set(encoder["inputs"]) >= {"image1", "image2"}
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in api.values())
    with pytest.raises(ValueError):
        compile_anchor(prompt="p", width=1280, height=704, seed=2, filename_prefix="x",
                       references=["1", "2", "3", "4"])
    assert MAX_REFERENCES == 3
    with pytest.raises(ValueError):
        compile_anchor(prompt="p", width=1281, height=704, seed=2, filename_prefix="x")


def test_missing_nodes_names_what_the_server_lacks() -> None:
    assert "TextEncodeQwenImageEditPlus" in missing_nodes({"KSampler": {}})
    api = compile_anchor(prompt="p", width=768, height=768, seed=1, filename_prefix="x",
                         references=["a"])
    offered = {node["class_type"] for node in api.values()}
    assert missing_nodes(offered) == []


# ── The definition ──────────────────────────────────────────────────────────


def test_the_definition_offers_five_performer_pictures() -> None:
    text = DEFINITION.read_text(encoding="utf-8")
    for slot in range(1, 6):
        assert f"role: performer_{slot}" in text
    assert "performers: true" in text
    assert "lyrics: true" in text
