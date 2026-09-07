"""The skin hold inside the chain adapter (7 Sep 2026).

The hold runs on each rendered window before its last frame becomes the
next window's reference, so the delivered seam frame and the seed carry the
same skin; a held window is its own prepared part. A source within one
window never sees any of it, and a window the hold cannot help is joined
exactly as it always was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_character_replacement import _input, _job, _still
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _service
from worker.adapters.base import AdapterJob
from worker.adapters.character_replacement import CharacterReplacementAdapter
from worker.comfy.ltx_graphs import ReplacementEdits, compile_character_replacement
from worker.core.config import settings
from worker.media.ffmpeg import ffmpeg
from worker.media.probe import probe_media

FPS = 24
CANVAS = (184, 320)
SKIN = "0xe1b496"
SOURCE_SKIN = "0xc89678"
GREY = "0x5a5a5a"
FACE = (60, 40, 60, 80)
HANDS = (40, 180, 100, 90)
SILHOUETTE = [(50, 30, 84, 100), (30, 170, 120, 110)]


def _drawboxes(boxes: list[tuple[tuple[int, int, int, int], str]]) -> str:
    return ",".join(
        f"drawbox=x={x}:y={y}:w={w}:h={h}:color={colour}@1:t=fill" for (x, y, w, h), colour in boxes
    )


async def _skin_source(dest: Path, seconds: float) -> Path:
    """A source clip whose performer's skin silhouette is where the hands and
    face of the render are."""
    await ffmpeg(
        [
            "-f", "lavfi", "-i", f"color=c={GREY}:s=184x320:r={FPS}:d={seconds + 0.5:.3f}",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-vf", _drawboxes([(box, SOURCE_SKIN) for box in SILHOUETTE]),
            "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest), "-y",
        ]
    )
    return dest


async def _drifting_render(dest: Path, frames: int) -> Path:
    """What a window renders: the face holds, the hands darken across it."""
    fade = f"(1-0.4*min(T*{FPS}/{frames - 1},1))"
    chain = (
        f"{_drawboxes([(FACE, SKIN)])},"
        f"geq=r='if(between(X,{HANDS[0]},{HANDS[0] + HANDS[2]})*"
        f"between(Y,{HANDS[1]},{HANDS[1] + HANDS[3]}),225*{fade},r(X,Y))'"
        f":g='if(between(X,{HANDS[0]},{HANDS[0] + HANDS[2]})*"
        f"between(Y,{HANDS[1]},{HANDS[1] + HANDS[3]}),180*{fade},g(X,Y))'"
        f":b='if(between(X,{HANDS[0]},{HANDS[0] + HANDS[2]})*"
        f"between(Y,{HANDS[1]},{HANDS[1] + HANDS[3]}),150*{fade},b(X,Y))'"
    )
    await ffmpeg(
        [
            "-f", "lavfi", "-i", f"color=c={GREY}:s=184x320:r={FPS}:d={frames / FPS + 1:.3f}",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-vf", chain, "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(dest), "-y",
        ]
    )
    return dest


@pytest.fixture
def small_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two-second windows on a small canvas: the whole chain, quickly."""
    monkeypatch.setattr(settings, "character_replacement_canvas", CANVAS)
    monkeypatch.setattr(settings, "character_replacement_max_seconds", 2)
    monkeypatch.setattr(settings, "character_replacement_skin_hold", True)


@needs_ffmpeg
async def test_the_hold_runs_on_every_window_and_the_seed_comes_from_the_held_frames(
    tmp_path: Path, small_windows: None
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await _skin_source(tmp_path / "source.mp4", 4.2)
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _drifting_render(tmp_path / "unused.mp4", 49))
    fake.output_queue = [
        await _drifting_render(tmp_path / "render0.mp4", 49),
        await _drifting_render(tmp_path / "render1.mp4", 49),
    ]
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
            prompt="a man in a grey coat",
        ),
        on_progress,
    )

    metadata = json.loads((workspace / "character-replacement.json").read_text(encoding="utf-8"))
    assert metadata["skin_hold"] is True
    assert metadata["skin_target"] is not None
    assert metadata["skin_source_y"] is not None
    records = [window["skin_hold"] for window in metadata["windows"]]
    assert len(records) == 2
    assert all(record is not None for record in records)
    assert [record["frames"] for record in records] == [49, 49]
    assert records[0]["applied"] is True, records[0]
    # The seed for window 1 came from the held frames, so the seed pass's own
    # skin step stood down.
    assert metadata["windows"][1]["skin_correction"] == {
        "applied": False,
        "reason": "seed taken from the held frames",
    }
    # The delivered length is the source's own timeline, as without the hold.
    info = await probe_media(result.path)
    assert info.duration_seconds == pytest.approx(97 / FPS, abs=0.05)


@needs_ffmpeg
async def test_a_window_the_hold_cannot_help_is_joined_as_it_always_was(
    tmp_path: Path, small_windows: None
) -> None:
    """Renders with no skin in them: the hold refuses, records why, and the
    chain delivers exactly what it delivered before."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 4.2, audio=True, size="184x320")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(
        await make_clip(tmp_path / "render.mp4", 49 / FPS, audio=True, size="184x320")
    )
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    result = await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
            prompt="a man in a grey coat",
        ),
        on_progress,
    )

    metadata = json.loads((workspace / "character-replacement.json").read_text(encoding="utf-8"))
    records = [window["skin_hold"] for window in metadata["windows"]]
    assert all(record is not None and record["applied"] is False for record in records), records
    info = await probe_media(result.path)
    assert info.duration_seconds == pytest.approx(97 / FPS, abs=0.05)


@needs_ffmpeg
async def test_a_source_within_one_window_never_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "character_replacement_skin_hold", True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 8.6, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(
        await make_clip(tmp_path / "render.mp4", 193 / FPS, audio=True, size="144x256")
    )
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _job(
            workspace,
            [
                _input("source_video", source, "video"),
                _input("reference_image", reference, "image"),
            ],
            prompt="a man in a grey coat",
        ),
        on_progress,
    )

    assert not (workspace / "character-replacement.json").exists()
    assert not list(workspace.glob("hold-*"))


def _job_with(**execution: object) -> AdapterJob:
    return AdapterJob(
        job_id="j",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement", **execution},
    )


def test_the_hold_is_on_by_default_and_the_override_wins() -> None:
    assert settings.character_replacement_skin_hold is True
    assert CharacterReplacementAdapter.holds_skin(_job_with()) is True
    assert CharacterReplacementAdapter.holds_skin(_job_with(skin_hold="false")) is False
    assert CharacterReplacementAdapter.holds_skin(_job_with(skin_hold=True)) is True


def test_the_ripple_strength_is_the_graphs_own_unless_a_deployment_says_otherwise() -> None:
    assert settings.character_replacement_ripple_strength is None
    assert CharacterReplacementAdapter.ripple_strength(_job_with()) is None
    assert CharacterReplacementAdapter.ripple_strength(_job_with(ripple_strength="1.45")) == 1.45
    nonsense = _job_with(ripple_strength="nonsense")
    assert CharacterReplacementAdapter.ripple_strength(nonsense) is None
    assert CharacterReplacementAdapter.ripple_strength(_job_with(ripple_strength=0)) is None


def test_the_ripple_override_moves_one_number_and_nothing_else() -> None:
    from tests.test_ltx_graphs import _graph

    graph = _graph("character_replacement")
    base = dict(
        positive="a man",
        negative="nobody",
        video="v.mp4",
        image="i.png",
        seconds=8,
        width=1280,
        height=736,
        seed_base=None,
        filename_prefix="zolexai/job/output",
    )
    as_shipped = compile_character_replacement(graph, ReplacementEdits(**base))
    stronger = compile_character_replacement(
        graph, ReplacementEdits(**base, lora_strength=1.45)
    )
    [shipped_lora] = [e for e in as_shipped.values() if e["class_type"] == "LoraLoaderModelOnly"]
    [strong_lora] = [e for e in stronger.values() if e["class_type"] == "LoraLoaderModelOnly"]
    assert shipped_lora["inputs"]["strength_model"] == 1.35
    assert strong_lora["inputs"]["strength_model"] == 1.45
    assert strong_lora["inputs"]["lora_name"] == shipped_lora["inputs"]["lora_name"]
    for nid, entry in as_shipped.items():
        if entry["class_type"] == "LoraLoaderModelOnly":
            continue
        assert stronger[nid] == entry, nid
