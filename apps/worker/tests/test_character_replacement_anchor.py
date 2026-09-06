"""Colour anchoring of chained seeds (6 Sep 2026: "he went darker through the video").

Each chained window is seeded from a RENDERED frame, and the graph renders
that look a little darker and flatter than it was given; unattended, the
loss compounds window after window. Before a window's last frame becomes
the next reference it is now matched — luminance level and spread, chroma
means, all bounded — to the first window's own rendering just after its
handoff. Measured here with real ffmpeg statistics on real files; what the
model does with the anchored seed is a GPU measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_character_replacement import _input, _job, _still
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _service
from worker.adapters.base import AdapterJob
from worker.adapters.character_replacement import (
    ANCHOR_FRAMES,
    ANCHOR_GAIN_RANGE,
    ANCHOR_LUMA_OFFSET_LIMIT,
    CharacterReplacementAdapter,
    ColourAnchor,
)
from worker.core.config import settings
from worker.media.ffmpeg import ffmpeg

FPS = 24


async def _render(path: Path, frames: int, *, dark_tail: bool = False) -> Path:
    """A window's render; with `dark_tail` its last second is darker and
    flatter — the drift the graph introduces, exaggerated to be measurable."""
    clip = await make_clip(path, frames / FPS, audio=True, size="144x256")
    if not dark_tail:
        return clip
    darkened = path.with_name(f"{path.stem}-dark.mp4")
    await ffmpeg(
        [
            "-i",
            str(clip),
            "-vf",
            "eq=brightness=-0.08:contrast=0.8:enable='gte(t,7)'",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(darkened),
            "-y",
        ]
    )
    return darkened


def _two_windows(workspace: Path, source: Path, reference: Path, **execution) -> AdapterJob:
    return AdapterJob(
        job_id="job-cr-1",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        inputs=[_input("source_video", source, "video"), _input("reference_image", reference, "image")],
        execution={"runtime": "character_replacement", **execution},
        output_content_type="video/mp4",
        workspace=workspace,
    )


def _metadata(workspace: Path) -> dict:
    return json.loads((workspace / "character-replacement.json").read_text(encoding="utf-8"))


@needs_ffmpeg
async def test_a_darkened_seed_is_brought_back_to_the_first_windows_look(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "unused.mp4", 241))
    fake.output_queue = [
        await _render(tmp_path / "r0.mp4", 241, dark_tail=True),
        await _render(tmp_path / "r1.mp4", 241),
    ]
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(_two_windows(workspace, source, reference), on_progress)

    metadata = _metadata(workspace)
    anchor = metadata["anchor"]
    assert anchor is not None and anchor["y_high"] > anchor["y_low"]
    correction = metadata["windows"][1]["seed_correction"]
    assert metadata["windows"][0]["seed_correction"] is None
    # The seed was darker and flatter than the anchor; the correction lifts
    # and stretches it, within the bounds.
    assert correction["seed_y_mean"] < correction["anchor_y_mean"] - 5
    assert correction["seed_y_high"] < correction["anchor_y_high"] - 5
    assert 1.0 < correction["gain"] <= ANCHOR_GAIN_RANGE[1]
    # The offset restores the level AFTER the gain, so its sign depends on
    # where the seed's mean sits; only its bound is a contract.
    assert abs(correction["y_offset"]) <= ANCHOR_LUMA_OFFSET_LIMIT
    # The anchored frame exists beside the raw one, and it is what window 1 got.
    assert (workspace / "reference01.png").exists()
    assert (workspace / "reference01-anchored.png").exists()
    assert [name for name, _ in fake.uploads][2] == "zolex_job-cr-1_reference01.png"
    # Measured: the uploaded reference now sits at the anchor's level.
    uploaded = await adapter._measure_colour(
        _two_windows(workspace, source, reference), workspace / "zolex_job-cr-1_reference01.png", None
    )
    assert abs(uploaded.y_mean - anchor["y_mean"]) < 4.0
    assert abs(uploaded.spread - (anchor["y_high"] - anchor["y_low"])) < 12.0


@needs_ffmpeg
async def test_a_seed_already_at_the_anchor_passes_through_untouched(tmp_path: Path) -> None:
    """The anchor IS the seed's own statistics: nothing to correct, so the
    frame goes back as it is and no anchored copy is written."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    frame = await _still(workspace / "reference01.png")
    adapter = CharacterReplacementAdapter(service=_service(FakeLtxComfy(workspace / "none.mp4")))
    job = _job(workspace, [])
    anchor = await adapter._measure_colour(job, frame, None)

    returned, correction = await adapter._anchor_seed(job, frame, anchor, 1)

    assert returned == frame
    assert correction["gain"] == 1.0 and correction["y_offset"] == 0.0
    assert not (workspace / "reference01-anchored.png").exists()


@needs_ffmpeg
async def test_anchoring_can_be_switched_off_and_photo_mode_never_needs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "unused.mp4", 241))
    fake.output_queue = [
        await _render(tmp_path / "r0.mp4", 241, dark_tail=True),
        await _render(tmp_path / "r1.mp4", 241),
    ]
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _two_windows(workspace, source, reference, anchor_reference=False), on_progress
    )
    metadata = _metadata(workspace)
    assert metadata["anchor"] is None
    assert metadata["windows"][1]["seed_correction"] is None
    assert not (workspace / "reference01-anchored.png").exists()

    # Photo mode: every window is seeded from the customer's picture, so
    # there is no drift to anchor and no anchor is measured.
    monkeypatch.setattr(settings, "character_replacement_chain_reference", "photo")
    workspace2 = tmp_path / "ws2"
    workspace2.mkdir()
    fake2 = FakeLtxComfy(await _render(tmp_path / "render2.mp4", 241))
    adapter2 = CharacterReplacementAdapter(service=_service(fake2))
    await adapter2.run(_two_windows(workspace2, source, reference), on_progress)
    metadata2 = _metadata(workspace2)
    assert metadata2["anchor"] is None
    assert all(w["seed_correction"] is None for w in metadata2["windows"])


def test_the_anchor_frames_skip_the_handoff() -> None:
    first, last = ANCHOR_FRAMES
    assert first == 4  # frames 0-3 are the photo and the graph's handoff from it
    assert last - first + 1 == FPS  # one second of the first window's own rendering
    anchor = ColourAnchor(y_mean=98.0, y_low=36.0, y_high=178.0, u_mean=118.8, v_mean=136.8)
    assert anchor.spread == 142.0
