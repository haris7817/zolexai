"""The hands clause for chained character replacement (7 Sep 2026).

Measured on the client's clip: the face holds the photo's skin tone but the
hands darken inside every window; nothing in the text ever spoke about them.
The clause is the one signal that speaks DURING a window. It is added only to
chained jobs, between the pack's lead sentence and the customer's own words,
and a source within one window is byte for byte what it was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import make_clip, needs_ffmpeg
from tests.test_character_replacement import _input, _job, _still
from tests.test_character_replacement_chain import _render, _submission_facts
from tests.test_ltx_comfy import FakeLtxComfy, _recorder, _service
from worker.adapters.base import AdapterJob
from worker.adapters.character_replacement import CharacterReplacementAdapter
from worker.comfy.ltx_prompts import (
    CHARACTER_REPLACEMENT_LEAD,
    CHARACTER_REPLACEMENT_SKIN,
    character_replacement_prompt,
)
from worker.core.config import settings


def _positive(prompt: dict) -> str:
    [conditioning] = [e for e in prompt.values() if e["class_type"] == "LTXVConditioning"]
    return prompt[conditioning["inputs"]["positive"][0]]["inputs"]["text"]


def test_the_prompt_helper_is_byte_identical_without_the_clause() -> None:
    assert character_replacement_prompt("") == CHARACTER_REPLACEMENT_LEAD
    assert character_replacement_prompt("  a man in a red coat ") == (
        f"{CHARACTER_REPLACEMENT_LEAD} a man in a red coat"
    )
    with_clause = character_replacement_prompt("a man in a red coat", skin=CHARACTER_REPLACEMENT_SKIN)
    assert with_clause == f"{CHARACTER_REPLACEMENT_LEAD} {CHARACTER_REPLACEMENT_SKIN} a man in a red coat"
    assert character_replacement_prompt("", skin=CHARACTER_REPLACEMENT_SKIN) == (
        f"{CHARACTER_REPLACEMENT_LEAD} {CHARACTER_REPLACEMENT_SKIN}"
    )
    # Relational wording only: it names no colour, so it can pull no character
    # the wrong way; and it never asks for hands the source does not show.
    lowered = CHARACTER_REPLACEMENT_SKIN.lower()
    for colour in ("light", "dark", "brown", "white", "black", "pale", "tan"):
        assert f" {colour} " not in f" {lowered} "
    assert "whenever they are in view" in lowered
    assert "same skin tone as the face" in lowered


def test_the_clause_is_on_by_default_and_the_override_wins() -> None:
    assert settings.character_replacement_chain_skin_clause is True
    assert CharacterReplacementAdapter.chain_skin_clause(_job(Path("."), [])) is True
    on = AdapterJob(
        job_id="j",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement", "chain_skin_clause": True},
    )
    assert CharacterReplacementAdapter.chain_skin_clause(on) is True
    off = AdapterJob(
        job_id="j",
        workflow_id="character-replacement",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "character_replacement", "chain_skin_clause": "false"},
    )
    assert CharacterReplacementAdapter.chain_skin_clause(off) is False


@needs_ffmpeg
async def test_a_chained_job_puts_the_clause_in_every_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "character_replacement_chain_skin_clause", True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 20.0, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", 241))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
            prompt="a woman with silver hair in a red coat",
        ),
        on_progress,
    )

    assert len(fake.submissions) == 2
    for submission in fake.submissions:
        text = _positive(submission["prompt"])
        assert text == (
            f"{CHARACTER_REPLACEMENT_LEAD} {CHARACTER_REPLACEMENT_SKIN} "
            "a woman with silver hair in a red coat"
        )
    metadata = json.loads((workspace / "character-replacement.json").read_text(encoding="utf-8"))
    assert metadata["skin_clause"] is True
    # Nothing else about the windows moved.
    assert [_submission_facts(s["prompt"])["length"] for s in fake.submissions] == [10, 10]


@needs_ffmpeg
async def test_a_source_within_one_window_never_gets_the_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with the switch on, the single-window path is what it always was."""
    monkeypatch.setattr(settings, "character_replacement_chain_skin_clause", True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = await make_clip(tmp_path / "source.mp4", 8.6, audio=True, size="144x256")
    reference = await _still(tmp_path / "reference.png")
    fake = FakeLtxComfy(await _render(tmp_path / "render.mp4", 193))
    adapter = CharacterReplacementAdapter(service=_service(fake))
    on_progress, _ = _recorder()

    await adapter.run(
        _job(
            workspace,
            [_input("source_video", source, "video"), _input("reference_image", reference, "image")],
            prompt="a man with short black curls and a charcoal suit",
        ),
        on_progress,
    )

    assert len(fake.submissions) == 1
    assert _positive(fake.submitted["prompt"]) == (
        f"{CHARACTER_REPLACEMENT_LEAD} a man with short black curls and a charcoal suit"
    )
    assert not (workspace / "character-replacement.json").exists()
