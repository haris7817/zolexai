"""No written language in the picture.

Client report, 10 Sep 2026, on a delivered file: the captions are burned into
the generated pixels, there is no subtitle stream, and `ffmpeg -sn` cannot
touch them. Their fix — a no-text clause in the POSITIVE prompt, because the
distilled workflow runs CFG 1.0 and a negative prompt has nothing to act
through there.

The ordering is the whole point of these, and it is not obvious: the clause
has to be composed on last, after the guideline rewrite and after the spoken
lines, or it gets paraphrased by one and rejected by the other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.base import AdapterJob
from worker.adapters.ltx_hd import WORKFLOW_ID, LtxHdAdapter
from worker.core.config import settings
from worker.prompt.no_text import NO_VISIBLE_TEXT, without_visible_text


def _job(workspace: Path, **params) -> AdapterJob:
    return AdapterJob(
        job_id="hd-job",
        workflow_id=WORKFLOW_ID,
        workflow_version="1",
        prompt="a photorealistic skydiving sequence",
        parameters={"duration": "8s", **params},
        execution={"runtime": "ltx_hd"},
        workspace=workspace,
    )


def test_the_clause_is_the_clients_text_and_not_a_paraphrase_of_it() -> None:
    """They will read this back off a prompt trace, so it is stored as the
    string they wrote rather than as our rewording of it."""
    assert NO_VISIBLE_TEXT == (
        "No visible written language appears anywhere in the video. "
        "No subtitles, captions, closed captions, dialogue text, lower thirds, "
        "labels, signs, logos, or watermarks. Spoken dialogue is heard only "
        "through the audio and is never displayed visually."
    )


def test_the_clause_is_appended_by_default_and_only_once() -> None:
    """Default off is the client's `allow_captions = False`. Appending twice
    would put the same prohibition in the box twice, which is the failure the
    guideline pack's §7 is actually warning about."""
    out = without_visible_text("a street at night")
    assert out == f"a street at night {NO_VISIBLE_TEXT}"
    assert without_visible_text(out) == out
    assert out.count("No visible written language") == 1


def test_a_customer_who_wants_a_sign_in_frame_can_have_one() -> None:
    """The clause forbids signs and shopfronts as well as captions, so it
    needs an escape hatch or it becomes a defect of its own."""
    assert without_visible_text("a neon shopfront", allow_captions=True) == "a neon shopfront"


def test_an_empty_prompt_gets_the_clause_without_a_leading_space() -> None:
    assert without_visible_text("") == NO_VISIBLE_TEXT
    assert without_visible_text("   ") == NO_VISIBLE_TEXT


def test_the_job_decides_before_the_deployment_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`allow_captions` is a creative choice about one video, not a
    deployment lever — the opposite of `delivery` — so a job's own answer
    wins over the setting in both directions."""
    adapter = LtxHdAdapter()
    assert adapter._allow_captions(_job(tmp_path)) is False

    monkeypatch.setattr(settings, "ltx_allow_captions", True)
    assert adapter._allow_captions(_job(tmp_path)) is True
    assert adapter._allow_captions(_job(tmp_path, allow_captions=False)) is False

    monkeypatch.setattr(settings, "ltx_allow_captions", False)
    assert adapter._allow_captions(_job(tmp_path, allow_captions=True)) is True
    for truthy in ("true", "True", "yes", "on", "1", True):
        assert adapter._allow_captions(_job(tmp_path, allow_captions=truthy)) is True
    for falsy in ("false", "no", "off", "0", "", False):
        assert adapter._allow_captions(_job(tmp_path, allow_captions=falsy)) is False


def test_the_clause_lands_after_the_spoken_lines_not_before_them() -> None:
    """Auto Dialogue writes quoted lines into the prompt. The clause says
    those lines are heard and never displayed, so it only makes sense after
    them — and composing it last is also what keeps the guideline rewriter
    from paraphrasing it and our own §7 validator from rejecting it.
    """
    with_lines = 'a newsroom. The anchor says "Good evening."'
    out = without_visible_text(with_lines)
    assert out.index("Good evening.") < out.index("No visible written language")
    assert out.endswith("never displayed visually.")
