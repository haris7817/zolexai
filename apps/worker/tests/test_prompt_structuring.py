"""Deterministic prompt structuring and time-pinned multi-shot prompts.

Both features exist because the production runtime is the CFG-distilled
checkpoint: it cannot be pushed toward a prompt at inference, so the prompt
itself has to carry the discipline. The rules encoded here were measured by
hand on 2026-08-16 — explicit counts and colours, each restated as a
persistence rule, held a scene that drifted without them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    invocations,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
)
from worker.adapters.ltx import LtxAdapter
from worker.longform import plan_section_prompts, structure_prompt

# ── structure_prompt: what it adds ───────────────────────────────────────


def test_the_original_prompt_survives_verbatim_as_the_first_block() -> None:
    """The entire reason this is rules rather than a model: nothing the user
    typed may be rewritten, reordered or paraphrased."""
    prompt = "Two cars racing on a Los Angeles road, matte black and pearl white"
    structured = structure_prompt(prompt)
    assert structured.startswith(prompt)


def test_counts_become_persistence_rules() -> None:
    structured = structure_prompt("two cars racing through a desert")
    assert "Exactly 2 cars" in structured
    assert "only cars" in structured


def test_colours_become_persistence_rules() -> None:
    structured = structure_prompt("a matte black car chases a pearl white car")
    lowered = structured.lower()
    assert "matte black" in lowered.split("continuity")[1]
    assert "pearl white" in lowered.split("continuity")[1]
    assert "stays matte black" in lowered


def test_digit_counts_work_too() -> None:
    structured = structure_prompt("3 dancers on a rooftop")
    assert "Exactly 3 dancers" in structured


def test_no_negative_phrasing_is_ever_added() -> None:
    """The distilled model has no negation mechanism — 'no cuts, no other
    cars' reads as 'cuts, other cars' with extra steps. Every added rule must
    be stated positively."""
    for prompt in (
        "two cars racing",
        "a red balloon over the city",
        "a quiet mountain lake at dawn",
    ):
        added = structure_prompt(prompt)[len(prompt):].lower()
        assert "no " not in added, f"negative phrasing in: {added!r}"
        assert "don't" not in added and "do not" not in added


def test_an_already_structured_prompt_is_left_alone() -> None:
    prompt = "Persistent: a neon city\nSection 1: the chase begins"
    assert structure_prompt(prompt) == prompt


def test_a_plain_prompt_still_gains_the_generic_continuity_block() -> None:
    """Even with nothing to extract, the continuity block is the measured
    lever against drift, so it is always present."""
    structured = structure_prompt("a quiet mountain lake at dawn")
    assert "CONTINUITY" in structured
    assert "same subjects" in structured.lower()


def test_prose_connectives_are_not_mistaken_for_nouns() -> None:
    """'black and white footage' must not pin a noun called 'and'."""
    structured = structure_prompt("black and white footage of a train")
    assert " and stays" not in structured
    assert "The black" not in structured or "black and" not in structured.split("CONTINUITY")[1]


# ── structure_prompt v2 (execution.prompt_structuring_v2) ────────────────
#
# Everything here is gated: with v2 OFF the output must be byte-identical to
# what has always shipped, and the ON behaviour answers the three audited
# defects (21 Aug 2026 diagnosis) — the camera contradiction, the exit-blind
# presence assertion, and the continuity bullet read as one section's
# dialogue.


def test_v2_off_is_byte_identical_to_the_shipped_block() -> None:
    for prompt in (
        "two cars racing through a desert",
        "A locked-off static camera. A woman stands at a window.",
        "The woman leaves. She does not return.",
    ):
        assert structure_prompt(prompt) == structure_prompt(prompt, v2=False)


def test_v2_splits_the_presence_and_camera_clauses_into_separate_bullets() -> None:
    structured = structure_prompt("a quiet mountain lake at dawn", v2=True)
    block = structured.split("CONTINUITY")[1]
    assert "still present at the end" in block
    assert "keeps moving" in block
    # And they are separate bullets, so one can hold while the other yields.
    for line in block.splitlines():
        assert not (
            "still present at the end" in line and "keeps moving" in line
        ), "presence and camera must not share a bullet"


def test_v2_answers_a_static_camera_request_with_a_static_rule() -> None:
    """The B5 contradiction: a user typing 'The camera never moves' used to
    receive 'the camera keeps moving' in the same prompt."""
    structured = structure_prompt(
        "A locked-off static camera on a tripod. The camera never moves. "
        "A woman stands at a window in a quiet room.",
        v2=True,
    )
    added = structured.split("CONTINUITY")[1].lower()
    assert "keeps moving" not in added
    assert "holds perfectly still" in added


def test_v2_withholds_presence_assertions_when_the_user_says_someone_leaves() -> None:
    """Presence claims about departed people are the measured rendered-ghost
    failure (GPU, 20 Aug 2026)."""
    structured = structure_prompt(
        "A man walks out of the room and never comes back.", v2=True
    )
    added = structured.split("CONTINUITY")[1].lower()
    assert "still present at the end" not in added
    assert "count in every frame" not in added
    # Identity constancy still holds for whoever is on screen.
    assert "same faces, clothing and colours" in added


def test_v2_keeps_presence_assertions_when_nobody_leaves() -> None:
    structured = structure_prompt("two cars racing through a desert", v2=True)
    added = structured.split("CONTINUITY")[1].lower()
    assert "still present at the end" in added
    assert "count in every frame" in added


def test_v2_structuring_is_idempotent() -> None:
    """The v1 header matched neither _ALREADY_STRUCTURED nor _PERSISTENT_LINE,
    so structure_prompt would restructure its own output if called twice."""
    once = structure_prompt("a quiet mountain lake at dawn", v2=True)
    assert structure_prompt(once, v2=True) == once


def test_v2_bullets_reach_every_section_instead_of_one() -> None:
    """The A5 misclassification: the internal colon in '- One continuous
    scene:' read as a dialogue turn and the rule reached exactly one section,
    presented as something to perform."""
    structured = structure_prompt("a quiet mountain lake at dawn", v2=True)
    prompts = plan_section_prompts(structured, 3, total_seconds=90.0, v2=True)
    for prompt in prompts:
        assert "One continuous scene" in prompt
        assert "still present at the end" in prompt
    for prompt in prompts:
        head, _, tail = prompt.partition("NEW ACTION OR DIALOGUE")
        assert "One continuous scene" not in tail, (
            "a continuity rule must never be a section's assigned action"
        )


def test_v2_never_adds_negative_phrasing_either() -> None:
    for prompt in (
        "two cars racing",
        "A locked-off static camera. A woman stands at a window.",
        "The woman leaves and never comes back.",
    ):
        added = structure_prompt(prompt, v2=True)[len(prompt):].lower()
        assert "no " not in added, f"negative phrasing in: {added!r}"
        assert "don't" not in added and "do not" not in added and "never" not in added


# ── plan_section_prompts v2 ──────────────────────────────────────────────


def test_v2_off_section_prompts_are_byte_identical() -> None:
    assert plan_section_prompts(_STORYBOARD, 8, total_seconds=120.0) == (
        plan_section_prompts(_STORYBOARD, 8, total_seconds=120.0, v2=False)
    )


def test_v2_section_one_is_not_told_to_continue_from_a_predecessor() -> None:
    """Section 1 has no predecessor pass: an instruction referencing a
    nonexistent frame is caption noise on a runtime that reads captions as
    content."""
    prompts = plan_section_prompts(_STORYBOARD, 3, total_seconds=120.0, v2=True)
    assert "predecessor" not in prompts[0]
    assert "established previously" not in prompts[0]
    assert prompts[0].startswith("LONG-FORM VIDEO — SECTION 1 OF 3.")
    # Later sections keep the continuation register in full.
    for prompt in prompts[1:]:
        assert "Continue directly from the predecessor frame." in prompt
        assert "established previously" in prompt


def test_v2_single_pass_requests_remain_byte_for_byte_unchanged() -> None:
    assert plan_section_prompts(_STORYBOARD, 1, total_seconds=120.0, v2=True) == [
        _STORYBOARD
    ]


def test_v2_user_bullet_lists_stay_persistent_even_with_colons() -> None:
    """A bullet is a constraint or a description, never a `Name: "line"`
    dialogue turn."""
    prompt = (
        "A rainy street.\n"
        "- Wardrobe: the woman wears a red coat\n"
        "- Lighting: neon signs reflect in puddles"
    )
    prompts = plan_section_prompts(prompt, 2, total_seconds=60.0, v2=True)
    for section in prompts:
        assert "red coat" in section
        assert "neon signs" in section


# ── Timestamped multi-shot prompts ───────────────────────────────────────


_STORYBOARD = """A music video for an upbeat pop song.
0:00-0:10: the singer walks alone through neon streets
0:30-0:45: chorus — wide shot, dancers join her
[1:30-1:45] final chorus on a rooftop at dawn
"""


def test_timed_shots_land_in_the_sections_their_timestamps_name() -> None:
    """The client's storyboard prompts say WHERE a shot belongs. Allocating
    them by count put the chorus in the wrong minute of the song."""
    prompts = plan_section_prompts(_STORYBOARD, 8, total_seconds=120.0)

    # 8 sections of 15s: 0:00-0:10 -> section 0, 0:30-0:45 -> section 2
    # (midpoint 37.5s), 1:30-1:45 -> section 6 (midpoint 97.5s).
    assert "neon streets" in prompts[0]
    assert "dancers join her" in prompts[2]
    assert "rooftop at dawn" in prompts[6]
    # And nowhere else.
    for index in (1, 3, 4, 5, 7):
        assert "dancers join her" not in prompts[index]
        assert "rooftop at dawn" not in prompts[index]


def test_mm_ss_and_seconds_forms_both_parse() -> None:
    prompt = "0:05-0:10: a door opens\n20s-25s: the lights come on"
    prompts = plan_section_prompts(prompt, 3, total_seconds=30.0)
    assert "door opens" in prompts[0]
    assert "lights come on" in prompts[2]


def test_prose_ranges_are_not_mistaken_for_timestamps() -> None:
    """'3-4 cars pass by' is a count, not a shot at second three."""
    prompts = plan_section_prompts(
        "A busy street.\n3-4 cars pass by constantly.", 2, total_seconds=20.0
    )
    # The line must appear in EVERY section (persistent), not be scheduled.
    assert all("cars pass by" in p for p in prompts)


def test_without_total_seconds_timed_lines_still_schedule_in_order() -> None:
    """No duration to map against — contiguous allocation is the fallback,
    and the shots stay in the user's order."""
    prompts = plan_section_prompts(_STORYBOARD, 3)
    assert "neon streets" in prompts[0]
    assert "rooftop at dawn" in prompts[2]


def test_single_pass_requests_remain_byte_for_byte_unchanged() -> None:
    assert plan_section_prompts(_STORYBOARD, 1, total_seconds=120.0) == [_STORYBOARD]


# ── Adapter wiring ───────────────────────────────────────────────────────


def test_structuring_is_off_unless_the_workflow_asks(workspace: Path) -> None:
    """Default execution carries no flag, and the prompt reaches the model
    verbatim — pinned elsewhere too, but this is the pairing test for the
    flag being ON below."""
    job = make_job(workspace)
    cmd = LtxAdapter()._command(job, 2.0, workspace / "out.mp4")
    assert cmd[cmd.index("--prompt") + 1] == job.prompt


@needs_ffmpeg
async def test_the_flag_structures_the_prompt_the_model_receives(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run()` rebuilds the job with the structured prompt before dispatch,
    so every handler and every section plan inherits it — proven on the argv
    the model actually receives."""
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0, audio=True)
    )
    job = make_job(
        workspace, execution={"runtime": "ltx", "prompt_structuring": True}
    )
    await collect(job)

    argv = invocations(log)[0]
    sent = argv[argv.index("--prompt") + 1]
    assert sent.startswith(job.prompt), "the user's words must lead, verbatim"
    assert "CONTINUITY" in sent