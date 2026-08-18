"""Director mode: `prompt_mode: director` on Text to Video.

Two things this suite exists to keep true:

**Standard is byte-identical to what has served production.** Director mode is
reachable only through an explicit request parameter on the one workflow that
declares it; a job without it must produce the exact argv, structuring and
sectioning the pre-feature worker produced.

**The plan is global and the contract is enforced in code.** Dialogue the user
wrote survives verbatim; over-packed speech is trimmed rather than rushed;
every line has an owner that exists; timestamps never reach the model prompt;
each long-form section carries only its own events, so a section can never
restart the conversation.
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
    value_of,
)
from worker.adapters.base import AdapterError
from worker.director import (
    DirectorFailure,
    DirectorPlanError,
    compile_section_prompts,
    create_director_plan,
    parse_plan,
    wants_director,
)
from worker.director.provider import DirectorRequest, GemmaDirectorProvider

IDEA = "A detective confronts a corrupt police chief in his office at night."


def raw_plan(**overrides) -> dict:
    plan = {
        "scene": "A dim police chief's office at night, lit by a single desk lamp.",
        "tone": "tense",
        "ambience": "a low lamp hum and muffled city traffic",
        "characters": [
            {
                "id": "detective",
                "role": "detective",
                "appearance": "a weathered man in a rumpled gray suit",
                "voice": "low and gravelly",
            },
            {
                "id": "chief",
                "role": "police chief",
                "appearance": "a heavyset man in a dark green uniform",
                "voice": "deep and strained",
            },
        ],
        "timeline": [
            {
                "start": 0,
                "end": 4,
                "action": "The detective walks toward the desk",
                "camera": "medium shot, static",
                "speaker": None,
                "dialogue": None,
                "delivery": None,
            },
            {
                "start": 4,
                "end": 7,
                "action": "The detective leans forward",
                "camera": "medium close-up, static",
                "speaker": "detective",
                "dialogue": "You knew what was happening.",
                "delivery": "low and accusing",
            },
            {
                "start": 8,
                "end": 12,
                "action": "The police chief rises from his chair",
                "camera": "medium shot, static",
                "speaker": "chief",
                "dialogue": "You don't understand what was at stake.",
                "delivery": "strained and defensive",
            },
        ],
    }
    plan.update(overrides)
    return plan


def parsed(duration: float = 12.0, idea: str = IDEA, **overrides):
    return parse_plan(
        raw_plan(**overrides), idea=idea, duration_seconds=duration, language="english"
    )


def director_job(workspace: Path, **overrides):
    defaults = dict(
        prompt=IDEA,
        parameters={
            "duration": "2s",
            "aspect_ratio": "16:9",
            "quality": "High",
            "prompt_mode": "director",
        },
        execution={"runtime": "ltx", "prompt_structuring": True},
    )
    return make_job(workspace, **{**defaults, **overrides})


class CannedProvider:
    """A provider returning queued raw plans — no subprocess, no model."""

    def __init__(self, *plans) -> None:
        self.plans = list(plans)
        self.requests: list[DirectorRequest] = []

    async def generate_plan(self, request: DirectorRequest) -> dict:
        self.requests.append(request)
        return self.plans.pop(0)


def install_provider(monkeypatch: pytest.MonkeyPatch, provider) -> None:
    monkeypatch.setattr(
        GemmaDirectorProvider, "generate_plan", provider.generate_plan, raising=True
    )


# ── The mode is opt-in and scoped ────────────────────────────────────────


def test_only_text_to_video_with_the_parameter_wants_director(workspace: Path) -> None:
    """Image to Video shares the generation handler, so the scope check is the
    worker-side belt to the API's validation braces."""
    assert wants_director(director_job(workspace))
    assert not wants_director(make_job(workspace))
    assert not wants_director(
        director_job(workspace, parameters={"duration": "2s", "prompt_mode": "standard"})
    )
    assert not wants_director(director_job(workspace, workflow_id="image-to-video"))


@needs_ffmpeg
async def test_a_standard_job_never_touches_the_planner(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No parameter, no planner subprocess, and the structured prompt is the
    exact text the pre-feature worker sent."""
    from worker.longform.enhance import structure_prompt

    provider = CannedProvider()  # would raise IndexError if consulted
    install_provider(monkeypatch, provider)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(make_job(workspace, execution={"runtime": "ltx", "prompt_structuring": True}))

    assert provider.requests == []
    argv = invocations(log)[0]
    assert value_of(argv, "--prompt") == structure_prompt("a cinematic drone shot over a coastline")


# ── Plan validation ──────────────────────────────────────────────────────


def test_a_valid_plan_parses_with_its_events_in_order() -> None:
    plan = parsed()
    assert [c.id for c in plan.characters] == ["detective", "chief"]
    assert [e.start for e in plan.timeline] == [0, 4, 8]
    assert plan.spoken_words == 12


def test_the_literal_string_null_is_a_missing_speaker() -> None:
    """Instruct models write "null" as text as often as JSON null — observed
    on the 18 Aug pilot run."""
    events = raw_plan()["timeline"]
    events[0]["speaker"] = "null"
    plan = parsed(timeline=events)
    assert plan.timeline[0].speaker is None


def test_a_plan_missing_its_pieces_lists_every_problem() -> None:
    with pytest.raises(DirectorPlanError) as raised:
        parse_plan(
            {"scene": "", "characters": [], "timeline": []},
            idea=IDEA,
            duration_seconds=10.0,
            language="english",
        )
    text = str(raised.value)
    assert "scene" in text and "characters" in text and "timeline" in text


def test_dialogue_without_a_known_owner_is_refused() -> None:
    events = raw_plan()["timeline"]
    events[1]["speaker"] = "narrator"
    with pytest.raises(DirectorPlanError, match="unknown speaker"):
        parsed(timeline=events)


def test_overpacked_speech_is_trimmed_from_the_end_not_rushed() -> None:
    """A 5-second video cannot hold 12 spoken words at 2 wps; the tail lines
    lose their dialogue (keeping their action) rather than the pacing
    doubling."""
    events = raw_plan()["timeline"]
    for event, (start, end) in zip(events, [(0, 1.5), (1.5, 3.5), (3.5, 5)], strict=True):
        event["start"], event["end"] = start, end
    plan = parsed(duration=5.0, timeline=events)
    assert plan.spoken_words <= 5.0 * 2.0 * 1.15
    assert plan.timeline[1].dialogue is not None  # earlier line survives
    assert plan.timeline[2].dialogue is None  # later line trimmed
    assert plan.timeline[2].action  # the beat itself remains


def test_the_users_own_line_is_never_trimmed_and_never_rewritten() -> None:
    idea = 'A farewell. Woman says: "Please don\'t leave." Man replies: "I have to go."'
    events = [
        {
            "start": 0,
            "end": 4,
            "action": "The woman reaches out",
            "camera": "two-shot",
            "speaker": "detective",
            "dialogue": "Please don't leave.",
            "delivery": "soft",
        },
        {
            "start": 4,
            "end": 8,
            "action": "The man turns away",
            "camera": "two-shot",
            "speaker": "chief",
            "dialogue": "I have to go.",
            "delivery": "calm",
        },
    ]
    plan = parse_plan(
        raw_plan(timeline=events), idea=idea, duration_seconds=8.0, language="english"
    )
    assert [e.dialogue for e in plan.timeline] == ["Please don't leave.", "I have to go."]

    events[1]["dialogue"] = "I must depart."  # a paraphrase is a contract breach
    with pytest.raises(DirectorPlanError, match="dropped or rewrote"):
        parse_plan(raw_plan(timeline=events), idea=idea, duration_seconds=8.0, language="english")


# ── Compilation ──────────────────────────────────────────────────────────


def test_a_single_pass_gets_one_caption_with_every_line_quoted_exactly() -> None:
    [caption] = compile_section_prompts(parsed(), 1, total_seconds=12.0)
    assert '"You knew what was happening."' in caption
    assert '"You don\'t understand what was at stake."' in caption
    assert "low and accusing voice" in caption
    assert "Initially" in caption


def test_no_timestamps_or_labels_ever_reach_the_prompt() -> None:
    """No official LTX source supports timestamp syntax and the enhancer
    prompts forbid it — timing is the worker's job, not the model's."""
    import re

    for caption in compile_section_prompts(parsed(), 2, total_seconds=12.0):
        assert not re.search(r"\d\s*[-–]\s*\d+\s*(?:s\b|sec)", caption)
        assert "SECTION" not in caption
        assert ":" not in caption.replace('":', "").replace(':"', "")


def test_sections_split_the_dialogue_instead_of_repeating_it() -> None:
    """The long-form failure this feature must not reintroduce: each section
    carries only its own lines, and later sections say they continue."""
    first, second = compile_section_prompts(parsed(), 2, total_seconds=12.0)
    assert '"You knew what was happening."' in first
    assert "at stake" not in first
    assert '"You don\'t understand what was at stake."' in second
    assert "You knew what was happening" not in second
    assert "continue mid-scene" in second
    # Identity is restated in EVERY section — the text channel of continuity.
    for caption in (first, second):
        assert "rumpled gray suit" in caption
        assert "dark green uniform" in caption


def test_an_empty_section_holds_the_scene_without_inventing_events() -> None:
    events = raw_plan()["timeline"][:2]  # everything lands in the first half
    for event in events:
        event["end"] = min(event["end"], 5)
    [_, second] = compile_section_prompts(parsed(timeline=events), 2, total_seconds=12.0)
    assert "no one speaks" in second


def test_character_ids_leaking_into_prose_are_humanised() -> None:
    events = raw_plan()["timeline"]
    events[1]["camera"] = "close-up on chief, static"
    events[1]["action"] = "detective slams the folder in front of chief"
    [caption] = compile_section_prompts(parsed(timeline=events), 1, total_seconds=12.0)
    assert "on the police chief" in caption
    assert "front of the police chief" in caption


# ── Orchestration: retry once, then a clean failure ──────────────────────


async def test_a_refused_first_plan_is_retried_with_sampling(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CannedProvider({"scene": "nonsense"}, raw_plan())
    plan = await create_director_plan(director_job(workspace), 12.0, provider=provider)
    assert [r.sample for r in provider.requests] == [False, True]
    assert plan.spoken_words > 0


async def test_two_refused_plans_fail_the_job_with_customer_safe_copy(
    workspace: Path,
) -> None:
    provider = CannedProvider({"scene": "bad"}, {"scene": "bad"})
    with pytest.raises(DirectorFailure) as raised:
        await create_director_plan(director_job(workspace), 12.0, provider=provider)
    assert "standard prompt mode" in raised.value.user_message
    assert "attempt 2" in raised.value.internal_detail


# ── End to end through the adapter ───────────────────────────────────────


@needs_ffmpeg
async def test_a_director_job_renders_the_compiled_caption_not_the_idea(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idea is planned, the caption is rendered, and `structure_prompt`'s
    CONTINUITY block stays out — the compiler owns the whole text."""
    short_plan = raw_plan(
        timeline=[
            {
                "start": 0,
                "end": 1,
                "action": "The detective walks toward the desk",
                "camera": "medium shot, static",
                "speaker": None,
                "dialogue": None,
                "delivery": None,
            },
            {
                "start": 1,
                "end": 2,
                "action": "The detective leans forward",
                "camera": "medium close-up, static",
                "speaker": "detective",
                "dialogue": "You knew.",
                "delivery": "low and accusing",
            },
        ]
    )
    install_provider(monkeypatch, CannedProvider(short_plan))
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    result, reported = await collect(director_job(workspace))

    argv = invocations(log)[0]
    prompt = value_of(argv, "--prompt")
    assert '"You knew."' in prompt
    assert IDEA not in prompt  # the idea is an input, not the prompt
    assert "CONTINUITY" not in prompt  # structuring skipped by design
    assert result.duration_seconds == pytest.approx(2.0, abs=1.0)
    assert any("Directing" in message for _, _, message in reported)


@needs_ffmpeg
async def test_a_failed_planner_fails_the_job_before_any_gpu_time(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Director mode never silently falls back to a dialogue-less standard
    render — the user asked for a planned scene, and a clean error beats a
    surprise."""
    install_provider(monkeypatch, CannedProvider({"scene": "bad"}, {"scene": "bad"}))
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    with pytest.raises(AdapterError) as raised:
        await collect(director_job(workspace))

    assert raised.value.retriable is False
    assert invocations(log) == []
