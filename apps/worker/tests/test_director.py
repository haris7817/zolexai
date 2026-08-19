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
    """A provider returning queued raw plans — no subprocess, no model.

    Once the queue is down to its last entry it keeps returning it, which is
    what a real model does when asked again: it answers. Popping to empty
    instead would make every test brittle to how many attempts the
    orchestration happens to make.
    """

    def __init__(self, *plans) -> None:
        self.plans = list(plans)
        self.requests: list[DirectorRequest] = []

    async def generate_plan(self, request: DirectorRequest) -> dict:
        self.requests.append(request)
        return self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]


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


def test_short_clips_get_a_proportionally_tighter_speech_budget() -> None:
    """The establishing head is excluded from the word budget, so it costs a
    short clip a far larger share of its allowance than a long one — which is
    where over-packing actually produced run-together delivery."""
    from worker.director.plan import speech_budget

    assert speech_budget(15.0) == 25  # (15 - 2.5) * 2
    assert speech_budget(60.0) == 115
    # Proportionally: a 15s clip loses ~17% of a flat budget, a 60s clip ~4%.
    assert speech_budget(15.0) / 15.0 < speech_budget(60.0) / 60.0


def test_the_line_target_keeps_someone_speaking_throughout() -> None:
    """A target ABOVE the measured repetition threshold, separate from the
    ceiling. Below ~0.2 lines/second the model fills the silence by repeating
    itself — a 60s plan carrying 7 lines said one of them twice."""
    from worker.director.plan import spoken_line_budget, target_spoken_lines

    assert target_spoken_lines(20.0) == 5
    assert target_spoken_lines(60.0) == 15
    # The failing 60s render had 7 lines; the target must be well clear of it.
    assert target_spoken_lines(60.0) > 7
    # Every duration lands above the density that measured clean.
    for seconds in (10.0, 15.0, 20.0, 30.0, 60.0):
        assert target_spoken_lines(seconds) / seconds >= 0.2
    # A target is not a ceiling, and never exceeds one.
    assert target_spoken_lines(20.0) < spoken_line_budget(20.0)
    assert target_spoken_lines(5.0) == 2  # never below a two-line exchange


def test_pacing_problems_names_every_silence_a_customer_would_hear() -> None:
    """Sparse plans are reported, not raised: a thin conversation still makes
    a valid video, and failing the job over it helps nobody."""
    from worker.director.plan import pacing_problems

    events = [
        {
            "start": 0, "end": 2, "action": "The detective steps in",
            "camera": "medium shot", "speaker": "detective",
            "dialogue": "You knew.", "delivery": "low",
        },
        {
            "start": 40, "end": 44, "action": "The chief looks up",
            "camera": "close-up", "speaker": "chief",
            "dialogue": "I did what I had to.", "delivery": "strained",
        },
    ]
    problems = pacing_problems(parsed(duration=60.0, timeline=events))
    joined = " ".join(problems)
    assert "only 2 spoken lines" in joined       # too few for 60s
    assert "38-second silence" in joined         # the gap in the middle
    assert "last 16 seconds" in joined           # nothing lands at the end

    # A well-paced plan reports nothing at all.
    assert pacing_problems(parsed(duration=12.0)) == []


def test_back_to_back_spoken_lines_are_separated_by_a_pause_cue() -> None:
    """Two quoted lines in a row with only "A moment later" between them get
    delivered as one utterance. The pause is stated, per the official pacing
    guidance, whenever the preceding event also spoke."""
    events = raw_plan()["timeline"]
    events[0]["speaker"] = "detective"
    events[0]["dialogue"] = "You knew."
    events[0]["delivery"] = "low"
    [caption] = compile_section_prompts(parsed(timeline=events), 1, total_seconds=12.0)
    assert "After a short pause" in caption
    assert "A beat of silence passes, and then the" in caption  # no stray comma
    # The silent-beat case keeps its ordinary transition.
    [plain] = compile_section_prompts(parsed(), 1, total_seconds=12.0)
    assert "A moment later" in plain


def test_the_caption_asks_for_each_line_once_without_using_a_negative() -> None:
    """A 60s render spoke three of fourteen lines twice, each repeat seconds
    after the original. The instruction against it must be POSITIVE: this
    runtime has no negation mechanism, so "no line is repeated" reads as a
    request for repetition (the rule `enhance.py` is built around)."""
    [caption] = compile_section_prompts(parsed(), 1, total_seconds=12.0)
    assert "spoken a single time" in caption
    assert "moves forward to the next speaker" in caption
    for banned in ("do not repeat", "never repeat", "no line is repeated", "without repeating"):
        assert banned not in caption.lower()

    # A section with no dialogue has no lines to say once, and should not
    # carry an instruction about them.
    silent = raw_plan()["timeline"][:1]
    silent[0]["speaker"] = None
    silent[0]["dialogue"] = None
    [quiet] = compile_section_prompts(parsed(timeline=silent), 1, total_seconds=12.0)
    assert "spoken a single time" not in quiet


def test_a_distinctive_word_reused_across_lines_is_reported() -> None:
    """Every line unique, and the exchange still sounds machine-written: a
    customer noticed "excellent" twice before any of our checks did. Ordinary
    words are exempt — a conversation cannot avoid "the"."""
    from worker.director.plan import repeated_vocabulary, vocabulary_problems

    events = raw_plan()["timeline"]
    events[1]["dialogue"] = "That was an excellent decision."
    events[2]["dialogue"] = "Excellent work, truly."
    plan = parsed(timeline=events)
    assert repeated_vocabulary(plan) == ["excellent"]
    assert "excellent" in vocabulary_problems(plan)[0]

    # Structural words repeat in any real conversation and must not be flagged.
    events[1]["dialogue"] = "You knew about the money."
    events[2]["dialogue"] = "You knew about the risk."
    assert repeated_vocabulary(parsed(timeline=events)) == []


def test_continuity_facts_are_restated_in_every_section() -> None:
    """The customer symptoms this targets — a prop that comes back different
    after being off screen, a person who flickers out — are the unguided
    runtime dropping a constraint the prompt only implied. Restating is the
    measured lever, so it must survive into EVERY pass, not just the first."""
    hat = "the red felt Santa hat stays the same red felt hat every time it appears"
    plan = parsed(continuity=[hat, "exactly two people are in the room"])
    for caption in compile_section_prompts(plan, 2, total_seconds=12.0):
        assert "red felt Santa hat" in caption
        assert "Exactly two people are in the room" in caption
        # Continuous presence, phrased as presence.
        assert "stay fully visible in the frame" in caption


def test_continuity_is_phrased_as_what_stays_never_as_what_to_avoid() -> None:
    """This runtime has no negation mechanism (`enhance.py`), so a banned
    thing reads as a requested one. Every added constraint is positive."""
    plan = parsed(continuity=["the blue mug stays the same blue mug throughout"])
    [caption] = compile_section_prompts(plan, 1, total_seconds=12.0)
    tail = caption[caption.index("Under the voices") :].lower()
    for banned in ("does not", "do not", "never ", "without ", "no longer"):
        assert banned not in tail


def test_a_plan_with_no_continuity_facts_still_pins_the_cast() -> None:
    """The list is the planner's to fill and may be empty; identity and
    presence are ours and are always stated."""
    [caption] = compile_section_prompts(parsed(), 1, total_seconds=12.0)
    assert "keep exactly the same faces" in caption
    assert "stay fully visible in the frame" in caption


def test_a_move_phrase_that_already_has_a_verb_keeps_it() -> None:
    """Planners write moves both ways. A noun phrase gets a verb supplied; a
    phrase that already has one — including a sequenced "then cuts to …" —
    must not collect a second ("makes a then cuts to a medium shot")."""
    events = raw_plan()["timeline"]
    events[0]["camera"] = "medium shot, subtle push-in"
    events[1]["camera"] = "close-up, then cuts to a medium shot of both"
    [caption] = compile_section_prompts(parsed(timeline=events), 1, total_seconds=12.0)
    assert "camera makes a subtle push-in" in caption
    assert "camera then cuts to a medium shot of both" in caption
    assert "makes a then" not in caption


def test_character_ids_leaking_into_prose_are_humanised() -> None:
    events = raw_plan()["timeline"]
    events[1]["camera"] = "close-up on chief, static"
    events[1]["action"] = "detective slams the folder in front of chief"
    [caption] = compile_section_prompts(parsed(timeline=events), 1, total_seconds=12.0)
    assert "on the police chief" in caption
    assert "front of the police chief" in caption


# ── Orchestration: retry once, then a clean failure ──────────────────────


async def test_a_sparse_plan_is_retried_with_the_pacing_complaint_attached(
    workspace: Path,
) -> None:
    """The retry is never a blind repeat: it carries what was wrong, so the
    planner can fix the silence rather than re-roll and hope."""
    sparse = raw_plan(
        timeline=[
            {
                "start": 0, "end": 3, "action": "The detective steps in",
                "camera": "medium shot", "speaker": "detective",
                "dialogue": "You knew.", "delivery": "low",
            },
            {
                "start": 40, "end": 44, "action": "The chief looks away",
                "camera": "close-up", "speaker": "chief",
                "dialogue": "I did what I had to.", "delivery": "strained",
            },
        ]
    )
    provider = CannedProvider(sparse, raw_plan())
    plan = await create_director_plan(director_job(workspace), 60.0, provider=provider)

    assert len(provider.requests) == 2
    notes = " ".join(provider.requests[1].notes)
    assert "spoken lines" in notes  # the count complaint reached the retry
    assert "silence" in notes       # and so did the gap
    assert plan.spoken_lines >= 2


async def test_a_plan_that_only_paces_badly_still_ships(workspace: Path) -> None:
    """Sparse dialogue is a worse video, not a broken one. After the retry the
    best available plan is used rather than failing a job the customer is
    watching — the complaint is in the log, not in their face."""
    sparse = raw_plan(
        timeline=[
            {
                "start": 0, "end": 3, "action": "The detective steps in",
                "camera": "medium shot", "speaker": "detective",
                "dialogue": "You knew.", "delivery": "low",
            },
        ]
    )
    plan = await create_director_plan(
        director_job(workspace), 60.0, provider=CannedProvider(sparse)
    )
    assert plan.spoken_lines == 1


async def test_the_hosted_planner_is_preferred_and_falls_back_when_unusable(
    workspace: Path,
) -> None:
    """Hosted first because it plans in seconds where the local checkpoint
    costs 18-26s of the GPU the render is waiting for — but an outage or a
    missing key must only make the feature slower, never absent."""
    from worker.director import DirectorProviderUnavailable

    class Unusable:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_plan(self, request: DirectorRequest) -> dict:
            self.calls += 1
            raise DirectorProviderUnavailable("CEREBRAS_API_KEY is not set")

    hosted, local = Unusable(), CannedProvider(raw_plan())
    plan = await create_director_plan(
        director_job(workspace), 12.0, providers=[hosted, local]
    )

    assert plan.spoken_lines >= 2
    # Asked once, then abandoned — an unusable provider must not burn its retry.
    assert hosted.calls == 1
    assert len(local.requests) == 1


def test_the_chain_is_local_only_until_a_key_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node with no credential plans locally and says so in the log, rather
    than every job failing against a service it was never given access to."""
    from worker.core.config import settings
    from worker.director import (
        CerebrasDirectorProvider,
        GemmaDirectorProvider,
        default_providers,
    )

    monkeypatch.setattr(settings, "cerebras_api_key", "")
    assert [type(p) for p in default_providers()] == [GemmaDirectorProvider]

    monkeypatch.setattr(settings, "cerebras_api_key", "sk-test")
    assert [type(p) for p in default_providers()] == [
        CerebrasDirectorProvider,
        GemmaDirectorProvider,
    ]


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
                "end": 4,
                "action": "The detective leans forward",
                "camera": "medium close-up, static",
                "speaker": "detective",
                "dialogue": "You knew.",
                "delivery": "low and accusing",
            },
        ]
    )
    install_provider(monkeypatch, CannedProvider(short_plan))
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 5.0, audio=True))

    # 5s is the product's shortest offered length; below it the establishing
    # head leaves no speech budget at all, which is the correct answer for a
    # duration the menu never offers.
    result, reported = await collect(
        director_job(
            workspace,
            parameters={
                "duration": "5s",
                "aspect_ratio": "16:9",
                "quality": "High",
                "prompt_mode": "director",
            },
        )
    )

    argv = invocations(log)[0]
    prompt = value_of(argv, "--prompt")
    assert '"You knew."' in prompt
    assert IDEA not in prompt  # the idea is an input, not the prompt
    assert "CONTINUITY" not in prompt  # structuring skipped by design
    assert result.duration_seconds == pytest.approx(5.0, abs=1.0)
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
