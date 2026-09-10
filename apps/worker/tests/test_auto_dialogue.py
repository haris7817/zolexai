"""Automatic dialogue — the client's pack, on this platform's measurements.

Two things are being pinned here and they pull in opposite directions.

The first is **restraint**: this feature edits the prompt a customer wrote, on
workflows a client is testing, so every gate that leaves the prompt alone
matters more than the path that changes it. Most of these tests are about not
speaking.

The second is **the measurements**. The client's pack writes one line per
section; `worker/director/plan.py` records nine GPU renders saying a clip with
too few lines echoes itself. Where the two disagree, the measurement wins, and
that decision is pinned so nobody quietly restores the single line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.base import AdapterError, AdapterJob
from worker.core.config import settings
from worker.dialogue import add_auto_dialogue, enabled_for
from worker.dialogue.decide import (
    AUTO_DIALOGUE_WORKFLOWS,
    Dialogue,
    Line,
    Speaker,
    compose,
    line_target,
    no_eligible_speaker,
    skip_reason,
    word_budget,
)
from worker.dialogue.provider import (
    DialogueRejected,
    DialogueRequest,
    DialogueUnavailable,
    parse,
    system_prompt,
)
from worker.longform.language import soundscape_clause, supplied_dialogue

SCENE = "A taxi driver picks up a passenger outside a rain-soaked station at night"


def _job(
    prompt: str = SCENE, workflow: str = "text-to-video", layout: str = "paragraph", **parameters
) -> AdapterJob:
    # `layout` pins the composition path per test: the older tests describe
    # the paragraph/beats layouts and keep them; the native tests opt in.
    return AdapterJob(
        job_id="dialogue-job",
        workflow_id=workflow,
        workflow_version="1",
        prompt=prompt,
        parameters={"duration": "15s", **parameters},
        execution={"runtime": "ltx_comfy", "auto_dialogue_layout": layout},
        workspace=Path("."),
    )


def _answer(lines: int = 4, **overrides) -> dict:
    body = {
        "has_speaker": True,
        "speakers": [
            {"id": "driver", "description": "the taxi driver", "voice": "low and weary"},
            {"id": "rider", "description": "the passenger", "voice": "clipped"},
        ],
        "lines": [
            {"speaker": "driver", "text": "Where to tonight?"},
            {"speaker": "rider", "text": "The old harbour road."},
            {"speaker": "driver", "text": "That is a long way in this."},
            {"speaker": "rider", "text": "I know. Just drive."},
        ][:lines],
    }
    body.update(overrides)
    return body


class _Writer:
    """A provider that answers with whatever the test hands it."""

    name = "stub"

    def __init__(self, answer=None, raises: Exception | None = None) -> None:
        self._answer = answer if answer is not None else _answer()
        self._raises = raises
        self.calls = 0

    async def write(self, request: DialogueRequest) -> dict:
        self.calls += 1
        self.seen_system = request.system_text
        if self._raises is not None:
            raise self._raises
        return self._answer


# ── When it stands back ────────────────────────────────────────────────────


def test_it_is_off_until_a_deployment_turns_it_on() -> None:
    """The default changes nothing about any video anyone renders today."""
    assert enabled_for(_job()) is False
    assert skip_reason(_job(), 15.0, enabled=False) == "disabled"


def test_a_job_may_ask_for_it_either_way() -> None:
    assert enabled_for(_job(auto_dialogue=True)) is True
    assert enabled_for(_job(auto_dialogue="false")) is False


@pytest.mark.parametrize(
    "prompt",
    [
        "A quiet mountain lake at dawn, no dialogue",
        "A product shot of a watch, ambience only",
        "A silent scene in an empty church",
        "A city street, no voiceover",
    ],
)
def test_a_prompt_that_asks_for_silence_is_never_given_words(prompt: str) -> None:
    assert skip_reason(_job(prompt), 15.0, enabled=True) == "forbidden_by_prompt"


@pytest.mark.parametrize(
    "prompt",
    [
        'A detective leans in and says, "You were there that night."',
        "A shopper asks the assistant about the price",
        "Dialogue: two friends argue about a bill",
    ],
)
def test_a_prompt_that_already_has_a_say_is_left_alone(prompt: str) -> None:
    """Broader than this codebase's own quoted-words rule, deliberately: the
    question here is whether the CUSTOMER already had a say, and someone who
    wrote a speech verb did."""
    assert skip_reason(_job(prompt), 15.0, enabled=True) == "dialogue_already_present"


def test_a_speech_verb_outside_the_clients_list_does_not_block() -> None:
    """"A woman explains the product" is a scene that WANTS words: the customer
    described someone talking and wrote nothing for them to say, which is the
    gap this feature was asked for. The client's list is kept exactly as they
    wrote it rather than widened towards every speech verb, because each verb
    added is a scene that silently stops being filled."""
    assert skip_reason(_job("A woman explains the product to camera"), 15.0, enabled=True) == ""


def test_a_muted_video_is_not_given_words() -> None:
    """Writing lines for a soundtrack that gets thrown away would spend a
    model call to make the picture worse — every word in a prompt moves it."""
    assert skip_reason(_job(sound="false"), 15.0, enabled=True) == "sound_off"
    muted = _job()
    muted.execution["soundscape"] = False
    assert skip_reason(muted, 15.0, enabled=True) == "sound_off"


def test_director_mode_keeps_its_own_dialogue() -> None:
    """Director plans lines with locks and exits this module knows nothing
    about. Two planners writing for one video is the failure being prevented."""
    assert (
        skip_reason(_job(prompt_mode="director"), 15.0, enabled=True)
        == "director_mode_owns_the_dialogue"
    )


def test_only_the_single_pass_workflows_are_eligible() -> None:
    """Extend and Video to Video describe a continuation and a restyle, not a
    scene to invent — the line `_DIRECTOR_WORKFLOWS` draws, for that reason."""
    assert AUTO_DIALOGUE_WORKFLOWS == {"text-to-video", "image-to-video", "text-to-video-hd"}
    for workflow in ("extend-video", "video-to-video", "music-video", "character-replacement"):
        assert (
            skip_reason(_job(workflow=workflow), 15.0, enabled=True) == "workflow_not_eligible"
        ), workflow


def test_a_clip_too_short_to_hold_a_line_gets_none() -> None:
    assert skip_reason(_job(), 4.0, enabled=True) == "too_short_to_speak"


def test_an_eligible_job_passes_every_gate() -> None:
    assert skip_reason(_job(), 15.0, enabled=True) == ""


# ── The measurement the client's pack does not carry ───────────────────────


def test_it_asks_for_a_conversation_not_one_line() -> None:
    """The pack writes one line per section. Nine GPU renders on 18-19 Aug
    2026 say a 15-second clip carrying two echoes its last word, so the line
    count comes from `target_spoken_lines` instead."""
    assert line_target(8) == 2
    assert line_target(15) == 4
    assert line_target(30) == 8
    # And the words stay inside the pacing ceiling that measurement set.
    assert word_budget(8) == 11
    assert word_budget(15) == 25


def test_the_writer_is_told_that_too_few_lines_is_the_worse_failure() -> None:
    """A model handed only a maximum treats it as a problem to stay clear of —
    the lyrics writer learned that expensively (given 'at most 9' it wrote 6)."""
    text = system_prompt(DialogueRequest(prompt=SCENE, seconds=15))
    assert "about 4 lines" in text
    assert "worse failure than too many" in text
    assert "25 spoken words" in text


# ── What the prompt ends up saying ─────────────────────────────────────────


def test_the_customers_prompt_survives_byte_for_byte() -> None:
    enriched = compose(SCENE, _dialogue())
    assert enriched.startswith(SCENE)


def test_the_lines_become_quoted_speech_the_soundtrack_rule_recognises() -> None:
    """The whole mechanism: `soundscape_clause` hands the soundtrack to the
    scene when nobody speaks and to the people on screen when somebody does,
    and it decides on quoted words. Before, this prompt got silence."""
    assert supplied_dialogue(SCENE) is False
    assert "No one speaks" in soundscape_clause(SCENE, {}, {})

    enriched = compose(SCENE, _dialogue())
    assert supplied_dialogue(enriched) is True
    clause = soundscape_clause(enriched, {}, {})
    assert "No one speaks" not in clause
    assert "spoken a single time" in clause


# ── Who is allowed to speak ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("prompt", "blocked"),
    [
        ("A massive Tyrannosaurus Rex emerges from the jungle and roars", True),
        ("A lion stalks through tall grass at dawn", True),
        ("A red sports car drifts around a corner", True),
        # A human in the scene can speak, even with an animal beside them.
        ("A woman walks her dog through the park", False),
        ("Two women talk on a park bench", False),
        ("A chef plates a dish in a busy kitchen", False),
        # Explicit permission, in the shapes a customer actually writes.
        ("A talking dog says hello to the postman", False),
        ("The dinosaur says it is hungry", False),
        ("An anthropomorphic fox in a waistcoat", False),
    ],
)
def test_only_a_scene_with_somebody_human_gets_dialogue(prompt: str, blocked: bool) -> None:
    """Client report, 9 Sep 2026: a Tyrannosaurus was given "This valley
    belongs to me alone. None shall challenge my reign."

    Nothing had gone wrong mechanically. The writer was asked for one to four
    visible speaking characters, and a dinosaur is visible. The pack's own
    instruction — return `has_speaker: false` when nobody human could speak —
    is advice to a language model, and advice is not a gate.
    """
    assert no_eligible_speaker(_job(prompt)) is blocked


def test_the_gate_runs_before_the_writer_is_ever_called() -> None:
    """The ordering the client asked for. A writer that is asked "who speaks
    here?" about a dinosaur will answer, so it is not asked."""
    writer = _Writer()
    job = _job("A massive Tyrannosaurus Rex emerges from the jungle and roars")
    assert skip_reason(job, 15.0, enabled=True) == "no_eligible_speaker"
    assert writer.calls == 0


async def test_a_creature_scene_is_told_to_roar_instead_of_speak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saying nobody speaks is not enough on its own: an animal with a mouth
    and no instruction about it gets one anyway. This says what happens
    instead, and terminates the customer's sentence so the two do not run
    together.

    This is the DEPLOYMENT-DEFAULT path — dialogue the customer never asked
    for — which is the case the 9 Sep 2026 rule was written for, after a
    Tyrannosaurus was given four lines. It is unchanged.
    """
    import worker.dialogue as module

    # No `auto_dialogue` parameter: the feature is on because the deployment
    # says so, not because this customer asked.
    monkeypatch.setattr(settings, "auto_dialogue_enabled", True)
    job = _job("A massive Tyrannosaurus Rex emerges from the jungle and roars")
    writer = _Writer()
    result = await module.add_auto_dialogue(job, 15.0, providers=[writer])
    assert writer.calls == 0
    assert '"' not in result.prompt
    assert "never to form speech" in result.prompt
    assert result.prompt.startswith(job.prompt + ".")


async def test_ticking_the_box_over_a_dinosaur_makes_the_dinosaur_speak() -> None:
    """Client instruction, 10 Sep 2026: "when video is generated of animals
    with dialogue on they should speak."

    The 9 Sep rule stood every creature down, including for a customer who
    had explicitly turned Auto Dialogue on. That reading hands them the
    silent video they just declined — on an animal prompt there is nothing
    else the switch could mean. So an explicit request now licenses the
    non-human to speak, and the deployment default still refuses (the test
    above).
    """
    import worker.dialogue as module

    job = _job(
        "A massive Tyrannosaurus Rex emerges from the jungle and roars",
        auto_dialogue=True,
    )
    assert no_eligible_speaker(job) is False
    writer = _Writer()
    result = await module.add_auto_dialogue(job, 15.0, providers=[writer])
    # The writer IS consulted, and the roar clause does not muzzle the lines.
    assert writer.calls == 1
    assert '"' in result.prompt
    assert "never to form speech" not in result.prompt


def test_a_deployment_can_license_a_talking_animal() -> None:
    from dataclasses import replace as _replace

    job = _job("A massive Tyrannosaurus Rex emerges from the jungle")
    assert no_eligible_speaker(job) is True
    allowed = _replace(job, execution={**job.execution, "allow_nonhuman_speech": True})
    assert no_eligible_speaker(allowed) is False


def test_a_possessive_is_not_a_spoken_line() -> None:
    """Measured on a live job, 9 Sep 2026.

    A straight apostrophe is also a possessive, and the unguarded character
    class matched from the first "captain's" to the second — reading sixty
    words of scene description as quoted speech. The consequence was the exact
    fault this regex exists to prevent: `soundscape_clause` took its
    supplied-dialogue branch and told the model someone says those words, on a
    prompt where nobody speaks at all. It also made Auto Dialogue skip every
    such job as `dialogue_already_present`.

    Rare before the LTX 2.5 rewrite and common after it, because a rewritten
    prompt is longer, more literary, and full of possessives.
    """
    possessives = (
        "The camera pushes in toward the captain's face, capturing the salt "
        "spray on his skin and the distant cry of the captain's crew."
    )
    assert supplied_dialogue(possessives) is False
    assert supplied_dialogue("It's the captain's ship and it's getting late") is False

    # A real single-quoted line still counts: the guard is about where the
    # mark sits, not about which mark it is.
    assert supplied_dialogue("He turns and says 'Get out now' before leaving") is True
    assert supplied_dialogue('She says, "Where to tonight?"') is True


def test_a_line_that_ends_in_punctuation_is_not_given_a_second_stop() -> None:
    """`"Where to?".` is a stop the model reads as part of the line. The
    Director compiler's `_sentence` rule, reproduced."""
    dialogue = Dialogue(
        speakers=(Speaker("driver", "the taxi driver"),),
        lines=(Line("driver", "Hop in. Where to?"), Line("driver", "Nasty night")),
    )
    enriched = compose(SCENE, dialogue)
    assert '"Hop in. Where to?"' in enriched
    assert '".' not in enriched
    # A line with no terminator of its own still gets one.
    assert '"Nasty night."' in enriched


def test_the_first_line_carries_the_voice_and_later_ones_do_not_restate_it() -> None:
    enriched = compose(SCENE, _dialogue())
    assert 'The taxi driver says in a low and weary voice, "Where to tonight?"' in enriched
    assert enriched.count("low and weary") == 1


def test_the_hd_path_carries_its_own_anti_repeat_rule() -> None:
    """No soundscape clause runs on the HD path, and a five-word line in a
    fifteen-second video loops without one (measured 28 Aug 2026)."""
    without = compose(SCENE, _dialogue())
    assert "spoken a single time" not in without
    with_rule = compose(SCENE, _dialogue(), add_speech_rule=True)
    assert "spoken a single time" in with_rule


def test_the_beats_layout_separates_lines_with_cues_and_changes_nothing_else() -> None:
    """Same lines, same speakers, same quotes — with a beat between them.
    The paragraph layout is byte-identical to before, so the default
    render does not move while the experiment runs."""
    paragraph = compose(SCENE, _dialogue())
    beats = compose(SCENE, _dialogue(), layout="beats")
    assert paragraph == compose(SCENE, _dialogue(), layout="paragraph")
    assert beats.startswith(SCENE)
    assert beats.startswith(SCENE + "\n\nEarly on, the taxi driver says")
    assert "After a short pause, the passenger says" in beats
    assert "Near the end, the taxi driver says" in beats
    lines = ('"Where to tonight?"', '"The old harbour road."', '"That is a long way in this."')
    for quoted in lines:
        assert quoted in beats and quoted in paragraph
    # The cues are the only difference.
    stripped = beats
    for cue in ("Early on, t", "After a short pause, t", "Near the end, t"):
        stripped = stripped.replace(cue, "T")
    assert stripped == paragraph


def test_the_language_is_named_once_not_twice() -> None:
    """A line written in one language under a sentence naming another is a
    contradiction the model resolves by mumbling."""
    enriched = compose(SCENE, _dialogue(language="Spanish"))
    assert "Spanish" not in enriched  # the downstream clause owns that
    assert "in Spanish" in compose(SCENE, _dialogue(language="Spanish"), add_speech_rule=True)


def _dialogue(language: str = "") -> Dialogue:
    return Dialogue(
        speakers=(
            Speaker("driver", "the taxi driver", "low and weary"),
            Speaker("rider", "the passenger", "clipped"),
        ),
        lines=(
            Line("driver", "Where to tonight?"),
            Line("rider", "The old harbour road."),
            Line("driver", "That is a long way in this."),
        ),
        language=language,
    )


# ── Reading the writer's answer ────────────────────────────────────────────


def _request(seconds: float = 15.0) -> DialogueRequest:
    return DialogueRequest(prompt=SCENE, seconds=seconds)


def test_a_scene_with_nobody_to_speak_is_left_silent() -> None:
    assert parse({"has_speaker": False}, _request()) is None


def test_a_repeated_line_is_refused() -> None:
    """The artefact the whole feature exists to prevent. Arriving from the
    writer rather than the renderer makes it no less a repeat."""
    answer = _answer(2)
    answer["lines"][1] = {"speaker": "rider", "text": "Where to tonight?"}
    with pytest.raises(DialogueRejected, match="same thing"):
        parse(answer, _request())


def test_a_line_from_an_undeclared_speaker_is_refused() -> None:
    answer = _answer(2)
    answer["lines"][1] = {"speaker": "stranger", "text": "Let me in."}
    with pytest.raises(DialogueRejected, match="undeclared"):
        parse(answer, _request())


def test_a_plan_over_the_word_budget_is_refused() -> None:
    answer = _answer(2)
    answer["lines"][0]["text"] = " ".join(["word"] * 60)
    with pytest.raises(DialogueRejected, match="word budget"):
        parse(answer, _request())


def test_embedded_quotes_are_stripped_rather_than_escaped() -> None:
    """A nested quote ends the quoted span early: the model then speaks half a
    line and reads the rest as prose."""
    answer = _answer(1)
    answer["lines"][0]["text"] = 'He called it "the old road" once'
    dialogue = parse(answer, _request())
    assert dialogue is not None
    assert '"' not in dialogue.lines[0].text


def test_only_speakers_who_actually_speak_survive() -> None:
    dialogue = parse(_answer(1), _request())
    assert dialogue is not None
    assert [s.id for s in dialogue.speakers] == ["driver"]


# ── End to end, without a model ────────────────────────────────────────────


async def test_a_written_line_reaches_the_job() -> None:
    writer = _Writer()
    job = await add_auto_dialogue(_job(auto_dialogue=True), 15.0, providers=[writer])
    assert writer.calls == 1
    assert job.prompt.startswith(SCENE)
    assert '"Where to tonight?"' in job.prompt


async def test_a_failed_writer_renders_the_customers_own_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail open where the feature came from the DEPLOYMENT.

    Nobody asked for dialogue on this job — a deployment default did — so the
    customer asked for a video and the right outcome is the video they asked
    for. The parameter is deliberately absent: with it set, the contract is
    the opposite one, below.
    """
    monkeypatch.setattr(settings, "auto_dialogue_enabled", True)
    original = _job()
    for failure in (
        DialogueUnavailable("no key"),
        DialogueRejected("nonsense"),
        RuntimeError("something else entirely"),
    ):
        job = await add_auto_dialogue(original, 15.0, providers=[_Writer(raises=failure)])
        assert job.prompt == SCENE, failure


async def test_a_failed_writer_fails_a_job_that_asked_for_dialogue() -> None:
    """Fail closed where the CUSTOMER asked.

    The client's report on 8 Sep 2026: Auto Dialogue on, a video delivered
    with an audio track and nobody speaking, and nothing anywhere saying so.
    A silent video that looks like a success is the one outcome this must
    never produce, so an explicit request that could not be written stops the
    job with a message the customer can act on.
    """
    for failure in (
        DialogueUnavailable("no key"),
        DialogueRejected("nonsense"),
        RuntimeError("something else entirely"),
    ):
        with pytest.raises(AdapterError) as caught:
            await add_auto_dialogue(
                _job(auto_dialogue=True), 15.0, providers=[_Writer(raises=failure)]
            )
        assert "Auto Dialogue is on" in caught.value.user_message, failure
        assert caught.value.retriable is True


async def test_a_prompt_that_already_speaks_is_not_a_failure() -> None:
    """The refusals that are ANSWERS still render, switch on or not.

    A customer who wrote their own quoted line asked for dialogue and has it;
    failing that job would be failing it for succeeding.
    """
    spoken = _job('A taxi driver says, "Where to tonight?"', auto_dialogue=True)
    job = await add_auto_dialogue(spoken, 15.0, providers=[_Writer(raises=RuntimeError("x"))])
    assert job.prompt == spoken.prompt


async def test_no_writer_at_all_fails_a_job_that_asked_for_dialogue() -> None:
    with pytest.raises(AdapterError):
        await add_auto_dialogue(_job(auto_dialogue=True), 15.0, providers=[])


async def test_a_second_writer_gets_a_turn_after_a_failure() -> None:
    first = _Writer(raises=DialogueUnavailable("rate limited"))
    second = _Writer()
    job = await add_auto_dialogue(_job(auto_dialogue=True), 15.0, providers=[first, second])
    assert second.calls == 1
    assert '"Where to tonight?"' in job.prompt


async def test_no_speaker_is_an_answer_not_a_failure() -> None:
    """A landscape stays a landscape; asking the next writer would only be
    shopping for a different answer to a question already answered."""
    first = _Writer(answer={"has_speaker": False})
    second = _Writer()
    job = await add_auto_dialogue(_job(auto_dialogue=True), 15.0, providers=[first, second])
    assert second.calls == 0
    assert job.prompt == SCENE


async def test_a_skipped_job_never_reaches_a_writer() -> None:
    writer = _Writer()
    job = await add_auto_dialogue(_job(auto_dialogue=False), 15.0, providers=[writer])
    assert writer.calls == 0
    assert job.prompt == SCENE


# ── The client's native format (second revision, 8 Sep 2026) ───────────────


def _native_answer(seconds: int = 30, words: int | None = None) -> dict:
    """A script shaped like the client's own test fixture: two speakers, a
    connected exchange, the word count landing inside their range."""
    from worker.dialogue.native import word_range

    low, high = word_range(seconds)
    total = words if words is not None else (low + high) // 2
    pool = (
        "I used to rush through nights like this without noticing anything around me "
        "but the rain makes the whole city feel quieter tonight that little cafe still "
        "looks warm and inviting yet I think this peaceful walk is exactly what I needed "
        "maybe peace begins when we finally stop running and simply breathe and listen "
        "to the sound of the wet street and the hum of the late trams going home"
    ).split()
    chosen = (pool * 3)[:total]   # long enough for any range under test
    half = len(chosen) // 2
    return {
        "visual_prompt": "A taxi driver picks up a passenger outside a rain-soaked station at night, in three connected shots.",
        "ambience": "Rain, idling engine and distant traffic continue across every cut.",
        "speakers": [
            {"speaker_id": "person_a", "visual_identity": "the taxi driver in a grey cap",
             "voice_description": "middle-aged low gravelly voice with an unhurried local accent"},
            {"speaker_id": "person_b", "visual_identity": "the passenger in a wet dark coat",
             "voice_description": "younger light clear voice with a quick careful pace"},
            {"speaker_id": "person_c", "visual_identity": "a station guard who never speaks",
             "voice_description": "deep flat voice"},
        ],
        "dialogue_turns": [
            {"speaker_id": "person_a", "text": " ".join(chosen[:half])},
            {"speaker_id": "person_b", "text": " ".join(chosen[half:])},
        ],
    }


def test_the_native_format_locks_each_voice_once_with_no_cues_or_clocks() -> None:
    from worker.dialogue.native import compose_native_prompt, validate_script

    plan = validate_script(_native_answer(30), 30)
    prompt = compose_native_prompt(plan, "English")
    # one stable voice per speaker, stated exactly once
    assert prompt.count("middle-aged low gravelly voice") == 1
    assert prompt.count("younger light clear voice") == 1
    # turns are says/replies — no prose cue, no manner, no timestamp
    assert 'person_a says, "' in prompt and 'person_b replies, "' in prompt
    for banned in ("After a short pause", "Early on", "Near the end", "says in a"):
        assert banned not in prompt
    import re as _re
    assert not _re.search(r"\d+\.\d+-\d+\.\d+s", prompt)
    # pacing is one sentence about brief pauses; ambience sits UNDER the voices
    assert "no longer than 250 milliseconds" in prompt
    assert "continuous underneath the voices" in prompt
    assert "the only sounds are the ones the scene itself makes" not in prompt
    assert "The spoken language is English." in prompt


def test_two_speakers_are_told_the_listener_keeps_her_mouth_shut() -> None:
    """Client report, 9 Sep 2026: on a two-hander both women moved their
    mouths for the whole clip though only one spoke at a time.

    Nothing in the prompt was wrong. It locked the voices, the order and the
    pauses — and said nothing whatever about the person who is NOT speaking.
    Given two visible speakers and no instruction about the silent one, the
    model animates both. The fix is a constraint on the listener.
    """
    from worker.dialogue.native import compose_native_prompt, validate_script

    prompt = compose_native_prompt(validate_script(_native_answer(30), 30), "English")
    assert "listens in silence with their lips closed and still" in prompt
    assert "Only one mouth moves at any moment" in prompt
    assert "voices never overlap" in prompt
    assert "speak strictly one at a time, in the order written" in prompt


def test_the_listener_rule_does_not_repeat_who_anybody_is() -> None:
    """The voice locks already say "person_a is the woman on the left". The
    turn-taking sentence leans on that rather than restating it — repetition
    in a positive prompt is what the guideline pack's own validation warns
    about, and it is why the writer is told to put the position first."""
    from worker.dialogue.native import (
        NativePlan,
        NativeSpeaker,
        NativeTurn,
        compose_native_prompt,
    )

    plan = NativePlan(
        visual_prompt="Two women sit on a park bench.",
        ambience="Distant traffic.",
        speakers=(
            NativeSpeaker("person_a", "the woman on the left in a red coat", "warm and low"),
            NativeSpeaker("person_b", "the woman on the right in denim", "lighter and quicker"),
        ),
        turns=(
            NativeTurn("person_a", "Did you hear about Maria?"),
            NativeTurn("person_b", "I did, this morning."),
        ),
    )
    prompt = compose_native_prompt(plan)
    assert prompt.count("the woman on the left in a red coat") == 1
    assert prompt.count("the woman on the right in denim") == 1


def test_a_monologue_is_not_told_about_turn_taking() -> None:
    """One speaker cannot overlap themselves, and the sentence would only
    spend tokens on a scene it does not describe."""
    from worker.dialogue.native import (
        NativePlan,
        NativeSpeaker,
        NativeTurn,
        compose_native_prompt,
    )

    solo = NativePlan(
        visual_prompt="A man walks a dog at dawn.",
        ambience="Wind in the trees.",
        speakers=(NativeSpeaker("person_a", "the man in the grey coat", "low and even"),),
        turns=(NativeTurn("person_a", "Come on, boy, nearly home."),),
    )
    prompt = compose_native_prompt(solo)
    assert "lips closed" not in prompt
    assert "Only one mouth moves" not in prompt


def test_the_writer_is_told_to_anchor_identity_to_the_frame() -> None:
    """Wardrobe does not tell one face from another; a position does. Without
    this the listener rule has nothing decidable to attach to."""
    from worker.dialogue.native import system_prompt

    text = system_prompt(30)
    assert "the woman on the left" in text
    assert "left-to-right order" in text
    assert "before any description of" in text


def test_the_native_validator_enforces_the_clients_rules() -> None:
    import pytest as _pytest

    from worker.dialogue.native import NativeDialogueRejected, validate_script

    # the 30 s range is 48–68 words; ~40 is the client's rejected render
    with _pytest.raises(NativeDialogueRejected):
        validate_script(_native_answer(30, words=40), 30)
    with _pytest.raises(NativeDialogueRejected):
        validate_script(_native_answer(30, words=80), 30)
    # under is strict (47 fails); a small overshoot is tolerated (74 = 68 + 10%
    # passes, 76 does not) — measured: the writer lands a few words over on
    # rich prompts, and the alternative was a silent video
    with _pytest.raises(NativeDialogueRejected):
        validate_script(_native_answer(30, words=47), 30)
    assert validate_script(_native_answer(30, words=74), 30).total_words == 74
    with _pytest.raises(NativeDialogueRejected):
        validate_script(_native_answer(30, words=76), 30)
    # a line under three words
    short = _native_answer(30)
    short["dialogue_turns"].append({"speaker_id": "person_a", "text": "Best day"})
    with _pytest.raises(NativeDialogueRejected):
        validate_script(short, 30)
    # a repeated line
    twice = _native_answer(30)
    twice["dialogue_turns"].append(dict(twice["dialogue_turns"][0]))
    with _pytest.raises(NativeDialogueRejected):
        validate_script(twice, 30)
    # a speaker the scene never declared (the client's Squidward)
    ghost = _native_answer(30)
    ghost["dialogue_turns"][1]["speaker_id"] = "person_z"
    with _pytest.raises(NativeDialogueRejected):
        validate_script(ghost, 30)
    # a declared speaker who never speaks is dropped rather than described
    plan = validate_script(_native_answer(30), 30)
    assert [s.speaker_id for s in plan.speakers] == ["person_a", "person_b"]


def test_format_labels_are_stripped_from_the_visual_prompt_whatever_the_writer_kept() -> None:
    """The client's review: "vertical video (16:9)" contradicting itself, and
    a shot list ending at 10 s — both in the customer's text. The writer is
    told to drop such labels and mostly does, but kept "vertical video" as a
    style word on a 16:9 job. Orientation is the customer's aspect choice,
    so the labels go deterministically too."""
    from worker.dialogue.native import strip_format_labels, validate_script

    assert strip_format_labels(
        "Create a hyper-realistic cinematic 15-second vertical video (16:9) of a race at night."
    ) == "Create a hyper-realistic cinematic of a race at night."
    assert strip_format_labels("A 4K 1920x1080 landscape shot, 30 seconds, aspect ratio 9:16.") == "A shot."
    # words that only look like labels survive
    assert strip_format_labels("The 3 friends walk 2 blocks.") == "The 3 friends walk 2 blocks."
    answer = _native_answer(30)
    answer["visual_prompt"] = "Hyper-realistic cinematic vertical video in downtown Los Angeles at night."
    assert validate_script(answer, 30).visual_prompt == "Hyper-realistic cinematic in downtown Los Angeles at night."


def test_the_word_range_follows_the_clients_table_and_extends_between_rows() -> None:
    from worker.dialogue.native import word_range

    assert word_range(8) == (12, 18)
    assert word_range(10) == (16, 22)
    assert word_range(15) == (24, 34)
    assert word_range(30) == (48, 68)
    # between rows, the table's own slope (1.6–2.27 words per second)
    low, high = word_range(20)
    assert (low, high) == (32, 45)


async def test_the_native_layout_replaces_the_prompt_with_the_validated_screenplay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's design rewrites the customer's text into `visual_prompt`
    (their instruction: keep the story, strip duration/resolution/aspect
    labels) — that is what fixes "vertical video (16:9)" and a shot list that
    stops at 10 s, both of which came from the customer's own prompt."""
    monkeypatch.setattr(settings, "auto_dialogue_layout", "native")
    writer = _Writer(answer=_native_answer(30))
    job = await add_auto_dialogue(_job(auto_dialogue=True, duration="30s", layout="native"), 30.0, providers=[writer])
    assert writer.calls == 1
    assert job.prompt.startswith("A taxi driver picks up a passenger")
    assert 'person_a says, "' in job.prompt
    assert "After a short pause" not in job.prompt
    # the writer was handed the client's instruction, with the 30 s range
    assert "Total spoken words must be 48-68" in writer.seen_system


async def test_a_writer_that_misses_the_range_once_is_told_what_to_fix_and_retried() -> None:
    """Measured 8 Sep 2026: the hosted writer returned 37 words for a 24–34
    range and the job fell open to a silent video. One retry, naming the
    rule, is what turns that into a pass."""

    class _Sequence:
        name = "sequence"

        def __init__(self) -> None:
            self.calls = 0
            self.users: list[str] = []

        async def write(self, request: DialogueRequest) -> dict:
            self.calls += 1
            self.users.append(request.user_text)
            return _native_answer(30, words=80) if self.calls == 1 else _native_answer(30)

    writer = _Sequence()
    job = await add_auto_dialogue(
        _job(auto_dialogue=True, duration="30s", layout="native"), 30.0, providers=[writer]
    )
    assert writer.calls == 2
    assert "Your previous script was rejected" in writer.users[1]
    assert "48-68" in writer.users[1]
    assert 'person_a says, "' in job.prompt


async def test_a_native_script_the_validator_refuses_falls_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "auto_dialogue_layout", "native")
    monkeypatch.setattr(settings, "auto_dialogue_enabled", True)
    bad = _Writer(answer=_native_answer(30, words=40))
    job = await add_auto_dialogue(_job(duration="30s", layout="native"), 30.0, providers=[bad])
    assert job.prompt == SCENE
