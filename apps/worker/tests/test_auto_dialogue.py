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

from worker.adapters.base import AdapterJob
from worker.dialogue import add_auto_dialogue, enabled_for
from worker.dialogue.decide import (
    AUTO_DIALOGUE_WORKFLOWS,
    Dialogue,
    Line,
    Speaker,
    compose,
    line_target,
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


def _job(prompt: str = SCENE, workflow: str = "text-to-video", **parameters) -> AdapterJob:
    return AdapterJob(
        job_id="dialogue-job",
        workflow_id=workflow,
        workflow_version="1",
        prompt=prompt,
        parameters={"duration": "15s", **parameters},
        execution={"runtime": "ltx_comfy"},
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


async def test_a_failed_writer_renders_the_customers_own_prompt() -> None:
    """Fail open. The customer asked for a video, not for dialogue, and the
    right outcome is the video they asked for."""
    original = _job(auto_dialogue=True)
    for failure in (
        DialogueUnavailable("no key"),
        DialogueRejected("nonsense"),
        RuntimeError("something else entirely"),
    ):
        job = await add_auto_dialogue(original, 15.0, providers=[_Writer(raises=failure)])
        assert job.prompt == SCENE, failure


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
