"""The client's LTX 2.5 guideline pack: routing, validation, and the seam.

The pack itself (`worker/prompt/ltx25/guidelines/`) is their text and is not
asserted on here — it is data, and a test that restated it would only fail the
next time they revise it. What is asserted is everything this codebase decided:
which form a prompt gets, which of their rules a machine can enforce, where the
rewrite sits relative to Auto Dialogue and the deterministic structuring, and
that every failure leaves the customer's own words alone.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from worker.adapters.base import AdapterJob
from worker.core.config import settings
from worker.dialogue.provider import DialogueRejected, DialogueUnavailable
from worker.prompt.ltx25 import LTX25_WORKFLOWS, apply_guidelines, enabled_for, rewrite
from worker.prompt.ltx25.compose import system_prompt, user_prompt
from worker.prompt.ltx25.forms import Form, guideline, select, terminology
from worker.prompt.ltx25.validate import check

SCENE = "A taxi driver picks up a passenger outside a rain-soaked station at night"

WRITTEN = (
    "A medium shot of a taxi driver at the wheel outside a rain-soaked station "
    "at night, lit by the amber wash of a streetlamp through the windscreen. "
    "The passenger settles into the back seat as the wipers sweep. The camera "
    "pushes in slowly and settles on the driver's eyes in the mirror. Rain "
    "drums on the roof and the indicator ticks steadily."
)


def _job(prompt: str = SCENE, workflow: str = "text-to-video", **execution) -> AdapterJob:
    return AdapterJob(
        job_id="ltx25-job",
        workflow_id=workflow,
        workflow_version="1",
        prompt=prompt,
        parameters={"duration": "15s"},
        execution={"runtime": "ltx_comfy", "ltx25_guidelines": True, **execution},
        workspace=Path("."),
    )


class _Writer:
    """A prompt writer that answers, or fails in a stated way."""

    name = "fake"

    def __init__(self, *answers: str, raises: Exception | None = None) -> None:
        self._answers = list(answers) or [WRITTEN]
        self._raises = raises
        self.calls = 0
        self.systems: list[str] = []
        self.users: list[str] = []

    async def write(self, request: Any) -> dict[str, Any]:
        self.calls += 1
        self.systems.append(request.system_text)
        self.users.append(request.user or "")
        if self._raises is not None:
            raise self._raises
        return {"prompt": self._answers[min(self.calls, len(self._answers)) - 1]}


# ── The pack is present and readable ───────────────────────────────────────


def test_every_guideline_file_ships_and_loads() -> None:
    """The pack is data in the wheel and the image. A missing file is a
    silently degraded prompt, so it is caught here rather than on a GPU."""
    for name in (
        "core",
        "single-shot",
        "multi-shot",
        "screenplay",
        "audio-dialogue",
        "dub-it",
        "validation",
    ):
        assert guideline(name).startswith("LTX-2.5"), name
    bank = terminology()
    assert bank["schema_version"] == "1.0"
    assert bank["editing_transitions"] and bank["continuous_camera_language"]


# ── Form selection: the client's routing, in their order ───────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A woman walks through a neon-lit market as the camera tracks beside her",
         Form.SINGLE_SHOT),
        ("A chef plates a dish, then a hard cut transitions to the diner tasting it",
         Form.MULTI_SHOT),
        ("A quick montage of the city waking up", Form.MULTI_SHOT),
        ('One says, "Did you hear?" The other replies, "I did."', Form.SCREENPLAY),
        ("A conversation between two colleagues in a lift", Form.SCREENPLAY),
    ],
)
def test_the_form_follows_what_the_customer_asked_for(text: str, expected: Form) -> None:
    assert select(text, workflow_id="text-to-video") == expected


def test_one_quoted_line_is_not_a_screenplay() -> None:
    """The commonest prompt on this platform is a single-shot clip that
    happens to speak. Reformatting those into screenplays would change nearly
    every generation, which is the opposite of what the pack is for."""
    one = 'A barista slides a cup across the counter and says, "Careful, it is hot."'
    assert select(one, workflow_id="text-to-video") == Form.SINGLE_SHOT


def test_image_to_video_stays_one_take_unless_a_cut_is_named() -> None:
    """The client's caveat, and `core.txt` §5: an opening image anchors one
    continuous take. Dialogue alone must not move it off single-shot."""
    talking = 'The woman in the photo turns and says, "You came back." She smiles, "I did."'
    assert select(talking, workflow_id="image-to-video", anchored=True) == Form.SINGLE_SHOT
    cut = "The subject smiles, then the view cuts to the crowd behind her"
    assert select(cut, workflow_id="image-to-video", anchored=True) == Form.MULTI_SHOT


def test_a_deployment_can_pin_the_form() -> None:
    assert select(SCENE, workflow_id="text-to-video", requested="multi-shot") == Form.MULTI_SHOT
    assert select(SCENE, workflow_id="text-to-video", requested="nonsense") == Form.SINGLE_SHOT


def test_dub_it_is_vendored_but_unreachable() -> None:
    """No workflow here replaces speech in a source video, so nothing selects
    the form. The guide is carried anyway: the day such a tool exists this is
    a routing line rather than a research task."""
    assert guideline("dub-it").startswith("LTX-2.5 DUB-IT")
    assert select("dub this clip into French", workflow_id="text-to-video") != Form.DUB_IT


# ── Validation: the half a machine can decide ──────────────────────────────


def test_an_edit_inside_a_single_take_is_an_error() -> None:
    report = check("She turns, then a hard cut transitions to the street.", Form.SINGLE_SHOT)
    assert report.result == "INVALID"
    assert [f.rule for f in report.errors] == ["F1"]


def test_a_multi_shot_prompt_must_name_its_cuts() -> None:
    report = check("A chef plates a dish. The diner tastes it.", Form.MULTI_SHOT)
    assert [f.rule for f in report.errors] == ["F2"]


def test_generation_settings_may_not_reach_the_prompt() -> None:
    """`core.txt` §4 / `validation.txt` §3. A prompt that states the canvas
    spends conditioning on something the graph already holds."""
    report = check("A 16:9 shot at 24 fps with seed 42, a chef plates a dish.", Form.SINGLE_SHOT)
    assert {f.rule for f in report.errors} == {"S3"}


def test_ordinary_english_is_not_mistaken_for_a_setting() -> None:
    """The check has to survive prose. "She steps off the kerb" is not a
    sampler setting, and a rule that fired on it would reject good prompts
    forever."""
    clean = (
        "She steps off the kerb into four lanes of traffic as a lorry fades "
        "from view behind her, and the score of the match plays on a radio."
    )
    assert not check(clean, Form.SINGLE_SHOT).errors


def test_dialogue_is_never_invented() -> None:
    """The likeliest way a rewriter damages this product: the customer asked
    for a scene, the writer decided it would be better with a line in it."""
    report = check('The chef says, "Service!"', Form.SINGLE_SHOT, dialogue_allowed=False)
    assert [f.rule for f in report.errors] == ["A7"]


def test_a_supplied_line_must_come_back_word_for_word() -> None:
    supplied = ("Service!",)
    assert not check(
        'The chef says, "Service!" as steam rises.', Form.SINGLE_SHOT, supplied_lines=supplied
    ).errors
    paraphrased = check(
        'The chef says, "Order up!" as steam rises.', Form.SINGLE_SHOT, supplied_lines=supplied
    )
    assert [f.rule for f in paraphrased.errors] == ["A7"]


def test_a_silent_video_carries_no_sound_direction() -> None:
    report = check("Pans hiss and music swells.", Form.SINGLE_SHOT, sound_on=False)
    assert [f.rule for f in report.errors] == ["A4"]


def test_a_prompt_made_of_prohibitions_is_a_warning_not_an_error() -> None:
    """`validation.txt` §11. The universal negative is submitted separately;
    restating it in the positive box is waste, not a failure."""
    report = check(
        "A chef with no extra fingers, no blur, no watermark, no text, never "
        "changing identity, without flicker, plating a dish.",
        Form.SINGLE_SHOT,
    )
    assert report.result == "VALID_WITH_WARNINGS"
    assert [f.rule for f in report.warnings] == ["P11"]


# ── What the writer is told ────────────────────────────────────────────────


def test_only_the_selected_form_guide_is_sent() -> None:
    """The client was explicit that this is a selection, not a concatenation:
    every rule and technical term in every prompt is the failure mode."""
    text = system_prompt(Form.SINGLE_SHOT)
    assert "LTX-2.5 SINGLE-SHOT RULES" in text
    assert "LTX-2.5 MULTI-SHOT RULES" not in text
    assert "LTX-2.5 SCREENPLAY-STYLE RULES" not in text
    assert "LTX-2.5 DUB-IT RULES" not in text
    assert "LTX-2.5 PROMPTING CORE" in text


def test_the_vocabulary_is_offered_as_a_menu() -> None:
    """`terminology.json` is a bank, and a writer handed all of it writes
    prompts made of it. The camera and edit lists are binding; the rest is
    explicitly optional."""
    text = system_prompt(Form.SINGLE_SHOT)
    assert "MENU, not a checklist" in text
    assert "Never add a subject, style, effect or sound" in text
    assert "no phrase from `editing_transitions` may appear" in text


def test_a_silent_job_is_told_so_instead_of_being_sent_the_audio_guide() -> None:
    text = system_prompt(Form.SINGLE_SHOT, sound_on=False)
    assert "THIS VIDEO IS SILENT" in text
    assert "LTX-2.5 AUDIO AND DIALOGUE RULES" not in text


def test_the_customers_own_words_are_quoted_not_summarised() -> None:
    text = user_prompt(SCENE, Form.SINGLE_SHOT)
    assert SCENE in text
    assert "never change the subject, the setting, the action or the intent" in text


# ── The seam ───────────────────────────────────────────────────────────────


async def test_a_rewritten_job_carries_the_new_prompt() -> None:
    writer = _Writer()
    job = await apply_guidelines(_job(), providers=[writer])
    assert job.prompt == WRITTEN
    assert writer.calls == 1


async def test_a_rewritten_job_stands_the_deterministic_structuring_down() -> None:
    """`core.txt` §7 and `validation.txt` §11: the rewritten prompt already
    carries its own continuity, and `worker/longform/enhance.py` would append
    a second copy of it."""
    job = await apply_guidelines(
        _job(prompt_structuring=True), providers=[_Writer()]
    )
    assert job.execution["prompt_structuring"] is False


async def test_a_job_that_is_not_rewritten_keeps_its_structuring() -> None:
    job = _job(prompt_structuring=True)
    unchanged = await apply_guidelines(job, providers=[_Writer(raises=RuntimeError("x"))])
    assert unchanged.prompt == SCENE
    assert unchanged.execution["prompt_structuring"] is True


@pytest.mark.parametrize(
    "failure",
    [DialogueUnavailable("no key"), DialogueRejected("nonsense"), RuntimeError("something")],
)
async def test_every_failure_renders_the_customers_own_prompt(failure: Exception) -> None:
    """Fail open, always. Unlike Auto Dialogue there is no explicit request to
    honour — the customer never asked for this — so there is no promise to
    break by falling back to their words."""
    job = await apply_guidelines(_job(), providers=[_Writer(raises=failure)])
    assert job.prompt == SCENE


async def test_no_writer_at_all_renders_the_customers_own_prompt() -> None:
    assert (await apply_guidelines(_job(), providers=[])).prompt == SCENE


async def test_an_invalid_prompt_gets_one_corrective_round() -> None:
    """The dialogue writer's measured shape: a strict validator refuses a
    first answer more often than not, and one round naming the broken rule
    turns most of those into a pass."""
    bad = "She turns, then a hard cut transitions to the street."
    writer = _Writer(bad, WRITTEN)
    job = await apply_guidelines(_job(), providers=[writer])
    assert writer.calls == 2
    assert "rejected by the validator" in writer.users[1]
    assert "F1" in writer.users[1]
    assert job.prompt == WRITTEN


async def test_a_writer_that_stays_invalid_falls_open() -> None:
    bad = "She turns, then a hard cut transitions to the street."
    writer = _Writer(bad, bad)
    job = await apply_guidelines(_job(), providers=[writer])
    assert writer.calls == 2
    assert job.prompt == SCENE


async def test_a_writer_that_invents_dialogue_is_refused() -> None:
    """End to end, because this is the fault with the worst consequence: a
    video that says words the customer never wrote."""
    invented = 'The taxi driver says, "Where to tonight?" as the wipers sweep.'
    writer = _Writer(invented, invented)
    job = await apply_guidelines(_job(), providers=[writer])
    assert job.prompt == SCENE


async def test_a_customers_own_quoted_line_survives_the_rewrite() -> None:
    spoken = 'A taxi driver picks up a passenger and says, "Where to tonight?"'
    kept = (
        "A medium shot of a taxi driver at the wheel at night as rain streaks "
        'the windscreen. He turns and says, "Where to tonight?" while the '
        "wipers sweep and the indicator ticks."
    )
    writer = _Writer(kept)
    job = await apply_guidelines(_job(spoken), providers=[writer])
    assert job.prompt == kept
    assert "THE SPOKEN LINES ARE FIXED" in writer.users[0]


# ── Where it does and does not run ─────────────────────────────────────────


async def test_it_is_off_unless_a_deployment_asks() -> None:
    plain = AdapterJob(
        job_id="j",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt=SCENE,
        parameters={},
        execution={"runtime": "ltx_comfy"},
        workspace=Path("."),
    )
    assert enabled_for(plain) is False
    writer = _Writer()
    assert (await apply_guidelines(plain, providers=[writer])).prompt == SCENE
    assert writer.calls == 0


async def test_the_deployment_default_can_turn_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ltx25_guidelines_enabled", True)
    plain = replace(_job(), execution={"runtime": "ltx_comfy"})
    assert enabled_for(plain) is True


@pytest.mark.parametrize("workflow", ["extend-video", "video-to-video", "character-replacement",
                                      "music-video"])
async def test_workflows_whose_prompt_is_not_a_scene_are_left_alone(workflow: str) -> None:
    """Extend continues a clip, Video to Video restyles one, Character
    Replacement's prompt is a character description inside the pack's own lead
    sentence, and Music Video has a song. None of them is a standalone scene,
    and rewriting one to the rules of a scene breaks what it is part of."""
    assert workflow not in LTX25_WORKFLOWS
    writer = _Writer()
    job = await apply_guidelines(_job(workflow=workflow), providers=[writer])
    assert job.prompt == SCENE
    assert writer.calls == 0


async def test_director_mode_keeps_its_own_prompt() -> None:
    """Two writers for one prompt is the failure the dialogue module refuses
    for the same reason."""
    job = _job()
    director = replace(job, parameters={**job.parameters, "prompt_mode": "director"})
    writer = _Writer()
    assert (await apply_guidelines(director, providers=[writer])).prompt == SCENE
    assert writer.calls == 0


async def test_a_silent_job_is_never_given_a_soundtrack() -> None:
    """The whole path, with sound off: the writer is told, and a prompt that
    directs sound anyway is refused rather than rendered."""
    job = _job()
    silent = replace(job, parameters={**job.parameters, "sound": False})
    noisy = "A taxi waits at the kerb as music swells over the rain."
    writer = _Writer(noisy, noisy)
    result = await apply_guidelines(silent, providers=[writer])
    assert "THIS VIDEO IS SILENT" in writer.systems[0]
    assert "LTX-2.5 AUDIO AND DIALOGUE RULES" not in writer.systems[0]
    assert result.prompt == SCENE


def test_the_silent_check_is_a_backstop_not_a_guarantee() -> None:
    """Stated because the limit is real and worth knowing before trusting it.

    "Rain drums on the roof" is an audio direction and also a physical
    description; no word list separates the two. The instruction does the
    work — a silent job is told plainly that nothing is heard — and this
    check only catches what survives it. Widening the list until this case
    fails would start rejecting sound-free prompts, and a false ERROR throws
    away a good rewrite.
    """
    subtle = "Rain drums on the roof of the waiting taxi."
    assert not check(subtle, Form.SINGLE_SHOT, sound_on=False).errors
    blatant = "Rain drums on the roof while music swells."
    assert [f.rule for f in check(blatant, Form.SINGLE_SHOT, sound_on=False).errors] == ["A4"]


async def test_the_form_and_verdict_are_available_without_a_render() -> None:
    written = await rewrite(_job(), providers=[_Writer()])
    assert written is not None
    prompt, form, report = written
    assert prompt == WRITTEN
    assert form is Form.SINGLE_SHOT
    assert report.result == "VALID"
