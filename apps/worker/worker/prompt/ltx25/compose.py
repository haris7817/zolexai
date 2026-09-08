"""The instruction the prompt writer is given: which guidelines, in what order.

`core.txt` is always sent, then exactly one form guide, then
`audio-dialogue.txt` whenever the video has a soundtrack, then the parts of
`validation.txt` the writer can act on. The client was explicit that this is
a selection and not a concatenation — *"it must select the correct section,
not insert every rule and technical term into every prompt"* — and
`terminology.json` is the sharpest case: it is a vocabulary bank, and a
writer handed all of it writes prompts made of it.

So the bank is sent in two halves. The camera and edit vocabularies go every
time, because they are what makes the form boundary decidable — the
difference between "the camera pushes in" and "a hard cut transitions to" is
the difference between two of the four forms. The style, lighting, texture
and sound banks go as an explicitly optional menu, with the instruction to
take only what the scene needs and to invent nothing to use them up.
"""

from __future__ import annotations

import json

from worker.dialogue.provider import JSON_BEGIN, JSON_END
from worker.prompt.ltx25.forms import Form, guideline, terminology

#: The provider chain's own markers, not new ones: `_extract_json` looks for
#: exactly these before it falls back to hunting for a brace, and a second
#: pair would work only by that fallback.
BEGIN = JSON_BEGIN
END = JSON_END


def _vocabulary(form: Form) -> str:
    bank = terminology()
    rules = bank.get("semantic_rules", {})
    always = {
        "continuous_camera_language": bank.get("continuous_camera_language", []),
        "editing_transitions": bank.get("editing_transitions", []),
    }
    optional = {
        key: bank[key]
        for key in ("style_categories", "visual_details", "sound_and_voice",
                    "film_characteristics", "scale_indicators",
                    "pacing_and_temporal_effects", "visual_effects")
        if key in bank
    }
    parts = [
        "CONTROLLED VOCABULARY",
        "These two lists decide the form and must be used precisely:",
        json.dumps(always, indent=2),
        "What each means:",
        json.dumps(rules, indent=2),
        "",
        "The lists below are a MENU, not a checklist. Take only the few terms "
        "the scene actually needs and ignore the rest. Never add a subject, "
        "style, effect or sound to the scene merely to use a term from it.",
        json.dumps(optional, indent=2),
    ]
    if form is Form.MULTI_SHOT:
        parts.insert(
            2,
            "This prompt HAS cuts: every one of them must be named with a "
            "phrase from `editing_transitions`.",
        )
    elif form is Form.SINGLE_SHOT:
        parts.insert(
            2,
            "This prompt is ONE take: use `continuous_camera_language` only, "
            "and no phrase from `editing_transitions` may appear.",
        )
    return "\n".join(parts)


def system_prompt(form: Form, *, sound_on: bool = True) -> str:
    """Everything the writer is bound by, assembled for this one job."""
    blocks = [
        "You are a prompt writer for the LTX 2.5 video model. You rewrite a "
        "customer's description into a single finished generation prompt that "
        "follows the guidelines below exactly.",
        guideline("core"),
        f"THE FORM FOR THIS PROMPT IS {form.name}. Its rules:",
        guideline(form.guide),
    ]
    if sound_on:
        blocks.append(guideline("audio-dialogue"))
    else:
        blocks.append(
            "THIS VIDEO IS SILENT. Write no ambience, music, speech, singing "
            "or sound effects of any kind, and no quoted dialogue. Do not "
            "mention sound to say there is none."
        )
    blocks.append(_vocabulary(form))
    blocks.append(guideline("validation"))
    blocks.append(
        "Return JSON only, between "
        f"{BEGIN} and {END}, with one key: prompt (the finished prompt text, "
        "and nothing else — no heading, no explanation, no commentary). "
        "Output the finished prompt, not an account of how you wrote it."
    )
    return "\n\n".join(blocks)


def user_prompt(
    text: str,
    form: Form,
    *,
    language: str = "",
    supplied_lines: tuple[str, ...] = (),
    max_speakers: int = 0,
    dialogue_allowed: bool = True,
    anchored: bool = False,
) -> str:
    """The job itself: the customer's words, and what may not be changed.

    The customer's text is quoted rather than paraphrased into instructions,
    and the constraints that follow are the ones a rewrite can silently
    break — their meaning, their dialogue, their language, and the fact that
    an image already answers half the question.
    """
    parts = [
        "Rewrite this customer description as one finished LTX 2.5 prompt.",
        "",
        "CUSTOMER DESCRIPTION:",
        text.strip(),
        "",
        "CONSTRAINTS",
        "- Keep what they asked for. You may add craft — shot scale, lighting, "
        "camera behaviour, audible sources, continuity anchors — but never "
        "change the subject, the setting, the action or the intent.",
        "- Everything they were specific about survives verbatim: names, "
        "numbers, colours, wardrobe, brands, places.",
    ]
    if anchored:
        parts.append(
            "- An INPUT IMAGE supplies the first frame. Do not re-describe what "
            "is already visible in it. Write what changes: performance, "
            "movement, camera, environment, audio, and the ending state."
        )
    if not dialogue_allowed:
        parts.append(
            "- NOBODY SPEAKS. Write no quoted lines and invent no dialogue, "
            "narration or voice-over. If the description implies conversation, "
            "show it through visible behaviour instead."
        )
    elif supplied_lines:
        listed = "; ".join(f'"{line}"' for line in supplied_lines)
        parts.append(
            f"- THE SPOKEN LINES ARE FIXED: {listed}. Reproduce each one word "
            "for word inside double quotation marks, keep it with the speaker "
            "it belongs to, add no line and drop none."
        )
        if max_speakers:
            parts.append(f"- At most {max_speakers} people speak. Add no further speakers.")
    if language:
        parts.append(
            f"- The spoken language is {language}. Do not translate any line "
            "away from it."
        )
    parts += [
        "",
        f"Return the JSON described above, between {BEGIN} and {END}.",
    ]
    return "\n".join(parts)


def retry_prompt(base: str, instruction: str) -> str:
    """The second and final ask, naming what failed.

    The dialogue writer's measured shape (8 Sep 2026): a writer's first
    answer misses a strict validator more often than not, and one corrective
    round that names the broken rule turns most of those into a pass.
    """
    return (
        f"{base}\n\nYour previous prompt was rejected by the validator. "
        f"{instruction} Fix exactly that, change nothing else, and return the "
        "complete JSON again."
    )


__all__ = ["BEGIN", "END", "retry_prompt", "system_prompt", "user_prompt"]
