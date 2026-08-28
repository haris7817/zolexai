"""What language the generated audio speaks.

Both video engines write their own audio from the same prompt that draws the
picture. Neither takes a separate audio prompt and neither takes a language
flag — LTX's command builder has `--audio-path` for conditioning and nothing
else, and H3's graph is driven entirely by its segment prompts. So the prompt
is the only lever there is, and until 28 Aug 2026 neither engine's standard
path said a single word about language.

That is not silence. It is the model choosing, and a customer put the result
plainly: *"What language you got the backend? Because everything i hit sound
in images comes with a language that is not english?"*

LTX's Director mode never showed the fault, which is the tell. Its plan puts
real quoted dialogue in the prompt, so the language rides in the words
themselves; `DirectorPlan.language` exists but has only ever reached a log
line. The standard path — the default, and what most customers use — had
neither the words nor the field.

This module lives apart from `prompts.py` for an unglamorous reason:
`prompts.py` imports `h3_prompts`, and both need what is here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Languages the video runtime's vendor documents as validated for generated
#: speech. Mirrors `DIALOGUE_LANGUAGES` in `worker.director.provider` and in
#: the API's workflow registry; the three lists must not drift apart.
SPOKEN_LANGUAGE_NAMES: dict[str, str] = {
    "english": "English",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "russian": "Russian",
}


def spoken_language_sentence(language: str | None) -> str:
    """One sentence fixing the language of any speech the model generates.

    Phrased as a CONDITIONAL, and that is the whole care in this function. A
    prompt that says "the woman speaks English" invents a speaking woman in a
    silent scene, because a text encoder cannot hear an instruction as
    optional and naming a noun is how you summon it. "If anyone speaks" binds
    the language to speech the scene was going to contain anyway and adds
    nothing to one that was not. It is the same shape as the music-video
    performance line ("if a person is visible singing, their mouth…"), which
    exists for the same reason.

    An unknown or absent language returns "" — the prompt then reads exactly
    as it did before this existed.
    """
    name = SPOKEN_LANGUAGE_NAMES.get(str(language or "").strip().lower())
    if name is None:
        return ""
    return f"If anyone speaks or sings, they do so in {name}."


def spoken_language_clause(
    parameters: Mapping[str, Any],
    execution: Mapping[str, Any],
    plan_language: str = "",
) -> str:
    """The language sentence for one job, or "" when it does not apply.

    Shared by both engines, because both had the same silence. Resolution,
    most specific first:

      1. the customer's `dialogue_language`, when they named a real one;
      2. a plan's own language, because a plan that quotes dialogue has
         already written it in that language and this sentence must not
         contradict the words beside it;
      3. `execution.spoken_language`, the deployment default.

    `auto` at step 1 is not an answer, it is the absence of one, and it falls
    through — an unstated language is the reported bug, not a neutral default.

    Skipped when the customer turned sound off: the result is muted after
    rendering either way, so the sentence would buy nothing, and it is not
    free — every word in the prompt moves the picture.
    """
    if str(parameters.get("sound", True)).strip().lower() in ("false", "no", "off", "0"):
        return ""
    requested = str(parameters.get("dialogue_language") or "").strip().lower()
    if requested in SPOKEN_LANGUAGE_NAMES:
        return spoken_language_sentence(requested)
    if str(plan_language or "").strip().lower() in SPOKEN_LANGUAGE_NAMES:
        return spoken_language_sentence(plan_language)
    return spoken_language_sentence(execution.get("spoken_language", "english"))
