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

import re
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


#: Quoted text in the customer's own prompt — the ONLY thing that licenses
#: generated speech. A speech VERB does not: "a woman explains the product"
#: describes what she is doing, not what she says, and a model given a verb
#: with no words invents them or reads the caption instead. That distinction
#: is the client's backend rule of 28 Aug 2026, and it is stricter than the
#: verb list this replaced.
_QUOTED_RE = re.compile(r"[\"“‘'][^\"”’']{3,}[\"”’']")


def supplied_dialogue(prompt: str) -> bool:
    """Whether the customer explicitly wrote words for someone to say."""
    return bool(_QUOTED_RE.search(str(prompt or "")))


def soundscape_clause(
    prompt: str,
    parameters: Mapping[str, Any],
    execution: Mapping[str, Any],
    plan_language: str = "",
) -> str:
    """One sentence giving this video's soundtrack an owner.

    The narration leak is the reason this is a soundscape and not just a
    language. Measured 18 Aug 2026 (`research-2026-08-18-director-idea-mode`,
    TC2): where a section's dialogue did not cover its window, the model
    filled the uncovered seconds by READING THE CAPTION'S OWN SENTENCES ALOUD
    as narration. Director mode fixed it by appending a described-silence beat
    — the compiler's note puts it exactly right, "so the soundtrack has an
    owner all the way to the last frame".

    Standard mode has no dialogue at all, so its window is uncovered end to
    end and the caption owns the whole soundtrack. Client, 28 Aug 2026: "the
    video came out good and I hit audio but what it is reading, I guess, is
    the prompt on the backend."

    So the sentence splits on what the customer actually asked for:

      * they asked for talking → the speech is named as the scene's own,
        spoken naturally, in a stated language. The audio has an owner and it
        is the person on screen.
      * they did not → the ambience is named as the whole soundtrack, which
        is the described-silence fix applied one layer out.

    Both are stated positively. This runtime has no negation mechanism, so
    "no narration" contributes the token *narration*; naming what the
    soundtrack IS crowds out what it must not be, and that is the only lever
    that has ever worked here.

    Returns "" when sound is off — the result is muted after rendering, and
    every word in the prompt moves the picture.
    """
    if not execution.get("soundscape", True):
        # The off switch, and it exists because this sentence was measured
        # doing harm. Image to Video, 28 Aug 2026: two renders at 15:50 and
        # 15:53 (360 frames, 576x1024, 47s each) were good; the 17:37 render
        # with the same geometry, same engine and the same 48s wall clock came
        # back with "the image is just moving". The only difference was this
        # sentence, telling the model the soundtrack was ambience — which is a
        # statement that the scene is quiet and still, and the picture obeyed.
        #
        # The narration leak it was written for is real and still unfixed
        # wherever this is off. That is a trade a deployment makes per
        # workflow, with a GPU A/B, and not one a default should make silently.
        return ""
    if str(parameters.get("sound", True)).strip().lower() in ("false", "no", "off", "0"):
        return ""

    dialogue = supplied_dialogue(prompt)
    if not dialogue:
        # THE RULE. Speech is generated only when the customer supplied the
        # words; with none supplied, the model is told plainly that nobody
        # speaks, and the soundtrack is handed to the scene instead.
        #
        # "No one speaks" is a negation, which this file otherwise avoids
        # because the runtime has no negation mechanism. It earns the
        # exception on evidence: Director mode's compiler emits exactly that
        # phrasing and its narration leak went away (research-2026-08-18,
        # TC2c). The general rule loses to the measurement.
        #
        # The second half is what stops it going silent. "Ambience" was the
        # first wording and ambience MEANS room tone — the Director research
        # measured its described-silence tail at -45 dBFS, inaudible on a
        # phone, and the client reported that same evening as "video dont
        # have the sound". Naming the sounds the scene MAKES points the
        # soundtrack at events: footsteps, traffic, a door, an engine.
        return "No one speaks. The only sounds are the ones the scene itself makes."

    language = _resolve(parameters, execution, plan_language)
    spoken = f" in {language}" if language else ""
    # Dialogue WAS supplied, so speech is licensed — but only these words,
    # only once, and only for as long as they take.
    #
    # "A single time" is Director's phrasing, and it is there for a measured
    # reason: a 60s render on 19 Aug spoke three of its fourteen lines twice.
    # Without it, a five-word line in a twenty-second video loops — a client
    # on 28 Aug 2026 got "Never mess with the family" repeated for the full
    # twenty seconds, which is the same uncovered-window failure as the
    # narration leak wearing a different hat: the model has audio time to
    # fill and only one thing in the prompt to fill it with.
    #
    # So the tail gets an owner in the same breath. Naming what the rest of
    # the video sounds like is what stops the line being stretched over it.
    return (
        f"The only words anyone speaks are the quoted lines above, spoken"
        f"{spoken} by the people on screen in natural conversational voices. "
        "Each line is spoken a single time and is not repeated; for the rest "
        "of the video the only sounds are the ones the scene itself makes."
    )


def _resolve(
    parameters: Mapping[str, Any],
    execution: Mapping[str, Any],
    plan_language: str = "",
) -> str:
    """The language name to use, most specific first, or "".

      1. the customer's `dialogue_language`, when they named a real one;
      2. a plan's own language, because a plan that quotes dialogue has
         already written it in that language and this must not contradict the
         words beside it;
      3. `execution.spoken_language`, the deployment default.

    `auto` at step 1 is not an answer, it is the absence of one, and it falls
    through — an unstated language is the reported bug, not a neutral default.
    """
    requested = str(parameters.get("dialogue_language") or "").strip().lower()
    if requested in SPOKEN_LANGUAGE_NAMES:
        return SPOKEN_LANGUAGE_NAMES[requested]
    planned = str(plan_language or "").strip().lower()
    if planned in SPOKEN_LANGUAGE_NAMES:
        return SPOKEN_LANGUAGE_NAMES[planned]
    default = str(execution.get("spoken_language", "english")).strip().lower()
    return SPOKEN_LANGUAGE_NAMES.get(default, "")
