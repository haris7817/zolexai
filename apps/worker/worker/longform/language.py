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


#: Words that mean the customer asked for talking. Speech VERBS plus the nouns
#: that name a talking format, because "a podcast interview" never conjugates
#: anything. Quoted text counts too and is matched separately.
_SPEECH_WORDS = (
    "speak", "speaks", "speaking", "talk", "talks", "talking",
    "say", "says", "saying", "said", "tell", "tells", "telling",
    "explain", "explains", "explaining", "describe", "describes", "describing",
    "announce", "announces", "announcing", "address", "addresses", "addressing",
    "argue", "argues", "arguing", "answer", "answers", "answering",
    "ask", "asks", "asking", "reply", "replies", "replying",
    "greet", "greets", "greeting", "whisper", "whispers", "whispering",
    "shout", "shouts", "shouting", "recount", "recounts", "recounting",
    "sing", "sings", "singing", "narrate", "narrates", "narrating",
    "chat", "chats", "chatting", "recite", "recites", "reciting",
    "conversation", "dialogue", "dialog", "interview", "podcast",
    "monologue", "speech", "voice-over", "voiceover", "narration",
    "presenter", "anchor", "host", "vlog", "testimonial",
)

_SPEECH_RE = re.compile(r"\b(?:" + "|".join(_SPEECH_WORDS) + r")\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“‘'][^\"”’']{3,}[\"”’']")


def implies_speech(prompt: str) -> bool:
    """Whether the customer's own words ask for anyone to talk.

    Deliberately generous about what counts and deliberately dumb about how it
    decides. Getting this wrong in the "yes" direction costs a sentence saying
    speech should sound natural, in a video with no speech in it. Getting it
    wrong in the "no" direction costs the ambience sentence in a video that
    does talk, which is the milder of the two only because the speech itself
    still happens.
    """
    text = str(prompt or "")
    return bool(_SPEECH_RE.search(text) or _QUOTED_RE.search(text))


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
    if str(parameters.get("sound", True)).strip().lower() in ("false", "no", "off", "0"):
        return ""
    if not implies_speech(prompt):
        # "The sounds the scene itself makes" and not "the scene's ambience",
        # which is what this said first. Ambience MEANS room tone, and room
        # tone is quiet by definition — the Director research measured its
        # described-silence tail at -45 dBFS, inaudible on a phone. Asking a
        # gun being raised for "ambience" gets a near-silent video, which the
        # client reported the same evening as "video don't have the sound".
        #
        # This keeps the property that matters — the soundtrack has an owner,
        # so the model stops reading the caption aloud to fill it — while
        # pointing that owner at events rather than at the room: footsteps,
        # traffic, a door, whatever the scene is actually doing.
        return "The soundtrack is the sounds the scene itself makes."
    language = _resolve(parameters, execution, plan_language)
    if not language:
        return "Any talking is spoken aloud by the people on screen, in natural conversational voices."
    # "by the people on screen" is the load-bearing half: it hands the audio to
    # someone visible, which is what the caption was doing in their absence.
    # Not "to each other" — one presenter talking to camera is the common case
    # and that phrasing invents a second person for them to talk to.
    return (
        f"Any talking is spoken aloud by the people on screen in {language}, "
        "in natural conversational voices."
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
