"""When a video should speak, and what the prompt must say for it to.

Adapted from the client's `ltx25_auto_dialogue_backend` pack (7 Sep 2026).
Their `auto_dialogue.py` is the source of this module's policy — the two
pattern sets, the fail-open posture, and the rule that a prompt which already
carries dialogue or explicitly refuses it is returned untouched. Those are
kept as they wrote them.

**What is not kept is the single line.** The pack writes one spoken line per
section, which is right for the eight-second section it was written against
and wrong for anything longer on this runtime. `worker/director/plan.py`
records the measurement: across nine GPU renders on 18-19 Aug 2026, plans
below roughly 0.2 lines per second were the ones that failed, and a 15-second
clip carrying two lines echoed its last word. Dead air is not neutral — given
seconds the prompt says nothing about, the model fills them by repeating a
line it already spoke or by reading the caption's own prose aloud. So the
number of lines and the word budget come from `target_spoken_lines` and
`speech_budget`, the functions that already carry that measurement, rather
than from a second set of constants that would drift away from them.

## Why writing quoted lines is the whole feature

`worker/longform/language.py` already decides the soundtrack's owner, and it
turns on one thing: whether the customer's prompt contains quoted words. With
none, it tells the model plainly that *"No one speaks"* — the measured fix for
the model narrating its own caption. With them, it licenses exactly those
words, spoken once, in a stated language.

That mechanism is complete and it works. What it has never had is a way to
get quoted words into a prompt the customer wrote without any. This module is
that way, and nothing downstream needed changing to receive it: the enriched
prompt takes the path a prompt with dialogue has always taken.

## Where it applies

Text to Video, Image to Video and Text to Video HD — single-pass generation,
where one plan covers the whole soundtrack. Deliberately not Extend Video or
Video to Video: their prompts describe a continuation or a restyle rather than
a scene to invent, which is the same line `_DIRECTOR_WORKFLOWS` draws for the
same reason. Not Music Video, which has a song.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from worker.adapters.base import AdapterJob
from worker.director.plan import (
    MAX_CHARACTERS,
    speech_budget,
    spoken_line_budget,
    target_spoken_lines,
)
from worker.longform.language import supplied_dialogue

#: The workflows a generated line may reach. One pass, one scene, one plan.
AUTO_DIALOGUE_WORKFLOWS = frozenset(
    {"text-to-video", "image-to-video", "text-to-video-hd"}
)

#: The client's pack, verbatim. A customer who asks for silence gets silence,
#: and the ask is honoured before any model is consulted.
_NO_DIALOGUE_PATTERNS = (
    r"\bno dialogue\b",
    r"\bwithout dialogue\b",
    r"\bno speech\b",
    r"\bwithout speech\b",
    r"\bno voice(?:over)?\b",
    r"\bsilent scene\b",
    r"\bambience only\b",
    r"\bsound effects only\b",
)

#: Also the client's pack. Broader than this codebase's own `supplied_dialogue`
#: — it fires on a speech VERB ("a woman explains the product") where ours
#: requires quoted words — and the breadth is right for THIS question. Ours
#: answers "may the model speak?", where naming a verb without words is the
#: measured cause of invented speech. This one answers "did the customer
#: already have a say in the dialogue?", and someone who wrote a speech verb
#: did. Both are checked, and either one is enough to stand back.
_EXISTING_DIALOGUE_PATTERNS = (
    r"\bdialogue\s*:",
    r"\bvoice[ -]?over\s*:",
    r"\bnarration\s*:",
    r"\b(?:says?|speaks?|whispers?|shouts?|yells?|asks?|replies?|answers?)\b",
    r"\blip[ -]?sync\b",
    r"\bspoken line\b",
)

_SOUND_OFF = frozenset({"false", "no", "off", "0"})


@dataclass(frozen=True)
class Speaker:
    """One visible person who may be given words."""

    id: str
    description: str
    """The noun phrase the prompt refers to them by: "the taxi driver"."""

    voice: str = ""
    """Audible manner, woven in the first time they speak: "low and weary"."""


@dataclass(frozen=True)
class Line:
    speaker: str
    text: str
    delivery: str = ""


@dataclass(frozen=True)
class Dialogue:
    speakers: tuple[Speaker, ...]
    lines: tuple[Line, ...]
    language: str = ""

    @property
    def words(self) -> int:
        return sum(len(line.text.split()) for line in self.lines)

    def speaker(self, speaker_id: str) -> Speaker:
        for entry in self.speakers:
            if entry.id == speaker_id:
                return entry
        raise KeyError(speaker_id)


# ── Whether to ask at all ──────────────────────────────────────────────────


def prompt_forbids_dialogue(prompt: str) -> bool:
    return any(re.search(p, prompt, re.IGNORECASE) for p in _NO_DIALOGUE_PATTERNS)


def prompt_has_dialogue(prompt: str) -> bool:
    return any(re.search(p, prompt, re.IGNORECASE) for p in _EXISTING_DIALOGUE_PATTERNS)


def sound_is_on(job: AdapterJob) -> bool:
    """Whether this video will keep its audio.

    Both switches, because either one silences the result and writing lines
    for a soundtrack that gets thrown away spends a language model call to
    make the picture worse — every word in a prompt moves the image.
    """
    if not job.execution.get("soundscape", True):
        return False
    return str(job.parameters.get("sound", True)).strip().lower() not in _SOUND_OFF


def skip_reason(job: AdapterJob, seconds: float, *, enabled: bool) -> str:
    """Why this job gets no generated dialogue, or "" to go ahead.

    A string rather than a bool so the log says which gate closed. Every one
    of these is a reason to leave the prompt exactly as the customer wrote it.
    """
    if not enabled:
        return "disabled"
    if job.workflow_id not in AUTO_DIALOGUE_WORKFLOWS:
        return "workflow_not_eligible"
    if str(job.parameters.get("prompt_mode") or "").strip().lower() == "director":
        # Director mode plans its own dialogue, with locks and exits this
        # module has no idea about. Two planners writing lines for one video
        # is the failure the client's pack was built to prevent.
        return "director_mode_owns_the_dialogue"
    if not sound_is_on(job):
        return "sound_off"
    prompt = job.prompt.strip()
    if not prompt:
        return "empty_prompt"
    if prompt_forbids_dialogue(prompt):
        return "forbidden_by_prompt"
    if supplied_dialogue(prompt) or prompt_has_dialogue(prompt):
        return "dialogue_already_present"
    if seconds < 5.0:
        # Under `ESTABLISH_SECONDS` plus a line's worth of air there is no
        # speakable window; `speech_budget` agrees, returning almost nothing.
        return "too_short_to_speak"
    return ""


# ── What to ask for ────────────────────────────────────────────────────────


def word_budget(seconds: float) -> int:
    """Spoken words this clip can hold — the measured pacing ceiling."""
    return max(4, speech_budget(seconds))


def line_target(seconds: float) -> int:
    """Lines to aim for, from the density that separated clean renders."""
    return target_spoken_lines(seconds)


def line_ceiling(seconds: float) -> int:
    return spoken_line_budget(seconds)


def speaker_ceiling() -> int:
    return MAX_CHARACTERS


# ── What the prompt ends up saying ─────────────────────────────────────────


def _speech_verb(line: Line, speaker: Speaker, already_spoke: set[str]) -> str:
    """ "says in a low and weary voice" — the compiler's shape, reused.

    Per-line delivery outranks the speaker's standing voice; the standing
    voice is spent the first time they speak so the voice itself is
    established once rather than restated at every line.
    """
    manner = (line.delivery or "").strip().rstrip(".")
    if not manner and speaker.id not in already_spoke:
        manner = speaker.voice.strip().rstrip(".")
    already_spoke.add(speaker.id)
    if not manner:
        return "says"
    manner = re.sub(r"\s+voice$", "", manner, flags=re.IGNORECASE)
    article = "an" if manner[:1].lower() in "aeiou" else "a"
    return f"says in {article} {manner} voice"


def speech_rule(language: str = "") -> str:
    """The sentence that stops an injected line being stretched or looped.

    Lifted verbatim from `soundscape_clause`'s supplied-dialogue branch, whose
    measurements it carries: a five-word line in a twenty-second video loops
    (a client got "Never mess with the family" repeated for the full twenty
    seconds on 28 Aug 2026), because the model has audio time to fill and only
    one thing in the prompt to fill it with. "A single time" is what stops the
    repeat, and naming the tail's sounds is what stops the stretch.

    Needed only where no soundscape clause runs downstream. On the ComfyUI
    text-to-video path one does, and emitting this as well would be the same
    rule stated twice.
    """
    spoken = f" in {language}" if language else ""
    return (
        f"The only words anyone speaks are the quoted lines above, spoken"
        f"{spoken} by the people on screen in natural conversational voices. "
        "Each line is spoken a single time and is not repeated; for the rest "
        "of the video the only sounds are the ones the scene itself makes."
    )


def compose(prompt: str, dialogue: Dialogue, *, add_speech_rule: bool = False) -> str:
    """The customer's prompt, verbatim, then the lines as quoted speech.

    The customer's text is never rewritten or reordered — the result CONTAINS
    it as its first block, byte for byte, which is the same contract
    `structure_prompt` holds and for the same reason: a prompt the customer
    can no longer recognise is a prompt they cannot debug.

    The sentences are shaped exactly like the Director compiler's, because
    that shape is the one measured to be delivered verbatim rather than
    narrated. Nothing here names the language: `soundscape_clause` does that,
    once, downstream, and two sentences naming it would be two chances to
    disagree.
    """
    if not dialogue.lines:
        return prompt
    spoke: set[str] = set()
    sentences = []
    for line in dialogue.lines:
        try:
            speaker = dialogue.speaker(line.speaker)
        except KeyError:
            continue
        verb = _speech_verb(line, speaker, spoke)
        sentences.append(f'{speaker.description} {verb}, "{_spoken(line.text)}"')
    if not sentences:
        return prompt
    block = " ".join(_sentence(_capfirst(s)) for s in sentences)
    if add_speech_rule:
        block = f"{block} {speech_rule(dialogue.language)}"
    return f"{prompt.rstrip()}\n\n{block}"


def _spoken(text: str) -> str:
    """The line as it appears inside the quotation marks.

    Its terminal punctuation goes INSIDE the quote, which is where a reader
    and a text encoder both expect it. The alternative — closing the quote and
    then adding a stop — produces `"Where to?".`, and the model reads that
    trailing stop as part of what it is meant to say.
    """
    body = text.strip()
    return body if not body or body[-1] in ".!?" else body + "."


def _sentence(text: str, terminal: str = ".") -> str:
    """Terminate a clause without doubling a stop.

    `_sentence` in the Director compiler, reproduced: a clause ending in a
    closing quote is already finished, and "Where to?"." is a stop the model
    reads as part of the line.
    """
    body = text.strip()
    if not body:
        return ""
    if body[-1] in '.!?"':
        return body
    return body + terminal


def _capfirst(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


__all__ = [
    "AUTO_DIALOGUE_WORKFLOWS",
    "Dialogue",
    "Line",
    "Speaker",
    "compose",
    "line_ceiling",
    "line_target",
    "prompt_forbids_dialogue",
    "prompt_has_dialogue",
    "skip_reason",
    "sound_is_on",
    "speech_rule",
    "speaker_ceiling",
    "word_budget",
]
