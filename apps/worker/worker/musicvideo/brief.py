"""The customer's music-video prompt as a structured creative brief.

Client report, 9 Sep 2026: a customer asked for a Beverly Hills penthouse
full of hay bales, a hot-pink Lamborghini towing a tractor down a country
road, and a Nashville barn turned neon nightclub — and received a generic
country-pop performance video. Their diagnosis was right and the cause is in
their own package: `zolex_music_worker.director.expand_direction` matches the
prompt against a keyword→genre table and takes every location, palette and
camera move from a hard-coded profile. The customer's text is stored as
`original_direction` and never read again; `compile_shot_prompts` builds each
render prompt from tables alone. Not one noun the customer wrote reaches a
shot.

This module is the first half of the fix: read the prompt into a brief the
planner can be held to. `worker/musicvideo/enforce.py` is the second half,
where the brief is written into every shot and checked before any GPU time.

## Priority order (the client's, kept exactly)

  1. the customer's prompt — story, locations, characters, props, actions,
     camera, prohibitions;
  2. performer photos — identity;
  3. a reference video — camera, composition, pacing;
  4. lyrics and audio — timing, emotion, lip-sync;
  5. backend defaults — only what the customer left unsaid.

## Two extractors, one contract

The hosted writer (the same chain Auto Dialogue uses) reads the prompt into
JSON. Behind it, `heuristic_brief` does the same job with regular
expressions: every sentence is an event, "in a / at the …" phrases are
locations, a short list catches vehicles and instruments. It is cruder, and
that is fine, because the contract is not "understand the prompt" — it is
"the customer's own words reach every shot, in order". The heuristic keeps
the sentences verbatim, so it satisfies that by construction.

Concrete things the customer named — locations, vehicles, props, events —
are **mandatory** (the client's rule). Adjectives are preferred. Nothing the
customer asked for is ever optional.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from worker.core.logging import get_logger
from worker.dialogue.provider import (
    JSON_BEGIN,
    JSON_END,
    DialogueProvider,
    DialogueRejected,
    DialogueRequest,
    DialogueUnavailable,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class Event:
    """One thing the customer said happens, in the order they said it."""

    action: str
    location_index: int = 0
    """Index into `CreativeBrief.locations`; -1 when the event names none."""


@dataclass
class CreativeBrief:
    original_prompt: str
    theme: str = ""
    locations: list[str] = field(default_factory=list)
    """Mandatory, in story order."""
    characters: list[str] = field(default_factory=list)
    props: list[str] = field(default_factory=list)
    """Vehicles, instruments, objects — mandatory."""
    events: list[Event] = field(default_factory=list)
    """Chronological story beats — mandatory."""
    camera: str = ""
    look: str = ""
    prohibited: list[str] = field(default_factory=list)
    single_shot: bool = False
    """The customer asked for one continuous take: no planner cuts."""
    no_captions: bool = True
    source: str = "heuristic"

    @property
    def mandatory(self) -> list[str]:
        """Every string that must survive into at least one shot prompt."""
        seen: list[str] = []
        for item in [*self.locations, *self.props, *(e.action for e in self.events)]:
            text = item.strip()
            if text and text.casefold() not in {s.casefold() for s in seen}:
                seen.append(text)
        return seen

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mandatory"] = self.mandatory
        return data


# ── Single-shot detection ──────────────────────────────────────────────────

#: Client, 9 Sep 2026: "one continuous shot", "static camera", "same
#: framing" or "no cuts" must bypass the multi-shot planner.
_SINGLE_SHOT_PATTERNS = (
    r"\bone\s+continuous\s+(?:shot|take)\b",
    r"\bsingle\s+continuous\s+(?:shot|take)\b",
    r"\b(?:one|single)[- ]shot\b",
    r"\b(?:one|single)\s+take\b",
    r"\bcontinuous\s+take\b",
    r"\bunbroken\s+(?:shot|take)\b",
    r"\bno\s+cuts?\b",
    r"\bwithout\s+(?:any\s+)?cuts?\b",
    r"\bstatic\s+camera\b",
    r"\bsame\s+framing\b",
    r"\bfixed\s+(?:camera|frame|framing)\b",
    r"\block(?:ed)?[- ]off\s+(?:camera|shot)\b",
)


def wants_single_shot(prompt: str) -> bool:
    return any(re.search(p, prompt, re.IGNORECASE) for p in _SINGLE_SHOT_PATTERNS)


def forbids_captions(prompt: str) -> bool:
    """A customer who said "no captions" gets the caption policy locked.
    The default is also no captions — a music video with generated text on
    it is a defect — so this only records that they said so."""
    return bool(
        re.search(r"\bno\s+(?:captions?|subtitles?|(?:generated\s+)?text|on[- ]screen\s+text)\b",
                  prompt, re.IGNORECASE)
    )


# ── The heuristic extractor ────────────────────────────────────────────────

_LOCATION_RE = re.compile(
    r"\b(?:in|inside|at|across|through|within|down|along|to|into|onto)\s+"
    r"(?:a|an|the|this|that)\s+([^,.;:]{4,90}?)"
    r"(?=[,.;:]|\s+(?:where|while|as|and\s+\w+ing|then|before|after)\b|$)",
    re.IGNORECASE,
)

#: Things that must be on screen if the customer named them. Short on
#: purpose: it catches the concrete nouns a music video is most often about
#: and the heuristic keeps whole sentences anyway, so a missed prop is still
#: in the prompt — it is only not tracked as its own mandatory line.
_PROP_WORDS = (
    r"lamborghini|ferrari|porsche|mustang|cadillac|pickup|truck|tractor|"
    r"motorcycle|motorbike|bike|jeep|convertible|limousine|limo|helicopter|"
    r"jet|boat|yacht|horse|"
    r"guitar|banjo|fiddle|violin|piano|keyboard|drums?|drum kit|saxophone|"
    r"trumpet|microphone|mic|turntables?|"
    r"disco ball|mason jar|martini|champagne|moonshine|tablet|phone|"
    r"saddle|hay bales?|neon sign|bonfire|campfire|umbrella|cigarette"
)
_PROP_RE = re.compile(
    rf"\b((?:[a-z][a-z-]*\s+){{0,3}}(?:{_PROP_WORDS}))\b", re.IGNORECASE
)

#: Head nouns that follow a place preposition without being a place.
_NOT_A_PLACE_RE = re.compile(
    r"^(?:supplied\s+|original\s+|whole\s+|same\s+|current\s+)?"
    r"(?:song|track|music|beat|rhythm|tempo|bass|chorus|verse|lyrics?|melody|"
    r"camera|lens|frame|framing|audience|viewers?|crowd|screen|tablet|phone|"
    r"streams?|service|signal|bow|drop|climax|end|start|beginning|finish)\b",
    re.IGNORECASE,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"“])")

_PROHIBIT_RE = re.compile(
    r"\bno\s+([a-z][a-z\- ]{2,40}?)(?=[,.;]|\s+(?:and|or)\b|$)", re.IGNORECASE
)


def heuristic_brief(prompt: str) -> CreativeBrief:
    """A brief from the prompt's own sentences, with nothing invented.

    Every sentence becomes an event, verbatim. That is what guarantees the
    customer's words reach the shots even when no writer is available.
    """
    text = " ".join(prompt.split())
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    locations: list[str] = []
    for match in _LOCATION_RE.finditer(text):
        phrase = match.group(1).strip()
        # "into a mason jar" is a prop being used, not a place, and "to the
        # supplied song" is the soundtrack: a phrase whose head noun is on
        # the prop list, or is the song, the camera or the audience, is not a
        # location. Treating one as a place demotes the sentence around it
        # from a lock on every shot to a beat for one of them.
        if re.match(rf"^(?:{_PROP_WORDS})\b", phrase, re.IGNORECASE):
            continue
        if _NOT_A_PLACE_RE.match(phrase):
            continue
        if phrase.casefold() not in {loc.casefold() for loc in locations}:
            locations.append(phrase)

    props: list[str] = []
    for match in _PROP_RE.finditer(text):
        phrase = _clean_prop(match.group(1))
        if phrase and phrase.casefold() not in {p.casefold() for p in props}:
            props.append(phrase)

    events: list[Event] = []
    locks: list[str] = []
    for sentence in sentences:
        if _looks_like_direction_only(sentence) or _is_global_lock(sentence, locations):
            locks.append(sentence.rstrip("."))
            continue
        index = -1
        for i, location in enumerate(locations):
            if location.casefold() in sentence.casefold():
                index = i
                break
        events.append(Event(action=sentence.rstrip("."), location_index=index))
    # Events with no location inherit the last one seen: the story does not
    # teleport between sentences.
    current = 0
    fixed: list[Event] = []
    for event in events:
        if event.location_index >= 0:
            current = event.location_index
        fixed.append(Event(action=event.action, location_index=current))

    prohibited = [m.group(1).strip().rstrip(".") for m in _PROHIBIT_RE.finditer(text)]
    theme = (locks[0] if locks else (sentences[0] if sentences else text)).rstrip(".")[:200]

    return CreativeBrief(
        original_prompt=prompt,
        theme=theme,
        locations=locations,
        props=props,
        events=fixed,
        look="; ".join(locks[1:])[:400],
        prohibited=prohibited,
        single_shot=wants_single_shot(text),
        no_captions=True,
        source="heuristic",
    )


_LEADING_NOISE = re.compile(
    r"^(?:(?:shows?|drives?|pours?|dances?|switches|embodying|filled|strangely|beside|"
    r"into|onto|from|to|with|on|of|her|his|their|the|a|an|tiny|huge|massive)\s+)+",
    re.IGNORECASE,
)


def _clean_prop(phrase: str) -> str:
    """"drives a hot-pink Lamborghini" -> "hot-pink Lamborghini". The capture
    starts wherever the window before the noun starts; the noun phrase is
    what has to be on screen."""
    cleaned = _LEADING_NOISE.sub("", phrase.strip()).strip()
    return cleaned if len(cleaned) >= 3 else ""


_GLOBAL_LOCK_RE = re.compile(
    r"\b(?:remains?|stays?|consistent|throughout|identity|lip[- ]?sync|"
    r"music\s+video\s+(?:blending|combining|mixing|about))\b",
    re.IGNORECASE,
)


def _is_global_lock(sentence: str, locations: list[str]) -> bool:
    """A sentence about the whole video — "the same singer remains visually
    consistent throughout", "a music video blending X and Y" — is a lock on
    every shot, not a beat for one of them."""
    if any(loc.casefold() in sentence.casefold() for loc in locations):
        return False
    return bool(_GLOBAL_LOCK_RE.search(sentence))


def _looks_like_direction_only(sentence: str) -> bool:
    """A trailing sentence of adjectives — "Natural body movement, accurate
    lip-sync, beat-matched cuts…" — is style direction, not a story event.
    It still reaches the prompt through `look`/restrictions; it should not
    be handed to one shot as if something happens in it."""
    lowered = sentence.casefold()
    verbs = re.findall(r"\b(?:is|are|was|were|has|have|walks?|drives?|pours?|shows?|"
                       r"performs?|dances?|sings?|opens?|reveals?|transitions?|cuts?|"
                       r"switches?|balances?|loses?|stopped|stops?|emerges?|watches)\b",
                       lowered)
    return not verbs and lowered.count(",") >= 3


# ── The hosted extractor ───────────────────────────────────────────────────

_SYSTEM = (
    "You turn a customer's music-video description into a structured creative "
    "brief. Read only what they wrote; invent nothing. Return JSON only, between "
    f"{JSON_BEGIN} and {JSON_END}, with keys: theme (one line), locations (array "
    "of strings, in story order, each a concrete place the customer named, "
    "quoting their own words), characters (array of strings), props (array of "
    "strings: vehicles, instruments, objects they named, their words), events "
    "(array of objects {action: string in their words, location_index: integer "
    "index into locations}, in the order the story happens), camera (string, "
    "their camera instructions or empty), look (string, lighting/colour/style "
    "words they used or empty), prohibited (array of strings of things they said "
    "must not appear), single_shot (boolean: true only if they asked for one "
    "continuous shot, no cuts, static camera or same framing). Keep every "
    "location, vehicle, prop and story event they named — omit none. Do not "
    "add locations or events from the genre; if they gave one location, "
    "locations has one entry."
)


def _user(prompt: str) -> str:
    return f"CUSTOMER DESCRIPTION:\n{prompt.strip()}\n\nReturn the JSON described above."


def _parse(raw: dict[str, Any], prompt: str) -> CreativeBrief:
    def strings(key: str) -> list[str]:
        value = raw.get(key) or []
        if not isinstance(value, list):
            raise DialogueRejected(f"brief.{key} is not a list")
        out: list[str] = []
        for item in value:
            text = " ".join(str(item or "").split()).strip()
            if text and text.casefold() not in {s.casefold() for s in out}:
                out.append(text)
        return out

    locations = strings("locations")
    events: list[Event] = []
    for item in raw.get("events") or []:
        if isinstance(item, dict):
            action = " ".join(str(item.get("action") or "").split()).strip()
            try:
                index = int(item.get("location_index", 0))
            except (TypeError, ValueError):
                index = 0
        else:
            action, index = " ".join(str(item).split()).strip(), 0
        if action:
            bounded = min(max(index, 0), max(len(locations) - 1, 0))
            events.append(Event(action=action.rstrip("."), location_index=bounded))
    if not events and not locations:
        raise DialogueRejected("brief has neither events nor locations")

    return CreativeBrief(
        original_prompt=prompt,
        theme=" ".join(str(raw.get("theme") or "").split())[:200],
        locations=locations,
        characters=strings("characters"),
        props=strings("props"),
        events=events,
        camera=" ".join(str(raw.get("camera") or "").split())[:300],
        look=" ".join(str(raw.get("look") or "").split())[:300],
        prohibited=strings("prohibited"),
        single_shot=bool(raw.get("single_shot")) or wants_single_shot(prompt),
        no_captions=True,
        source="writer",
    )


async def extract_brief(
    prompt: str,
    *,
    providers: list[DialogueProvider] | None,
    job_id: str = "",
) -> CreativeBrief:
    """The brief, from the writer when one answers and the heuristic when not.

    Never raises. The heuristic is not a degraded mode — it is the guarantee.
    A writer that drops a location fails the same coverage check the
    heuristic passes by construction, so the customer's words are reached
    either way; the writer only adds a cleaner split into beats.
    """
    text = prompt.strip()
    if not text:
        return CreativeBrief(original_prompt=prompt)
    for provider in providers or []:
        name = getattr(provider, "name", type(provider).__name__)
        try:
            request = DialogueRequest(prompt=text, seconds=0.0, system=_SYSTEM, user=_user(text))
            brief = _parse(await provider.write(request), text)
        except DialogueUnavailable as exc:
            logger.info("music_video_brief_unavailable",
                        extra={"job_id": job_id, "provider": name, "detail": str(exc)})
            continue
        except DialogueRejected as exc:
            logger.warning("music_video_brief_rejected",
                           extra={"job_id": job_id, "provider": name, "detail": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 — the heuristic is behind this
            logger.warning("music_video_brief_failed",
                           extra={"job_id": job_id, "provider": name,
                                  "error": type(exc).__name__})
            continue
        # The writer's brief is merged OVER the heuristic's, never instead of
        # it: anything the heuristic found in the customer's words that the
        # writer dropped is put back, because a dropped noun is the whole
        # fault this exists to stop.
        fallback = heuristic_brief(text)
        for location in fallback.locations:
            if not any(_overlaps(location, known) for known in brief.locations):
                brief.locations.append(location)
        for prop in fallback.props:
            if not any(_overlaps(prop, known) for known in brief.props):
                brief.props.append(prop)
        if not brief.events:
            brief.events = fallback.events
        brief.prohibited = brief.prohibited or fallback.prohibited
        return brief
    return heuristic_brief(text)


def _overlaps(a: str, b: str) -> bool:
    """Two phrases about the same thing, loosely: one contains the other's
    longest word. "hot-pink Lamborghini" and "a hot-pink Lamborghini towing a
    farm tractor" are one prop, not two."""
    words_a = [w for w in re.findall(r"[a-z]{4,}", a.casefold())]
    words_b = set(re.findall(r"[a-z]{4,}", b.casefold()))
    if not words_a or not words_b:
        return a.casefold() == b.casefold()
    longest = max(words_a, key=len)
    return longest in words_b


# ── Coverage ───────────────────────────────────────────────────────────────


@dataclass
class Coverage:
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        total = len(self.covered) + len(self.missing)
        return 1.0 if total == 0 else len(self.covered) / total

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {"fraction": round(self.fraction, 3), "covered": self.covered,
                "missing": self.missing}


def coverage(brief: CreativeBrief, shot_prompts: list[str]) -> Coverage:
    """Which mandatory items appear in at least one shot prompt.

    The client's preflight rule: a plan whose shots lack a mandatory
    location, vehicle, prop or event is rejected before any GPU time. The
    comparison is on the customer's own words, case-insensitively, after
    whitespace is collapsed — no fuzzy matching, because a fuzzy pass is how
    "a barn" gets to count for "a Nashville barn turned neon nightclub".
    """
    haystack = " ".join(" ".join(p.split()).casefold() for p in shot_prompts)
    result = Coverage()
    for item in brief.mandatory:
        needle = " ".join(item.split()).casefold()
        (result.covered if needle in haystack else result.missing).append(item)
    return result


__all__ = [
    "Coverage",
    "CreativeBrief",
    "Event",
    "coverage",
    "extract_brief",
    "forbids_captions",
    "heuristic_brief",
    "wants_single_shot",
]


def dumps(brief: CreativeBrief) -> str:
    return json.dumps(brief.to_dict(), indent=2, ensure_ascii=False)
