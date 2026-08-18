"""The DirectorPlan: what the planner must produce, and what makes one valid.

The planner is a language model, and language models return *plausible* JSON,
not correct JSON. Every rule the product depends on is therefore enforced here,
deterministically, after parsing — the model is asked nicely in the system
prompt and then checked coldly in code:

  * dialogue the USER wrote is preserved verbatim (the one contract that can
    never bend: a rewritten user line is worse than a failed job);
  * speech fits the clip — a 10-second video cannot hold a monologue, and an
    over-packed plan is trimmed from the end rather than spoken at 2x;
  * every line has an owner that exists, so speaker identity cannot drift by
    construction;
  * events sit inside the video's actual duration, because the timeline is
    consumed by the WORKER (bucketing events into generation windows).
    Timestamps never reach the model prompt — no official LTX source supports
    timestamp syntax, and the enhancer system prompts explicitly forbid it.

Validation failures raise `DirectorPlanError` with every problem listed, which
is what makes a single retry meaningful: the provider can log exactly why the
first plan was refused.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

#: The pacing ceiling: how many spoken words one second of video can carry.
#: Conversational speech runs ~2.5 words/second; planning at 2 leaves air for
#: reactions, pauses and silence — which is both the official prompting
#: recommendation ("break long sentences into shorter phrases with acting
#: directions between them") and the hedge against the model mumbling through
#: an over-packed soundtrack.
WORDS_PER_SECOND = 2.0

#: Grace multiplier before trimming kicks in. A plan a few words over budget
#: is fine; one at double the budget is not.
_BUDGET_SLACK = 1.15

#: More characters than this cannot hold stable identities in one generated
#: scene; the planner is told two or three is the sweet spot.
MAX_CHARACTERS = 4

_MAX_EVENTS = 24

#: Quoted spans in the user's idea that the plan must carry verbatim. Double
#: quotes only (straight or curly): single quotes cannot be told apart from
#: the apostrophe inside "don't", and a false quote boundary would demand the
#: planner reproduce half a line. Two-word minimum, so a quoted title like
#: "Heat" does not become mandatory dialogue.
_USER_QUOTE = re.compile(r"[\"“]([^\"“”]{2,300})[\"”]")


class DirectorPlanError(ValueError):
    """The planner's output is not an acceptable plan. Lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass(frozen=True)
class DirectorCharacter:
    id: str
    role: str
    """Short noun phrase the prose refers to them by: "detective", "police chief"."""
    appearance: str
    """Concrete visible description, held constant across every section."""
    voice: str = ""
    """Audible description ("low, gravelly"), woven in when they first speak."""


@dataclass(frozen=True)
class DirectorEvent:
    start: float
    end: float
    action: str
    camera: str = ""
    speaker: str | None = None
    dialogue: str | None = None
    delivery: str | None = None
    """Audible manner of the line ("low and accusing") — becomes ".. says in a
    low and accusing voice" in the compiled caption."""


@dataclass(frozen=True)
class DirectorPlan:
    scene: str
    tone: str
    language: str
    duration_seconds: float
    ambience: str
    characters: tuple[DirectorCharacter, ...]
    timeline: tuple[DirectorEvent, ...]

    def character(self, character_id: str) -> DirectorCharacter:
        for entry in self.characters:
            if entry.id == character_id:
                return entry
        raise KeyError(character_id)

    @property
    def spoken_words(self) -> int:
        return sum(len((event.dialogue or "").split()) for event in self.timeline)


def required_quotes(idea: str) -> list[str]:
    """Dialogue the user wrote themselves, which the plan must not touch."""
    return [match.strip() for match in _USER_QUOTE.findall(idea) if len(match.split()) >= 2]


def _normalise_line(text: str) -> str:
    """Case- and whitespace-insensitive comparison form for verbatim checks.

    Terminal punctuation is ignored too: a planner that carries "Please don't
    leave." where the idea wrote "Please don't leave" has preserved the line.
    """
    cleaned = re.sub(r"\s+", " ", text).strip().strip(".!?¡¿ ").casefold()
    return cleaned.replace("’", "'").replace("‘", "'")


def parse_plan(
    raw: object,
    *,
    idea: str,
    duration_seconds: float,
    language: str,
) -> DirectorPlan:
    """Parses and validates the planner's JSON into a `DirectorPlan`.

    Raises `DirectorPlanError` naming every violated rule at once, so a retry
    prompt (or a log reader) sees the full diagnosis rather than the first
    symptom.
    """
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise DirectorPlanError(["the planner did not return a JSON object"])

    scene = str(raw.get("scene") or "").strip()
    tone = str(raw.get("tone") or "").strip()
    ambience = str(raw.get("ambience") or "").strip()
    if not scene:
        problems.append("the plan has no scene description")

    characters = _parse_characters(raw.get("characters"), problems)
    timeline = _parse_timeline(
        raw.get("timeline"), {c.id for c in characters}, duration_seconds, problems
    )

    if problems:
        raise DirectorPlanError(problems)

    plan = DirectorPlan(
        scene=scene,
        tone=tone,
        language=language,
        duration_seconds=duration_seconds,
        ambience=ambience,
        characters=tuple(characters),
        timeline=tuple(timeline),
    )
    plan = _enforce_speech_budget(plan, idea)
    _require_user_dialogue(plan, idea)
    return plan


# ── Parsing pieces ───────────────────────────────────────────────────────


def _parse_characters(raw: object, problems: list[str]) -> list[DirectorCharacter]:
    if not isinstance(raw, list) or not raw:
        problems.append("the plan has no characters")
        return []
    if len(raw) > MAX_CHARACTERS:
        problems.append(
            f"{len(raw)} characters is more than one scene can keep consistent "
            f"(limit {MAX_CHARACTERS})"
        )
        return []

    characters: list[DirectorCharacter] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            problems.append(f"character {index + 1} is not an object")
            continue
        identifier = re.sub(r"\W+", "_", str(entry.get("id") or "").strip().lower()).strip("_")
        role = str(entry.get("role") or "").strip()
        appearance = str(entry.get("appearance") or "").strip()
        voice = str(entry.get("voice") or "").strip()
        if not identifier:
            problems.append(f"character {index + 1} has no id")
            continue
        if identifier in seen:
            problems.append(f"character id '{identifier}' appears twice")
            continue
        if not role or not appearance:
            problems.append(f"character '{identifier}' is missing a role or appearance")
            continue
        seen.add(identifier)
        characters.append(
            DirectorCharacter(id=identifier, role=role, appearance=appearance, voice=voice)
        )
    return characters


def _parse_timeline(
    raw: object,
    character_ids: set[str],
    duration_seconds: float,
    problems: list[str],
) -> list[DirectorEvent]:
    if not isinstance(raw, list) or not raw:
        problems.append("the plan has no timeline")
        return []
    if len(raw) > _MAX_EVENTS:
        problems.append(f"{len(raw)} timeline events is too many (limit {_MAX_EVENTS})")
        return []

    events: list[DirectorEvent] = []
    for index, entry in enumerate(raw):
        label = f"event {index + 1}"
        if not isinstance(entry, dict):
            problems.append(f"{label} is not an object")
            continue
        try:
            start = float(entry.get("start"))
            end = float(entry.get("end"))
        except (TypeError, ValueError):
            problems.append(f"{label} has non-numeric timing")
            continue
        if math.isnan(start) or math.isnan(end) or start < 0 or end <= start:
            problems.append(f"{label} has an invalid time range {start}-{end}")
            continue
        if start >= duration_seconds + 0.5:
            problems.append(f"{label} starts at {start:g}s, beyond the {duration_seconds:g}s video")
            continue
        end = min(end, duration_seconds)

        action = str(entry.get("action") or "").strip()
        camera = str(entry.get("camera") or "").strip()
        speaker_raw = entry.get("speaker")
        # Instruct models write the LITERAL STRING "null"/"none" as often as
        # JSON null (observed on the pilot run, 18 Aug 2026).
        speaker = (
            re.sub(r"\W+", "_", str(speaker_raw).strip().lower()).strip("_")
            if speaker_raw
            else None
        )
        if speaker in ("null", "none", ""):
            speaker = None
        dialogue = str(entry.get("dialogue") or "").strip().strip('"“”') or None
        delivery = str(entry.get("delivery") or "").strip() or None

        if dialogue and not speaker:
            problems.append(f"{label} has dialogue with no speaker")
            continue
        if speaker and speaker not in character_ids:
            problems.append(f"{label} names unknown speaker '{speaker}'")
            continue
        if not action and not dialogue:
            problems.append(f"{label} has neither action nor dialogue")
            continue
        events.append(
            DirectorEvent(
                start=start,
                end=end,
                action=action,
                camera=camera,
                speaker=speaker,
                dialogue=dialogue,
                delivery=delivery,
            )
        )

    events.sort(key=lambda event: (event.start, event.end))
    return events


# ── Post-parse contracts ─────────────────────────────────────────────────


def _enforce_speech_budget(plan: DirectorPlan, idea: str) -> DirectorPlan:
    """Trims planner-invented dialogue until the plan fits its duration.

    Later lines go first — cutting a conversation's tail preserves its logic,
    cutting its head rewrites it. Lines the user wrote in the idea are never
    trimmed: if the USER over-packed the clip, that is their call to make, and
    honouring it beats silently editing them.
    """
    budget = int(plan.duration_seconds * WORDS_PER_SECOND * _BUDGET_SLACK)
    if plan.spoken_words <= budget:
        return plan

    protected = {_normalise_line(quote) for quote in required_quotes(idea)}
    events = list(plan.timeline)
    for index in range(len(events) - 1, -1, -1):
        if plan.spoken_words <= budget:
            break
        event = events[index]
        if not event.dialogue or _normalise_line(event.dialogue) in protected:
            continue
        stripped = replace(
            event,
            dialogue=None,
            delivery=None,
            action=event.action or "holds the moment in silence",
        )
        events[index] = stripped
        plan = replace(plan, timeline=tuple(events))
    return plan


def _require_user_dialogue(plan: DirectorPlan, idea: str) -> None:
    """Every line the user quoted in the idea must appear in the plan, exactly.

    This is Director mode's version of the platform rule that user text is
    never rewritten. The enhancer never paraphrases a prompt; the planner never
    paraphrases a line.
    """
    quotes = required_quotes(idea)
    if not quotes:
        return
    spoken = [_normalise_line(event.dialogue) for event in plan.timeline if event.dialogue]
    missing = [
        quote
        for quote in quotes
        if _normalise_line(quote) not in spoken
        and not any(_normalise_line(quote) in line for line in spoken)
    ]
    if missing:
        raise DirectorPlanError(
            [f'the plan dropped or rewrote the user\'s own line "{quote}"' for quote in missing]
        )
