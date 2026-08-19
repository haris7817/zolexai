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

#: The pacing ceiling: how many spoken words one second of SPEAKABLE video can
#: carry. Conversational speech runs ~2.5 words/second; planning at 2 leaves
#: air for reactions, pauses and silence — which is both the official
#: prompting recommendation ("break long sentences into shorter phrases with
#: acting directions between them") and the hedge against the model mumbling
#: through an over-packed soundtrack.
WORDS_PER_SECOND = 2.0

#: Seconds at the head of a clip that belong to establishing the shot rather
#: than to speech, and are therefore excluded from the word budget.
#:
#: This is what makes the budget bite harder on short clips, which is where
#: over-packing actually hurts: it costs a 60-second video 4% of its allowance
#: and a 10-second video a quarter of it. Measured 18 Aug 2026 — a 15s render
#: whose plan opened with two quick lines delivered them as one run-together
#: utterance, while longer clips carrying MORE words per second stayed clean.
ESTABLISH_SECONDS = 2.5

#: Grace multiplier before trimming kicks in. A plan a few words over budget
#: is fine; one at double the budget is not.
_BUDGET_SLACK = 1.15


def speech_budget(duration_seconds: float) -> int:
    """Spoken words this clip can hold, before the grace multiplier."""
    speakable = max(0.0, duration_seconds - ESTABLISH_SECONDS)
    return int(speakable * WORDS_PER_SECOND)


#: Screen time one spoken line should occupy, including its reaction beat.
#:
#: MEASURED, and the number the whole feature's pacing rests on. Across nine
#: GPU renders on 18-19 Aug 2026, plans denser than ~0.2 lines/second were
#: clean and plans below it were not: a 60s clip carrying 7 lines (0.12/s)
#: opened with a 12.8-SECOND silence and then spoke one line twice, and a 15s
#: clip carrying 2 (0.13/s) echoed its last word. A 20s clip carrying 6
#: (0.30/s) delivered all six verbatim.
#:
#: The failure mode is worth naming precisely, because it looks like two bugs
#: and is one: **dead air is not neutral.** Given seconds the plan says
#: nothing about, the model fills them — by repeating a line it already said,
#: or by reading the caption's own prose aloud. Density is the cure for both,
#: and separation (the pause cues in `compiler.py`) is what keeps density from
#: turning into run-together delivery. They are independent levers; an earlier
#: revision cut density to fix run-together and produced exactly the sparse,
#: repetitive output a customer then reported.
TARGET_SECONDS_PER_LINE = 4.0

#: Fewer lines than this is not a conversation, whatever the duration.
_MINIMUM_LINES = 2

#: Longest stretch of the timeline that may pass with nobody speaking.
#:
#: Six seconds is a long beat on screen and about where the measured failures
#: began: the 60s plan that repeated itself opened with 12.8 seconds of
#: silence. Checked after parsing and reported back to the planner as a
#: correction rather than enforced by rejection — see `pacing_problems`.
MAX_SILENT_GAP = 6.0


def spoken_line_budget(duration_seconds: float) -> int:
    """The CEILING on separate spoken lines — the far bound, not the aim.

    Kept because the other direction is a real failure too: past roughly one
    line every two seconds there is no room for the reactions and pauses that
    make an exchange readable.
    """
    return max(_MINIMUM_LINES, int(duration_seconds // 2))


def target_spoken_lines(duration_seconds: float) -> int:
    """How many spoken lines a plan should actually aim to produce.

    A TARGET, stated separately from the ceiling above, because a model handed
    only a maximum treats it as a problem to stay clear of. The lyrics writer
    learned this the expensive way three days earlier: given "at most 9" it
    wrote 6, and the song was mostly instrumental. Same model family, same
    arithmetic, same result here — told "about 3" alongside "at most", the
    planner wrote 2 and the render echoed itself.
    """
    # Rounded UP, not to nearest: the two errors are not symmetric. One line
    # too many is a slightly busy scene; one too few is the silence the model
    # fills by repeating itself. (`round` would also hand a 10-second clip 2
    # lines rather than 3, via banker's rounding on the exact .5.)
    by_density = math.ceil(duration_seconds / TARGET_SECONDS_PER_LINE)
    return min(spoken_line_budget(duration_seconds), max(_MINIMUM_LINES, by_density))


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
    continuity: tuple[str, ...] = ()
    """Facts that must look the same in every frame — a red hat that stays the
    same red hat after it is taken off and put back on, a jacket that does not
    change colour when someone turns around.

    This is the Director-mode counterpart of the CONTINUITY block
    `worker/longform/enhance.py` appends to a standard prompt, and it exists
    for the same measured reason (16 Aug 2026): the distilled runtime is
    unguided, so a constraint the prompt only implies has no mechanism to hold
    it, and restating one explicitly is the lever that actually works.
    Customer-reported symptoms this targets: a prop that comes back subtly
    different after being off screen, and a person who flickers out for a
    frame or two mid-shot."""

    def character(self, character_id: str) -> DirectorCharacter:
        for entry in self.characters:
            if entry.id == character_id:
                return entry
        raise KeyError(character_id)

    @property
    def spoken_words(self) -> int:
        return sum(len((event.dialogue or "").split()) for event in self.timeline)

    @property
    def spoken_lines(self) -> int:
        return sum(1 for event in self.timeline if (event.dialogue or "").strip())

    @property
    def seconds_per_spoken_line(self) -> float | None:
        """Screen seconds per spoken line — the density that decides pacing.

        Logged on every plan because it is the single number that separated a
        clean render from a repetitive one across every GPU measurement:
        comfortably under `TARGET_SECONDS_PER_LINE` was clean, well above it
        was not.
        """
        lines = self.spoken_lines
        return round(self.duration_seconds / lines, 1) if lines else None


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
        continuity=_parse_continuity(raw.get("continuity")),
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
    budget = int(speech_budget(plan.duration_seconds) * _BUDGET_SLACK)
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


def _parse_continuity(raw: object) -> tuple[str, ...]:
    """Immutable facts, deduplicated and bounded.

    Bounded because this text is repeated in EVERY section's caption: a long
    list stops being emphasis and starts crowding out the scene itself.
    """
    if not isinstance(raw, list):
        return ()
    seen: list[str] = []
    for entry in raw:
        text = str(entry or "").strip().rstrip(".")
        if text and text.lower() not in {s.lower() for s in seen}:
            seen.append(text)
    return tuple(seen[:6])


#: Words too ordinary for their reuse to be noticeable. A conversation cannot
#: avoid "the" or "you"; it very much can avoid saying "excellent" twice.
_COMMON_WORDS = frozenset(
    """a an and the this that these those is are was were be been am i you he she it we they
    me him her us them my your his hers its our their to of in on at for with from by as
    but or so if then than there here what who whom whose which when where why how not no
    yes do does did done have has had will would can could shall should may might must
    just now well okay ok about into over under out up down off again more most very
    all any some one two three too also like get got go going come came know knew think
    thought say said says tell told take took make made want need let put back still
    even only ever never always because while after before""".split()
)

#: Below this length a word is structural rather than distinctive.
_MIN_DISTINCTIVE = 4


def repeated_vocabulary(plan: DirectorPlan) -> list[str]:
    """Distinctive words the dialogue uses in more than one line.

    A customer noticed it before any of our checks did: a character says
    "excellent", and two lines later says "excellent" again. Each line is
    unique — the line-level repetition rules all pass — but the exchange reads
    as written by something with a small vocabulary, which is exactly what it
    was.

    Reported as a correction rather than raised, for the same reason as pacing:
    a repeated adjective makes a weaker video, not a broken one.
    """
    spoken = [(event.dialogue or "").strip() for event in plan.timeline]
    seen: dict[str, int] = {}
    for line in spoken:
        # Per LINE, not per occurrence: a word repeated inside one sentence is
        # usually deliberate ("no, no, no"), and it is the echo ACROSS lines
        # that reads as a limited vocabulary.
        for word in {
            match.lower()
            for match in re.findall(r"[^\W\d_]+", line, re.UNICODE)
            if len(match) >= _MIN_DISTINCTIVE and match.lower() not in _COMMON_WORDS
        }:
            seen[word] = seen.get(word, 0) + 1
    return sorted(word for word, count in seen.items() if count > 1)


def vocabulary_problems(plan: DirectorPlan) -> list[str]:
    """`repeated_vocabulary`, phrased as corrections the planner can act on."""
    repeats = repeated_vocabulary(plan)
    if not repeats:
        return []
    return [
        "these words are used in more than one line — replace all but the "
        f"first with different wording: {', '.join(repeats[:8])}"
    ]


def pacing_problems(plan: DirectorPlan) -> list[str]:
    """Where this plan would leave the soundtrack empty, in the planner's terms.

    Deliberately NOT part of `parse_plan`'s validation. Everything that raises
    there is a correctness contract — a rewritten user line, a speaker who does
    not exist — and failing a job over those is right. Pacing is a quality
    judgement: a sparse plan still renders a valid video, just a worse one. So
    it is measured here, handed back to the planner as a correction it can act
    on, and accepted on the final attempt rather than costing the customer
    their job.

    The stakes are still real. Dead air is where the model repeats a line it
    already spoke or reads the caption prose aloud, which is what a customer
    reports as "it says the same thing twice".
    """
    spoken = [event for event in plan.timeline if (event.dialogue or "").strip()]
    if not spoken:
        return []

    problems: list[str] = []
    target = target_spoken_lines(plan.duration_seconds)
    if len(spoken) < max(2, target - 1):
        problems.append(
            f"only {len(spoken)} spoken lines for a {plan.duration_seconds:g}-second "
            f"video — write about {target}, spread across the whole timeline"
        )

    opening = spoken[0].start
    if opening > MAX_SILENT_GAP:
        problems.append(
            f"nobody speaks for the first {opening:g} seconds — start the dialogue "
            "within a couple of seconds of the opening"
        )

    for earlier, later in zip(spoken, spoken[1:], strict=False):
        gap = later.start - earlier.end
        if gap > MAX_SILENT_GAP:
            problems.append(
                f"a {gap:g}-second silence between the line ending at "
                f"{earlier.end:g}s and the next at {later.start:g}s — add lines so "
                f"no gap exceeds {MAX_SILENT_GAP:g} seconds"
            )

    tail = plan.duration_seconds - spoken[-1].end
    if tail > MAX_SILENT_GAP:
        problems.append(
            f"the last {tail:g} seconds have no dialogue — the closing line should "
            "land near the end of the video"
        )
    return problems


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
