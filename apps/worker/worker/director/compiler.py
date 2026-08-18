"""DirectorPlan → the prompts the model actually reads.

The target register is the one the LTX 2.5 runtime was trained on, taken from
its own enhancer system prompt and the official prompting guides (read
2026-08-18): chronological prose with "Initially… / A moment later…"
transitions, dialogue quoted exactly with an audible delivery description,
camera framing woven in as sentences, physical acting rather than emotion
labels — and NO timestamps, labels or section headers, which no official
source supports and the enhancer prompts explicitly forbid.

Timing therefore lives entirely on this side of the prompt: each timeline
event is bucketed into the generation window containing its midpoint (the
same rule `plan_section_prompts` uses for user-written timed lines), and each
section's caption carries ONLY its own events. The global plan is written
once, before any section renders — a section can never re-invent the
conversation, which is precisely the long-form dialogue-restart failure this
design exists to prevent.

What every section repeats on purpose: the scene, and each character's full
appearance. Restating identity per pass is the measured fix for drift on the
distilled runtime (16 Aug 2026) — continuity comes from the pinned predecessor
frame plus this repetition, never from narrative summary.
"""

from __future__ import annotations

import re
from dataclasses import replace

from worker.director.plan import DirectorCharacter, DirectorEvent, DirectorPlan

_TRANSITIONS = ("A moment later", "Then", "A beat later", "Next", "Soon after")

_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_PRONOUN = re.compile(r"^(?:he|she|they)\s+", re.IGNORECASE)


def compile_section_prompts(
    plan: DirectorPlan,
    section_total: int,
    *,
    total_seconds: float,
) -> list[str]:
    """One caption per generation section, from one global plan.

    Sections are the same even windows the chain renders (`plan_segments`
    splits evenly, so window = total / count). A single-pass job simply gets a
    one-element list — the whole plan as one caption.
    """
    section_total = max(1, section_total)
    window = total_seconds / section_total
    buckets: list[list[DirectorEvent]] = [[] for _ in range(section_total)]
    for event in plan.timeline:
        midpoint = (event.start + event.end) / 2
        index = min(section_total - 1, max(0, int(midpoint / window)))
        buckets[index].append(event)

    return [
        _compile_section(
            plan,
            events,
            first=index == 0,
            window_end=(index + 1) * window,
        )
        for index, events in enumerate(buckets)
    ]


# ── One section ──────────────────────────────────────────────────────────


def _compile_section(
    plan: DirectorPlan,
    events: list[DirectorEvent],
    *,
    first: bool,
    window_end: float,
) -> str:
    introduced: set[str] = set()
    sentences: list[str] = [_sentence(plan.scene)]

    if first:
        if plan.characters:
            sentences.append(_cast_sentence(plan.characters, introduced))
    else:
        sentences.append(
            _sentence(
                _capfirst(
                    _join_roles([_full_subject(c, introduced) for c in plan.characters])
                    + " continue mid-scene, identical to before in face, clothing and "
                    "voice, without repeating any earlier action or line"
                )
            )
        )

    if not events:
        sentences.append(
            "The conversation rests for now; everyone holds the scene with small "
            "natural movements, glances and breathing, and no one speaks."
        )
    else:
        previous_camera = ""
        for position, event in enumerate(events):
            camera = _humanise(event.camera.strip(), plan)
            if camera and camera.lower() != previous_camera.lower():
                sentences.append(_camera_sentence(camera))
                previous_camera = camera
            sentences.append(
                _event_sentence(
                    plan,
                    event,
                    transition=_transition(position, len(events), first),
                    introduced=introduced,
                )
            )
        # A tail the plan says nothing about is not neutral: on TC2 the model
        # filled ~10 uncovered seconds by READING the caption's camera and
        # ambience sentences aloud as narration. Describe the tail as lived
        # silence so the soundtrack has an owner all the way to the last frame.
        if window_end - events[-1].end > 2.5:
            sentences.append(
                "For the remaining seconds the exchange settles: they hold each "
                "other's gaze with small natural movements, and the room's "
                "ambience is the only sound."
            )

    sentences.append(_ambience_sentence(plan))
    if plan.characters:
        sentences.append(
            _sentence(
                _capfirst(
                    _join_roles([_subject(c) for c in plan.characters])
                    + " keep exactly the same faces, clothing and voices for the "
                    "entire video"
                )
            )
        )
    return " ".join(filter(None, sentences))


def _transition(position: int, total: int, first_section: bool) -> str:
    if position == 0:
        return "Initially" if first_section else ""
    if position == total - 1 and total >= 3:
        return "Finally"
    return _TRANSITIONS[(position - 1) % len(_TRANSITIONS)]


# ── Sentence builders ────────────────────────────────────────────────────


def _cast_sentence(characters: tuple[DirectorCharacter, ...], introduced: set[str]) -> str:
    """Who is here, described concretely, before anything happens.

    Slightly ahead of the "begin with action" ideal, deliberately: identity
    stated early and repeated is the measured anti-drift lever, and a dialogue
    scene that loses a face has lost everything.
    """
    parts = [_full_subject(entry, introduced) for entry in characters]
    return _sentence(_capfirst(_join_roles(parts) + " are here from the first frame"))


def _event_sentence(
    plan: DirectorPlan,
    event: DirectorEvent,
    *,
    transition: str,
    introduced: set[str],
) -> str:
    speaker = plan.character(event.speaker) if event.speaker else None
    event = _humanised_event(event, plan)

    # A speaker with no line is an ACTION beat, not an empty quote: planners
    # tag silent reaction events with the reacting character, and rendering
    # them as `says, ""` handed the model a blank to fill — which it filled by
    # narrating the caption (tc2b, 18 Aug).
    if speaker is None or not (event.dialogue or "").strip():
        body = event.action.strip().rstrip(".")
        if transition:
            return _sentence(f"{transition}, {body[0].lower()}{body[1:]}" if body else transition)
        return _sentence(_capfirst(body))

    subject = _full_subject(speaker, introduced)
    action = _action_clause(event.action, speaker)
    says = _speech_verb(event, speaker, introduced)
    line = (event.dialogue or "").strip()

    if action:
        core = f'{subject} {action}, and {says}, "{line}"'
    else:
        core = f'{subject} {says}, "{line}"'
    if transition:
        core = f"{transition}, {core[0].lower()}{core[1:]}"
    else:
        core = _capfirst(core)
    return _sentence(core)


def _speech_verb(event: DirectorEvent, speaker: DirectorCharacter, spoken: set[str]) -> str:
    """ "says in a low and accusing voice" — the officially endorsed shape.

    Delivery (per line) outranks the character's standing voice description;
    the standing voice is used the first time they speak in a section so the
    voice itself stays consistent across passes.
    """
    manner = (event.delivery or "").strip().rstrip(".")
    if not manner and f"voice:{speaker.id}" not in spoken:
        manner = speaker.voice.strip().rstrip(".")
    spoken.add(f"voice:{speaker.id}")
    if not manner:
        return "says"
    manner = re.sub(r"\s+voice$", "", manner, flags=re.IGNORECASE)
    article = "an" if manner[:1].lower() in "aeiou" else "a"
    return f"says in {article} {manner} voice"


def _action_clause(action: str, speaker: DirectorCharacter) -> str:
    """The event's action, rebased onto the speaking subject when possible.

    Planners write actions with their own subjects ("The detective leans
    forward"). Stripping a leading mention of the speaker (or a pronoun) lets
    the action and the line share one subject — "the detective leans forward,
    and says…" — instead of naming him twice in a row. An action about someone
    ELSE is left alone and precedes the line as its own sentence fragment.
    """
    text = action.strip().rstrip(".")
    if not text:
        return ""
    stripped = _PRONOUN.sub("", _ARTICLE.sub("", text))
    role = speaker.role.strip()
    if stripped.lower().startswith(role.lower()):
        remainder = stripped[len(role) :].strip()
        if remainder:
            return remainder[0].lower() + remainder[1:]
        return ""
    if _PRONOUN.match(text):
        rest = _PRONOUN.sub("", text)
        return rest[0].lower() + rest[1:] if rest else ""
    return text[0].lower() + text[1:]


def _camera_sentence(camera: str) -> str:
    """ "A medium close-up frames the moment, and the camera remains static." """
    parts = [part.strip() for part in re.split(r"[,;]", camera) if part.strip()]
    shot = parts[0] if parts else camera.strip()
    move = ", ".join(parts[1:])
    shot = _ARTICLE.sub("", shot)
    shot = shot[0].lower() + shot[1:] if shot else shot
    article = "An" if shot[:1].lower() in "aeiou" else "A"
    if not move or "static" in move.lower():
        return f"{article} {shot} frames the moment, and the camera remains static."
    move = _normalise_move(move)
    return f"{article} {shot} frames the moment as the camera {move}."


_MOVE_VERBS = re.compile(
    r"^(?:pans?|pushes|pulls?|tilts?|tracks?|dollies|zooms?|moves?|drifts?|"
    r"glides?|circles?|rises?|descends?|follows?|holds?|remains?|stays?)\b",
    re.IGNORECASE,
)


def _normalise_move(move: str) -> str:
    text = move.strip().rstrip(".")
    text = re.sub(r"^(?:the\s+)?camera\s+", "", text, flags=re.IGNORECASE)
    if not text:
        return "remains steady"
    text = text[0].lower() + text[1:]
    # Planners write moves as noun phrases ("subtle push-in") as often as verb
    # phrases ("pushes in slowly"); a noun phrase needs a verb to sit after
    # "as the camera …".
    if not _MOVE_VERBS.match(text):
        article = "an" if text[:1] in "aeiou" else "a"
        return f"makes {article} {text}"
    return text


def _ambience_sentence(plan: DirectorPlan) -> str:
    ambience = plan.ambience.strip().rstrip(".")
    if ambience:
        body = ambience[0].lower() + ambience[1:]
        return f"Under the voices, {body}, with no background music."
    return "Quiet natural room ambience continues under the voices, with no background music."


def _humanise(text: str, plan: DirectorPlan) -> str:
    """Character ids swapped for their role words, anywhere prose will carry.

    Planners leak ids into action and camera text ("close-up on boss_marcos")
    despite being told not to — observed on the pilot run. An id reaching the
    model prompt reads as a nonsense token, so this is a deterministic safety
    net, not a formatting nicety.

    Two-phase, via markers: existing role mentions (article included) are
    absorbed FIRST, longest role first, so replacing the id "chief" can never
    rewrite the middle of an already-correct "police chief" — the naive
    single pass produced "the police the police chief".
    """
    if not text:
        return text
    order = sorted(
        range(len(plan.characters)),
        key=lambda i: len(plan.characters[i].role),
        reverse=True,
    )
    for index in order:
        character = plan.characters[index]
        role = _ARTICLE.sub("", character.role.strip())
        marker = f"\x00{index}\x00"
        # Every alternative absorbs an optional preceding article: replacing a
        # bare "robot" inside "the robot" with "the humanoid robot" without
        # eating its article produced "the the humanoid robot" (TC2, 18 Aug).
        tokens = [role] + [
            token
            for token in (character.id, character.id.replace("_", " "))
            if token and token.lower() != role.lower()
        ]
        # Planners also shorten a multi-word role to its head noun ("the
        # robot" for "humanoid robot" — tc2b). Fold that alias in when no
        # other character could be meant by it.
        head = role.split()[-1].lower()
        if len(role.split()) > 1 and len(head) >= 4:
            others = " ".join(
                f"{c.role} {c.id}" for c in plan.characters if c is not character
            ).lower()
            if head not in others:
                tokens.append(head)
        for token in tokens:
            text = re.sub(
                rf"\b(?:the\s+|an?\s+)?{re.escape(token)}\b",
                marker,
                text,
                flags=re.IGNORECASE,
            )
    for index, character in enumerate(plan.characters):
        text = text.replace(f"\x00{index}\x00", _subject(character))
    # Replacements land lowercased; restore capitals at sentence starts.
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def _humanised_event(event: DirectorEvent, plan: DirectorPlan) -> DirectorEvent:
    return replace(event, action=_humanise(event.action, plan))


# ── Small prose helpers ──────────────────────────────────────────────────


def _subject(character: DirectorCharacter) -> str:
    """ "the police chief" — lowercased, because roles are common nouns and a
    planner-capitalised "Interviewer" reads as a name mid-sentence."""
    role = _ARTICLE.sub("", character.role.strip()).lower()
    return f"the {role}"


def _full_subject(character: DirectorCharacter, introduced: set[str]) -> str:
    """Full appearance on first mention per section, short role afterwards.

    Re-introducing the appearance every section (not just section one) is
    intentional: each section is a separate model pass, and the text is the
    only channel besides the pinned frame that carries identity into it.
    """
    if character.id in introduced:
        return _subject(character)
    introduced.add(character.id)
    appearance = character.appearance.strip().rstrip(".")
    appearance = appearance[0].lower() + appearance[1:] if appearance else ""
    return f"{_subject(character)}, {appearance},"


def _join_roles(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _capfirst(text: str) -> str:
    """Uppercase only the first character — `str.capitalize` would lowercase
    every proper noun and colour word after it."""
    return text[0].upper() + text[1:] if text else text


def _sentence(text: str, terminal: str = ".") -> str:
    body = text.strip()
    if not body:
        return ""
    if body[-1] in '.!?"':
        return body
    return body + terminal
