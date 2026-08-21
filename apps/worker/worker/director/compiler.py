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

from worker.core.logging import get_logger
from worker.director.plan import DirectorCharacter, DirectorEvent, DirectorPlan

logger = get_logger(__name__)

_TRANSITIONS = ("A moment later", "Then", "A beat later", "Next", "Soon after")

#: Used instead of the above when the PRECEDING event also spoke.
#:
#: Two quoted lines in a row with only "A moment later" between them read as
#: one continuous utterance, and the model delivers them that way — measured
#: 18 Aug 2026, where a 15s render fused its opening two lines despite
#: carrying fewer words per second than a clean 20s one. An explicit pause is
#: the official pacing lever ("break long sentences into shorter phrases with
#: acting directions between them … explicit direction on pacing").
_PAUSE_TRANSITIONS = (
    "After a short pause",
    "A beat of silence passes, and then",
    "The room holds still for a moment, and then",
)

_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_PRONOUN = re.compile(r"^(?:he|she|they)\s+", re.IGNORECASE)

#: Determiners that already introduce the noun they precede, so a role word
#: substituted after one must NOT bring an article of its own.
_QUANTIFIER = (
    r"(?:one|two|three|four|five|six|both|each|every|another|either|neither|no)"
)


def compile_section_prompts(
    plan: DirectorPlan,
    section_total: int,
    *,
    total_seconds: float,
    camera_continuity: bool = False,
) -> list[str]:
    """One caption per generation section, from one global plan.

    Sections are the same even windows the chain renders (`plan_segments`
    splits evenly, so window = total / count). A single-pass job simply gets a
    one-element list — the whole plan as one caption.

    ``camera_continuity`` (execution.director_camera_continuity) carries the
    shot each section ends on into the next one: a section that opens on the
    same camera says the shot CONTINUES instead of announcing it as a fresh
    framing event at every seam. Off (the default), captions are byte-identical
    to what has always shipped.
    """
    section_total = max(1, section_total)
    window = total_seconds / section_total
    buckets: list[list[DirectorEvent]] = [[] for _ in range(section_total)]
    for event in plan.timeline:
        midpoint = (event.start + event.end) / 2
        index = min(section_total - 1, max(0, int(midpoint / window)))
        buckets[index].append(event)

    # The camera each section INHERITS: the last stated shot of any earlier
    # bucket. Computed with the same humanisation the caption loop applies, so
    # equality checks compare what the model would actually read.
    entering = ""
    entering_cameras: list[str] = []
    for events in buckets:
        entering_cameras.append(entering)
        for event in events:
            camera = _humanise(event.camera.strip(), plan)
            if camera:
                entering = camera

    # Presence follows the BUCKETS, not the clock. An exit event whose nominal
    # end spills a few seconds past a boundary still RENDERS entirely in the
    # section that owns it — so from the next section's first frame the
    # character is gone, whatever the plan's timestamps say. Who a section
    # opens with is everyone not exited in an earlier bucket; who survives it
    # is everyone not exited in this bucket either.
    departed: set[str] = set()
    casts: list[list[DirectorCharacter]] = []
    survivor_sets: list[list[DirectorCharacter]] = []
    for events in buckets:
        casts.append([c for c in plan.characters if c.id not in departed])
        for event in events:
            departed.update(event.exits)
        survivor_sets.append([c for c in plan.characters if c.id not in departed])

    captions = [
        _compile_section(
            plan,
            events,
            first=index == 0,
            cast=casts[index],
            survivors=survivor_sets[index],
            window_end=(index + 1) * window,
            entering_camera=entering_cameras[index] if camera_continuity else "",
        )
        for index, events in enumerate(buckets)
    ]
    # The semantic state each section renders under, in the log — this is the
    # first question a repetition or reappearance report needs answered, and
    # the captions themselves are too long to read at a glance.
    logger.info(
        "director_sections",
        extra={
            "sections": section_total,
            "language": plan.language,
            "per_section": [
                {
                    "index": index,
                    "events": len(events),
                    "spoken_lines": sum(
                        1 for e in events if (e.dialogue or "").strip()
                    ),
                    "present": [c.id for c in casts[index]],
                    "departed": [
                        c.id
                        for c in plan.characters
                        if c.id not in {x.id for x in casts[index]}
                    ],
                }
                for index, events in enumerate(buckets)
            ],
        },
    )
    return captions


# ── One section ──────────────────────────────────────────────────────────


def _compile_section(
    plan: DirectorPlan,
    events: list[DirectorEvent],
    *,
    first: bool,
    cast: list[DirectorCharacter],
    survivors: list[DirectorCharacter],
    window_end: float,
    entering_camera: str = "",
) -> str:
    """One caption. `cast` is who the section OPENS with; `survivors` is who
    is still there when it ends.

    Presence is the state this compiler exists to keep straight: every
    sentence that asserts constancy is scoped to survivors, because "present
    and solid in every single frame", said about a man who walks out
    mid-video, is a rendered ghost — measured on the GPU, 20 Aug 2026, where
    he stood back in the kitchen for the final twelve seconds while the
    soundtrack said he was gone.
    """
    introduced: set[str] = set()
    sentences: list[str] = [_sentence(plan.scene)]

    if first:
        if cast and plan.source_anchored:
            sentences.append(_anchored_cast_sentence(tuple(cast), introduced))
        elif cast:
            sentences.append(_cast_sentence(tuple(cast), introduced))
    elif cast:
        verb = "continues" if len(cast) == 1 else "continue"
        sentences.append(
            _sentence(
                _capfirst(
                    _join_roles([_full_subject(c, introduced) for c in cast])
                    + f" {verb} mid-scene, identical to before in face, clothing "
                    "and voice, without repeating any earlier action or line"
                )
            )
        )
        if len(cast) < len(plan.characters):
            # Someone left in an earlier section. The scene is restated in
            # terms of who REMAINS — never by naming the departed, because on
            # this runtime a name in the caption is a request for its owner.
            sentences.append(_remaining_sentence(cast))
    else:
        sentences.append(
            "The room continues on its own, calm and still, exactly the same "
            "place as before."
        )

    if not events:
        sentences.append(
            "The conversation rests for now; everyone holds the scene with small "
            "natural movements, glances and breathing, and no one speaks."
        )
    else:
        # With an entering camera (director_camera_continuity), the section
        # already knows what shot it inherits: an opening event on the SAME
        # shot states it as a continuation, never as a fresh framing event —
        # a fresh "A medium shot frames the moment" at every seam reads as a
        # cut to a new setup. With the flag off `entering_camera` is "" and
        # this loop behaves exactly as it always has.
        previous_camera = entering_camera
        camera_stated = False
        previous_spoke = False
        for position, event in enumerate(events):
            camera = _humanise(event.camera.strip(), plan)
            if camera and camera.lower() != previous_camera.lower():
                sentences.append(_camera_sentence(camera))
                previous_camera = camera
                camera_stated = True
            elif camera and entering_camera and not camera_stated:
                sentences.append(_camera_continues_sentence(camera))
                camera_stated = True
            speaks = bool((event.dialogue or "").strip())
            sentences.append(
                _event_sentence(
                    plan,
                    event,
                    transition=_transition(
                        position,
                        len(events),
                        first,
                        after_speech=previous_spoke and speaks,
                    ),
                    introduced=introduced,
                )
            )
            previous_spoke = speaks
            if event.exits:
                # The action said what happened; this says what is TRUE from
                # here on. Resulting state, not repeated action — "From this
                # moment, only the woman remains" gives the model a scene to
                # hold, where re-describing the departure would be an
                # invitation to perform it again.
                remaining = [
                    c for c in plan.characters if c.id in plan.present_ids(event.end)
                ]
                sentences.append(_after_exit_sentence(remaining))
        # A tail the plan says nothing about is not neutral: on TC2 the model
        # filled ~10 uncovered seconds by READING the caption's camera and
        # ambience sentences aloud as narration. Describe the tail as lived
        # silence so the soundtrack has an owner all the way to the last frame.
        if window_end - events[-1].end > 2.5:
            sentences.append(_settle_sentence(survivors))

    # Each line said once, stated POSITIVELY.
    #
    # Measured 19 Aug 2026: a 60s render spoke three of its fourteen lines
    # twice, each repeat landing seconds after the original — the model
    # filling a line's remaining screen time by saying it again rather than
    # moving on. The obvious wording ("no line is repeated") is exactly what
    # this runtime cannot use: it has no negation mechanism, so a banned thing
    # reads as a requested one (see `worker/longform/enhance.py`). Framed as
    # forward motion instead, which is the same instruction the model can act
    # on.
    if any((event.dialogue or "").strip() for event in events):
        sentences.append(
            "Each line of dialogue is spoken a single time, and the exchange "
            "moves forward to the next speaker as soon as it lands."
        )

    sentences.append(_ambience_sentence(plan))
    sentences.extend(_continuity_sentences(plan, survivors))
    return " ".join(filter(None, sentences))


def _remaining_sentence(cast: list[DirectorCharacter]) -> str:
    """Who the scene holds now, after an earlier departure — stated as what IS."""
    roles = _join_roles([_subject(c) for c in cast])
    if len(cast) == 1:
        return _sentence(
            _capfirst(f"{roles} is alone in the scene now, and stays alone")
        )
    return _sentence(
        _capfirst(f"only {roles} are in the scene now, and it stays that way")
    )


def _after_exit_sentence(remaining: list[DirectorCharacter]) -> str:
    """The resulting state the moment a departure completes."""
    if not remaining:
        return (
            "From this moment the room stands quiet and empty, and the scene "
            "continues on its own."
        )
    roles = _join_roles([_subject(c) for c in remaining])
    if len(remaining) == 1:
        return _sentence(
            _capfirst(f"from this moment, {roles} is alone in the scene")
        )
    return _sentence(
        _capfirst(f"from this moment, only {roles} remain in the scene")
    )


def _settle_sentence(survivors: list[DirectorCharacter]) -> str:
    """The described-silence closing beat, worded for who is actually left.

    "They hold each other's gaze" was written for a scene that still has two
    people in it; said over a woman alone at the sink it re-invents a partner
    to hold the gaze of.
    """
    if len(survivors) == 1:
        subject = _subject(survivors[0])
        return (
            f"For the remaining seconds the scene settles: {subject} holds the "
            "moment with small natural movements, and the room's ambience is "
            "the only sound."
        )
    if not survivors:
        return (
            "For the remaining seconds the room rests quiet and still, and its "
            "ambience is the only sound."
        )
    return (
        "For the remaining seconds the exchange settles: they hold each "
        "other's gaze with small natural movements, and the room's "
        "ambience is the only sound."
    )


def _continuity_sentences(
    plan: DirectorPlan, survivors: list[DirectorCharacter]
) -> list[str]:
    """What must not drift, restated at the end of every section.

    Three customer-reported symptoms share one cause and one lever. A person
    who flickers out for a few frames, a prop that returns subtly different
    after being off screen, and a face that shifts between passes are all the
    unguided runtime having no mechanism to hold a constraint the prompt only
    implied. Restating constraints explicitly is the measured fix (16 Aug
    2026), and it is why `enhance.py` appends the same kind of block to a
    standard prompt.

    Everything here is phrased as what STAYS, never as what to avoid: this
    runtime has no negation mechanism, so "the hat does not change" reads as a
    changing hat.

    Scoped to `survivors` — who is still in the scene when this section ends —
    because these are the strongest sentences in the caption and they must
    never argue with a departure. "Present and solid in every single frame",
    said about the man who walked out, is precisely the rendered ghost the
    20 Aug measurement caught.
    """
    sentences: list[str] = []
    if survivors:
        subjects = _join_roles([_subject(c) for c in survivors])
        one = len(survivors) == 1
        keeps, stays = ("keeps", "stays") if one else ("keep", "stay")
        faces, voices = ("face", "voice") if one else ("faces", "voices")
        if plan.source_anchored:
            # Constancy anchored to the FRAME rather than to a described look:
            # on this path the text was forbidden to invent appearances, so
            # the conditioned image is the only honest referent. In sections
            # after the first, "the first frame" resolves to that pass's own
            # conditioned frame 0 — the predecessor's final image — which
            # carries the same identity forward, so the sentence stays true at
            # every seam.
            sentences.append(
                _sentence(
                    _capfirst(subjects)
                    + f" {keeps} exactly the same {faces}, clothing, hair, colours "
                    f"and {voices} they have in the first frame, for the entire video"
                )
            )
        else:
            sentences.append(
                _sentence(
                    _capfirst(subjects)
                    + f" {keeps} exactly the same {faces}, clothing and {voices} "
                    "for the entire video"
                )
            )
        # Continuous presence, stated as presence. The flicker is a person
        # briefly absent, so the constraint is that they are always there.
        sentences.append(
            _sentence(
                _capfirst(subjects)
                + f" {stays} fully visible in the frame from the first frame to "
                "the last, present and solid in every single frame"
            )
        )
    for fact in plan.continuity:
        sentences.append(_sentence(_capfirst(_humanise(fact, plan))))
    return sentences


def _transition(
    position: int, total: int, first_section: bool, *, after_speech: bool = False
) -> str:
    if position == 0:
        return "Initially" if first_section else ""
    if after_speech:
        # A pause cue outranks "Finally" on the closing line: landing an
        # exchange matters less than the last two lines staying separable.
        return _PAUSE_TRANSITIONS[(position - 1) % len(_PAUSE_TRANSITIONS)]
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


def _anchored_cast_sentence(
    characters: tuple[DirectorCharacter, ...], introduced: set[str]
) -> str:
    """The cast of a source-anchored plan: the picture is the description.

    The video's first frame IS the uploaded photograph, pinned at full
    conditioning strength — so the strongest honest identity statement is
    "exactly as this frame shows them", not a text description the planner was
    forbidden to invent. Appearance the idea itself stated still rides along
    via `_full_subject`; a character without one is named by role alone.
    """
    parts = [_full_subject(entry, introduced) for entry in characters]
    return _sentence(
        _capfirst(
            _join_roles(parts)
            + " are already present in the opening frame, and they keep exactly "
            "the appearance that frame shows for the whole video"
        )
    )


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
            return _sentence(_joined(transition, body) if body else transition)
        return _sentence(_capfirst(body))

    subject = _full_subject(speaker, introduced)
    action = _action_clause(event.action, speaker)
    says = _speech_verb(event, speaker, introduced)
    line = (event.dialogue or "").strip()

    if action:
        core = f'{subject} {action}, and {says}, "{line}"'
    else:
        core = f'{subject} {says}, "{line}"'
    core = _joined(transition, core) if transition else _capfirst(core)
    return _sentence(core)


def _joined(transition: str, body: str) -> str:
    """Transition + clause, with the comma only where English wants one.

    A transition that already ends in a conjunction ("…, and then") runs
    straight into its clause; one that does not takes a comma.
    """
    lowered = body[0].lower() + body[1:]
    separator = " " if re.search(r"\b(?:and|then|but|so)$", transition) else ", "
    return f"{transition}{separator}{lowered}"


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


def _camera_continues_sentence(camera: str) -> str:
    """ "The same medium shot continues from the previous moment…"

    A section is a separate model pass with no memory: the pinned frame
    carries the inherited framing in pixels, and this sentence carries it in
    text as the same shot still going — the continuation phrasing exists so a
    seam does not read as a cut to a new setup.
    """
    parts = [part.strip() for part in re.split(r"[,;]", camera) if part.strip()]
    shot = parts[0] if parts else camera.strip()
    move = ", ".join(parts[1:])
    shot = _ARTICLE.sub("", shot)
    shot = shot[0].lower() + shot[1:] if shot else shot
    if not move or "static" in move.lower():
        return (
            f"The same {shot} continues from the previous moment, and the "
            "camera remains static."
        )
    move = _normalise_move(move)
    return f"The same {shot} continues from the previous moment as the camera {move}."


#: A move phrase that already has its own verb needs no "makes a" in front of
#: it. An optional leading adverb is part of the match because planners write
#: sequenced moves ("then cuts to a medium shot"), and treating one of those as
#: a noun phrase produced "as the camera makes a then cuts to…" (TC3, 18 Aug).
_MOVE_VERBS = re.compile(
    r"^(?:(?:and\s+|then\s+|slowly\s+|briefly\s+|before\s+)*)"
    r"(?:pans?|pushes|pulls?|tilts?|tracks?|dollies|zooms?|moves?|drifts?|"
    r"glides?|circles?|rises?|descends?|follows?|holds?|remains?|stays?|"
    r"cuts?|switches|shifts?|settles?|reframes?)\b",
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
        bare = f"\x01{index}\x01"
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
            # A counted mention already has its determiner: "one woman" must
            # not become "one the woman". Counting sentences are exactly what
            # survives grounding on the anchored path ("Two people are
            # present: one woman and one robot"), so this is the phrasing the
            # customer's caption carries every section.
            text = re.sub(
                rf"\b({_QUANTIFIER})\s+(?:the\s+|an?\s+)?{re.escape(token)}\b",
                lambda match, bare=bare: f"{match.group(1)} {bare}",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf"\b(?:the\s+|an?\s+)?{re.escape(token)}\b",
                marker,
                text,
                flags=re.IGNORECASE,
            )
    for index, character in enumerate(plan.characters):
        text = text.replace(f"\x00{index}\x00", _subject(character))
        text = text.replace(f"\x01{index}\x01", _bare_subject(character))
    # Replacements land lowercased; restore capitals at sentence starts.
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def _humanised_event(event: DirectorEvent, plan: DirectorPlan) -> DirectorEvent:
    return replace(event, action=_humanise(event.action, plan))


# ── Small prose helpers ──────────────────────────────────────────────────


def _subject(character: DirectorCharacter) -> str:
    """ "the police chief" — lowercased, because roles are common nouns and a
    planner-capitalised "Interviewer" reads as a name mid-sentence."""
    return f"the {_bare_subject(character)}"


def _bare_subject(character: DirectorCharacter) -> str:
    """ "police chief" — for positions that supply their own determiner."""
    return _ARTICLE.sub("", character.role.strip()).lower()


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
    if not appearance:
        # Anchored plans may leave appearance empty on purpose — the uploaded
        # image is the description, and "the woman, ," is not a sentence.
        return _subject(character)
    appearance = appearance[0].lower() + appearance[1:]
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
