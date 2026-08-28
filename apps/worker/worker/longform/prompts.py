"""Deterministic section prompts for long-form generation.

Long video used to send the same prompt to every model pass.  The predictable
result was equally repeated action and dialogue.  This module separates text
that describes persistent context from text that clearly advances a sequence,
then assigns each sequential unit to exactly one section.

This is deliberately not an LLM rewrite.  User-authored fragments are copied
verbatim, so names, colours, counts, quoted dialogue and camera directions are
never paraphrased behind the user's back.  Explicit ``Persistent:`` and
``Section N:`` lines are preferred; the fallback recognises dialogue and common
sequence markers conservatively.  Ambiguous prose remains persistent rather
than being silently reinterpreted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from worker.longform.h3_prompts import parse_timed_sections

_SECTION_LINE = re.compile(
    r"^\s*(?:section\s+\d+(?:\s*/\s*(?:\d+|\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?:\s*(?:s|sec|seconds))?))?|\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:s|sec|seconds)?)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

#: A shot pinned to a time range, the way the client's own example prompts
#: write them: "0:00-0:10 a woman walks in", "[0:30 – 0:45]: chorus, wide shot",
#: "15s-30s: drone pullback". At least one side must be unambiguous time —
#: mm:ss, or seconds with an explicit s/sec suffix — so prose like "3-4 cars
#: pass" never matches.
_TIMED_LINE = re.compile(
    r"^\s*\[?\s*"
    r"(\d{1,3}:\d{2}|\d+(?:\.\d+)?\s*s(?:ec(?:onds)?)?)"
    r"\s*(?:-|–|—|to)\s*"
    r"(\d{1,3}:\d{2}|\d+(?:\.\d+)?\s*s(?:ec(?:onds)?)?)"
    r"\s*\]?\s*[:\-–—]?\s*(.+)$",
    re.IGNORECASE,
)


def _parse_timestamp(raw: str) -> float:
    text = raw.strip().lower()
    if ":" in text:
        minutes, seconds = text.split(":")
        return int(minutes) * 60 + int(seconds)
    return float(re.sub(r"\s*s(?:ec(?:onds)?)?$", "", text))


@dataclass(frozen=True)
class _Action:
    text: str
    start: float | None = None
    end: float | None = None

    @property
    def timed(self) -> bool:
        return self.start is not None
_PERSISTENT_LINE = re.compile(
    r"^\s*(?:persistent|continuity|subject(?:s)?|scene|constraints?)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_DIALOGUE_LINE = re.compile(r"^\s*[^:\n]{1,48}:\s+.+$")
#: A list bullet. Under v2 a bulleted line is never read as a `Name: "line"`
#: dialogue turn: a bullet is a constraint or a description, and an internal
#: colon inside one ("- One continuous scene: the camera …") was being
#: performed as one section's dialogue — the silent reinterpretation this
#: module's contract forbids. Ambiguity keeps it persistent instead.
_BULLET_LINE = re.compile(r"^\s*[-•*]\s+")
_SEQUENCE_START = re.compile(
    r"^\s*(?:first|then|next|after(?:wards|\s+that)?|meanwhile|finally|lastly)\b",
    re.IGNORECASE,
)
_INLINE_SEQUENCE = re.compile(
    r"(?:,\s*|;\s*|\s+)(?=(?:then|next|after(?:wards|\s+that)?|meanwhile|finally|lastly)\b)",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def plan_section_prompts(
    master_prompt: str,
    section_total: int,
    *,
    total_seconds: float | None = None,
    v2: bool = False,
    dialogue: bool = True,
) -> list[str]:
    """Return one prompt per section without replaying sequential material.

    A single-pass request is byte-for-byte unchanged.  For a chain, explicit
    section lines, dialogue turns and sequence-marked sentences are allocated
    contiguously.  Everything else is an unchanging reference shared by all
    sections.  Empty later sections receive only a continuation instruction;
    they never receive an earlier action again merely to fill space.

    When the prompt pins shots to time ranges ("0:30-0:45: chorus, wide shot")
    and `total_seconds` is known, each timed shot lands in the section whose
    window contains its midpoint — the client's multi-shot prompts say WHERE
    an action belongs, and allocating them by count instead of by time was
    putting the chorus shot in the wrong minute of the song. Sections are
    treated as even windows; musical-boundary sections deviate a little from
    even, which moves a shot by at most a couple of seconds at a seam.

    ``v2`` (execution.prompt_structuring_v2): bulleted lines classify as
    persistent rather than as dialogue turns, and section 1 — which has no
    predecessor pass — is not told to continue from a predecessor frame or to
    keep what "established previously" established. Off (the default), the
    output is byte-identical to what has always shipped.
    """
    if section_total <= 1:
        return [master_prompt]

    persistent, actions = _separate(master_prompt, v2=v2)
    if total_seconds and any(action.timed for action in actions):
        assigned, persistent = _distribute_timed(
            actions, section_total, total_seconds, persistent
        )
    else:
        assigned = _distribute([a.text for a in actions], section_total)
    prompts: list[str] = []

    for index, current in enumerate(assigned, start=1):
        # Section 1 has no predecessor: under v2 it is not asked to continue
        # from a frame that does not exist or to keep what "established
        # previously" established — an instruction that references a
        # nonexistent thing is caption noise on a runtime that reads captions
        # as content (the shared section-1 preamble issue, noted 21 Aug).
        opening = v2 and index == 1
        if opening:
            lines = [f"LONG-FORM VIDEO — SECTION 1 OF {section_total}."]
        else:
            lines = [
                f"LONG-FORM CONTINUATION — SECTION {index} OF {section_total}.",
                "Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.",
            ]
        # The scaffolding outweighs a short user prompt several times over, so
        # every generic noun in it is a suggestion the model will happily take:
        # a music video for "a car on an empty road" came back as a singer with
        # a crowd behind her, every time (client report, 27 Aug). Naming the
        # constraints as the ONLY inventory is the counterweight.
        if persistent:
            lines += ["PERSISTENT USER CONSTRAINTS (verbatim):", persistent]
            lines.append(
                "These constraints are the complete inventory of this video: "
                "introduce no person, crowd, performer, location or object "
                "they do not name."
            )
        subject = "ACTION OR DIALOGUE" if dialogue else "ACTION"
        lines.append(f"NEW {subject} FOR THIS SECTION ONLY (verbatim):")
        lines.append(current or "Continue naturally from the preceding section without introducing a new event.")
        finish = (
            " Complete this section's assigned dialogue before the section ends."
            if dialogue
            else ""
        )
        if opening:
            if finish:
                lines.append(finish.strip())
        else:
            lines.append(
                "Continue directly from the predecessor frame. Do not replay, "
                "restart or summarise any earlier action." + finish
            )
        prompts.append("\n".join(lines))

    return prompts


def _separate(master_prompt: str, *, v2: bool = False) -> tuple[str, list[_Action]]:
    # The inline paragraph format first — "[0–6s] aerial shot … [6–12s] he
    # enters …" flowing through one paragraph, the way the client actually
    # writes scripts. The line-based patterns below cannot see it (nothing
    # matches at line starts), which on the H3 path put the whole script in
    # every segment until 26 Aug. Same parser, same contract: the preamble
    # is the identity/style brief (persistent, restated every section) and
    # each block is a shot pinned to its own moment, all verbatim.
    preamble, blocks = parse_timed_sections(master_prompt)
    if blocks:
        return preamble.strip(), [
            _Action(text, start, end) for start, end, text in blocks
        ]

    lines = [line for line in master_prompt.splitlines() if line.strip()]
    persistent: list[str] = []
    actions: list[_Action] = []
    undecided: list[str] = []

    for line in lines:
        timed = _TIMED_LINE.match(line)
        if timed:
            start = _parse_timestamp(timed.group(1))
            end = _parse_timestamp(timed.group(2))
            if end >= start:
                actions.append(_Action(timed.group(3).strip(), start, end))
                continue
        section = _SECTION_LINE.match(line)
        if section:
            actions.append(_Action(section.group(1).strip()))
            continue
        fixed = _PERSISTENT_LINE.match(line)
        if fixed:
            persistent.append(fixed.group(1).strip())
            continue
        if v2 and _BULLET_LINE.match(line):
            persistent.append(line.strip())
            continue
        if _DIALOGUE_LINE.match(line) or _SEQUENCE_START.match(line):
            actions.append(_Action(line.strip()))
            continue
        undecided.append(line)

    # A prose paragraph often carries several dialogue turns on one line. Split
    # it only at sentence boundaries and only classify fragments with a strong
    # sequencing signal. Ambiguous descriptions stay persistent.
    for line in undecided:
        sequence = _INLINE_SEQUENCE.split(line.strip())
        if len(sequence) > 1:
            actions.extend(
                _Action(fragment.strip()) for fragment in sequence if fragment.strip()
            )
            continue
        fragments = _SENTENCE_BOUNDARY.split(line.strip())
        for fragment in fragments:
            if (
                _SEQUENCE_START.match(fragment)
                or _DIALOGUE_LINE.match(fragment)
                or '"' in fragment
                or "“" in fragment
                or "”" in fragment
            ):
                actions.append(_Action(fragment))
            else:
                persistent.append(fragment)

    # If nothing was recognisably sequential, the prompt is probably a visual
    # style/scene brief. Repeating it is correct; inventing a story split is not.
    if not actions:
        return master_prompt, []
    return "\n".join(persistent), actions


def _distribute(actions: list[str], total: int) -> list[str]:
    if not actions:
        return [""] * total
    width = math.ceil(len(actions) / total)
    return ["\n".join(actions[index * width : (index + 1) * width]) for index in range(total)]


def _distribute_timed(
    actions: list[_Action],
    total: int,
    total_seconds: float,
    persistent: str,
) -> tuple[list[str], str]:
    """Timed shots land in the section whose window holds their midpoint.

    A prompt that pins shots to timestamps is deliberate about WHERE things
    happen, so the untimed leftovers in the same prompt are treated as
    persistent context rather than shuffled into whichever section had room —
    a stray untimed line jumping the queue ahead of "0:30: the chorus" would
    reorder the user's own storyboard.
    """
    window = total_seconds / total
    buckets: list[list[tuple[float, str]]] = [[] for _ in range(total)]

    for action in actions:
        if not action.timed:
            persistent = f"{persistent}\n{action.text}".strip()
            continue
        midpoint = (action.start + (action.end if action.end is not None else action.start)) / 2
        index = min(total - 1, max(0, int(midpoint / window)))
        buckets[index].append((action.start, action.text))

    assigned = [
        "\n".join(text for _, text in sorted(bucket, key=lambda item: item[0]))
        for bucket in buckets
    ]
    return assigned, persistent


# Re-exported so existing importers keep working; the definitions live in
# `worker.longform.language`, which `h3_prompts` imports too — this module
# already imports h3_prompts, so the shared code cannot live here.
from worker.longform.language import (  # noqa: E402
    SPOKEN_LANGUAGE_NAMES,
    spoken_language_clause,
    spoken_language_sentence,
)
