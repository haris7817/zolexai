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

_SECTION_LINE = re.compile(
    r"^\s*(?:section\s+\d+(?:\s*/\s*(?:\d+|\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?:\s*(?:s|sec|seconds))?))?|\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:s|sec|seconds)?)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)
_PERSISTENT_LINE = re.compile(
    r"^\s*(?:persistent|continuity|subject(?:s)?|scene|constraints?)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_DIALOGUE_LINE = re.compile(r"^\s*[^:\n]{1,48}:\s+.+$")
_SEQUENCE_START = re.compile(
    r"^\s*(?:first|then|next|after(?:wards|\s+that)?|meanwhile|finally|lastly)\b",
    re.IGNORECASE,
)
_INLINE_SEQUENCE = re.compile(
    r"(?:,\s*|;\s*|\s+)(?=(?:then|next|after(?:wards|\s+that)?|meanwhile|finally|lastly)\b)",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def plan_section_prompts(master_prompt: str, section_total: int) -> list[str]:
    """Return one prompt per section without replaying sequential material.

    A single-pass request is byte-for-byte unchanged.  For a chain, explicit
    section lines, dialogue turns and sequence-marked sentences are allocated
    contiguously.  Everything else is an unchanging reference shared by all
    sections.  Empty later sections receive only a continuation instruction;
    they never receive an earlier action again merely to fill space.
    """
    if section_total <= 1:
        return [master_prompt]

    persistent, actions = _separate(master_prompt)
    assigned = _distribute(actions, section_total)
    prompts: list[str] = []

    for index, current in enumerate(assigned, start=1):
        lines = [
            f"LONG-FORM CONTINUATION — SECTION {index} OF {section_total}.",
            "Keep the same subjects, identities, faces, clothing, colours, object counts, vehicles, environment and camera direction established previously.",
        ]
        if persistent:
            lines += ["PERSISTENT USER CONSTRAINTS (verbatim):", persistent]
        lines.append("NEW ACTION OR DIALOGUE FOR THIS SECTION ONLY (verbatim):")
        lines.append(current or "Continue naturally from the preceding section without introducing a new event.")
        lines.append(
            "Continue directly from the predecessor frame. Do not replay, restart or summarise any earlier action or dialogue. Complete this section's assigned dialogue before the section ends."
        )
        prompts.append("\n".join(lines))

    return prompts


def _separate(master_prompt: str) -> tuple[str, list[str]]:
    lines = [line for line in master_prompt.splitlines() if line.strip()]
    persistent: list[str] = []
    actions: list[str] = []
    undecided: list[str] = []

    for line in lines:
        section = _SECTION_LINE.match(line)
        if section:
            actions.append(section.group(1).strip())
            continue
        fixed = _PERSISTENT_LINE.match(line)
        if fixed:
            persistent.append(fixed.group(1).strip())
            continue
        if _DIALOGUE_LINE.match(line) or _SEQUENCE_START.match(line):
            actions.append(line.strip())
            continue
        undecided.append(line)

    # A prose paragraph often carries several dialogue turns on one line. Split
    # it only at sentence boundaries and only classify fragments with a strong
    # sequencing signal. Ambiguous descriptions stay persistent.
    for line in undecided:
        sequence = _INLINE_SEQUENCE.split(line.strip())
        if len(sequence) > 1:
            actions.extend(fragment.strip() for fragment in sequence if fragment.strip())
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
                actions.append(fragment)
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
