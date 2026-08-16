"""Deterministic prompt structuring — the adherence rules, applied by code.

## Why this exists, and why it is rules rather than a model

The production runtime is the CFG-distilled checkpoint: guidance is baked in,
so there is no dial that pushes it toward the prompt at inference. The one
lever that measurably works was found by hand on 2026-08-16: "two cars"
drifted colours across a 30-second clip, and the identical scene held
perfectly when the prompt stated the count explicitly and repeated each colour
as a persistence rule. The customer should not need to know that trick.

This module applies it mechanically. Rules, not a language model, for the same
reason `plan_section_prompts` is deterministic: a rewrite that paraphrases can
lose the exact names, counts and colours the user typed — which is the very
failure it would exist to prevent. Here the original prompt is emitted
verbatim, first, always; everything added is derived from it and appended
after it.

## What it deliberately avoids

Negative phrasing. The distilled model has no negation mechanism (no negative
prompt, no CFG), so "no cuts, no other cars" reads as "cuts, other cars" with
extra steps. Every added constraint is stated positively: "the two cars remain
the only vehicles", never "no additional vehicles appear".
"""

from __future__ import annotations

import re

# ── What gets restated ───────────────────────────────────────────────────

#: Colour words worth pinning. Compound shades ("matte black") are matched
#: before their plain form so the user's exact wording is what gets repeated.
_COLOURS = (
    "matte black", "pearl white", "jet black", "pure white",
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "silver", "golden", "gold", "grey", "gray", "brown", "teal",
    "cyan", "crimson", "maroon", "navy", "beige", "turquoise",
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_COLOUR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _COLOURS) + r")\s+([a-z]+(?:\s+[a-z]+)?)\b",
    re.IGNORECASE,
)

#: "two cars", "3 dancers", "exactly two red cars" — a count followed shortly
#: by a plural noun. Conservative: only plural nouns, only within two words.
_COUNT_PATTERN = re.compile(
    r"\b(?:exactly\s+)?(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+((?:[a-z]+\s+){0,2}?[a-z]+s)\b",
    re.IGNORECASE,
)

#: Words that end a colour/count noun phrase but are not the noun ("black and
#: white footage", "two of them"). A match whose noun is one of these is noise.
_NOT_NOUNS = frozenset(
    "and or of the a an in on at is are with to for as its it's very more most "
    "less then than seconds minutes s videos video times".split()
)

#: Prompts that are already structured are left entirely alone: the user (or
#: the section planner) has taken control, and stacking two structures produces
#: contradictions rather than adherence.
_ALREADY_STRUCTURED = re.compile(
    r"^\s*(persistent|section\s+\d+|continuity)\s*:", re.IGNORECASE | re.MULTILINE
)


def _clean_noun(noun: str) -> str | None:
    words = [w for w in noun.lower().split() if w not in _NOT_NOUNS]
    if not words:
        return None
    return " ".join(words)


def structure_prompt(prompt: str) -> str:
    """The user's prompt, verbatim, followed by derived continuity rules.

    Never rewrites, reorders or paraphrases a single word of the input — the
    output CONTAINS the input as its first block, byte for byte. A prompt that
    already carries explicit structure (Persistent:/Section N:) is returned
    unchanged.
    """
    text = prompt.strip()
    if not text or _ALREADY_STRUCTURED.search(text):
        return prompt

    rules: list[str] = []
    seen: set[str] = set()

    for match in _COUNT_PATTERN.finditer(text):
        count_raw, noun_raw = match.group(1), match.group(2)
        noun = _clean_noun(noun_raw)
        if noun is None or noun in seen:
            continue
        seen.add(noun)
        count = _NUMBER_WORDS.get(count_raw.lower(), count_raw)
        rules.append(
            f"Exactly {count} {noun} appear, and they remain the only {noun.split()[-1]} "
            "on screen for the entire video."
        )

    for match in _COLOUR_PATTERN.finditer(text):
        colour, noun_raw = match.group(1).lower(), match.group(2)
        noun = _clean_noun(noun_raw)
        if noun is None:
            continue
        key = f"{colour} {noun}"
        if key in seen:
            continue
        seen.add(key)
        rules.append(
            f"The {colour} {noun.split()[-1]} stays {colour} from the first frame "
            "to the last."
        )

    lines = [text, "", "CONTINUITY (fixed for the entire video):"]
    lines += [f"- {rule}" for rule in rules]
    lines += [
        "- The same subjects keep the same faces, clothing, colours and count "
        "in every frame.",
        "- One continuous scene: the camera keeps moving through the same "
        "environment, and every subject present at the start is still present "
        "at the end.",
    ]
    return "\n".join(lines)
