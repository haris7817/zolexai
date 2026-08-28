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
#: The sentence every v2 block ends up containing. v2 emits prose with no
#: heading, so this is what "already structured" looks like to a second pass.
_V2_MARKER = "The same subjects keep the same faces"

_ALREADY_STRUCTURED = re.compile(
    r"^\s*(persistent|section\s+\d+|continuity)\s*:", re.IGNORECASE | re.MULTILINE
)

#: A user-asserted camera hold. When one is present, the v2 continuity block
#: answers it with a positive static rule instead of appending "the camera
#: keeps moving" to the same prompt — the runtime has no negation mechanism,
#: so the user's own "never moves" is the weaker phrasing of the two and the
#: worker must not be the one supplying the stronger opposite.
_STATIC_CAMERA = re.compile(
    r"locked[- ]off"
    r"|static camera"
    r"|fixed camera"
    r"|camera\s+(?:stays|remains|is|holds)\s+(?:perfectly\s+)?(?:still|static|fixed|locked)"
    r"|camera\s+(?:never|doesn'?t|does\s+not|won'?t)\s+move"
    r"|on a tripod"
    r"|no camera movement",
    re.IGNORECASE,
)

#: Departure vocabulary. A prompt that says someone leaves must not also carry
#: "every subject present at the start is still present at the end": presence
#: assertions about departed people are the measured rendered-ghost failure
#: (GPU, 20 Aug 2026), and Director mode scopes its constancy sentences to
#: survivors for exactly this reason. Standard mode has no presence model, so
#: the v2 block simply withholds the presence and count assertions when the
#: user's own text says somebody goes.
_EXIT_WORDS = re.compile(
    r"\b(?:leaves?|leaving"
    r"|exits?|exiting"
    r"|walks?\s+(?:out|away|off)|walked\s+(?:out|away|off)"
    r"|departs?|departed"
    r"|drives?\s+(?:away|off)|drove\s+(?:away|off)"
    r"|runs?\s+(?:away|off)|ran\s+(?:away|off)"
    r"|never\s+(?:returns?|comes?\s+back)"
    r"|(?:doesn'?t|does\s+not|won'?t)\s+(?:return|come\s+back)"
    r"|disappears?|vanish(?:es)?)\b",
    re.IGNORECASE,
)


def _clean_noun(noun: str) -> str | None:
    words = [w for w in noun.lower().split() if w not in _NOT_NOUNS]
    if not words:
        return None
    return " ".join(words)


def structure_prompt(prompt: str, *, v2: bool = False) -> str:
    """The user's prompt, verbatim, followed by derived continuity rules.

    Never rewrites, reorders or paraphrases a single word of the input — the
    output CONTAINS the input as its first block, byte for byte. A prompt that
    already carries explicit structure (Persistent:/Section N:) is returned
    unchanged.

    ``v2`` (execution.prompt_structuring_v2) changes only the appended block,
    never the user's text: the presence clause and the camera clause become
    separate bullets so they can hold or yield independently; the camera
    clause answers a user-asserted static camera with a static rule instead of
    contradicting it; the presence and count assertions are withheld when the
    user's own text says somebody leaves; and the block header is one this
    module's own ``_ALREADY_STRUCTURED`` recognises, so structuring is
    idempotent. Off (the default), the output is byte-identical to what has
    always shipped.
    """
    text = prompt.strip()
    # v2's output is prose and carries no header, so the header regex below
    # cannot recognise it on a second pass. Its own constancy sentence is the
    # marker instead — present in every v2 block, and not a phrase a customer
    # writes. Without this, structuring its own output appends a second copy
    # of every rule.
    if _V2_MARKER in text:
        return text
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

    if not v2:
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

    # v2. The header carries its qualifier AFTER the colon so the block
    # matches _ALREADY_STRUCTURED and _PERSISTENT_LINE on a second pass —
    # the v1 header's parenthetical sat before the colon and matched neither,
    # so structure_prompt would restructure its own output if ever called
    # twice.
    # Prose, not a labelled list. This block used to open "CONTINUITY: these
    # rules hold for the entire video." above a set of "- " bullets, and on
    # 28 Aug a customer's image-to-video SPOKE it: "the same rules hold for
    # the entire video" at 0:12, "same subjects keep the same faces ... every
    # frame" at 0:15. The runtime writes its audio from this same text, and a
    # document-shaped block reads as something to be read out loud. It is
    # also the wrong shape for the text encoder -- Lightricks instruct their
    # own enhancer to emit no headings, markdown or leading special
    # characters, which is why the music-video module is prose too.
    #
    # Every constraint below is unchanged in content. Only the furniture is
    # gone, and the furniture is what the model was reciting.
    # The user's own text closes with a full stop before anything is appended
    # to it. Without one the constraint sentences fuse onto it -- "a person
    # telling how to use a gun The same subjects keep..." -- which reads as
    # one run-on to the encoder and leaves the section planner no sentence
    # boundary to work from.
    opening_text = text.rstrip()
    if opening_text and opening_text[-1] not in ".!?":
        opening_text += "."
    sentences = [opening_text]
    sentences += [rule.rstrip().rstrip(".") + "." for rule in rules]
    if _EXIT_WORDS.search(text):
        # Someone leaves. Identity constancy still holds for whoever is on
        # screen; presence and count assertions would argue with the user's
        # own story, so they are withheld.
        sentences.append(
            "The same subjects keep the same faces, clothing and colours "
            "in every frame."
        )
    else:
        sentences.append(
            "The same subjects keep the same faces, clothing, colours and "
            "count in every frame."
        )
        sentences.append(
            "Every subject present at the start is still present at the end."
        )
    if _STATIC_CAMERA.search(text):
        sentences.append(
            "It is one continuous scene and the camera holds perfectly still "
            "in the same environment from the first frame to the last."
        )
    else:
        sentences.append(
            "It is one continuous scene and the camera keeps moving through "
            "the same environment."
        )
    return " ".join(sentences)
