"""Which LTX 2.5 prompt form a job is asking for, and the text that governs it.

`core.txt` §1 makes the choice the first decision and warns against making it
by accident: *"A camera pan, track, orbit, tilt, push-in, or pullback is
movement inside one shot. A hard cut, match cut, dissolve, or other named edit
begins another shot."* That sentence is the whole router — everything below
is it, applied to a customer's text.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from functools import cache, lru_cache
from importlib import resources


class Form(StrEnum):
    """The four forms `core.txt` defines."""

    SINGLE_SHOT = "single_shot"
    MULTI_SHOT = "multi_shot"
    SCREENPLAY = "screenplay"
    DUB_IT = "dub_it"

    @property
    def guide(self) -> str:
        return {
            Form.SINGLE_SHOT: "single-shot",
            Form.MULTI_SHOT: "multi-shot",
            Form.SCREENPLAY: "screenplay",
            Form.DUB_IT: "dub-it",
        }[self]


@cache
def guideline(name: str) -> str:
    """One vendored guideline file, verbatim.

    Cached because these are read once per job and never change at runtime;
    read through `importlib.resources` rather than a path relative to
    `__file__` so a zipped or relocated install still finds them.
    """
    return (
        resources.files("worker.prompt.ltx25")
        .joinpath("guidelines", f"{name}.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


@lru_cache(maxsize=1)
def terminology() -> dict:
    return json.loads(
        resources.files("worker.prompt.ltx25")
        .joinpath("guidelines", "terminology.json")
        .read_text(encoding="utf-8")
    )


# ── Reading the customer's text ────────────────────────────────────────────

#: `terminology.json`'s `editing_transitions`, as patterns. An edit named in
#: the customer's own words is the strongest signal there is that they want
#: cuts, and `validation.txt` §1 makes one inside SINGLE_SHOT an ERROR — so
#: the same list decides the form and later checks the result.
_EDIT_PATTERNS = (
    r"\bhard\s+cuts?\b",
    r"\bmatch\s+cuts?\b",
    r"\bjump\s+cuts?\b",
    r"\bsmash\s+cuts?\b",
    r"\bquick\s+cuts?\b",
    r"\brapid\s+cuts?\b",
    r"\bcross[- ]?cut",
    r"\binter[- ]?cut",
    r"\bcuts?\s+(?:to|back|away|between)\b",
    r"\bcutting\s+(?:to|between)\b",
    r"\bdissolves?\s+(?:to|into)\b",
    r"\bfades?\s+(?:to|into|out|in)\b",
    r"\bmontage\b",
    r"\btransitions?\s+to\b",
    r"\bthen\s+we\s+see\b",
    r"\bseries\s+of\s+shots\b",
    r"\bsecond\s+shot\b",
    r"\bnext\s+shot\b",
    r"\bshot\s+two\b",
    r"\bscene\s+2\b",
)

#: Screenplay signals. Sustained dialogue is the documented trigger
#: (`screenplay.txt`: "sustained dialogue, several performance beats,
#: multiple speakers, or precise dramatic sequencing"), and quoted lines are
#: how sustained dialogue looks in a prompt. Two or more of them is the bar —
#: one quoted line is a single-shot clip that happens to speak, which is the
#: commonest prompt this product receives and must not be reformatted.
_SCREENPLAY_PATTERNS = (
    r"\bscreenplay\b",
    r"\bscript\s+format\b",
    r"\bdialogue\s*:",
    r"\bINT\.\s",
    r"\bEXT\.\s",
    r"\bconversation\s+between\b",
    r"\bthey\s+(?:talk|argue|discuss|debate)\b",
    r"\bback\s+and\s+forth\b",
)

def named_edits(text: str) -> list[str]:
    """Every editing transition the text names, in order of appearance."""
    found: list[tuple[int, str]] = []
    for pattern in _EDIT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append((match.start(), match.group(0)))
    return [phrase for _, phrase in sorted(found)]


def quoted_lines(text: str) -> list[str]:
    """Spoken lines, as `audio-dialogue.txt` requires them to be written.

    Double quotes only — an apostrophe in "don't" is not a spoken line, and
    counting it as one would route half the prompts on this platform into the
    screenplay form.
    """
    return [
        match.group(1).strip()
        for match in re.finditer(r"[\"“]([^\"“”]{2,}?)[\"”]", text)
        if match.group(1).strip()
    ]


def looks_screenplay(text: str) -> bool:
    if any(re.search(p, text, re.IGNORECASE) for p in _SCREENPLAY_PATTERNS):
        return True
    return len(quoted_lines(text)) >= 2


def looks_multi_shot(text: str) -> bool:
    return bool(named_edits(text))


def select(
    text: str,
    *,
    workflow_id: str,
    requested: str = "",
    anchored: bool = False,
) -> Form:
    """The form this prompt should be written in.

    The client's routing order (8 Sep 2026), with their two caveats kept:

      * **Dub-It only for replacing speech in a source video.** No workflow
        here does that, so nothing selects it — the guide is vendored and the
        constant exists so the day one arrives it is a routing line, not a
        research task. Their own words: *"The two women talking in the park
        use normal text-to-video dialogue — not Dub-It."*
      * **Image to Video stays single-shot unless a cut is explicitly
        requested**, which `core.txt` §5 states as well ("Preserve one
        continuous take unless a cut away from the opening image is
        explicitly requested"). An anchored job therefore needs a NAMED edit
        to leave single-shot; screenplay alone will not move it, because a
        talking head anchored to a photograph is one take.

    `requested` is an explicit override (`execution.ltx25_form`), because a
    heuristic over prose will be wrong sometimes and the answer to that is a
    switch, not a rewrite.
    """
    override = requested.strip().lower().replace("-", "_")
    if override in {form.value for form in Form}:
        return Form(override)

    if workflow_id == "dub-it":  # no such workflow today; see the docstring
        return Form.DUB_IT

    multi = looks_multi_shot(text)
    if anchored:
        return Form.MULTI_SHOT if multi else Form.SINGLE_SHOT
    if looks_screenplay(text):
        return Form.SCREENPLAY
    if multi:
        return Form.MULTI_SHOT
    return Form.SINGLE_SHOT


__all__ = [
    "Form",
    "guideline",
    "looks_multi_shot",
    "looks_screenplay",
    "named_edits",
    "quoted_lines",
    "select",
    "terminology",
]
