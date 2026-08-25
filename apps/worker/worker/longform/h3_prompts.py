"""Per-segment prompt discipline for the H3 Extender long-form path.

Why this exists, measured rather than assumed: the client pack's 60-second
run with its shipped placeholder prompts drifted badly — the performer's
wardrobe changed in segment 2, segment 4 collapsed into a still portrait of
the identity reference, segment 5 reset the scene. The same graph, same seeds
and same references with prompts that re-state the subject every segment held
one man, one coat, one room and one camera across all five segments
(25 Aug 2026, `docs/internal/client-h3-comfyui-results.md` §8).

So the rule is the same one the LTX audit established for our own chain: a
continuation prompt that does not re-describe what must persist is an
invitation to reinvent it. Every segment prompt produced here re-states:

  subject identity · clothing · environment · persistent props · camera
  what changes THIS segment · what must NOT reappear · how it hands off

Deterministic templates, no model in the loop — the discipline is structure,
not prose quality, and a compiler that phoned an LLM per segment would make
long-form latency worse for no measured benefit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class H3ScenePlan:
    """What must stay true across a long-form generation.

    `subject` should read as an identity description ("a man in his fifties
    with a short grey beard"), because it is repeated verbatim in every
    segment — that repetition is the mechanism, exactly as in the proven run.
    """

    subject: str
    wardrobe: str = ""
    environment: str = ""
    props: tuple[str, ...] = ()
    camera: str = "The camera holds one steady shot."
    beats: tuple[str, ...] = ()
    """Optional per-segment action. Shorter than the segment count is fine —
    missing beats become natural continuation."""

    departures: dict[int, str] = field(default_factory=dict)
    """1-based segment index → what leaves the scene AT THE END of that
    segment, permanently. Later segments state its absence explicitly, because
    presence-blind continuation prompts are how departed people resurrect —
    the exact failure `DirectorEvent.exits` fixed on the LTX chain."""

    reference_labels: tuple[str, ...] = ()
    """Reference tags to keep alive in every segment ("<Picture 1>", …), so
    the model keeps reading the references instead of its own last guess."""

    timed_beats: tuple[tuple[float, float, str], ...] = ()
    """(start_s, end_s, action) blocks parsed from a customer's own
    "[0–6s] …" script. Mapped onto segments at compile time so each segment
    receives ONLY its own slice of the story — the 26 Aug military-rescue
    audit showed what happens otherwise: the full five-shot script rode into
    both segments as "the subject", segment 1 raced the whole mission and
    segment 2 re-enacted it from the top."""


#: "[0–6s]", "[12-18 s]" — en dash, em dash or hyphen, optional decimals.
_TIMED_SECTION = re.compile(
    r"\[\s*(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)\s*s\s*\]", re.IGNORECASE
)


def parse_timed_sections(
    prompt: str,
) -> tuple[str, tuple[tuple[float, float, str], ...]]:
    """(preamble, timed blocks) from a prompt that scripts its own timeline.

    Fewer than two markers means the prompt is not a timeline — returned
    unchanged so ordinary prompts keep the free-text path. The preamble
    (everything before the first marker) is where writers put the identity
    description, which is exactly what the subject slot wants.
    """
    matches = list(_TIMED_SECTION.finditer(prompt))
    if len(matches) < 2:
        return prompt, ()
    preamble = prompt[: matches[0].start()].strip()
    blocks: list[tuple[float, float, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        text = prompt[m.end() : end].strip().strip(";,").strip()
        if text:
            blocks.append((float(m.group(1)), float(m.group(2)), text))
    return preamble or prompt, tuple(blocks)


def _reference_bindings(plan: H3ScenePlan) -> tuple[str, str]:
    """(identity tag, scene clause) — each reference gets exactly one owner.

    The R2V mapping is a convention the prose must not blur: the FIRST label
    is the identity photograph, everything after it is scene. Offering every
    label as "the subject" let the model pick — on 25 Aug it picked the
    source video's singer over the customer's reference (review pack 03),
    and the 26 Aug 12-step sample picked the right person but kept his
    photo's backdrop. Worse, the opening segment never named the labels at
    all, so a single-segment video left the binding entirely to chance.
    """
    if not plan.reference_labels:
        return "", ""
    identity, *rest = plan.reference_labels
    identity_tag = f" (exactly the person in {identity})"
    scene_clause = (
        f"the location, lighting and framing of {', '.join(rest)}"
        if rest and not plan.environment
        else ""
    )
    return identity_tag, scene_clause


def _persistence_clause(plan: H3ScenePlan) -> str:
    identity_tag, scene_clause = _reference_bindings(plan)
    parts = [f"The subject is still {plan.subject}{identity_tag}"]
    if plan.wardrobe:
        parts.append(f"wearing the same {plan.wardrobe}")
    if plan.environment:
        parts.append(f"in the same {plan.environment}")
    elif scene_clause:
        parts.append(f"in {scene_clause}")
    for prop in plan.props:
        parts.append(f"with the same {prop}")
    return ", ".join(parts) + "."


def _absence_clause(plan: H3ScenePlan, segment: int) -> str:
    gone = [what for at, what in plan.departures.items() if at < segment]
    if not gone:
        return ""
    listed = "; ".join(gone)
    return (
        f" {listed} left the scene earlier and does not appear again — "
        "no reappearance, no replacement subject."
    )


def _beat(plan: H3ScenePlan, segment: int, segments: int = 1) -> str:
    if segment <= len(plan.beats) and plan.beats[segment - 1].strip():
        return plan.beats[segment - 1].strip().rstrip(".") + "."
    # No structured beats — every free-text production job. The old filler
    # ("The action continues naturally.") let each segment re-enact the whole
    # prompt: a 30s two-segment video told its story twice, with the climax
    # appearing before its own setup (client frame-audit, 26 Aug). Absent real
    # beats, the segments at least get an arc: begin, advance, conclude.
    if segments <= 1:
        return "The action plays out completely within this single shot."
    if segment == 1:
        return (
            "This is the OPENING of one single continuous story: establish "
            "the scene and begin the action — events that belong later must "
            "not happen yet."
        )
    if segment == segments:
        return (
            "This is the FINAL part of the same single take: carry the "
            "ongoing action to its conclusion and a clear final image — "
            "never restart or replay an earlier moment."
        )
    return (
        f"This is part {segment} of the same single take: continue the SAME "
        "ongoing action from exactly where it left off — never restart, "
        "repeat, or replay an earlier moment; what already happened stays "
        "finished."
    )


def _departure_clause(plan: H3ScenePlan, segment: int) -> str:
    what = plan.departures.get(segment)
    if not what:
        return ""
    return f" By the end of this segment {what} has left the frame completely."


def discipline_prompts(
    plan: H3ScenePlan, segments: int, total_seconds: float | None = None
) -> list[str]:
    """One prompt per segment, each self-sufficient about what persists."""
    if segments < 1:
        raise ValueError("segments must be >= 1")

    if plan.timed_beats and not plan.beats and total_seconds:
        # The customer scripted their own timeline — honour it: each block
        # joins the segment its midpoint falls in, so a segment carries only
        # its own slice of the story instead of the whole script.
        seg_len = total_seconds / segments
        per_segment: list[list[str]] = [[] for _ in range(segments)]
        for start, end, text in plan.timed_beats:
            mid = (start + end) / 2.0
            index = min(int(mid / seg_len), segments - 1)
            per_segment[index].append(text.rstrip("."))
        plan = replace(
            plan,
            beats=tuple(". Then ".join(parts) if parts else "" for parts in per_segment),
        )

    prompts: list[str] = []
    for segment in range(1, segments + 1):
        persistence = _persistence_clause(plan)
        beat = _beat(plan, segment, segments)
        departure = _departure_clause(plan, segment)
        absence = _absence_clause(plan, segment)

        if segment == 1:
            identity_tag, scene_clause = _reference_bindings(plan)
            opening = (
                f"One continuous cinematic shot: {plan.subject}{identity_tag}"
                + (f", wearing {plan.wardrobe}" if plan.wardrobe else "")
                + (f", in {plan.environment}" if plan.environment else "")
                + (f", in {scene_clause}" if scene_clause else "")
                + (
                    f", with {', '.join(plan.props)}"
                    if plan.props
                    else ""
                )
                + f". {plan.camera} {beat}{departure}"
            )
            handoff = (
                " End in a stable pose that the next segment can continue from."
                if segments > 1
                else " End cleanly with the subject fully visible."
            )
            prompts.append(opening + handoff)
            continue

        continuation = (
            "Continue directly from the exact prior final frame with no cut, "
            f"reset, replay, or new subject. {persistence} {plan.camera} "
            f"{beat}{departure}{absence}"
        )
        if segment == segments:
            continuation += (
                " Finish cleanly with the subject fully visible and no abrupt ending."
            )
        else:
            continuation += (
                " End in a stable handoff pose for the next segment."
            )
        prompts.append(continuation)
    return prompts


def plan_from_prompt(prompt: str, *, reference_labels: tuple[str, ...] = ()) -> H3ScenePlan:
    """The minimal plan when all we hold is the customer's free-text prompt.

    The whole prompt becomes the subject description — repeated per segment,
    which is precisely the proven fix. Structured fields (wardrobe, exits)
    arrive when a Director plan exists; a missing plan must not mean missing
    discipline.

    A prompt that scripts its own timeline ("[0–6s] …") is split instead:
    the preamble becomes the subject and each timed block becomes a beat for
    the segment it belongs to — the customer already did the Director's job.
    """
    subject, timed = parse_timed_sections(prompt)
    return H3ScenePlan(
        subject=subject.strip().rstrip("."),
        reference_labels=reference_labels,
        timed_beats=timed,
    )
