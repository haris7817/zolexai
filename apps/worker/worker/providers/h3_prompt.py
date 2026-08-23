"""The same ZolexAI plan, written the way MiniMax H3 documents.

Provider-specific compilation is the whole reason the Director plan is a
structured object rather than a paragraph. LTX and H3 must receive the same
STORY — the same characters, the same lines, the same beats, the same camera
intent — and they must receive it in different prose, because the two engines
document opposite conventions:

  * LTX: one flowing chronological paragraph, "Initially… / A moment later…",
    and its own enhancer prompts explicitly FORBID timestamps or labels.
  * H3: three named fields (`integrated_multimodal_description`,
    `overall_soundscape`, `non_diegetic_music`), numbered speaker ids, dialogue
    inside `<d>` tags carrying a language tag, a CLOSED camera-motion
    vocabulary, and timestamps on every shot after the first.

Writing one prompt for both would hand at least one of them a format its own
documentation rejects, and the benchmark would then be measuring our prose
rather than the models. So the semantic plan is compiled twice.

Nothing here is tuned. Wording that only a render can settle — how much
description H3 wants, whether its shot timestamps help or hurt — is a GPU
question, and the benchmark exists to answer it.

Source: MiniMaxAI/MiniMax-H3 docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and
docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md, read 2026-08-22.
"""

from __future__ import annotations

import re

from worker.director.plan import DirectorPlan

#: H3's documented motion vocabulary, verbatim from the base guide's table.
#: A camera phrase that cannot be expressed in these terms is passed through
#: as prose rather than silently dropped — but it is recorded as unmapped, so
#: the benchmark can tell "H3 ignored the request" from "we never asked".
H3_CAMERA_MOTIONS = (
    "Zoom In", "Zoom Out", "Push In", "Pull Out",
    "Pan Left", "Pan Right", "Truck Left", "Truck Right",
    "Tilt Up", "Tilt Down", "Pedestal Up", "Pedestal Down",
    "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV",
    "Roll Clockwise", "Roll Counterclockwise",
)

#: Our Director vocabulary to H3's. Only unambiguous pairs; anything else
#: stays prose. "push-in" and "dolly in" are the same move in both worlds;
#: "orbit" is H3's "Arc Shot"; a static camera is a first-class term there.
_MOTION_ALIASES: dict[str, str] = {
    "static": "Static Shot",
    "locked": "Static Shot",
    "still": "Static Shot",
    "push in": "Push In",
    "push-in": "Push In",
    "pushes in": "Push In",
    "dolly in": "Push In",
    "pull out": "Pull Out",
    "pulls out": "Pull Out",
    "dolly out": "Pull Out",
    "zoom in": "Zoom In",
    "zoom out": "Zoom Out",
    "pan left": "Pan Left",
    "pan right": "Pan Right",
    "tilt up": "Tilt Up",
    "tilt down": "Tilt Down",
    "orbit": "Arc Shot",
    "circles": "Arc Shot",
    "arc": "Arc Shot",
    "tracking": "Tracking Shot",
    "tracks": "Tracking Shot",
    "follows": "Tracking Shot",
    "handheld": "Shake Slightly",
    "crane up": "Pedestal Up",
    "crane down": "Pedestal Down",
    "pov": "POV",
}

#: The language tag inside `<d>`. Our Director speaks the Dub-It validated set;
#: H3 states eleven stable languages, a superset of it.
_LANGUAGE_TAGS = {
    "english": "English",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "russian": "Russian",
}


def map_camera_motion(camera: str) -> tuple[str, bool]:
    """(H3 motion term, mapped). Unmapped text is returned unchanged.

    Longest alias first, so "dolly in" never matches on the bare "in" of some
    other phrase and "pan left" beats a hypothetical "pan".
    """
    text = (camera or "").strip().lower()
    if not text:
        return "", False
    for alias in sorted(_MOTION_ALIASES, key=len, reverse=True):
        if alias in text:
            return _MOTION_ALIASES[alias], True
    return camera.strip(), False


def _shot_phrase(camera: str) -> str:
    """H3 wants the shot size as prose and the motion as an action in it."""
    parts = [p.strip() for p in re.split(r"[,;]", camera or "") if p.strip()]
    if not parts:
        return ""
    shot = parts[0]
    motion, _ = map_camera_motion(", ".join(parts[1:]) if len(parts) > 1 else "")
    if not motion:
        return f"The shot is a {shot.lower()}."
    if motion == "Static Shot":
        return f"The shot is a {shot.lower()} and the camera holds a Static Shot."
    return f"The shot is a {shot.lower()} and the camera performs a {motion} at slow speed."


def _timestamp(seconds: float) -> str:
    """`00:03.500` — H3's documented shot-timestamp format, two decimals."""
    minutes, rest = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{rest:06.3f}"


def compile_h3_section(
    plan: DirectorPlan,
    events,
    *,
    index: int,
    total: int,
    window_start: float,
    task_line: str = "",
) -> str:
    """One H3 prompt for one section of the SAME plan LTX would render.

    Timestamps are relative to this section, because each section is its own
    generation and H3 requires shot times inside the clip's duration.
    """
    speakers = {c.id: f"S{n + 1}" for n, c in enumerate(plan.characters)}
    lines: list[str] = []
    if task_line:
        lines += [task_line, ""]

    description: list[str] = [plan.scene.strip().rstrip(".") + "."]
    for character in plan.characters:
        appearance = character.appearance.strip().rstrip(".")
        tag = speakers[character.id]
        if appearance:
            description.append(
                f"The {character.role.strip().lower()} ({tag}) is {appearance}."
            )
        else:
            description.append(f"The {character.role.strip().lower()} ({tag}) is present.")

    for position, event in enumerate(events):
        clause: list[str] = []
        # The first shot of a clip takes no timestamp; later ones do.
        if position:
            clause.append(
                f"[Shot {position + 1}] At {_timestamp(max(0.0, event.start - window_start))},"
            )
        shot = _shot_phrase(event.camera)
        action = (event.action or "").strip().rstrip(".")
        if action:
            clause.append(f"{action}." if not clause else f"{action[0].lower()}{action[1:]}.")
        if shot:
            clause.append(shot)
        dialogue = (event.dialogue or "").strip()
        if dialogue and event.speaker in speakers:
            character = plan.character(event.speaker)
            tag = speakers[event.speaker]
            manner = (event.delivery or character.voice or "").strip().rstrip(".")
            manner_text = f" in a {manner} voice" if manner else ""
            language = _LANGUAGE_TAGS.get(plan.language.lower(), "English")
            clause.append(
                f"The {character.role.strip().lower()} ({tag}) says{manner_text}: "
                f"<d>[{language}]{dialogue}</d>"
            )
        if clause:
            description.append(" ".join(clause))

    for fact in plan.continuity:
        description.append(fact.strip().rstrip(".") + ".")

    ambience = plan.ambience.strip().rstrip(".") or "quiet natural room ambience"

    lines.append("integrated_multimodal_description: " + " ".join(description))
    lines.append("")
    lines.append(
        "overall_soundscape: " + ambience[0].upper() + ambience[1:] + " continues under the scene."
    )
    lines.append("")
    # The guide is explicit that non-diegetic music is its own field and that
    # abstract mood words do not belong in it.
    lines.append("non_diegetic_music: None.")
    return "\n".join(lines)


def compile_h3_text_section(
    prompt: str, *, index: int, total: int, task_line: str = ""
) -> str:
    """The standard-mode equivalent: the user's own text, in H3's fields.

    The customer's words are copied verbatim into the description exactly as
    the LTX path copies them — a benchmark comparing two engines must not be
    comparing two different prompts.
    """
    lines: list[str] = []
    if task_line:
        lines += [task_line, ""]
    body = prompt.strip()
    if total > 1:
        body = (
            f"{body} This is part {index + 1} of {total} of one continuous scene, "
            "continuing directly from the previous moment without restarting it."
            if index
            else body
        )
    lines.append("integrated_multimodal_description: " + body)
    lines.append("")
    lines.append("overall_soundscape: Natural ambience appropriate to the scene.")
    lines.append("")
    lines.append("non_diegetic_music: None.")
    return "\n".join(lines)
