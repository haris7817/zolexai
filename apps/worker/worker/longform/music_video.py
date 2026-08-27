"""Shot direction for music video — the layer that makes it look directed.

Lip-sync was solved by conditioning on the audio; the picture was not. A
music video reached every section with the same repeated prompt wrapped in
generic scaffolding, so a three-minute video was one composition held for
three minutes, and the scaffolding's own labels ("NEW ACTION FOR THIS
SECTION ONLY") rendered into the frame as garbled on-screen text — this
runtime reads captions as content, so a label in the prompt becomes a label
in the picture (client frame, 27 Aug 2026).

This module gives each section a ROLE and a SHOT, from the audio the worker
already analyses:

    vocal spans (stem separation)  → is anyone singing here?
    section loudness (RMS)         → is this the big moment or a quiet one?
    position in the song           → opening, middle, ending

    → intro / verse / chorus / bridge / outro
    → a framing and a camera move that suits that role and differs from the
      section before it

Deterministic, no model in the loop — the same discipline the rest of the
prompt layer follows. A song this cannot analyse still gets a plan: the
roles fall back to position alone, which is better than one shot repeated.

The rendered prompt is PROSE. No headings, no colons introducing a label,
no quoted markup — anything that looks like a caption is a caption to this
model.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Shot vocabulary per role. Framings first (what the audience sees), then a
#: movement. Chosen from terms video models demonstrably act on — shot size
#: and a single clear move — rather than film-school terms they average away.
_SHOTS: dict[str, tuple[tuple[str, str], ...]] = {
    "intro": (
        ("a wide establishing shot", "the camera drifts slowly forward"),
        ("a high wide shot", "the camera descends slowly toward the scene"),
        ("a slow wide reveal", "the camera glides sideways to uncover the scene"),
    ),
    "verse": (
        ("a medium shot at eye level", "the camera pushes in slowly"),
        ("a handheld medium shot", "the camera follows the movement, tracking"),
        ("a three-quarter medium shot", "the camera arcs gently around the subject"),
        ("a steady eye-level shot", "the camera holds still while the scene moves"),
    ),
    "chorus": (
        ("a close-up", "the camera pushes in hard and fast"),
        ("a low-angle hero shot", "the camera rises, looking up"),
        ("a sweeping wide shot", "the camera orbits around the subject"),
        ("a tight close-up", "the camera holds close as light moves across the frame"),
    ),
    "bridge": (
        ("a side-on profile shot", "the camera tracks slowly sideways"),
        ("a distant silhouette shot", "the camera stays back, barely moving"),
        ("an over-the-shoulder shot", "the camera drifts behind the subject"),
    ),
    "outro": (
        ("a slow pull-back to a wide shot", "the camera retreats, leaving space"),
        ("a final wide shot", "the camera settles and holds"),
    ),
}

#: A section counts as sung when this much of it carries vocal.
_SUNG_FRACTION = 0.35

#: A sung section is a chorus when its loudness clears the song's own median
#: by this margin — choruses are mixed hotter and denser than verses, which
#: is the only genre-proof signal available without a music-theory model.
_CHORUS_MARGIN = 1.06


@dataclass(frozen=True)
class ShotDirection:
    """One section's role and camera, ready to render into a prompt."""

    index: int
    role: str
    framing: str
    movement: str

    @property
    def camera_line(self) -> str:
        return f"Filmed as {self.framing}; {self.movement}."


def _role_for(
    index: int,
    total: int,
    sung: float,
    loudness: float,
    median_loudness: float,
) -> str:
    """Section role from what the audio is doing at that moment."""
    if index == 0 and sung < _SUNG_FRACTION:
        return "intro"
    if index == total - 1:
        return "outro" if sung < _SUNG_FRACTION else "chorus"
    if sung < _SUNG_FRACTION:
        # An instrumental passage in the middle of a song is a break.
        return "bridge"
    if median_loudness > 0 and loudness >= median_loudness * _CHORUS_MARGIN:
        return "chorus"
    return "verse"


def plan_shots(
    sections: list[tuple[float, float]],
    *,
    sung_fractions: list[float] | None = None,
    loudness: list[float] | None = None,
) -> list[ShotDirection]:
    """A shot per section: `sections` is [(start_seconds, duration), …].

    `sung_fractions` and `loudness` are parallel lists when the analysis is
    available. Without them the plan falls back to position alone — an
    opening, a middle, an ending — which still varies the picture.
    """
    total = len(sections)
    if total == 0:
        return []

    fractions = sung_fractions or [1.0] * total
    levels = loudness or [1.0] * total
    ordered = sorted(levels)
    median = ordered[len(ordered) // 2] if ordered else 0.0

    shots: list[ShotDirection] = []
    used_framings: list[str] = []
    for index, _ in enumerate(sections):
        role = _role_for(
            index,
            total,
            fractions[index] if index < len(fractions) else 1.0,
            levels[index] if index < len(levels) else 1.0,
            median,
        )
        options = _SHOTS[role]
        # Rotate through the role's options AND refuse the previous framing,
        # because two identical framings across a cut is the monotony this
        # module exists to remove.
        choice = options[index % len(options)]
        if used_framings and choice[0] == used_framings[-1] and len(options) > 1:
            choice = options[(index + 1) % len(options)]
        used_framings.append(choice[0])
        shots.append(
            ShotDirection(
                index=index, role=role, framing=choice[0], movement=choice[1]
            )
        )
    return shots


def section_prompt(
    subject: str,
    shot: ShotDirection,
    *,
    total: int,
    beat: str = "",
    performance: str = "",
) -> str:
    """One section's complete prompt, as prose.

    Structure follows what the picture needs, in the order the model reads
    best: what this is, who/what is in it, how it is filmed, what changes
    here, and the closing rule that nothing else may join the scene.

    No labels, no headings, no all-caps: this runtime renders text it reads
    as text in the picture.
    """
    subject = subject.strip().rstrip(".")
    parts = [f"{subject}."]
    if shot.index > 0:
        parts.append(
            "This continues the same unbroken performance, the same subject and "
            "the same place as a moment ago."
        )
    parts.append(shot.camera_line)
    if beat:
        parts.append(beat.strip().rstrip(".") + ".")
    if performance:
        parts.append(performance.strip())
    parts.append(
        "Nothing enters the scene that is not described above — no extra "
        "people, no crowd, no audience, no text, no logos, no captions and no "
        "watermark anywhere in the picture."
    )
    return " ".join(parts)
