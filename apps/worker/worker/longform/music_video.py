"""Shot direction for music video — the layer that makes it look directed.

Lip-sync was solved by conditioning on the audio; the picture was not. A
music video reached every section with the same repeated prompt wrapped in
generic scaffolding, so a three-minute video was one composition held for
three minutes (client report, 27 Aug 2026: "all the videos no matter the
prompt come out the same").

The scaffolding was also the wrong SHAPE of input. Lightricks' own prompt
enhancer is instructed to emit no headings, markdown or leading special
characters, so a prompt built from ALL-CAPS labels and "(verbatim)" markers
is out of distribution for the text encoder. (An earlier note here blamed
those labels for the garbled banner text in a client frame. That was a
guess: no vendor or practitioner source links prompt casing to burned-in
output text, while LTX-Video issue #188 documents spurious logos and
watermarks with no prompt cause at all, and a 2.3-era upscaler bug produced
exactly that artifact. Prose is right because it matches the training
distribution — not because it fixes the banners.)

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

The rendered prompt is PROSE: one flowing paragraph, plain language,
present tense, a camera move with an end state — the format Lightricks
documents for this model. Quoted speech is deliberately never emitted; in
audio-driven generation the model detects speech from the audio, and quotes
in the prompt are what it renders as subtitles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Shot vocabulary per role, built from the camera language Lightricks
#: publishes for this model — follows, tracks, pans across, circles around,
#: tilts upward, pushes in, pulls back, overhead view, handheld movement,
#: over-the-shoulder, wide establishing shot, static frame. Terms outside
#: that list are a gamble on a model that was not trained to hear them.
#:
#: Every move carries an END STATE ("pushes in until the face fills the
#: frame"), which their guide singles out as the difference between a move
#: the model finishes and one it drifts through. Language stays plain: the
#: vendor's own prompt enhancer is instructed to avoid intensified wording,
#: so "pushes in hard and fast" is worse input than "pushes in".
_SHOTS: dict[str, tuple[tuple[str, str], ...]] = {
    "intro": (
        ("a wide establishing shot", "the camera pushes in slowly until the scene fills the frame"),
        ("an overhead view", "the camera tilts down toward the scene and settles level"),
        ("a static frame", "the camera holds still while the scene moves through it"),
    ),
    "verse": (
        ("a medium shot at eye level", "the camera pushes in until the subject fills the frame"),
        ("a handheld medium shot", "the camera follows the movement and stays with the subject"),
        ("an over-the-shoulder shot", "the camera tracks forward past the shoulder"),
        ("a static frame at eye level", "the camera holds while the subject moves within it"),
    ),
    "chorus": (
        ("a close-up", "the camera pushes in until the face fills the frame"),
        ("a low-angle shot", "the camera tilts upward until the subject stands over it"),
        ("a wide shot", "the camera circles around the subject and comes to rest facing it"),
    ),
    "bridge": (
        ("a side-on medium shot", "the camera tracks sideways and stops"),
        ("a distant wide shot", "the camera holds still, far back"),
        ("an over-the-shoulder shot", "the camera drifts behind the subject and settles"),
    ),
    "outro": (
        ("a wide shot", "the camera pulls back until the whole scene is visible"),
        ("a static wide frame", "the camera settles and holds"),
    ),
}

#: A section counts as sung when this much of it carries vocal.
_SUNG_FRACTION = 0.35

#: A sung section is a chorus when its loudness clears the song's own median
#: by this margin — choruses are mixed hotter and denser than verses, which
#: is the only genre-proof signal available without a music-theory model.
_CHORUS_MARGIN = 1.06


#: Sentences that only tell the model what NOT to draw. A customer prompt
#: written in the house style of image-generation forums ends with a long one
#: — "No identity changes, face drift, clothing mutations, duplicate people,
#: warped hands, random objects, … text, logos, …" — and this repeats it into
#: every section of a three-minute video.
#:
#: It is removed for two documented reasons, neither of them a guess about
#: casing. Negative prompting is measured NOT to suppress these artifacts on
#: this model family (LTX-Video issue #188 documents logos and watermarks
#: surviving it), and the words are nouns: a text encoder has no operator for
#: "no", so "no duplicate people, warped hands" contributes the tokens
#: *duplicate people* and *warped hands* to the conditioning. Deleting the
#: sentence costs nothing that was working and stops the prompt naming the
#: failures it is asking to avoid.
_NEGATION_RE = re.compile(
    r"(?:^|(?<=[.!?]))\s*(?:no|avoid|without|never)\b[^.!?]*[.!?]",
    re.IGNORECASE,
)


def strip_negations(prompt: str) -> str:
    """The prompt with whole "no …" sentences removed.

    Conservative on purpose: only complete sentences that OPEN with a negating
    word go, because those carry no describable content by construction. A
    negation buried mid-sentence ("she walks without stopping") stays, since
    removing it would take real description with it. If the result would be
    empty the original is returned — a prompt made entirely of prohibitions is
    still the customer's prompt.
    """
    stripped = _NEGATION_RE.sub(" ", prompt)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or prompt.strip()


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
    identity: str = "",
) -> str:
    """One section's complete prompt, as prose.

    Structure follows what the picture needs, in the order the model reads
    best: what this is, who/what is in it, how it is filmed, what changes
    here, and the closing rule that nothing else may join the scene.

    Plain prose, no labels or headings — the shape Lightricks' own enhancer
    is instructed to produce, and therefore the shape this text encoder was
    trained on.
    """
    subject = subject.strip().rstrip(".")
    parts = [f"{subject}."]
    if shot.index > 0:
        parts.append(
            "This continues the same unbroken performance, the same subject and "
            "the same place as a moment ago."
        )
        if identity:
            # WHO is on screen, in visible attributes, taken from this video's
            # own opening section rather than from the customer's words.
            #
            # The failure this answers (client frame-audit, 28 Aug 2026): a
            # three-minute Latin R&B video whose singer was clean-shaven and
            # skin-faded for two minutes grew a moustache, a soul patch, a
            # longer textured haircut and face tattoos between 139s and 145s,
            # and his nose, brow and jaw drifted throughout. The prompt was
            # 250 words that never named one visible attribute of him — it
            # said "Preserve the lead singer's exact face, hairstyle, tattoos"
            # and left the model nothing to preserve, because there is no face
            # in that sentence. Text conditioning cannot follow an instruction
            # about pixels it was never given.
            #
            # A prompt that DOES describe its subject holds identity across
            # the same nine sections, which is the whole evidence for this
            # being a description problem and not a model one. So the worker
            # supplies the description the customer did not: the anchor still
            # is captioned once by the local vision checkpoint, in the
            # vocabulary that caption is written for (age, build, hair length
            # and style, facial hair NAMED when present, clothing colours),
            # and every later section is told who the person is instead of
            # being told to remember.
            parts.append(
                f"The performer stays the same person throughout: {identity.rstrip('.')}."
            )
    parts.append(shot.camera_line)
    if beat:
        parts.append(beat.strip().rstrip(".") + ".")
    if performance:
        parts.append(performance.strip())
    # Exclusivity, stated positively and naming nothing unwanted. The first
    # cut of this listed "no logos, no captions, no watermark" — but negative
    # prompting is documented to fail for exactly those artifacts on this
    # model (LTX-Video issue #188), and naming a noun is a way to summon it.
    parts.append("The scene contains only what this description names.")
    return " ".join(parts)
