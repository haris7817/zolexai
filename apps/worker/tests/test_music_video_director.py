"""Music video shot direction — the layer that makes it look directed.

Pinned against the two client reports that produced it: every video looking
identical (one composition held for the whole song) and prompt scaffolding
rendering into the frame as on-screen text.
"""

from __future__ import annotations

from worker.longform.music_video import plan_shots, section_prompt, strip_negations


def _windows(count: int, span: float = 20.0):
    return [(i * span, span) for i in range(count)]


def test_roles_come_from_what_the_audio_is_doing() -> None:
    shots = plan_shots(
        _windows(5),
        sung_fractions=[0.0, 0.5, 0.9, 0.05, 0.6],
        loudness=[0.10, 0.14, 0.30, 0.11, 0.15],
    )
    roles = [s.role for s in shots]
    assert roles[0] == "intro"       # opens instrumental
    assert roles[2] == "chorus"      # sung and well above the median
    assert roles[3] == "bridge"      # instrumental in the middle is a break
    assert roles[4] == "chorus"      # the song ends singing: the big finish


def test_an_instrumental_ending_is_an_outro() -> None:
    shots = plan_shots(
        _windows(3), sung_fractions=[0.0, 0.9, 0.0], loudness=[0.1, 0.2, 0.1]
    )
    assert [s.role for s in shots] == ["intro", "chorus", "outro"]


def test_no_two_neighbouring_sections_share_a_framing() -> None:
    """The monotony this module exists to remove."""
    for count in range(2, 9):
        shots = plan_shots(_windows(count))
        framings = [s.framing for s in shots]
        assert all(a != b for a, b in zip(framings, framings[1:], strict=False))


def test_a_song_with_no_analysis_still_gets_a_varied_plan() -> None:
    shots = plan_shots(_windows(4))
    assert len(shots) == 4
    assert len({s.framing for s in shots}) > 1


def test_the_prompt_is_plain_prose() -> None:
    """Lightricks' own prompt enhancer is instructed to emit no headings,
    markdown or leading special characters, so a prompt built from ALL-CAPS
    labels is out of distribution for this text encoder."""
    [shot] = plan_shots(_windows(1))
    text = section_prompt("A red sports car on a wet street", shot, total=1)

    # No headings, no all-caps labels, no markup that reads as a caption.
    assert "SECTION" not in text
    assert "PERSISTENT" not in text
    assert "(verbatim)" not in text
    assert "\n" not in text
    # Exclusivity is stated POSITIVELY and names nothing unwanted: negative
    # prompting is documented to fail for logos/watermarks on this model
    # (LTX-Video issue #188), and naming a noun is a way to summon it.
    assert "contains only what this description names" in text
    for summonable in ("logo", "watermark", "caption", "subtitle"):
        assert summonable not in text.lower()
    # The customer's subject leads.
    assert text.startswith("A red sports car on a wet street.")


def test_camera_direction_reaches_every_section() -> None:
    shots = plan_shots(_windows(4))
    for shot in shots:
        text = section_prompt("A singer at a microphone", shot, total=4)
        assert "Filmed as" in text
        assert shot.framing in text


def test_continuation_is_only_claimed_after_the_first_section() -> None:
    shots = plan_shots(_windows(3))
    first = section_prompt("A singer", shots[0], total=3)
    second = section_prompt("A singer", shots[1], total=3)
    assert "continues the same unbroken performance" not in first
    assert "continues the same unbroken performance" in second


def test_a_scripted_beat_lands_in_its_section() -> None:
    [shot] = plan_shots(_windows(1))
    text = section_prompt(
        "A singer", shot, total=1, beat="she steps into the rain"
    )
    assert "she steps into the rain." in text


def test_the_performer_is_described_in_every_later_section() -> None:
    """The 28 Aug client frame-audit: a three-minute video's singer was
    clean-shaven and skin-faded for two minutes, then grew a moustache, a soul
    patch, a different haircut and face tattoos between 139s and 145s. The
    prompt was 250 words of "Preserve the lead singer's exact face, hairstyle,
    tattoos" — an instruction about pixels the text encoder never saw, naming
    no visible attribute at all. Later sections must be TOLD who the person
    is, in the words the vision describer supplies from the video's own
    opening section."""
    facts = "a man in his late twenties, clean-shaven, very short black hair in a tight fade"
    shots = plan_shots(_windows(3))
    first = section_prompt("A Latin R&B singer", shots[0], total=3, identity=facts)
    later = section_prompt("A Latin R&B singer", shots[1], total=3, identity=facts)

    assert "clean-shaven" in later
    assert "tight fade" in later
    assert "stays the same person" in later
    # Section 1 IS the reference; describing it back to itself would only
    # compete with the customer's own words.
    assert "clean-shaven" not in first


def test_no_description_leaves_the_prompt_exactly_as_it_was() -> None:
    """Every describer failure returns "" — no checkpoint, a text-only
    checkpoint, a timeout, garbage. The render must not notice."""
    shots = plan_shots(_windows(2))
    assert section_prompt("A singer", shots[1], total=2, identity="") == (
        section_prompt("A singer", shots[1], total=2)
    )


def test_whole_prohibition_sentences_are_dropped() -> None:
    """A text encoder has no operator for "no": the customer's closing
    sentence contributed the tokens *duplicate people* and *warped hands* to
    every section of a three-minute video, and negative prompting is measured
    not to suppress these artifacts on this model family anyway
    (LTX-Video issue #188)."""
    subject = strip_negations(
        "A singer on a rooftop at night. No identity changes, face drift, "
        "duplicate people, warped hands, text, or logos. Cinematic 4K."
    )
    assert subject == "A singer on a rooftop at night. Cinematic 4K."


def test_a_negation_inside_a_sentence_keeps_its_description() -> None:
    """Only sentences that OPEN with a negating word carry no description by
    construction. "she walks without stopping" is a description."""
    text = "She walks without stopping along the pier."
    assert strip_negations(text) == text


def test_a_prompt_that_is_all_prohibitions_survives() -> None:
    """It is still the customer's prompt; an empty one would be worse."""
    text = "No text. No logos."
    assert strip_negations(text) == text
