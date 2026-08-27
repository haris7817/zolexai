"""Music video shot direction — the layer that makes it look directed.

Pinned against the two client reports that produced it: every video looking
identical (one composition held for the whole song) and prompt scaffolding
rendering into the frame as on-screen text.
"""

from __future__ import annotations

from worker.longform.music_video import plan_shots, section_prompt


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


def test_the_prompt_is_prose_with_no_caption_bait() -> None:
    """This runtime renders text it reads as text in the picture — the client
    frame that showed garbled banners and 'SECTION 1 ONLY' burned in."""
    [shot] = plan_shots(_windows(1))
    text = section_prompt("A red sports car on a wet street", shot, total=1)

    # No headings, no all-caps labels, no markup that reads as a caption.
    assert "SECTION" not in text
    assert "PERSISTENT" not in text
    assert "(verbatim)" not in text
    assert "\n" not in text
    # And it asks for a picture with no text in it at all.
    assert "no text" in text and "no watermark" in text
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
