"""Where the reference person lands in the composited identity anchor.

`scripts/person_anchor.py` runs in the LTX environment and needs torch, CUDA
and OpenCV to matte anything — but its placement arithmetic needs none of that,
and the placement arithmetic is what shipped wrong. The module keeps every
heavy import inside the function that uses it, so the geometry can be loaded
and tested here.

The bug these cover, measured 20 Aug 2026: a head-and-shoulders reference was
bottom-aligned to a full-body source person's box, which plants a bust on the
ground at the dancer's feet. The renders carried a woman standing motionless in
the road for the whole video, next to the dancer. A headshot is the commonest
thing a customer uploads against a full-body clip, so this was the common case.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "person_anchor.py"


def _load():
    spec = importlib.util.spec_from_file_location("person_anchor", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchor = _load()


# ── which kind of photo is this ───────────────────────────────────────────


def test_a_matte_running_off_the_bottom_is_a_crop_of_a_person() -> None:
    # A headshot: the subject's box reaches the photo's last row.
    assert anchor.is_truncated((100, 0, 600, 1000), 1000) is True


def test_a_matte_that_stops_short_of_the_bottom_is_a_whole_person() -> None:
    # Feet visible with floor beneath them.
    assert anchor.is_truncated((100, 40, 600, 880), 1000) is False


def test_the_tolerance_forgives_a_row_or_two_of_background() -> None:
    """A photo cropped to the chest rarely lands on the exact final row."""
    height = 1000
    almost = height - round(anchor.REFERENCE_TRUNCATION_TOLERANCE * height)
    assert anchor.is_truncated((100, 0, 600, almost), height) is True


# ── where the cutout goes ─────────────────────────────────────────────────

#: A dancer in a full-body wide shot: tall, narrow, feet near the frame's base.
FULL_BODY_SOURCE = (440, 60, 590, 520)
#: A head-and-shoulders portrait cut out of its photo: roughly square.
HEADSHOT_CUTOUT = (500, 540)


def test_a_headshot_hangs_from_the_head_not_the_feet() -> None:
    """The shipped regression: the bust must not be planted on the ground."""
    size, (_, paste_y) = anchor.place_cutout(
        FULL_BODY_SOURCE, HEADSHOT_CUTOUT, truncated=True
    )
    sx0, sy0, sx1, sy1 = FULL_BODY_SOURCE

    assert paste_y == sy0, "a truncated reference is aligned by its head"
    # The old behaviour put the cutout's bottom on the source person's feet.
    assert paste_y + size[1] != sy1


def test_a_headshot_falls_back_to_the_box_width_without_head_bands() -> None:
    """The fallback when either matte is not shaped like a person.

    Fitting a bust by HEIGHT into a full-body box shrinks it to a torso-sized
    lump, so the box's width is the better of the two crude options. Head
    matching, below, is better than either.
    """
    size, _ = anchor.place_cutout(FULL_BODY_SOURCE, HEADSHOT_CUTOUT, truncated=True)
    box_w = FULL_BODY_SOURCE[2] - FULL_BODY_SOURCE[0]
    assert size[0] == pytest.approx(anchor.FIT_WIDTH * box_w, abs=1)


def test_a_whole_figure_still_meets_the_same_ground() -> None:
    """The un-truncated case is unchanged: feet to the source person's feet."""
    cutout = (300, 900)  # a standing person, head to shoes
    size, (_, paste_y) = anchor.place_cutout(
        FULL_BODY_SOURCE, cutout, truncated=False
    )
    assert paste_y + size[1] == FULL_BODY_SOURCE[3]


def test_a_whole_figure_stays_inside_the_box_it_is_fitted_to() -> None:
    cutout = (300, 900)
    size, (paste_x, paste_y) = anchor.place_cutout(
        FULL_BODY_SOURCE, cutout, truncated=False
    )
    sx0, sy0, sx1, sy1 = FULL_BODY_SOURCE
    assert size[0] <= sx1 - sx0 and size[1] <= sy1 - sy0
    assert paste_x >= sx0 and paste_y >= sy0


def test_a_close_up_source_keeps_the_scale_it_already_had() -> None:
    """The framing that was already good keeps its size.

    In a speaking-to-camera shot the source person's own matte runs off the
    bottom of the frame too, so their box is tall and the binding constraint on
    a wide bust is WIDTH — under the old arithmetic as well as the new. The
    scale is therefore unchanged, which is why this case rendered well before
    the fix and still does.

    The head's POSITION does move: the old code hung the bust from the bottom
    of a box whose bottom is the frame edge, not a pair of feet, which sat the
    reference's head below the source person's head. Head-alignment is the
    correction, not a regression — the rendered A/B on the 15 s speaking
    fixture is near-identical either way.
    """
    close_up_source = (330, 40, 700, 576)
    box_w = close_up_source[2] - close_up_source[0]
    box_h = close_up_source[3] - close_up_source[1]

    size, (_, paste_y) = anchor.place_cutout(
        close_up_source, HEADSHOT_CUTOUT, truncated=True
    )
    old_scale = min(
        anchor.FIT_WIDTH * box_w / HEADSHOT_CUTOUT[0],
        anchor.FIT_HEIGHT * box_h / HEADSHOT_CUTOUT[1],
    )
    assert size[1] == pytest.approx(round(HEADSHOT_CUTOUT[1] * old_scale), rel=0.02)
    assert paste_y == close_up_source[1]


# ── heads, not bounding boxes ─────────────────────────────────────────────


# `head_band` itself walks a numpy matte, and this worker environment has no
# numpy by design — the matting runs in the LTX environment behind a CLI. What
# is testable here, and what actually shipped wrong, is the arithmetic that
# consumes the bands.


def test_a_matte_holding_more_than_a_person_no_longer_inflates_the_cutout() -> None:
    """The measured regression: BiRefNet mattes the salient OBJECT.

    On a seam frame of a woman standing beside a car it returned both as one
    region, so the "person's box" spanned most of the frame. Scaling a bust to
    that width produced a head half the frame high. Matching the HEADS instead
    is indifferent to whatever else the matte swept up.
    """
    person_and_car = (100, 60, 900, 520)
    bust = (500, 540)

    box_only, _ = anchor.place_cutout(person_and_car, bust, truncated=True)
    head_matched, _ = anchor.place_cutout(
        person_and_car, bust, truncated=True,
        source_head=(430, 500), reference_head=(120, 380),
    )

    assert box_only[1] > 700, "the old arithmetic really does produce a giant"
    assert head_matched[1] < box_only[1] / 4
    assert head_matched[1] < person_and_car[3] - person_and_car[1]


def test_the_two_heads_line_up() -> None:
    source_head = (430, 500)
    reference_head = (120, 380)
    size, (paste_x, _) = anchor.place_cutout(
        (100, 60, 900, 520), (500, 540), truncated=True,
        source_head=source_head, reference_head=reference_head,
    )
    scale = size[0] / 500
    landed = paste_x + (reference_head[0] + reference_head[1]) / 2 * scale
    assert landed == pytest.approx(sum(source_head) / 2, abs=1)


def test_head_matching_still_hangs_from_the_top() -> None:
    _, (_, paste_y) = anchor.place_cutout(
        FULL_BODY_SOURCE, HEADSHOT_CUTOUT, truncated=True,
        source_head=(470, 520), reference_head=(120, 380),
    )
    assert paste_y == FULL_BODY_SOURCE[1]


def test_a_whole_figure_ignores_the_head_bands() -> None:
    """Feet remain the landmark when the photo actually shows feet."""
    cutout = (300, 900)
    size, (_, paste_y) = anchor.place_cutout(
        FULL_BODY_SOURCE, cutout, truncated=False,
        source_head=(470, 520), reference_head=(120, 180),
    )
    assert paste_y + size[1] == FULL_BODY_SOURCE[3]
