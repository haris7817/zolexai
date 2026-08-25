"""The long-form prompt discipline, held to the failures it exists to prevent.

Every assertion here maps to a drift actually observed on the GPU (25 Aug
2026): wardrobe changing mid-run, the scene resetting, a departed subject
implied back into frame, a segment collapsing into a portrait because its
prompt no longer said who was on screen.
"""

from __future__ import annotations

from worker.longform.h3_prompts import (
    H3ScenePlan,
    discipline_prompts,
    plan_from_prompt,
)

PLAN = H3ScenePlan(
    subject="a man in his fifties with a short grey beard",
    wardrobe="heavy navy wool coat",
    environment="rehearsal room with the bulb-ringed mirror",
    props=("microphone",),
    camera="The camera holds one steady medium shot.",
)


def test_every_segment_restates_the_subject() -> None:
    prompts = discipline_prompts(PLAN, 5)
    assert len(prompts) == 5
    for prompt in prompts:
        assert "short grey beard" in prompt


def test_same_clothing_in_every_segment() -> None:
    for prompt in discipline_prompts(PLAN, 5):
        assert "navy wool coat" in prompt


def test_same_room_in_every_segment() -> None:
    for prompt in discipline_prompts(PLAN, 5):
        assert "rehearsal room" in prompt


def test_persistent_prop_carries_through() -> None:
    for prompt in discipline_prompts(PLAN, 5):
        assert "microphone" in prompt


def test_continuations_forbid_resets_and_new_subjects() -> None:
    prompts = discipline_prompts(PLAN, 5)
    for prompt in prompts[1:]:
        assert "Continue directly from the exact prior final frame" in prompt
        assert "no cut" in prompt
        assert "new subject" in prompt


def test_permanent_departure_never_reappears() -> None:
    plan = H3ScenePlan(
        subject="two friends at a kitchen table",
        environment="sunlit kitchen",
        departures={2: "the friend in the green jacket"},
    )
    prompts = discipline_prompts(plan, 5)
    # The departure happens IN segment 2…
    assert "left the frame completely" in prompts[1]
    # …and every later segment states the absence, which is what stops a
    # presence-blind continuation resurrecting them.
    for prompt in prompts[2:]:
        assert "does not appear again" in prompt
    # Segments before the exit say nothing about it.
    assert "does not appear again" not in prompts[0]
    assert "does not appear again" not in prompts[1]


def test_handoffs_chain_and_the_last_segment_concludes() -> None:
    prompts = discipline_prompts(PLAN, 5)
    for prompt in prompts[:-1]:
        assert "handoff" in prompt or "next segment" in prompt
    assert "Finish cleanly" in prompts[-1]


def test_single_segment_needs_no_handoff() -> None:
    (only,) = discipline_prompts(PLAN, 1)
    assert "next segment" not in only
    assert "fully visible" in only


def test_beats_land_in_their_own_segment_only() -> None:
    plan = H3ScenePlan(
        subject="a potter at her wheel",
        beats=("she centres the clay", "she raises the wall", "she trims the rim"),
    )
    prompts = discipline_prompts(plan, 5)
    assert "centres the clay" in prompts[0]
    assert "raises the wall" in prompts[1]
    assert "trims the rim" in prompts[2]
    # Segments past the supplied beats continue naturally instead of inventing.
    assert "continues naturally" in prompts[3]
    assert "continues naturally" in prompts[4]


def test_plan_from_prompt_repeats_the_whole_prompt_per_segment() -> None:
    plan = plan_from_prompt(
        "A drummer in a red jacket playing on a rooftop at dusk.",
        reference_labels=("<Picture 1>",),
    )
    prompts = discipline_prompts(plan, 5)
    for prompt in prompts:
        assert "drummer in a red jacket" in prompt
    for prompt in prompts[1:]:
        assert "<Picture 1>" in prompt


def test_two_references_get_one_owner_each() -> None:
    """The 25/26 Aug identity flip-flop: with both labels offered as 'the
    subject', the model chose — once the source video's singer (review pack
    03, wrong person), once the reference with its backdrop (sample 14,
    wrong scene). Picture 1 must own the person, Picture 2 the scene."""
    plan = plan_from_prompt(
        "He sings at the microphone.",
        reference_labels=("<Picture 1>", "<Picture 2>"),
    )
    [text] = discipline_prompts(plan, 1)
    assert "exactly the person in <Picture 1>" in text
    assert "location, lighting and framing of <Picture 2>" in text
    # Picture 2 must never be offered as an identity source.
    assert "person in <Picture 2>" not in text
