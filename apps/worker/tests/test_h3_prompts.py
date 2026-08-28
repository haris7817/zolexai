"""The long-form prompt discipline, held to the failures it exists to prevent.

Every assertion here maps to a drift actually observed on the GPU (25 Aug
2026): wardrobe changing mid-run, the scene resetting, a departed subject
implied back into frame, a segment collapsing into a portrait because its
prompt no longer said who was on screen.
"""

from __future__ import annotations

from worker.longform.h3_prompts import (
    _SEAM_SAFE_SHOTS,
    H3ScenePlan,
    discipline_prompts,
    plan_cameras,
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
    # Segments past the supplied beats get arc discipline instead of a
    # blank continuation — the 26 Aug frame-audit showed "continues
    # naturally" let a two-segment video re-enact its whole story twice.
    assert "never restart" in prompts[3]
    assert "conclusion" in prompts[4]  # the final segment must land an ending


def test_free_text_segments_get_an_arc_not_a_retelling() -> None:
    """The 26 Aug client frame-audit: a 30s two-segment story prompt told the
    story twice — climax at 6s, its own setup at 20s. Without structured
    beats every segment must know its place in ONE story.

    Since the 27 Aug identity/action split the arc is delivered differently:
    segment 1 carries the action and segment 2 carries the closing arc, so
    the action is stated once rather than re-issued. The property under test
    is the same one — the story must not be told twice."""
    plan = plan_from_prompt("A woman transforms into a mech and fights a demon.")
    first, second = discipline_prompts(plan, 2)
    assert "Transforms into a mech" in first     # the action happens here…
    assert "mech" not in second                  # …and is never re-issued
    assert "FINAL" in second
    assert "never restart" in second

    # A single-segment video needs no arc scaffolding, and keeps its action.
    [solo] = discipline_prompts(plan, 1)
    assert "FINAL" not in solo
    assert "Transforms into a mech" in solo


def test_an_arc_still_appears_when_the_prompt_has_no_separable_action() -> None:
    """A pure scene description cannot be split, so the arc beats remain the
    mechanism that stops the retelling."""
    plan = plan_from_prompt("A koi pond at dawn, mist over still water")
    prompts = discipline_prompts(plan, 2)
    assert "OPENING" in prompts[0]
    assert "FINAL" in prompts[1]


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


def test_a_customer_scripted_timeline_is_honoured() -> None:
    """The 26 Aug military-rescue audit: the customer scripted five timed
    shots and the compiler stuffed ALL of them into both segments as 'the
    subject' — so segment 1 raced the whole mission and segment 2 re-enacted
    it. A timeline prompt must be split: preamble = identity, each block =
    a beat for its own segment only."""
    prompt = (
        "30-second cinematic military rescue. The same battle-worn army "
        "soldier remains consistent throughout, wearing olive tactical gear. "
        "[0-6s] A wide aerial shot follows him through the jungle toward the "
        "compound. [6-12s] He enters the building, flashlight cutting through "
        "dust. [12-18s] He hears knocking and opens a concealed steel door. "
        "[18-24s] He descends into the bunker and frees three prisoners. "
        "[24-30s] He leads the group outside as a helicopter approaches."
    )
    plan = plan_from_prompt(prompt)
    assert plan.subject.startswith("30-second cinematic military rescue")
    assert "aerial" not in plan.subject  # the timeline is not the subject
    assert len(plan.timed_beats) == 5

    seg1, seg2 = discipline_prompts(plan, 2, total_seconds=30.0)
    # Segment 1 owns only its own slice of the story…
    assert "aerial" in seg1 and "flashlight" in seg1
    assert "helicopter" not in seg1 and "bunker" not in seg1
    # …and segment 2 owns the rest, including the payoff.
    assert "bunker" in seg2 and "helicopter" in seg2
    assert "aerial" not in seg2
    # Identity is re-stated at every scripted shot boundary, not once per
    # segment — the race audit counted four different hero cars inside
    # segments that named the subject a single time.
    for prompt in (seg1, seg2):
        assert prompt.count("battle-worn army") >= 2


def test_prompts_without_a_timeline_keep_the_free_text_path() -> None:
    plan = plan_from_prompt("A koi pond at dawn, one fish gliding slowly.")
    assert plan.timed_beats == ()
    assert plan.subject == "A koi pond at dawn, one fish gliding slowly"
    # A single stray time tag is not a timeline.
    plan2 = plan_from_prompt("A parade [0-6s] with floats.")
    assert plan2.timed_beats == ()


def test_the_action_leaves_the_subject_slot_so_segment_two_cannot_replay_it() -> None:
    """The 30s H3 loop, reproduced 27 Aug: with the whole prompt in the
    subject slot, segment 2 was told "the subject is still a man ... ordering
    a pizza" and ordered the pizza again — the second half re-enacted the
    first. Identity must persist; the action must not."""
    plan = plan_from_prompt(
        "A man in a hotel room talking on the phone ordering a pizza"
    )
    assert plan.subject == "A man in a hotel room"
    assert plan.beats == ("Talking on the phone ordering a pizza",)

    first, second = discipline_prompts(plan, 2)
    assert "ordering a pizza" in first          # it happens once…
    assert "ordering a pizza" not in second     # …and is never re-issued
    assert "A man in a hotel room" in second    # identity still persists
    assert "never restart or replay" in second


def test_wardrobe_stays_on_the_identity_side_of_the_split() -> None:
    """Appearance verbs ("wearing") must not be read as actions, or every
    later segment loses the clothing that identity depends on."""
    plan = plan_from_prompt(
        "A woman with long dark curly hair, wearing a deep emerald velvet "
        "jacket, sings into a vintage silver microphone"
    )
    assert "velvet jacket" in plan.subject
    assert plan.beats == ("Sings into a vintage silver microphone",)


def test_a_scene_description_with_no_action_is_left_alone() -> None:
    plan = plan_from_prompt("A koi pond at dawn, mist drifting over still water")
    assert plan.subject == "A koi pond at dawn, mist drifting over still water"
    assert plan.beats == ()


# ── The 28 Aug seam failure ──────────────────────────────────────────────
# A 30s T2V render (two 15s segments) cut hard at the boundary: segment 1 had
# been told to push in "until the subject fills the frame", so segment 2
# conditioned on a face with no floor, no body and no furniture in it. It read
# the head as a head-sized OBJECT lying on a desk, and built a whole room
# around that reading. Nothing about the identity discipline was wrong — the
# missing constraint was scale.


def test_no_segment_hands_off_a_frame_with_no_scale_in_it() -> None:
    """Every non-final segment must end wide enough to show the scale."""
    prompts = discipline_prompts(PLAN, 4, cameras=plan_cameras(4))
    for prompt in prompts[:-1]:
        assert "so the next segment inherits the scale of the room" in prompt
    # The last segment has no successor to hand anything to.
    assert "inherits the scale" not in prompts[-1]


def test_planned_cameras_never_end_a_segment_tight() -> None:
    """The music-video director's "until the subject fills the frame" is what
    produced a scale-free handoff frame. No planned shot may end there."""
    for camera in plan_cameras(len(_SEAM_SAFE_SHOTS) + 2):
        assert "fills the frame" not in camera
        assert "close-up" not in camera.lower()
        # Each shot names the body-and-room end state the next one inherits.
        assert any(word in camera for word in ("room", "floor", "waist", "knees"))


def test_planned_cameras_still_vary_between_segments() -> None:
    """Holding scale must not regress into one locked-off frame for 30s."""
    cameras = plan_cameras(4)
    assert len(set(cameras)) == 4


def test_every_continuation_restates_the_scale() -> None:
    for prompt in discipline_prompts(PLAN, 5)[1:]:
        assert "same scale and camera distance" in prompt
        assert "life-size person" in prompt


def test_the_reported_prompt_no_longer_replays_its_own_action() -> None:
    """"telling" was not in the action vocabulary, so the whole prompt sat in
    the identity slot and segment 2 was told the subject "is still ... telling
    about aliens" — and told it again from the top."""
    plan = plan_from_prompt("Donald Trump at the oval office telling about aliens")
    assert plan.subject == "Donald Trump at the oval office"
    assert plan.beats == ("Telling about aliens",)
    first, second = discipline_prompts(plan, 2, total_seconds=30.0, cameras=plan_cameras(2))
    assert "Telling about aliens" in first
    assert "telling about aliens" not in second.lower()
    assert "Donald Trump at the oval office" in second


def test_every_segment_is_told_what_language_to_speak() -> None:
    """This engine writes its own audio per segment, from these prompts.

    Image to Video on Best routes here, which is exactly the case the client
    described on 28 Aug 2026 — "everything i hit sound in images comes with a
    language that is not english". The LTX fix shipped the same day did not
    reach this path, because H3 builds its prompts here and never mentioned
    language at all.

    On EVERY segment, not just the first: a video whose second half switches
    language is worse than one that never stated it.
    """
    from worker.longform.h3_prompts import discipline_prompts, plan_from_prompt
    from worker.longform.language import soundscape_clause

    plan = plan_from_prompt("a man in a grey coat talks to camera in a kitchen")
    sentence = soundscape_clause("a man in a kitchen", {}, {})
    prompts = discipline_prompts(plan, 3, total_seconds=30.0, spoken_language=sentence)

    assert len(prompts) == 3
    for prompt in prompts:
        assert prompt.endswith("The only sounds are the ones the scene itself makes.")


def test_no_language_sentence_leaves_the_segment_prompts_untouched() -> None:
    """"" is a real answer — sound off, or a language nobody recognises."""
    from worker.longform.h3_prompts import discipline_prompts, plan_from_prompt

    plan = plan_from_prompt("a man in a grey coat talks to camera in a kitchen")
    with_none = discipline_prompts(plan, 2, total_seconds=20.0)
    explicit = discipline_prompts(plan, 2, total_seconds=20.0, spoken_language="")
    assert with_none == explicit
    assert not any("No one speaks" in prompt for prompt in with_none)
