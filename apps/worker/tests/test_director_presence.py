"""Departures are permanent: the semantic state behind 60-second continuity.

The measurement this suite exists to keep fixed (GPU, 20 Aug 2026): a 60s
single-pass render had the man walk out at ~38s as planned — then flicker back
at 43-48s and stand fully returned for the final twelve seconds, while the
soundtrack said "He is finally gone." The cause was compiled, not stochastic:
the caption simultaneously carried the departure event and the standing
constancy order "present and solid in every single frame" about BOTH
characters, plus the planner's "Two people are in the kitchen" fact. The model
obeyed the caption.

So the state machine is small and absolute: an exit is the one irreversible
thing a plan can say. After it, the character is out of every cast sentence,
out of every constancy sentence, out of every people-count — and the scene is
restated in terms of who REMAINS, never of who left, because on this runtime a
name in the caption is a request for its owner.
"""

from __future__ import annotations

import pytest

from tests.test_director import IDEA, raw_plan
from worker.director import compile_section_prompts, parse_plan
from worker.director.plan import DirectorPlanError

GOODBYE_IDEA = (
    "A woman and a man share a quiet goodbye in a kitchen; midway through he "
    "walks out for good and she finishes the scene alone."
)


def goodbye_raw_plan(**overrides) -> dict:
    """The measured failure's shape: dialogue, a departure, a solo tail."""
    plan = {
        "scene": "A small kitchen at dusk, warm lamplight.",
        "tone": "tender",
        "ambience": "a kettle ticking as it cools",
        "characters": [
            {
                "id": "woman",
                "role": "woman",
                "appearance": "a woman in a green cardigan",
                "voice": "warm",
            },
            {
                "id": "man",
                "role": "man",
                "appearance": "a tall man in a denim jacket",
                "voice": "low",
            },
        ],
        "continuity": [
            "The woman wears a green cardigan",
            "Two people are in the kitchen",
        ],
        "timeline": [
            {
                "start": 0,
                "end": 8,
                "action": "The woman pours tea",
                "camera": "two-shot, static",
                "speaker": "man",
                "dialogue": "The train leaves at eight.",
                "delivery": "quiet",
            },
            {
                "start": 12,
                "end": 20,
                "action": "The woman nods",
                "camera": "medium close-up, static",
                "speaker": "woman",
                "dialogue": "Write to me when you arrive.",
                "delivery": "soft",
            },
            {
                "start": 24,
                "end": 34,
                "action": "The man opens the door and walks out",
                "camera": "wide shot, static",
                "speaker": "man",
                "dialogue": "I promise.",
                "delivery": "gentle",
                "exits": ["man"],
            },
            {
                "start": 38,
                "end": 46,
                "action": "The woman sits at the table",
                "camera": "medium shot, static",
                "speaker": "woman",
                "dialogue": "Eight o'clock, then.",
                "delivery": "steady",
            },
            {
                "start": 50,
                "end": 58,
                "action": "The woman washes the two cups",
                "camera": "medium close-up, static",
                "speaker": "woman",
                "dialogue": "Safe travels, love.",
                "delivery": "quiet and sure",
            },
        ],
    }
    plan.update(overrides)
    return plan


def goodbye_plan(**overrides):
    return parse_plan(
        goodbye_raw_plan(**overrides),
        idea=GOODBYE_IDEA,
        duration_seconds=60.0,
        language="english",
    )


# ── Parsing and validation ───────────────────────────────────────────────


def test_exits_parse_and_answer_presence_questions() -> None:
    plan = goodbye_plan()
    assert plan.has_exits
    assert plan.exit_time("man") == pytest.approx(34.0)
    assert plan.exit_time("woman") is None
    assert plan.present_ids(0.0) == ("woman", "man")
    assert plan.present_ids(33.0) == ("woman", "man")  # mid-departure: on screen
    assert plan.present_ids(34.0) == ("woman",)
    assert plan.present_ids(59.0) == ("woman",)


def test_the_goodbye_line_inside_the_exit_event_is_allowed() -> None:
    """Speaking while leaving is how goodbyes work; the ban starts after."""
    plan = goodbye_plan()
    exit_event = plan.timeline[2]
    assert exit_event.exits == ("man",)
    assert exit_event.dialogue == "I promise."


def test_speaking_after_your_own_exit_is_refused() -> None:
    events = goodbye_raw_plan()["timeline"]
    events[3]["speaker"] = "man"
    events[3]["dialogue"] = "One more thing."
    with pytest.raises(DirectorPlanError, match="after they left the scene"):
        goodbye_plan(timeline=events)


def test_exiting_twice_is_refused() -> None:
    events = goodbye_raw_plan()["timeline"]
    events[4]["exits"] = ["man"]
    events[4]["speaker"] = None
    events[4]["dialogue"] = None
    with pytest.raises(DirectorPlanError, match="exits twice"):
        goodbye_plan(timeline=events)


def test_an_unknown_exit_id_is_refused() -> None:
    events = goodbye_raw_plan()["timeline"]
    events[2]["exits"] = ["stranger"]
    with pytest.raises(DirectorPlanError, match="exits unknown character"):
        goodbye_plan(timeline=events)


def test_people_count_facts_are_dropped_when_someone_leaves() -> None:
    """"Two people are in the kitchen", restated after the man has gone, is
    the 20 Aug ghost wearing different words. Counts survive only in plans
    where nobody leaves."""
    plan = goodbye_plan()
    assert plan.continuity == ("The woman wears a green cardigan",)

    staying = parse_plan(
        raw_plan(continuity=["Two people are in the office"]),
        idea=IDEA,
        duration_seconds=12.0,
        language="english",
    )
    assert "Two people are in the office" in staying.continuity


# ── Compilation: the caption never argues with a departure ───────────────


def test_a_single_caption_stops_asserting_the_departed_mans_presence() -> None:
    """The exact contradiction the GPU rendered: departure event AND
    "present and solid in every single frame" about the man, in one text."""
    [caption] = compile_section_prompts(goodbye_plan(), 1, total_seconds=60.0)

    assert "walks out" in caption  # the action itself is kept
    assert "From this moment, the woman is alone in the scene." in caption
    # Constancy is survivor-scoped: the woman alone, correctly conjugated.
    assert "The woman keeps exactly the same face, clothing and voice" in caption
    assert "The woman stays fully visible in the frame" in caption
    # The man appears in his own events and NOWHERE in the standing orders.
    tail = caption.split("walks out")[1]
    assert "the man" not in tail
    assert "Two people" not in caption


def test_sections_after_a_departure_never_recast_the_departed() -> None:
    first, second = compile_section_prompts(goodbye_plan(), 2, total_seconds=60.0)

    # Section 1 carries the whole departure (event midpoint 29s < 30s) and
    # its constancy already speaks only of the survivor.
    assert "walks out" in first
    assert '"I promise."' in first
    assert "From this moment, the woman is alone in the scene." in first

    # Section 2 is the woman's alone — the man is not cast, not constancy,
    # not mentioned at all.
    assert "the man" not in second
    assert "continues mid-scene" in second
    assert "The woman is alone in the scene now, and stays alone." in second
    assert "The woman stays fully visible in the frame" in second
    assert '"Eight o\'clock, then."' in second
    assert '"I promise."' not in second  # no replay of completed dialogue


def test_a_late_exit_keeps_both_in_the_first_sections_constancy() -> None:
    """Presence scoping is per section, not global: before anyone leaves, the
    anti-flicker constancy still protects everyone."""
    events = goodbye_raw_plan()["timeline"]
    # Move the departure into the second half.
    events[2]["start"], events[2]["end"] = 36, 44
    events[3]["start"], events[3]["end"] = 46, 52
    events[4]["start"], events[4]["end"] = 54, 58
    first, second = compile_section_prompts(
        goodbye_plan(timeline=events), 2, total_seconds=60.0
    )

    assert "The woman and the man keep exactly the same faces" in first
    assert "The woman and the man stay fully visible" in first
    assert "walks out" in second
    assert "From this moment, the woman is alone in the scene." in second
    assert "The woman stays fully visible" in second
    assert "The woman and the man stay fully visible" not in second


def test_the_settling_tail_is_worded_for_who_is_left() -> None:
    """"They hold each other's gaze", said over a woman alone at the sink,
    re-invents a partner. The described-silence beat follows the survivors."""
    events = goodbye_raw_plan()["timeline"]
    events[4]["end"] = 50  # leave a >2.5s uncovered tail
    events[4]["start"] = 48
    [caption] = compile_section_prompts(
        goodbye_plan(timeline=events), 1, total_seconds=60.0
    )
    assert "the woman holds the moment" in caption
    assert "they hold each other's gaze" not in caption


def test_plans_without_exits_compile_exactly_as_before() -> None:
    """The baseline guarantee: every sentence the known-good 5/15/30s captions
    carry is untouched when nobody leaves."""
    plan = parse_plan(
        raw_plan(), idea=IDEA, duration_seconds=12.0, language="english"
    )
    [caption] = compile_section_prompts(plan, 1, total_seconds=12.0)
    assert (
        "The detective and the police chief keep exactly the same faces, "
        "clothing and voices for the entire video." in caption
    )
    assert (
        "The detective and the police chief stay fully visible in the frame "
        "from the first frame to the last, present and solid in every single "
        "frame." in caption
    )


# ── Geometry: 60 seconds re-enters the section machinery ─────────────────


def test_the_thirty_second_ceiling_splits_only_the_sixty_second_menu() -> None:
    """The production numbers, pinned: with the workflow's 30s ceiling, every
    menu duration below 60s is the same single pass it has always been, and
    60s becomes the two measured-clean 30s sections. This is the geometry half
    of the 20 Aug fix — the semantic half is everything above."""
    from pathlib import Path

    from tests.conftest import make_job
    from worker.adapters.ltx import LtxAdapter
    from worker.media import plan_segments

    adapter = LtxAdapter()
    job = make_job(
        Path("."), execution={"runtime": "ltx", "max_segment_seconds": 30}
    )
    per_pass = adapter._per_pass_seconds(job, (1024, 576))
    assert per_pass == 30.0

    for seconds, passes in ((5.0, 1), (10.0, 1), (15.0, 1), (30.0, 1), (60.0, 2)):
        segments = plan_segments(seconds, max_segment_seconds=per_pass)
        assert len(segments) == passes, f"{seconds}s -> {len(segments)} passes"
    halves = plan_segments(60.0, max_segment_seconds=per_pass)
    assert [s.duration_seconds for s in halves] == [30.0, 30.0]


def test_the_shipped_yaml_carries_the_ceiling_for_both_generation_workflows() -> None:
    """The ceiling lives in the workflow definitions the API bakes and serves;
    a worker deploy without the YAML change is inert. Read the real files so a
    stray revert is caught here rather than by a customer's 60s render."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "workflow-definitions"
    for name in ("text-to-video.yaml", "image-to-video.yaml"):
        text = (root / name).read_text(encoding="utf-8")
        assert re.search(r"^\s*max_segment_seconds:\s*30\s*$", text, re.M), name


# ── End to end: the chain renders each section under its own presence ────


async def test_a_sectioned_director_chain_keeps_the_departed_out_of_pass_two(
    workspace, fake_models, stub_repo, tmp_path, monkeypatch
) -> None:
    """The 20 Aug failure, replayed through the real adapter at test scale:
    the departure renders in pass one, and pass two's prompt carries the
    woman's world only — plus the predecessor frame, which is what actually
    shows the model a kitchen with one person in it."""
    from tests.conftest import (
        collect,
        conditioning_of,
        invocations,
        make_clip,
        make_job,
        render_stub,
        value_of,
    )
    from tests.test_director import CannedProvider, install_provider

    plan = goodbye_raw_plan(
        continuity=["The woman wears a green cardigan"],
        timeline=[
            {
                "start": 0,
                "end": 1.2,
                "action": "The man steps out through the gate",
                "camera": "two-shot, static",
                "speaker": "man",
                "dialogue": "Going.",
                "delivery": "quiet",
                "exits": ["man"],
            },
            {
                "start": 2.2,
                "end": 3.6,
                "action": "The woman waves",
                "camera": "medium shot, static",
                "speaker": "woman",
                "dialogue": "Stay well.",
                "delivery": "soft",
            },
        ],
    )
    install_provider(monkeypatch, CannedProvider(plan))
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True)
    )

    await collect(
        make_job(
            workspace,
            prompt=GOODBYE_IDEA,
            parameters={
                "duration": "4s",
                "aspect_ratio": "16:9",
                "quality": "High",
                "prompt_mode": "director",
            },
            execution={
                "runtime": "ltx",
                "prompt_structuring": True,
                "max_segment_seconds": 2,
            },
        )
    )

    calls = invocations(log)
    assert len(calls) == 2
    first, second = (value_of(argv, "--prompt") for argv in calls)

    assert '"Going."' in first
    assert "From this moment, the woman is alone in the scene." in first
    assert "the man" not in second
    assert '"Stay well."' in second
    assert '"Going."' not in second
    assert "The woman stays fully visible in the frame" in second
    # Continuity across the seam is the predecessor frame, as ever.
    assert conditioning_of(calls[1])[0][1] == 0


# ── The identity anchor obeys the two-image decoder measurements ─────────


def test_the_identity_anchor_rides_only_measured_two_image_counts(
    workspace,
) -> None:
    """20 Aug 2026, the night 60s I2V first chained in production: a
    720-frame pass with TWO conditioning images crashed the VAE decoder
    (CUBLAS_STATUS_INTERNAL_ERROR), 736 failed identically — so the
    render-extra-and-trim dodge is useless for this cell family — and
    120/240/360 passed. The anchor is therefore gated on the measured set:
    everywhere else the pass carries the seam frame alone, which is the
    single-image shape proven clean by the 60s validation."""
    from pathlib import Path

    from tests.conftest import make_job
    from worker.adapters.ltx import LtxAdapter

    adapter = LtxAdapter()
    job = make_job(workspace, execution={"runtime": "ltx"})
    still = Path("still.png")

    # Measured-safe counts carry the anchor, mid-window, at 0.2.
    for seconds, frames in ((5.0, 120), (10.0, 240), (15.0, 360)):
        anchor = adapter._identity_anchor(job, still, seconds)
        assert anchor is not None, f"{frames} frames should carry the anchor"
        assert anchor.frame_index == frames // 3
        assert anchor.strength == pytest.approx(0.2)

    # The production crash shape: 30s = 720 frames. Anchor dropped.
    assert adapter._identity_anchor(job, still, 30.0) is None

    # Strength zero is the existing off-switch, unchanged.
    off = make_job(
        workspace, execution={"runtime": "ltx", "i2v_reference_strength": 0}
    )
    assert off._cancelled is None  # jobs are plain dataclasses in tests
    assert adapter._identity_anchor(off, still, 5.0) is None


async def test_an_unmeasured_pass_keeps_the_seam_frame_alone(
    workspace, fake_models, stub_repo, tmp_path, monkeypatch
) -> None:
    """The whole-job proof at test scale: WITHOUT admitting the test count to
    the safe set, pass two of an I2V chain carries exactly one image — the
    predecessor frame — and the render completes."""
    from tests.conftest import (
        collect,
        conditioning_of,
        invocations,
        make_clip,
        render_stub,
    )
    from tests.test_ltx import make_i2v_job
    from worker.media import ffmpeg

    still = workspace / "still.png"
    await ffmpeg(
        ["-f", "lavfi", "-i", "testsrc2=size=896x512:rate=1", "-frames:v", "1", str(still)]
    )
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True)
    )

    await collect(
        make_i2v_job(
            workspace,
            still,
            parameters={"duration": "4s", "aspect_ratio": "16:9", "quality": "High"},
            execution={"runtime": "ltx", "max_segment_seconds": 2},
        )
    )

    calls = invocations(log)
    assert conditioning_of(calls[0]) == [(str(still), 0, 1.0)]
    second = conditioning_of(calls[1])
    assert len(second) == 1, "unmeasured count must carry the seam frame alone"
    assert second[0][1] == 0 and second[0][2] == 1.0


def test_the_anchor_guard_asks_about_the_count_the_renderer_will_land_on(
    workspace,
) -> None:
    """The 28 Aug identity regression, one level below the 27 Aug one.

    A conforming pipeline does not render its lattice count: it snaps up to
    the nearest entry in `measured_landings` and trims back afterwards. The
    audio tier lands on 121/241/385/481. The guard was asking about the
    lattice count instead, so a music-video section the onset planner pulled
    back to 18.5s — conforming 449, RENDERING 481, a count measured safe with
    two images — had its anchor dropped on a shape that never reached the
    decoder. Because that planner pulls nearly every seam off a round 20.0s,
    the anchor was live on almost no pass of a real three-minute video, and
    the client's singer drifted into a different man over nine sections.
    """
    from pathlib import Path

    from tests.conftest import make_job
    from worker.adapters.ltx import _A2VID, LtxAdapter

    adapter = LtxAdapter()
    job = make_job(workspace, execution={"runtime": "ltx"})
    still = Path("still.png")
    landings = _A2VID.measured_landings

    # 18.5s: 444 raw → 449 conforming → 481 rendered. The lattice count alone
    # says "unsafe"; the count the decoder actually sees says otherwise.
    assert adapter._identity_anchor(job, still, 18.5, conforming=True) is None
    anchor = adapter._identity_anchor(
        job, still, 18.5, conforming=True, landings=landings
    )
    assert anchor is not None
    # The index addresses the RENDERED timeline, and stays inside the frames
    # that survive the trim back to 444.
    assert anchor.frame_index == 481 // 3

    # Sections that genuinely land on an unmeasured two-image count are still
    # refused — the landing table is not a licence to skip the measurement.
    assert (
        adapter._identity_anchor(job, still, 10.0, conforming=True, landings=landings)
        is None  # 241 frames: a real landing, not in the two-image safe set
    )
