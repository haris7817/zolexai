"""Director mode on Image to Video: `prompt_mode: director` plus an upload.

The product principle under test: **the uploaded image defines WHO and WHAT;
Director mode defines WHAT HAPPENS.** Which decomposes into the same two
guarantees the T2V suite pins, plus two of its own:

* **Standard Image to Video is byte-identical to what has served production.**
  No parameter, no planner, no change of any kind.
* **The plan may not invent what the photograph already decided.** Anchored
  plans may leave a character's appearance to the image; compiled captions tie
  identity to the conditioned first frame instead of to a described look; the
  anchored planning brief exists only for anchored requests, so the T2V brief
  stays byte-identical.
* **The original upload stays the identity anchor across the whole chain** —
  which is the adapter's existing conditioning, asserted here THROUGH Director
  mode so a regression in either half is caught where it matters.
* **The optional vision step can only ever add.** Off by default, absorbed on
  failure, and the job's fate never depends on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    conditioning_of,
    invocations,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
    value_of,
)
from tests.test_director import CannedProvider, install_provider, raw_plan
from worker.core.config import settings
from worker.director import (
    compile_section_prompts,
    create_director_plan,
    parse_plan,
    source_image_facts,
    wants_director,
)
from worker.director.plan import ANCHORED_SCENE, DirectorPlanError
from worker.director.provider import DirectorRequest, system_prompt, user_prompt
from worker.media import ffmpeg

IDEA = "A woman and a robot on a bench discuss the future of education."


def anchored_raw_plan(**overrides) -> dict:
    """A plan the way the anchored brief asks for one: no invented looks."""
    plan = {
        "scene": "A bench in a park, as the photograph shows.",
        "tone": "warm",
        "ambience": "birdsong and a light breeze",
        "characters": [
            {"id": "woman", "role": "woman", "appearance": "", "voice": "warm and steady"},
            {"id": "robot", "role": "robot", "appearance": "", "voice": "soft and precise"},
        ],
        "continuity": ["Two figures share the bench"],
        "timeline": [
            {
                "start": 0,
                "end": 1,
                "action": "The woman turns toward the robot",
                "camera": "two-shot, static",
                "speaker": None,
                "dialogue": None,
                "delivery": None,
            },
            {
                "start": 1,
                "end": 3.5,
                "action": "The robot tilts its head",
                "camera": "two-shot, static",
                "speaker": "robot",
                "dialogue": "Teachers will remain.",
                "delivery": "soft and certain",
            },
            {
                "start": 5,
                "end": 7.5,
                "action": "The woman nods slowly",
                "camera": "medium close-up, static",
                "speaker": "woman",
                "dialogue": "Curiosity matters most.",
                "delivery": "quietly convinced",
            },
        ],
    }
    plan.update(overrides)
    return plan


def i2v_director_job(workspace: Path, image_path: Path | None, **overrides):
    defaults = dict(
        workflow_id="image-to-video",
        prompt=IDEA,
        parameters={
            "duration": "10s",
            "aspect_ratio": "16:9",
            "quality": "High",
            "prompt_mode": "director",
        },
        inputs=[staged_input("source_image", "image", "image/png", image_path)],
        execution={"runtime": "ltx", "prompt_structuring": True},
    )
    return make_job(workspace, **{**defaults, **overrides})


async def make_still(path: Path) -> Path:
    await ffmpeg(
        ["-f", "lavfi", "-i", "testsrc2=size=896x512:rate=1", "-frames:v", "1", str(path)]
    )
    return path


# ── Standard Image to Video is untouched ─────────────────────────────────


@needs_ffmpeg
async def test_a_standard_i2v_job_never_touches_the_planner(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No parameter, no planner, and the structured prompt is the exact text
    the pre-feature worker sent."""
    from worker.longform.enhance import structure_prompt

    provider = CannedProvider()  # would raise IndexError if consulted
    install_provider(monkeypatch, provider)
    still = await make_still(workspace / "still.png")
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True)
    )

    await collect(
        i2v_director_job(
            workspace,
            still,
            prompt="the water ripples gently",
            parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        )
    )

    assert provider.requests == []
    argv = invocations(log)[0]
    assert value_of(argv, "--prompt") == structure_prompt("the water ripples gently")
    assert conditioning_of(argv)[0] == (str(still), 0, 1.0)


def test_image_to_video_with_the_parameter_wants_director(workspace: Path) -> None:
    assert wants_director(i2v_director_job(workspace, None))
    assert not wants_director(
        i2v_director_job(
            workspace, None, parameters={"duration": "10s", "prompt_mode": "standard"}
        )
    )


# ── Anchored plans: the image owns appearance ────────────────────────────


def test_an_anchored_plan_may_leave_appearance_to_the_photograph() -> None:
    """On Image to Video the upload IS the appearance; a text-to-video plan
    with the same hole is still refused, because there the text is all the
    identity the render will ever have."""
    plan = parse_plan(
        anchored_raw_plan(),
        idea=IDEA,
        duration_seconds=10.0,
        language="english",
        source_anchored=True,
    )
    assert plan.source_anchored
    assert [c.appearance for c in plan.characters] == ["", ""]

    with pytest.raises(DirectorPlanError, match="missing a role or appearance"):
        parse_plan(
            anchored_raw_plan(), idea=IDEA, duration_seconds=10.0, language="english"
        )


def request_for(**overrides) -> DirectorRequest:
    defaults = dict(
        idea=IDEA, duration_seconds=10.0, language="auto", seed=1, sample=False
    )
    return DirectorRequest(**{**defaults, **overrides})


def test_the_anchored_brief_appends_only_for_source_anchored_requests() -> None:
    """The T2V brief must stay byte-identical — a drifting shared prompt is a
    behaviour change on the workflow this feature promised not to touch."""
    plain = system_prompt(request_for())
    anchored = system_prompt(request_for(source_anchored=True))

    assert "SOURCE IMAGE MODE" not in plain
    assert "SOURCE IMAGE MODE" in anchored
    assert anchored.startswith(plain)
    assert "NEVER invent visible details" in anchored


def test_photograph_facts_reach_the_planner_only_when_measured() -> None:
    """Facts are injected as evidence, never as an empty header the model
    would be invited to fill in itself."""
    without = user_prompt(request_for(source_anchored=True))
    assert "PHOTOGRAPH FACTS" not in without

    facts = "PEOPLE: one woman in a red coat; one silver robot"
    with_facts = user_prompt(request_for(source_anchored=True, image_facts=facts))
    assert "PHOTOGRAPH FACTS" in with_facts
    assert facts in with_facts


# ── Anchored compilation ─────────────────────────────────────────────────


def anchored_plan(duration: float = 10.0, **overrides):
    return parse_plan(
        anchored_raw_plan(**overrides),
        idea=IDEA,
        duration_seconds=duration,
        language="english",
        source_anchored=True,
    )


def test_anchored_compilation_anchors_identity_to_the_frame_not_to_prose() -> None:
    [caption] = compile_section_prompts(anchored_plan(), 1, total_seconds=10.0)

    assert "opening frame" in caption
    assert "the woman" in caption and "the robot" in caption
    # Constancy is tied to the conditioned frame, which is the only honest
    # referent when no appearance text exists.
    assert "they have in the first frame" in caption
    # The empty appearance must not leave grammatical debris.
    assert ", ," not in caption
    assert '"Teachers will remain."' in caption
    assert '"Curiosity matters most."' in caption


def test_idea_stated_appearance_still_rides_along_when_anchored() -> None:
    """The refusal is of INVENTED detail. A fact the user's own idea states is
    the measured anti-drift lever and keeps its place in every caption."""
    characters = anchored_raw_plan()["characters"]
    characters[0]["appearance"] = "wearing a yellow raincoat"
    plan = parse_plan(
        anchored_raw_plan(characters=characters),
        idea="A woman wearing a yellow raincoat talks to a robot about education.",
        duration_seconds=10.0,
        language="english",
        source_anchored=True,
    )
    [caption] = compile_section_prompts(plan, 1, total_seconds=10.0)
    assert "wearing a yellow raincoat" in caption


# ── Grounding: no claim about a photograph nobody looked at ──────────────


def test_an_invented_appearance_is_discarded_not_rendered() -> None:
    """The measured failure, 19 Aug 2026: given a photograph of a woman in a
    yellow raincoat, the hosted planner wrote a beige linen blouse on a clean
    first attempt. Nothing in the idea supports it, so it does not reach the
    caption — identity falls back to the frame, which is the truth anyway."""
    characters = anchored_raw_plan()["characters"]
    characters[0]["appearance"] = "mid-30s, slender, wearing a beige linen blouse"
    characters[1]["appearance"] = "white polished ceramic plating with blue LED eyes"
    plan = anchored_plan(characters=characters)

    assert [c.appearance for c in plan.characters] == ["", ""]
    [caption] = compile_section_prompts(plan, 1, total_seconds=10.0)
    assert "blouse" not in caption and "ceramic" not in caption


def test_an_invented_scene_becomes_a_pointer_at_the_frame() -> None:
    """The same run put the park bench in "a modern minimalist study". A
    caption cannot simply drop its opening sentence, so an ungrounded scene
    becomes the one true statement available: look at the frame."""
    plan = anchored_plan(scene="A modern minimalist study with soft natural light")
    assert plan.scene == ANCHORED_SCENE

    grounded = parse_plan(
        anchored_raw_plan(scene="A park bench under autumn leaves"),
        idea="A woman and a robot on a park bench under autumn leaves discuss education.",
        duration_seconds=10.0,
        language="english",
        source_anchored=True,
    )
    assert grounded.scene == "A park bench under autumn leaves"


def test_invented_continuity_is_dropped_because_every_section_repeats_it() -> None:
    """Continuity is restated at the end of EVERY section, so an invented fact
    there is drift pressure applied once per pass — the exact mechanism this
    mode exists to remove."""
    plan = anchored_plan(
        continuity=[
            "The woman wears a beige linen blouse",
            "The robot has white ceramic plating and blue LED eyes",
        ]
    )
    assert plan.continuity == ()

    for caption in compile_section_prompts(plan, 2, total_seconds=10.0):
        assert "blouse" not in caption and "ceramic" not in caption
        # What survives is the frame-anchored constancy the compiler owns,
        # which needs no visual claim to be true.
        assert "they have in the first frame" in caption


def test_measured_image_facts_ground_a_description_and_let_it_through() -> None:
    """The payoff for turning the vision step on: a description someone
    actually looked at is allowed to reach the caption, because it is now
    evidence rather than invention."""
    characters = anchored_raw_plan()["characters"]
    characters[0]["appearance"] = "wearing a yellow raincoat"
    plan = parse_plan(
        anchored_raw_plan(characters=characters, continuity=["The raincoat stays yellow"]),
        idea=IDEA,
        duration_seconds=10.0,
        language="english",
        source_anchored=True,
        grounding="PEOPLE: one woman wearing a yellow raincoat; one silver robot",
    )
    assert plan.characters[0].appearance == "wearing a yellow raincoat"
    assert plan.continuity == ("The raincoat stays yellow",)


def test_text_to_video_plans_are_never_grounded() -> None:
    """T2V has no photograph to contradict, and there the invented appearance
    IS the identity channel — the measured anti-drift lever. Grounding must
    not touch it."""
    plan = parse_plan(
        raw_plan(), idea="A detective confronts a chief.", duration_seconds=12.0,
        language="english",
    )
    assert plan.characters[0].appearance == "a weathered man in a rumpled gray suit"
    assert plan.scene.startswith("A dim police chief's office")


def test_anchored_sections_each_carry_only_their_own_events() -> None:
    """The global-plan-then-split design transfers whole: a later section can
    never restart the conversation, and every section restates the anchored
    constancy block."""
    first, second = compile_section_prompts(anchored_plan(), 2, total_seconds=10.0)

    assert '"Teachers will remain."' in first
    assert '"Teachers will remain."' not in second
    assert '"Curiosity matters most."' not in first
    assert '"Curiosity matters most."' in second
    for caption in (first, second):
        assert "they have in the first frame" in caption
        assert "stay fully visible in the frame" in caption
    assert "without repeating any earlier action or line" in second


# ── The optional vision step ─────────────────────────────────────────────


def vision_stub(tmp_path: Path, body: str) -> str:
    """A stand-in describer; returns the command string that runs it."""
    script = tmp_path / "vision_stub.py"
    script.write_text(body, encoding="utf-8")
    return f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}"'


ECHO_FACTS = """
import json, sys
request = json.loads(sys.stdin.read())
print("library noise the worker must skip")
print(request["begin_marker"])
print("PEOPLE: one woman in a red coat; one silver robot")
print("SETTING: a park bench at dusk")
print(request["end_marker"])
"""

CRASH = """
import sys
print("no multimodal weights here")
sys.exit(3)
"""


async def test_vision_is_off_by_default_and_spawns_nothing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def forbidden(*args, **kwargs):  # pragma: no cover - the assertion
        raise AssertionError("vision subprocess spawned while disabled")

    monkeypatch.setattr("asyncio.create_subprocess_exec", forbidden)
    still = tmp_path / "still.png"
    still.write_bytes(b"not read while disabled")

    assert await source_image_facts(i2v_director_job(workspace, still)) == ""


async def test_vision_facts_flow_from_the_subprocess_into_the_plan_request(
    workspace: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    still = tmp_path / "still.png"
    still.write_bytes(b"pixels stand in fine; the stub never opens the file")
    monkeypatch.setattr(settings, "director_vision_enabled", True)
    monkeypatch.setattr(
        settings, "director_vision_command", vision_stub(tmp_path, ECHO_FACTS)
    )

    provider = CannedProvider(anchored_raw_plan())
    plan = await create_director_plan(
        i2v_director_job(workspace, still), 10.0, provider=provider
    )

    assert plan.source_anchored
    request = provider.requests[0]
    assert request.source_anchored
    assert "one woman in a red coat" in request.image_facts
    assert "library noise" not in request.image_facts


async def test_a_failing_vision_step_degrades_to_no_facts_not_to_a_failed_job(
    workspace: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expected honest failure is a text-only checkpoint. One log line;
    the plan proceeds exactly as if the step were disabled."""
    still = tmp_path / "still.png"
    still.write_bytes(b"pixels")
    monkeypatch.setattr(settings, "director_vision_enabled", True)
    monkeypatch.setattr(
        settings, "director_vision_command", vision_stub(tmp_path, CRASH)
    )

    provider = CannedProvider(anchored_raw_plan())
    plan = await create_director_plan(
        i2v_director_job(workspace, still), 10.0, provider=provider
    )

    assert plan is not None
    assert provider.requests[0].image_facts == ""


# ── End to end through the adapter ───────────────────────────────────────


@needs_ffmpeg
async def test_an_i2v_director_chain_keeps_the_upload_as_the_identity_anchor(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two halves of the product principle, asserted together on a real
    two-pass chain: the compiled plan (not the idea) is what renders, its
    dialogue split across sections without replay — and every pass after the
    first still carries BOTH the predecessor frame and the original upload,
    exactly as standard Image to Video does."""
    plan = anchored_raw_plan(
        timeline=[
            {
                "start": 0,
                "end": 1.2,
                "action": "The woman turns toward the robot",
                "camera": "two-shot, static",
                "speaker": None,
                "dialogue": None,
                "delivery": None,
            },
            {
                "start": 1.2,
                "end": 1.9,
                "action": "The robot tilts its head",
                "camera": "two-shot, static",
                "speaker": "robot",
                "dialogue": "Ready?",
                "delivery": "soft",
            },
            {
                "start": 2.2,
                "end": 3.8,
                "action": "The woman smiles",
                "camera": "two-shot, static",
                "speaker": "woman",
                "dialogue": "Almost.",
                "delivery": "warm",
            },
        ]
    )
    install_provider(monkeypatch, CannedProvider(plan))
    still = await make_still(workspace / "still.png")
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True)
    )

    result, reported = await collect(
        i2v_director_job(
            workspace,
            still,
            parameters={
                "duration": "4s",
                "aspect_ratio": "16:9",
                "quality": "High",
                "prompt_mode": "director",
            },
            execution={"runtime": "ltx", "prompt_structuring": True, "max_segment_seconds": 2},
        )
    )

    calls = invocations(log)
    prompts = [value_of(argv, "--prompt") for argv in calls]

    # The compiled plan renders, not the idea, and sections never replay.
    assert IDEA not in prompts[0] and IDEA not in prompts[1]
    assert '"Ready?"' in prompts[0] and '"Ready?"' not in prompts[1]
    assert '"Almost."' not in prompts[0] and '"Almost."' in prompts[1]
    for prompt in prompts:
        assert "they have in the first frame" in prompt

    # Identity conditioning is untouched by Director mode: the upload opens
    # the video, and later passes carry it as a low-strength reference beside
    # the predecessor frame.
    assert conditioning_of(calls[0]) == [(str(still), 0, 1.0)]
    second = conditioning_of(calls[1])
    assert second[0] == (str(workspace / "segment-condition-0001.png"), 0, 1.0)
    assert second[1][0] == str(still), "the original identity anchor was dropped"
    assert second[1][1] > 0, "the identity anchor reset the continuation at frame zero"
    assert second[1][2] == pytest.approx(0.2)

    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)
    assert any("Directing" in message for _, _, message in reported)
