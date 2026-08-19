"""Director-aware extension: a Director video keeps being one when extended.

The lineage arrives from the API, resolved at creation time from the rows that
already existed (`source asset → producing job`), and stored in the extend
job's parameters as `director_lineage`. The worker's contract with it:

* **No lineage, no change.** A source with no Director ancestry — an upload, a
  standard generation, a pre-Director job — extends byte-identically to how
  extensions have always worked, and the planner is never consulted.
* **With lineage, the continuation is planned, not improvised**: the
  ancestor's language (never a silent fall-back to English), the ancestor's
  idea as the story-so-far the plan must move FORWARD from, and the anchored
  register — the finished video's last frame owns WHO and WHAT exactly the
  way an uploaded photograph does.
* **The original I2V upload rides along** as the `identity_image` input and
  conditions every extension pass at the same low-strength mid-window
  reference I2V chains have always used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    conditioning_of,
    invocations,
    make_clip,
    needs_ffmpeg,
    render_stub,
    staged_input,
    value_of,
)
from tests.test_director import CannedProvider, install_provider
from tests.test_ltx import make_extension_job
from worker.director import continuation_lineage
from worker.director.provider import DirectorRequest, system_prompt, user_prompt

LINEAGE = {
    "prompt_mode": "director",
    "dialogue_language": "spanish",
    "idea": "A woman and a robot on a bench discuss the future of education.",
    "prior_seconds": 30.0,
    "source_workflow": "image-to-video",
    "parent_job_id": "00000000-0000-0000-0000-00000000aaaa",
}


def continuation_raw_plan() -> dict:
    """A short, valid anchored continuation: no invented appearance."""
    return {
        "scene": "The same bench, exactly as the opening frame shows it.",
        "tone": "warm",
        "ambience": "birdsong and a light breeze",
        "characters": [
            {"id": "woman", "role": "woman", "appearance": "", "voice": "warm"},
            {"id": "robot", "role": "robot", "appearance": "", "voice": "soft"},
        ],
        "continuity": [],
        "timeline": [
            {
                "start": 0,
                "end": 1.0,
                "action": "The woman leans back thoughtfully",
                "camera": "two-shot, static",
                "speaker": None,
                "dialogue": None,
                "delivery": None,
            },
            {
                "start": 1.2,
                "end": 1.9,
                "action": "The robot turns its head toward her",
                "camera": "two-shot, static",
                "speaker": "robot",
                "dialogue": None,
                "delivery": None,
            },
        ],
    }


# ── Lineage detection ────────────────────────────────────────────────────


def test_lineage_is_read_only_when_it_names_director_mode(workspace: Path) -> None:
    directed = make_extension_job(
        workspace, None, parameters={"duration": "2s", "director_lineage": LINEAGE}
    )
    assert continuation_lineage(directed) == LINEAGE

    for parameters in (
        {"duration": "2s"},
        {"duration": "2s", "director_lineage": None},
        {"duration": "2s", "director_lineage": "director"},
        {"duration": "2s", "director_lineage": {"prompt_mode": "standard"}},
        {"duration": "2s", "director_lineage": {"prompt_mode": "director", "idea": ""}},
    ):
        job = make_extension_job(workspace, None, parameters=parameters)
        assert continuation_lineage(job) is None, parameters


# ── The continuation register ────────────────────────────────────────────


def continuation_request(**overrides) -> DirectorRequest:
    defaults = dict(
        idea="They stand up and start walking home together.",
        duration_seconds=10.0,
        language="spanish",
        seed=1,
        sample=False,
        source_anchored=True,
        prior_idea=LINEAGE["idea"],
        prior_seconds=30.0,
    )
    return DirectorRequest(**{**defaults, **overrides})


def test_the_continuation_register_appends_only_for_extensions() -> None:
    plain = system_prompt(
        continuation_request(prior_idea="", prior_seconds=0.0, source_anchored=False)
    )
    anchored = system_prompt(continuation_request(prior_idea=""))
    continuation = system_prompt(continuation_request())

    assert "CONTINUATION MODE" not in plain
    assert "CONTINUATION MODE" not in anchored
    assert "CONTINUATION MODE" in continuation
    assert continuation.startswith(anchored)


def test_the_story_so_far_reaches_the_planner_with_its_seconds() -> None:
    prompt = user_prompt(continuation_request())
    assert "THE STORY SO FAR" in prompt
    assert LINEAGE["idea"] in prompt
    assert "30 seconds of finished video" in prompt
    assert "IDEA: They stand up and start walking home together." in prompt

    fresh = user_prompt(continuation_request(prior_idea="", prior_seconds=0.0))
    assert "THE STORY SO FAR" not in fresh


async def test_the_plan_request_inherits_language_and_anchoring(
    workspace: Path,
) -> None:
    from worker.director import create_director_plan

    provider = CannedProvider(continuation_raw_plan())
    job = make_extension_job(
        workspace,
        None,
        prompt="They stand up and start walking home together.",
        parameters={"duration": "10s", "director_lineage": LINEAGE},
    )
    plan = await create_director_plan(job, 10.0, provider=provider, lineage=LINEAGE)

    request = provider.requests[0]
    assert request.language == "spanish"
    assert request.source_anchored is True
    assert request.prior_idea == LINEAGE["idea"]
    assert request.prior_seconds == pytest.approx(30.0)
    assert plan.source_anchored


async def test_an_unlisted_lineage_language_falls_back_to_auto(
    workspace: Path,
) -> None:
    from worker.director import create_director_plan

    provider = CannedProvider(continuation_raw_plan())
    job = make_extension_job(
        workspace, None, parameters={"duration": "10s"}
    )
    await create_director_plan(
        job, 10.0, provider=provider, lineage={**LINEAGE, "dialogue_language": "klingon"}
    )
    assert provider.requests[0].language == "auto"


# ── End to end through the adapter ───────────────────────────────────────


@needs_ffmpeg
async def test_a_director_lineage_extension_renders_the_continuation_plan(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, whole: lineage in the parameters means the planner runs,
    the compiled caption (not the raw extension prompt) reaches the model,
    and the ORIGINAL upload conditions the pass beside the seam frame."""
    install_provider(monkeypatch, CannedProvider(continuation_raw_plan()))
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    still = workspace / "identity.png"
    from worker.media import ffmpeg

    await ffmpeg(
        ["-f", "lavfi", "-i", "testsrc2=size=896x512:rate=1", "-frames:v", "1", str(still)]
    )
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True)
    )

    base = make_extension_job(workspace, source)
    job = make_extension_job(
        workspace,
        source,
        prompt="They stand up and start walking home together.",
        parameters={"duration": "2s", "director_lineage": LINEAGE},
        inputs=[
            *base.inputs,
            staged_input("identity_image", "image", "image/png", still),
        ],
    )
    result, reported = await collect(job)

    argv = invocations(log)[0]
    prompt = value_of(argv, "--prompt")
    # The compiled caption, not the user's extension text, and not the
    # standard sectioner's output — anchored to the opening frame, which for
    # an extension is the finished video's last moment. (The planner's own
    # scene wording was grounded away here: "The same bench…" carries words
    # the idea supplies, but "opening" is not among them, so the scene fell
    # back to the canonical anchored sentence. Stricter is fine.)
    assert "The scene continues exactly as the opening frame shows it." in prompt
    assert "the woman and the robot are already present in the opening frame" in prompt.lower()
    assert "They stand up and start walking home together." not in prompt
    assert "LONG-FORM CONTINUATION" not in prompt

    frames = conditioning_of(argv)
    assert frames[0][1] == 0 and frames[0][2] == 1.0  # the seam frame
    identity_refs = [f for f in frames if f[0] == str(still)]
    assert identity_refs, "the original upload was dropped from the extension"
    assert identity_refs[0][1] > 0
    assert identity_refs[0][2] == pytest.approx(0.2)

    assert any("Directing the continuation" in message for _, _, message in reported)
    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)


@needs_ffmpeg
async def test_an_extension_without_lineage_never_touches_the_planner(
    workspace: Path,
    fake_models: Path,
    stub_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standard extension, byte-identical: single-pass extensions have always
    sent the user's prompt verbatim, and the planner does not exist here."""
    provider = CannedProvider()  # would raise IndexError if consulted
    install_provider(monkeypatch, provider)
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True)
    )

    await collect(
        make_extension_job(
            workspace, source, prompt="the camera keeps drifting forward"
        )
    )

    assert provider.requests == []
    argv = invocations(log)[0]
    assert value_of(argv, "--prompt") == "the camera keeps drifting forward"
    assert conditioning_of(argv)[0][2] == 1.0
