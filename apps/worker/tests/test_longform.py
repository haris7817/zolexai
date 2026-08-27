"""The shared long-form layer: chaining, progress, musical cut points.

These tests use no model and, mostly, no ffmpeg. That is the point of the layer
existing: "a 73-second request becomes 30 + 30 + 13, each pass conditioned on
the last, with a bar that only moves forwards" is a property of this code, not
of any provider, and it should be provable in milliseconds.

Where a real file is genuinely needed — extracting the frame that chains one
pass to the next — ffmpeg is used and the test skips without it.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest

from tests.conftest import make_clip, make_job, make_track, needs_ffmpeg, recorder
from worker.adapters.base import AdapterError, JobCancelled
from worker.longform import (
    GENERATE_FROM,
    GENERATE_TO,
    ChainStep,
    StageReporter,
    band_for,
    plan_musical_boundaries,
    plan_section_prompts,
    render_chain,
)
from worker.longform.chain import plan_chain_segments as _plan
from worker.media import audio_onsets, detect_onsets

# ── Progress: forwards, always ───────────────────────────────────────────


async def test_progress_never_moves_backwards_however_it_is_called() -> None:
    """The bar the customer watches is the whole reason this class exists.

    Fifteen chained passes each computing their own percentage will, sooner or
    later, compute a lower one than the pass before. Clamping here means that
    can only ever be invisible, never a bar that jumps back to 20%.
    """
    on_progress, reported = recorder()
    stage = StageReporter(on_progress)

    await stage.generating(40, "…")
    await stage.generating(15, "…")  # a later pass computing a smaller number
    await stage.generating(55, "…")

    assert [value for _, value, _ in reported] == [40, 40, 55]


async def test_a_status_cannot_regress_because_the_api_would_kill_the_job() -> None:
    """A backwards status is not merely untidy: the API rejects it, the runner
    reads the rejection as a lost lease, and the job is abandoned mid-render.
    So a stage that reports out of order is corrected rather than forwarded."""
    on_progress, reported = recorder()
    stage = StageReporter(on_progress)

    await stage.stitching()
    await stage.generating(50, "a late report from an earlier stage")

    assert [status for status, _, _ in reported] == ["post_processing", "post_processing"]


async def test_the_stage_vocabulary_covers_the_whole_job_in_order() -> None:
    """preparing → probing → generating → stitching → muxing → finalizing →
    uploading, monotonic in both axes, and never naming a model or a GPU."""
    on_progress, reported = recorder()
    stage = StageReporter(on_progress)

    await stage.preparing()
    await stage.probing()
    await stage.section(1, 3, 20)
    await stage.section(3, 3, 80)
    await stage.stitching()
    await stage.muxing()
    await stage.finalizing()
    await stage.uploading()

    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)
    assert progress[-1] < 100, "completion is the platform's to declare, not a stage's"

    banned = ("ltx", "gpu", "cuda", "vast", "comfy", "nvfp4", "ffmpeg", "model")
    for _, _, message in reported:
        assert not any(word in message.lower() for word in banned), message


async def test_a_single_pass_job_never_mentions_sections() -> None:
    """Machinery is named only when there is machinery to explain — "Section 1
    of 1" is a leak of the implementation with nothing offered in return."""
    on_progress, reported = recorder()
    stage = StageReporter(on_progress)

    await stage.section(1, 1, 30)
    assert "section" not in reported[-1][2].lower()

    await stage.section(2, 4, 50)
    assert "Generating section 2 of 4…" == reported[-1][2]


async def test_section_progress_carries_machine_readable_timing() -> None:
    reports = []

    async def capture(status, progress, message, details=None) -> None:
        reports.append((status, progress, message, details))

    stage = StageReporter(capture)
    await stage.section(2, 3, 50, start_seconds=10.0, end_seconds=20.0)

    assert reports[-1][3] == {
        "phase": "generating",
        "section_index": 2,
        "section_total": 3,
        "section_start_seconds": 10.0,
        "section_end_seconds": 20.0,
    }


def test_dialogue_is_assigned_once_instead_of_replayed_per_section() -> None:
    prompt = '''Persistent: same woman, same silver robot, solid black visor
Section 1 / 0-10: MAYA: \"Where are we?\"
Section 2 / 10-20: ROBOT: \"Still moving.\"
Section 3 / 20-30: MAYA: \"Then keep going.\"'''  # noqa: E501

    planned = plan_section_prompts(prompt, 3)

    assert all("same woman, same silver robot, solid black visor" in item for item in planned)
    for line in (
        'MAYA: "Where are we?"',
        'ROBOT: "Still moving."',
        'MAYA: "Then keep going."',
    ):
        assert sum(line in item for item in planned) == 1
    assert "restart" in planned[1].lower()


def test_a_single_pass_prompt_remains_byte_for_byte_unchanged() -> None:
    prompt = "  two cars — black and pearl-white\nkeep exactly 2  "
    assert plan_section_prompts(prompt, 1) == [prompt]


def test_inline_then_actions_are_not_replayed() -> None:
    prompt = "black car starts, then white car overtakes, finally both stop"
    planned = plan_section_prompts(prompt, 3)

    for fragment in ("black car starts", "white car overtakes", "both stop"):
        assert sum(fragment in item for item in planned) == 1


@pytest.mark.parametrize("total", [1, 2, 3, 5, 12])
def test_bands_tile_the_generating_range_exactly(total: int) -> None:
    """Gaps leave the bar stalled between passes; overlaps make it jump back."""
    bands = [band_for(index, total) for index in range(total)]

    assert bands[0][0] == GENERATE_FROM
    assert bands[-1][1] == GENERATE_TO
    for (_, previous_end), (next_start, _) in zip(bands, bands[1:], strict=False):
        assert previous_end == next_start


# ── Planning ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("total", "ceiling", "expected"),
    [
        (60, 30, [30, 30]),
        # The directive's own example. Greedy chunking made this 30+30+13; the
        # windows are even now, so the same three passes are 24.33 each.
        (73, 30, [24, 24, 24]),
        (42, 30, [21, 21]),
        (30, 30, [30]),
        (5, 30, [5]),
    ],
)
def test_a_requested_length_becomes_passes_the_gpu_survives(
    total: float, ceiling: float, expected: list[float]
) -> None:
    """The safety property the whole product rests on: no length the product
    offers may reach the GPU as a single pass beyond its measured ceiling.

    The pass COUNT is what the ceiling determines, and it is unchanged by even
    windowing — 73s at a 30s ceiling is three passes either way. What changed is
    that the last one is no longer whatever happens to be left over, which on
    real uploads was sometimes a fraction of a second.
    """
    segments = _plan(total, ceiling, None)

    assert [round(s.duration_seconds) for s in segments] == expected
    assert all(s.duration_seconds <= ceiling for s in segments)
    assert sum(s.duration_seconds for s in segments) == pytest.approx(total)
    assert min(s.duration_seconds for s in segments) > ceiling / 10, (
        "a sliver pass costs a full model invocation to contribute a frame"
    )


def test_caller_supplied_cut_points_are_used_when_they_are_legal() -> None:
    segments = _plan(60, 30, [22.5, 45.0])
    assert [round(s.duration_seconds, 1) for s in segments] == [22.5, 22.5, 15.0]


def test_a_cut_point_that_would_oversize_a_pass_is_refused() -> None:
    """A timing layer is allowed to move a seam. It is not allowed to hand the
    GPU a pass longer than the one that ran out of memory."""
    with pytest.raises(ValueError, match="exceeds"):
        _plan(60, 30, [45.0])


# ── The chain ────────────────────────────────────────────────────────────


async def _run_chain(workspace: Path, total: float, ceiling: float, **kwargs):
    """Drives render_chain with a renderer that just records what it was asked."""
    steps: list[ChainStep] = []
    on_progress, reported = recorder()

    async def render(step: ChainStep) -> None:
        steps.append(step)
        step.output.write_bytes(b"rendered")

    parts = await render_chain(
        make_job(workspace),
        total,
        per_pass_seconds=ceiling,
        render=render,
        reporter=StageReporter(on_progress),
        chain_frames=False,
        **kwargs,
    )
    return parts, steps, reported


async def test_a_short_job_pays_for_no_chaining_machinery(workspace: Path) -> None:
    parts, steps, _ = await _run_chain(workspace, 5, 30)

    assert len(parts) == 1
    assert steps[0].is_first
    assert steps[0].section is None, "one pass is not a section of anything"
    assert steps[0].previous_frame is None


async def test_every_pass_receives_its_own_slice_of_the_bar(workspace: Path) -> None:
    _, steps, _ = await _run_chain(workspace, 90, 30)

    assert [step.section for step in steps] == [(1, 3), (2, 3), (3, 3)]
    assert [step.band for step in steps] == [band_for(index, 3) for index in range(3)]


async def test_a_pass_that_writes_nothing_fails_the_job(workspace: Path) -> None:
    """A renderer reporting success without producing a file would otherwise
    surface as a stitching error two steps later, pointing at the wrong thing."""
    on_progress, _ = recorder()

    async def render(step: ChainStep) -> None:
        return  # "succeeds", writes nothing

    with pytest.raises(AdapterError, match="wrote no file"):
        await render_chain(
            make_job(workspace),
            10,
            per_pass_seconds=30,
            render=render,
            reporter=StageReporter(on_progress),
        )


async def test_cancellation_is_honoured_between_passes(workspace: Path) -> None:
    """The cheapest place to stop a long job, and the one that matters: a
    cancelled four-pass render must not start pass two."""
    cancelled = asyncio.Event()
    started: list[int] = []
    on_progress, _ = recorder()

    async def render(step: ChainStep) -> None:
        started.append(step.index)
        step.output.write_bytes(b"rendered")
        cancelled.set()

    with pytest.raises(JobCancelled):
        await render_chain(
            make_job(workspace, _cancelled=cancelled),
            120,
            per_pass_seconds=30,
            render=render,
            reporter=StageReporter(on_progress),
            chain_frames=False,
        )

    assert started == [0], "the chain kept rendering after the job was cancelled"


@needs_ffmpeg
async def test_each_pass_is_conditioned_on_its_predecessors_last_frame(
    workspace: Path, tmp_path: Path
) -> None:
    """Continuity across a seam is entirely this frame. Without it, pass two is
    a different-looking video starting halfway through the result."""
    clip = await make_clip(tmp_path / "part.mp4", 1.0)
    seen: list[Path | None] = []
    on_progress, _ = recorder()

    async def render(step: ChainStep) -> None:
        seen.append(step.previous_frame)
        step.output.write_bytes(clip.read_bytes())

    await render_chain(
        make_job(workspace),
        3,
        per_pass_seconds=1,
        render=render,
        reporter=StageReporter(on_progress),
        seed_frame=None,
    )

    assert seen[0] is None
    assert seen[1] == workspace / "segment-condition-0001.png"
    assert seen[2] == workspace / "segment-condition-0002.png"
    assert all(frame.exists() for frame in seen[1:] if frame)


# ── Musical cut points ───────────────────────────────────────────────────


def test_a_track_inside_one_pass_needs_no_cut_points() -> None:
    assert plan_musical_boundaries(20, per_pass_seconds=30, onsets=[5, 10]) == []


def test_cuts_land_on_onsets_without_ever_oversizing_a_pass() -> None:
    """The two things that have to be true at once: the seam moves to a
    musical event, and the window it creates still fits the GPU."""
    onsets = [value * 0.5 for value in range(1, 240)]  # an event every 500ms
    boundaries = plan_musical_boundaries(120, per_pass_seconds=30, onsets=onsets)

    assert boundaries, "a track with obvious events should produce cut points"
    assert all(value in onsets for value in boundaries), "cuts must be ON events"

    windows = [
        end - start
        for start, end in zip([0.0, *boundaries], [*boundaries, 120.0], strict=False)
    ]
    assert all(width <= 30 + 1e-6 for width in windows)
    assert sum(windows) == pytest.approx(120)


@pytest.mark.parametrize("total", [4.0, 6.5, 30.0, 73.0, 240.0])
@pytest.mark.parametrize("ceiling", [1.0, 1.5, 3.0, 30.0])
@pytest.mark.parametrize("spacing", [0.0, 0.37, 2.0])
def test_no_arrangement_of_cut_points_can_oversize_a_pass(
    total: float, ceiling: float, spacing: float
) -> None:
    """A regression, and the reason this is a property test rather than an
    example.

    An earlier version folded an awkward final remainder into the previous
    window instead of cutting it, which quietly produced a window LONGER than
    the ceiling — the exact request that ran the GPU out of memory. Timing
    preferences are allowed to move a seam; they are not allowed to break the
    one invariant the ceiling exists to enforce.
    """
    onsets = (
        [] if not spacing else [value * spacing for value in range(1, int(total / spacing) + 1)]
    )
    boundaries = plan_musical_boundaries(total, per_pass_seconds=ceiling, onsets=onsets)

    cuts = [0.0, *boundaries, total]
    windows = [end - start for start, end in zip(cuts, cuts[1:], strict=False)]

    assert all(width > 0 for width in windows), "a zero-length pass is not a pass"
    assert all(width <= ceiling + 1e-6 for width in windows), windows
    assert sum(windows) == pytest.approx(total)


def test_a_track_with_no_detectable_events_falls_back_to_even_windows() -> None:
    """Timing analysis improves where a seam lands. It is never a prerequisite
    for producing the video at all."""
    boundaries = plan_musical_boundaries(120, per_pass_seconds=30, onsets=[])
    windows = [
        end - start
        for start, end in zip([0.0, *boundaries], [*boundaries, 120.0], strict=False)
    ]
    assert all(width <= 30 + 1e-6 for width in windows)


@pytest.mark.parametrize(
    ("total", "ceiling"),
    [
        # The measured cases: an MP3 probes a hair over its nominal length, and
        # greedily filling the ceiling put the leftover in a section of its own.
        (60.024, 20.0),
        (300.042, 60.0),
        (300.042, 20.0),
        (73.0, 30.0),
    ],
)
def test_cut_points_never_leave_a_sliver_section(total: float, ceiling: float) -> None:
    """No window may be a fraction of the others.

    A pass costs a 22B transformer loaded from host RAM before it draws its
    first frame, so a 0.18-second section is a full pass spent on four frames.
    Greedy filling produced exactly that on both measured tracks; nominal cuts
    with a bounded pull cannot, because the count is chosen before the cuts
    are placed and every cut lands at or before its nominal position.
    """
    onsets = [value * 0.517 for value in range(1, int(total / 0.517) + 1)]
    boundaries = plan_musical_boundaries(
        total, per_pass_seconds=ceiling, onsets=onsets
    )
    cuts = [0.0, *boundaries, total]
    windows = [end - start for start, end in zip(cuts, cuts[1:], strict=False)]

    assert len(windows) == math.ceil(total / ceiling - 1e-9), (
        "cutting on the music must not buy a pass the even plan would not have"
    )
    assert min(windows) >= 0.5 * max(windows), windows
    assert all(width <= ceiling + 1e-6 for width in windows)
    assert sum(windows) == pytest.approx(total)


def test_onset_detection_finds_nothing_in_a_flat_signal() -> None:
    """A steady tone has no events, and inventing some would put cuts in
    arbitrary places while claiming they were musical."""
    assert detect_onsets([0.4] * 500) == []


@needs_ffmpeg
async def test_onsets_are_found_in_a_track_that_actually_pulses(
    tmp_path: Path,
) -> None:
    """The detector against real decoded audio: a 120 BPM pulse should produce
    events at roughly two per second, not zero and not hundreds."""
    track = await make_track(tmp_path / "click.mp3", 8.0, beats_per_minute=120)
    onsets = await audio_onsets(track)

    assert onsets, "a pulsing track produced no detectable events"
    per_second = len(onsets) / 8.0
    assert 0.5 <= per_second <= 8.0, f"implausible event rate: {per_second:.1f}/s"


def test_inline_timed_paragraph_splits_like_the_h3_path() -> None:
    """The client's actual script format — "[0–6s] …" markers flowing through
    ONE paragraph — matched none of the line-based patterns, so the whole
    script rode into every section (the same failure the H3 path fixed on
    26 Aug). The inline parser now feeds the existing timed distribution:
    preamble = persistent identity restated per section, each block lands in
    the section holding its midpoint."""
    prompt = (
        "30-second cinematic military rescue. The same battle-worn army "
        "soldier remains consistent throughout, wearing olive tactical gear. "
        "[0-6s] A wide aerial shot follows him through the jungle toward the "
        "compound. [6-12s] He enters the building, flashlight cutting through "
        "dust. [12-18s] He hears knocking and opens a concealed steel door. "
        "[18-24s] He descends into the bunker and frees three prisoners. "
        "[24-30s] He leads the group outside as a helicopter approaches."
    )
    seg1, seg2 = plan_section_prompts(prompt, 2, total_seconds=30.0)
    # Identity preamble is restated in BOTH sections.
    for section in (seg1, seg2):
        assert "battle-worn army soldier" in section
    # Each section carries only its own slice of the timeline.
    assert "aerial" in seg1 and "flashlight" in seg1
    assert "helicopter" not in seg1 and "bunker" not in seg1
    assert "bunker" in seg2 and "helicopter" in seg2
    assert "aerial" not in seg2


def test_prompts_without_inline_markers_are_unchanged() -> None:
    prompt = "A koi pond at dawn, one fish gliding slowly beneath the mist."
    planned = plan_section_prompts(prompt, 2, total_seconds=10.0)
    # No timeline: the whole prompt stays the persistent brief in each section.
    assert all("koi pond" in section for section in planned)


def test_the_constraints_are_declared_the_complete_inventory() -> None:
    """Client report, 27 Aug: every music video came back looking the same —
    a singer with a crowd behind her — whatever the prompt asked for. The
    scaffolding outweighs a short prompt several times over, so its generic
    nouns read as suggestions. The constraints must be named as the ONLY
    inventory."""
    [section] = plan_section_prompts("A red sports car on an empty road", 1)
    # A single-pass request is still byte-identical: no scaffolding at all.
    assert section == "A red sports car on an empty road"

    sections = plan_section_prompts("A red sports car on an empty road", 3)
    for text in sections:
        assert "introduce no person, crowd, performer" in text


def test_music_video_sections_never_mention_dialogue() -> None:
    """The song IS the audio of a music video; three mentions of dialogue in
    the scaffolding were priming people talking on camera."""
    sections = plan_section_prompts(
        "A red sports car on an empty road", 3, dialogue=False
    )
    for text in sections:
        assert "dialogue" not in text.lower()

    # Workflows that DO write dialogue keep it — default is unchanged.
    spoken = plan_section_prompts("Two friends argue in a kitchen", 3)
    assert any("dialogue" in text.lower() for text in spoken)
