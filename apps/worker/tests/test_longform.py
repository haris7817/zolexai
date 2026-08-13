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
    render_chain,
)
from worker.longform.chain import _plan
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
        (73, 30, [30, 30, 13]),  # the directive's own example
        (42, 30, [30, 12]),
        (30, 30, [30]),
        (5, 30, [5]),
    ],
)
def test_a_requested_length_becomes_passes_the_gpu_survives(
    total: float, ceiling: float, expected: list[float]
) -> None:
    """The safety property the whole product rests on: a 60s render OOMed on
    the RTX 5090, so no length the product offers may reach the GPU whole."""
    segments = _plan(total, ceiling, None)

    assert [round(s.duration_seconds) for s in segments] == expected
    assert all(s.duration_seconds <= ceiling for s in segments)
    assert sum(s.duration_seconds for s in segments) == pytest.approx(total)


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
