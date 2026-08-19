"""Seam timing: a stitched restyle must not drift against its own soundtrack.

Video-to-video is the one workflow where picture and sound come from different
places: the picture is generated in sections, the audio is the source's own
track laid over the result once. That shape has a failure mode none of the
other workflows can express — every section can only be delivered as WHOLE
frames, so a section whose planned length is fractional comes back a sliver
long, and butt-joining the sections places each later one's content later
against the continuous audio. Measured arithmetically before the fix: +33ms
per seam on a 37s source at 30fps, +133ms of lip lag by the final section —
past the ~45ms threshold where a mouth visibly moves after its words.

Text-to-video never shows this because its sections carry their own generated
audio: any duration rounding moves both streams together, and its product
durations fit one pass anyway. That asymmetry — T2V clean, V2V slightly off —
was the reported symptom, and it is exactly what these tests hold down.

The fix: `_deliver_restyle` re-derives the chain's plan and pins every
normalized section to its planned frame count at the delivery rate, allocated
from cumulative boundaries so rounding cannot walk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
)
from worker.adapters.ltx import LtxAdapter
from worker.media import probe_media


def restyle_job(workspace: Path, source: Path, **overrides):
    defaults = dict(
        workflow_id="video-to-video",
        prompt="repaint it as a charcoal sketch",
        parameters={"aspect_ratio": "16:9"},
        inputs=[staged_input("source_video", "video", "video/mp4", source)],
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── The allocator, as arithmetic ─────────────────────────────────────────


def test_section_frames_come_from_cumulative_boundaries() -> None:
    """`round(end·fps) − round(start·fps)`, never per-section rounding.

    The 37s/5-pass case is the measured drift scenario: 7.4s sections are
    177.6 frames at 24fps, and independently ceiling each one is how +17ms a
    seam accumulated. Allocated cumulatively, the counts differ by at most a
    frame and their sum is the total to the frame.
    """
    adapter = LtxAdapter()
    rendered = [Path(f"part-{i}.mp4") for i in range(5)]

    counts = adapter._planned_section_frames(rendered, 37.0, 8.0, 30.0)
    assert counts is not None
    assert sum(counts) == round(37.0 * 30.0)
    assert all(abs(count - 7.4 * 30.0) <= 1 for count in counts)

    counts = adapter._planned_section_frames(rendered, 37.0, 8.0, 24.0)
    assert counts is not None
    assert sum(counts) == round(37.0 * 24.0)


def test_a_plan_that_does_not_match_what_rendered_declines_to_pin() -> None:
    """A mismatch means this arithmetic diverged from the chain's own plan.

    Delivering with the historical quantization is strictly better than
    cutting sections at lengths computed for a different plan."""
    adapter = LtxAdapter()
    assert (
        adapter._planned_section_frames([Path("only.mp4")], 37.0, 8.0, 24.0) is None
    )


# ── The delivery, end to end ─────────────────────────────────────────────


@needs_ffmpeg
async def test_sections_are_delivered_at_their_planned_frame_counts(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An awkward length — one whose even windows are NOT whole frames —
    must come back with every section cut to the plan, not to the render.

    3.7s at a 1s ceiling plans four 0.925s windows: 22.2 frames each at
    24fps. Before the fix each normalized section kept its own quantized
    length (23 frames) and the picture ended four slivers late; pinned, the
    counts are 22/22/23/22 — cumulative boundaries — and the join lands on
    the source timeline to the frame.
    """
    source = await make_clip(workspace / "source.mp4", 3.7, audio=True)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))
    job = restyle_job(
        workspace, source,
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    )
    await collect(job)

    expected = [
        round(3.7 * (k + 1) / 4 * 24) - round(3.7 * k / 4 * 24) for k in range(4)
    ]
    assert sum(expected) == round(3.7 * 24)

    sections = sorted(workspace.glob("normalized-section-*.mp4"))
    assert len(sections) == 4
    measured = [(await probe_media(part)).frame_count for part in sections]
    assert measured == expected

    output = await probe_media(workspace / "output.mp4")
    assert output.duration_seconds == pytest.approx(3.7, abs=0.1)
    assert output.has_audio is True
