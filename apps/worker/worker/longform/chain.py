"""The one long-form mechanism, shared by every workflow that needs one.

The client asked for five things that all reduce to the same problem: a
finished duration longer than a model can produce in one pass.

  | workflow        | total length is                  | first pass sees      |
  |-----------------|----------------------------------|----------------------|
  | text-to-video   | what the user picked             | nothing              |
  | image-to-video  | what the user picked             | the uploaded still   |
  | extend-video    | the extension the user picked    | the source's last    |
  | video-to-video  | the SOURCE's own duration        | the source's opening |
  | music-video     | the SONG's own duration          | nothing              |

Only the last column differs, so only the last column is a parameter. Everything
else — how many passes, how long each one may be, what conditions pass N, how
progress is spread across them, when cancellation is honoured, what happens to
a pass that fails — is this module, once.

**Nothing here decides the pass ceiling.** It is passed in, and it comes from
`settings.ltx_max_seconds` (or a workflow's lower override) rather than from a
number written down in five places. That is the property that keeps a request
the GPU cannot survive from ever being issued: the planner splits by the
measured ceiling, and no arithmetic anywhere multiplies past it.

This module does not know what a model is. `render` is a callable; the LTX
adapter passes one that shells out to a GPU, and a test passes one that writes
a fixture. That is what makes every long-form guarantee in this codebase
provable without hardware.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from worker.adapters.base import AdapterError, AdapterJob, cancellable
from worker.core.logging import get_logger
from worker.longform.progress import StageReporter, band_for
from worker.media import FfmpegError, Segment, extract_final_frame, plan_segments

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChainStep:
    """Everything a renderer needs for one pass, and nothing about the model."""

    segment: Segment
    total: int
    output: Path
    """Where this pass must write its file."""

    previous_frame: Path | None
    """
    The last picture of the pass before this one, or the caller's own seed for
    the first pass.

    Continuity across a seam is entirely this frame: a pass that starts from
    its predecessor's final image continues the shot, and one that starts cold
    begins a different-looking video halfway through the result. Renderers that
    have a stronger continuation signal available should prefer it and ignore
    this.
    """

    band: tuple[int, int]
    """The progress range this pass owns. Reporting outside it makes the
    customer's bar jump or stall."""

    @property
    def index(self) -> int:
        return self.segment.index

    @property
    def seconds(self) -> float:
        return self.segment.duration_seconds

    @property
    def is_first(self) -> bool:
        return self.segment.index == 0

    @property
    def section(self) -> tuple[int, int] | None:
        """(n, of) for customer copy — None when there is only one pass."""
        return (self.segment.index + 1, self.total) if self.total > 1 else None


RenderStep = Callable[[ChainStep], Awaitable[None]]
"""Produces one pass. Must write `step.output` and may report progress inside
`step.band`. Raising aborts the chain."""


async def render_chain(
    job: AdapterJob,
    total_seconds: float,
    *,
    per_pass_seconds: float,
    render: RenderStep,
    reporter: StageReporter,
    prefix: str = "segment",
    seed_frame: Path | None = None,
    chain_frames: bool = True,
    boundaries: list[float] | None = None,
) -> list[Path]:
    """Produces `total_seconds` of material as one or more passes.

    Returns the rendered parts in order. Assembly is the caller's job, because
    what "assembled" means differs per workflow — an extension puts the user's
    own footage in front, a music video lays a soundtrack over the whole thing.

    A length within one pass costs no chaining machinery at all: one render, no
    frame extraction, no section counter, no concat.
    """
    segments = _plan(total_seconds, per_pass_seconds, boundaries)
    total = len(segments)
    logger.info(
        "longform_plan",
        extra={
            "workflow_id": job.workflow_id,
            "total_seconds": round(total_seconds, 3),
            "per_pass_seconds": per_pass_seconds,
            "passes": total,
            "pass_seconds": [round(s.duration_seconds, 3) for s in segments],
        },
    )

    rendered: list[Path] = []
    previous_frame = seed_frame

    for segment in segments:
        # Between passes is the cheapest possible place to stop, and on a long
        # job it is the difference between releasing the GPU now and holding it
        # for another pass nobody will collect.
        job.raise_if_cancelled()

        step = ChainStep(
            segment=segment,
            total=total,
            output=job.workspace / f"{prefix}-{segment.index:04d}.mp4",
            previous_frame=previous_frame,
            band=band_for(segment.index, total),
        )
        await render(step)

        if not step.output.exists():
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=(
                    f"pass {segment.index} reported success but wrote no file at "
                    f"{step.output.name}"
                ),
            )
        rendered.append(step.output)

        if chain_frames and segment.index + 1 < total:
            previous_frame = await _final_frame(
                job, step.output, segment.index + 1, prefix
            )

    return rendered


def _plan(
    total_seconds: float, per_pass_seconds: float, boundaries: list[float] | None
) -> list[Segment]:
    """Even windows, or the caller's own cut points when it has better ones.

    A music video knows where the music changes and would rather cut there than
    at an arbitrary multiple of the pass ceiling. `boundaries` lets it say so;
    they are still validated against the ceiling here, so a timing layer can
    never widen a pass past what the GPU survives.
    """
    if not boundaries:
        return plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)

    cuts = [0.0, *sorted(boundaries), total_seconds]
    segments: list[Segment] = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        duration = end - start
        if duration <= 1e-6:
            continue
        if duration > per_pass_seconds + 1e-6:
            raise ValueError(
                f"boundary window {duration:.2f}s exceeds the {per_pass_seconds:.2f}s "
                "pass ceiling"
            )
        segments.append(
            Segment(index=len(segments), start_seconds=start, duration_seconds=duration)
        )
    if not segments:
        return plan_segments(total_seconds, max_segment_seconds=per_pass_seconds)
    return segments


async def _final_frame(job: AdapterJob, part: Path, index: int, prefix: str) -> Path:
    try:
        return await cancellable(
            job,
            extract_final_frame(part, job.workspace / f"{prefix}-condition-{index:04d}.png"),
        )
    except FfmpegError as exc:
        # A GENERATED part being unreadable is a generation flake, not a bad
        # upload — a retry can genuinely produce a readable one.
        raise AdapterError(
            "This generation could not be completed. Please try again.",
            internal_detail=f"pass {index - 1} produced an unreadable file: {exc}",
        ) from exc
