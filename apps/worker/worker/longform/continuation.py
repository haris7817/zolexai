"""The extension engine — chained continuation, stated as such.

Nothing in this product generates an unbounded video in one pass. A length
beyond one graph submission is produced as a chain: each pass renders up to
`per_pass_seconds` of new material conditioned on the last picture of what
came before, and the passes are stitched behind the source. This module is
that mechanism, once, for every workflow that continues footage:

  * Extend Video — the customer's clip plus N seconds of continuation;
  * any generation whose requested length exceeds one pass (not offered in
    the current ladder, but the engine does not know that).

## What a pass sees, and what is kept

Pass k is conditioned on the final frame of pass k−1 (or of the source, for
pass 0) — the FLF graph's first-frame input, at the strength the pack pins.
The graph renders that frame again at index 0, so every chained pass
starts with one picture the timeline already has. That frame is the
**overlap**: it is dropped before stitching (`SEAM_OVERLAP_FRAMES`), so a
30 s continuation contributes exactly 30 s × fps frames and no seam shows a
held frame.

## The customer's own first and last frame (6 Sep 2026)

An extension may carry two optional stills. `first_frame` replaces the
source's final frame as pass 0's conditioning picture — the customer chose
the hand-off rather than taking the frame the video happened to end on.
Nothing else changes: the same graph, the same one-frame overlap drop (the
rendered index-0 frame is the still itself, one frame at the timeline's
fps), the same length arithmetic. `last_frame` is the caller's business —
it goes to the FINAL pass's second image through `render_pass`; the engine
only records it here so `continuation.json` says what the run was given.

## Audio at the seams

Each pass carries its own generated soundtrack; the source keeps its own.
Joins are butt joins with a short fade in/out on every generated part
(`AUDIO_EDGE_FADE_SECONDS`) — a click guard, not a crossfade, because a
crossfade shortens the timeline by one fade per join and the delivered
length must be exactly what was promised. The seam is audible as a change
of room when the model's ambience differs between passes; that is a model
property, recorded in the metadata so it can be measured, not hidden.

## Metadata

Every continuation writes `continuation.json` beside its output: the source
probe, every pass (window, frames rendered and kept, conditioning frame,
seed, wall clock), the seam timestamps, and the promised versus measured
length. The benchmark and the GPU-day checklist read it.

STATUS: exercised end to end against a fake service (real ffmpeg, real
files); the model's behaviour at a seam is WAITING FOR GPU VALIDATION.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from worker.adapters.base import AdapterError, AdapterJob, cancellable
from worker.core.logging import get_logger
from worker.longform.chain import ChainStep, render_chain
from worker.longform.progress import StageReporter
from worker.media import (
    FfmpegError,
    MediaInfo,
    concat_segments,
    extract_final_frame,
    ffmpeg,
    normalize_clip,
    probe_media,
    verify_duration,
)

logger = get_logger(__name__)

#: Frames a chained pass repeats at its start — the conditioning picture,
#: rendered again at index 0 by the first-frame graph. Dropped at the seam.
SEAM_OVERLAP_FRAMES = 1

#: Fade applied to the start and end of every generated part's audio.
AUDIO_EDGE_FADE_SECONDS = 0.08

RenderPass = Callable[[ChainStep, Path | None], Awaitable[MediaInfo]]
"""Renders one pass: writes `step.output` conditioned on the given frame
(the previous part's last picture, or the seed still) and returns its probe.
The engine owns everything else."""


@dataclass(frozen=True)
class PassRecord:
    index: int
    start_seconds: float
    """Where this pass's contribution begins in the CONTINUATION timeline."""
    seconds: float
    conditioning_frame: str | None
    frames_rendered: int | None
    frames_kept: int
    wall_seconds: float
    width: int | None
    height: int | None


@dataclass
class ContinuationMetadata:
    job_id: str
    workflow_id: str
    engine: str
    fps: float
    per_pass_seconds: float
    added_seconds: float
    source: dict[str, object] | None
    passes: list[PassRecord] = field(default_factory=list)
    seams: list[float] = field(default_factory=list)
    """Timestamps in the FINAL timeline where one part ends and the next begins."""
    overlap_frames_per_seam: int = SEAM_OVERLAP_FRAMES
    audio: str = "source + generated per pass, edge-faded at seams"
    promised_seconds: float = 0.0
    measured_seconds: float | None = None
    status: str = "WAITING FOR GPU VALIDATION"
    first_frame: str | None = None
    """The customer's own first frame (file name), when one replaced the
    source's final frame as pass 0's conditioning picture."""
    last_frame: str | None = None
    """The customer's own last frame (file name), when the final pass was
    asked to end on it."""

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


async def continue_video(
    job: AdapterJob,
    *,
    source: Path,
    seconds: float,
    per_pass_seconds: float,
    fps: float,
    render_pass: RenderPass,
    reporter: StageReporter,
    engine: str = "ltx_comfy",
    output: Path | None = None,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
) -> tuple[Path, ContinuationMetadata]:
    """The customer's clip plus `seconds` of continuation, one file.

    The source is kept as-is in content (re-encoded once to the common
    parameters every part shares); the continuation is chained at
    `per_pass_seconds` per pass; the result is verified against the promised
    total length before it is returned.

    `first_frame`, when given, is pass 0's conditioning picture instead of
    the source's extracted final frame. `last_frame` is recorded in the
    metadata only; applying it to the final pass is `render_pass`'s job
    (it is the one that knows which pass is last and how its graph takes a
    second image).
    """
    info = await _probe(source)
    if not info.has_video or not info.duration_seconds:
        raise AdapterError(
            "That video could not be read. Please try another.",
            internal_detail=f"source probe: {info!r}",
            retriable=False,
        )
    target_fps = float(info.fps or fps)
    width, height = _even(info.width or 0), _even(info.height or 0)
    if not width or not height:
        raise AdapterError(
            "That video could not be read. Please try another.",
            internal_detail=f"source has no usable dimensions: {info!r}",
            retriable=False,
        )

    metadata = ContinuationMetadata(
        job_id=job.job_id,
        workflow_id=job.workflow_id,
        engine=engine,
        fps=target_fps,
        per_pass_seconds=per_pass_seconds,
        added_seconds=seconds,
        source={
            "name": source.name,
            "seconds": info.duration_seconds,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "has_audio": info.has_audio,
        },
        promised_seconds=info.duration_seconds + seconds,
        first_frame=first_frame.name if first_frame is not None else None,
        last_frame=last_frame.name if last_frame is not None else None,
    )

    await reporter.probing("Reading your video…")
    if first_frame is not None:
        # The customer chose the picture the continuation starts on; the
        # source's own final frame is not extracted at all.
        seed_frame = first_frame
    else:
        try:
            seed_frame = await cancellable(
                job, extract_final_frame(source, job.workspace / "continuation-seed.png")
            )
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"final-frame extraction failed: {exc}",
                retriable=False,
            ) from exc

    parts = await generate_chain(
        job,
        seconds=seconds,
        per_pass_seconds=per_pass_seconds,
        fps=fps,
        render_pass=render_pass,
        seed_frame=seed_frame,
        reporter=reporter,
        metadata=metadata,
        prefix="continuation",
    )

    await reporter.stitching()
    normalized: list[Path] = [
        await cancellable(
            job,
            normalize_clip(
                source,
                job.workspace / "continuation-part-source.mp4",
                width=width,
                height=height,
                fps=target_fps,
                audio=True,
            ),
        )
    ]
    timeline = info.duration_seconds
    for record, part in zip(metadata.passes, parts, strict=True):
        metadata.seams.append(round(timeline, 4))
        normalized.append(
            await _prepare_part(
                job,
                part,
                index=record.index,
                seconds=record.seconds,
                overlap_frames=SEAM_OVERLAP_FRAMES,
                width=width,
                height=height,
                fps=target_fps,
                render_fps=fps,
            )
        )
        timeline += record.seconds

    destination = output or job.workspace / "output.mp4"
    try:
        await cancellable(job, concat_segments(normalized, destination))
        measured = await verify_duration(destination, expected_seconds=metadata.promised_seconds)
    except FfmpegError as exc:
        raise AdapterError(
            "This generation could not be completed. Please try again.",
            internal_detail=f"continuation assembly failed: {exc}",
        ) from exc
    metadata.measured_seconds = measured
    metadata.write(job.workspace / "continuation.json")
    logger.info(
        "continuation_assembled",
        extra={
            "job_id": job.job_id,
            "source_seconds": round(info.duration_seconds, 3),
            "added_seconds": seconds,
            "passes": len(parts),
            "seams": metadata.seams,
            "measured_seconds": round(measured, 3),
            "first_frame": metadata.first_frame,
            "last_frame": metadata.last_frame,
        },
    )
    return destination, metadata


async def generate_chain(
    job: AdapterJob,
    *,
    seconds: float,
    per_pass_seconds: float,
    fps: float,
    render_pass: RenderPass,
    seed_frame: Path | None,
    reporter: StageReporter,
    metadata: ContinuationMetadata | None = None,
    prefix: str = "segment",
) -> list[Path]:
    """`seconds` of new material as one or more conditioned passes.

    Returns the raw parts (with their overlap frames still present). Passes
    are chained on each other's final frame by `render_chain`; this wrapper
    records what each pass was given and how long it took.
    """
    records = metadata.passes if metadata is not None else []

    async def render(step: ChainStep) -> None:
        started = time.monotonic()
        info = await render_pass(step, step.previous_frame)
        records.append(
            PassRecord(
                index=step.index,
                start_seconds=step.segment.start_seconds,
                seconds=step.seconds,
                conditioning_frame=step.previous_frame.name if step.previous_frame else None,
                frames_rendered=info.frame_count,
                frames_kept=frames_for(step.seconds, fps),
                wall_seconds=round(time.monotonic() - started, 1),
                width=info.width,
                height=info.height,
            )
        )

    return await render_chain(
        job,
        seconds,
        per_pass_seconds=per_pass_seconds,
        render=render,
        reporter=reporter,
        prefix=prefix,
        seed_frame=seed_frame,
        chain_frames=True,
    )


def frames_for(seconds: float, fps: float) -> int:
    """Frames a pass contributes after its overlap is dropped."""
    return max(1, int(round(seconds * fps)))


async def _prepare_part(
    job: AdapterJob,
    part: Path,
    *,
    index: int,
    seconds: float,
    overlap_frames: int,
    width: int,
    height: int,
    fps: float,
    render_fps: float,
) -> Path:
    """Drops the overlap, guards the audio edges, normalises to the timeline."""
    trimmed = part.with_name(f"{part.stem}-seam.mp4")
    start = overlap_frames / max(render_fps, 1e-6)
    info = await _probe(part)
    fade = AUDIO_EDGE_FADE_SECONDS
    fade_out_at = max(0.0, seconds - fade)
    audio_filter = (
        f"atrim=start={start:.6f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={fade},"
        f"afade=t=out:st={fade_out_at:.3f}:d={fade}"
    )
    args = [
        "-i",
        str(part),
        "-vf",
        f"trim=start_frame={overlap_frames},setpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    if info.has_audio:
        args += ["-af", audio_filter, "-c:a", "aac", "-b:a", "128k"]
    else:
        args += ["-an"]
    args += [str(trimmed), "-y"]
    try:
        await cancellable(job, ffmpeg(args))
        return await cancellable(
            job,
            normalize_clip(
                trimmed,
                job.workspace / f"continuation-part-{index + 1:04d}.mp4",
                width=width,
                height=height,
                fps=fps,
                audio=True,
                frames=frames_for(seconds, fps),
            ),
        )
    except FfmpegError as exc:
        raise AdapterError(
            "This generation could not be completed. Please try again.",
            internal_detail=f"continuation part {index} could not be prepared: {exc}",
        ) from exc


async def _probe(path: Path) -> MediaInfo:
    try:
        return await probe_media(path)
    except FfmpegError as exc:
        raise AdapterError(
            "That video could not be read. Please try another.",
            internal_detail=f"probe failed for {path.name}: {exc}",
            retriable=False,
        ) from exc


def _even(value: int) -> int:
    return value - (value % 2)


def passes_needed(seconds: float, per_pass_seconds: float) -> int:
    """How many chained passes a continuation costs — for copy and docs."""
    return max(1, math.ceil(seconds / per_pass_seconds - 1e-9))
