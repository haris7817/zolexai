"""Harness runtime — real media files, no model and no GPU.

## Why this exists

The mock adapter emits a PNG in six seconds. That was enough to prove the job
pipeline, and it proves nothing about the parts M2 actually depends on:
streaming a large file to storage, staging a source video on disk, planning and
stitching segments, measuring what was produced, surviving a stage longer than
the lease, cleaning up afterwards.

Waiting for a GPU to exercise those paths would mean debugging the platform and
the model at the same time, on rented hardware, against a deadline. This adapter
removes that: it drives the *entire* real code path using ffmpeg to synthesise
genuine, playable MP4 and MP3 output. Every long-form behaviour the client asked
for can be built and tested before any GPU exists.

Afterwards it stays useful as the runtime for CI and for local development,
where a model is neither available nor wanted.

## What it is not

It is not a fallback and it is not a demo. It produces test patterns and tones —
obviously synthetic to anyone who looks. **No workflow ships pointed at it**;
`runtime: harness` is set by hand for local testing only. If a customer ever saw
its output, that would be a routing bug, not a degraded mode.
"""

from __future__ import annotations

from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    parse_duration_seconds,
)
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media import (
    FfmpegError,
    OutputExpectation,
    concat_segments,
    duration_tolerance,
    ffmpeg,
    mux_audio,
    plan_segments,
    probe_media,
    tools_available,
    verify_output,
)

logger = get_logger(__name__)

_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (960, 540),
    "9:16": (540, 960),
    "1:1": (720, 720),
    "4:5": (720, 900),
}
_DEFAULT_DIMENSIONS = (960, 540)

#: Progress band reserved for segment rendering. Everything stays inside the
#: `generating` status because the API's status ranking is strictly forward —
#: a per-segment hop into `post_processing` and back would be rejected as an
#: illegal transition and would abandon the job.
_GENERATE_FROM = 15
_GENERATE_TO = 85


class HarnessAdapter:
    name = "harness"

    def supports(self, workflow_id: str) -> bool:
        return True

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        if not tools_available():
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail="ffmpeg/ffprobe not found on PATH",
                retriable=False,
            )

        await on_progress("preparing", 10, "Setting up your generation…")
        target_seconds = await self._target_duration(job)
        audio_only = job.execution.get("output_kind") == "audio"

        segments = plan_segments(
            target_seconds,
            max_segment_seconds=job.execution_int(
                "max_segment_seconds", settings.max_segment_seconds
            ),
        )
        logger.info(
            "harness_plan",
            extra={"target_seconds": target_seconds, "segments": len(segments)},
        )

        rendered: list[Path] = []
        for segment in segments:
            job.raise_if_cancelled()
            await on_progress(
                "generating",
                _progress_for(segment.index, len(segments)),
                _generating_message(segment.index, len(segments)),
            )
            rendered.append(
                await self._render_segment(
                    job, segment.index, segment.generate_seconds, audio_only
                )
            )

        await on_progress("post_processing", 90, "Polishing and encoding…")
        suffix = ".mp3" if audio_only else ".mp4"
        output = job.workspace / f"output{suffix}"

        # A workflow whose length came from an uploaded track must deliver that
        # track, whole — the same promise the real runtime makes. Without this
        # the harness would produce a "music video" carrying a synthetic tone,
        # which is precisely the quietly-wrong output this adapter exists to
        # rule out on every other path.
        track = job.input_for("source_audio")
        soundtrack = track.path if track and not audio_only else None

        try:
            joined = await concat_segments(
                rendered, job.workspace / f"joined{suffix}" if soundtrack else output
            )
            if soundtrack is not None:
                await mux_audio(joined, soundtrack, output)
            info = await verify_output(
                output,
                OutputExpectation(
                    expect_video=not audio_only,
                    expect_audio=soundtrack is not None,
                    expected_seconds=target_seconds,
                    tolerance_seconds=duration_tolerance(target_seconds),
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"assembly failed: {exc}",
            ) from exc

        measured = info.duration_seconds
        await on_progress("uploading", 95, "Almost ready…")

        return AdapterResult(
            path=output,
            content_type="audio/mpeg" if audio_only else "video/mp4",
            kind="audio" if audio_only else "video",
            duration_seconds=measured,
            width=info.width,
            height=info.height,
        )

    # ── Duration ─────────────────────────────────────────────────────────

    async def _target_duration(self, job: AdapterJob) -> float:
        """Source length when there is a source, otherwise the requested length.

        This is the shape the client's automatic-duration requirements need:
        video-to-video and music video take their length from the uploaded file,
        everything else from the request. Doing it here — against a real probe
        of a real file — is what makes the behaviour testable before the
        workflow metadata changes land in Batch 2.
        """
        for role in ("source_video", "source_audio"):
            item = job.input_for(role)
            if item is None or item.path is None:
                continue
            try:
                info = await probe_media(item.path)
            except FfmpegError as exc:
                raise AdapterError(
                    "That file could not be read. Please try another.",
                    internal_detail=f"probe of {role} failed: {exc}",
                    retriable=False,
                ) from exc
            if info.duration_seconds:
                return info.duration_seconds

        requested = parse_duration_seconds(job.parameters.get("duration"))
        if requested is None:
            raise AdapterError(
                "This generation could not be started.",
                internal_detail=f"no usable duration in {job.parameters!r}",
                retriable=False,
            )
        return requested

    # ── Rendering ────────────────────────────────────────────────────────

    async def _render_segment(
        self, job: AdapterJob, index: int, seconds: float, audio_only: bool
    ) -> Path:
        destination = job.workspace / f"segment-{index:04d}{'.mp3' if audio_only else '.mp4'}"
        try:
            await ffmpeg(
                self._segment_args(job, seconds, destination, audio_only),
                timeout=max(60.0, seconds * 10),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"segment {index} failed: {exc}",
            ) from exc
        return destination

    def _segment_args(
        self, job: AdapterJob, seconds: float, destination: Path, audio_only: bool
    ) -> list[str]:
        # `-t` before the output bounds the synthetic source, which is otherwise
        # infinite; lavfi generators do not end on their own.
        if audio_only:
            return [
                "-f", "lavfi", "-i", "sine=frequency=340:sample_rate=44100",
                "-t", f"{seconds:.3f}",
                "-c:a", "libmp3lame", "-b:a", "128k",
                str(destination),
            ]

        width, height = _DIMENSIONS.get(
            str(job.parameters.get("aspect_ratio") or ""), _DEFAULT_DIMENSIONS
        )
        return [
            "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100",
            "-t", f"{seconds:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k",
            # Every segment must share codec parameters or the stream-copy
            # concat falls back to a re-encode. A fixed GOP keeps the cut points
            # on keyframes.
            "-g", "24",
            str(destination),
        ]


def _progress_for(index: int, total: int) -> int:
    span = _GENERATE_TO - _GENERATE_FROM
    return _GENERATE_FROM + int(span * index / max(1, total))


def _generating_message(index: int, total: int) -> str:
    """Customer-facing copy. Mentions sections only when there are several.

    A single-pass job saying "Section 1 of 1" would expose machinery for no
    benefit; a four-minute job with no section counter looks stuck.
    """
    if total <= 1:
        return "This usually takes a couple of minutes."
    return f"Generating section {index + 1} of {total}…"
