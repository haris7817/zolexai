"""Character Replacement — a separate module on the client's LTX 2.5 graph.

Runtime `character_replacement`, one workflow (`character-replacement`),
one graph (`ltx25_character_replacement.json`). Kept apart from Video to
Video by design: that workflow restyles footage on the CLI runtime and is
untouched; this one regenerates a performance with a new person, and the
two share nothing but the ComfyUI service object.

## What the product is (client decision, 5 Sep 2026)

Exactly the delivered sample. The reference image provides the new
character AND the environment; the source video provides motion, camera and
timing. No background preservation, no compositing — the graph's Ripple
LoRA propagates the reference picture through the source's motion, and the
first frame of the result is the reference picture itself (measured on the
sample: frame 0 is the photo, frame 4 onward is the source's motion).

## What a job supplies to the graph

The source clip, the reference still (PNG-normalised), the length in whole
seconds, the canvas (the pack's 736×1280 budget, oriented like the source),
the prompt (the pack's lead sentence plus the customer's description of the
new character — the sample prompt shows that description is what carries
identity), the seed and the output prefix. Sampler, schedule, LoRA
strength, patches and switches are the pack's.

## The whole source, as a chain of windows (client request, 6 Sep 2026)

One run of the graph follows at most `character_replacement_max_seconds`
of source (10 s on the RTX PRO 6000: 85 GB of VRAM, 110 GiB of RAM; 20 s
was killed by the container's memory limit). The client wants the result to
follow the WHOLE video, as Video to Video does. So a longer source is cut
into whole-second windows on the graph's own 24 fps grid (the graph's loader
resamples every source to 24 fps — `force_rate` is wired to its "Set FPS"
primitive), each window is one unchanged run of the graph, and the pieces
are joined behind one another with the source's own soundtrack laid over
the whole result.

What carries the character across a seam is the graph's own mechanism: its
one reference picture is the first frame of what it renders. Window 0 gets
the customer's photo, as before. Window k gets the LAST frame window k−1
produced — the new character, in the new setting, at exactly the pose the
motion had reached — so the graph continues rather than restarts. That
frame is rendered again at index 0 of window k (the graph's first frame is
its reference), and that one duplicate is dropped at the seam, the same
overlap rule the extension engine uses. `character_replacement_chain_reference`
= `photo` is the alternative (the customer's picture for every window: no
drift over a long chain, a pose snap at every seam), kept as a setting so
the two can be compared on the GPU.

Nothing about the graph changes per window: same nodes, same models, same
LoRA, same canvas, same seeds (or the customer's seed plus the window
index). A source within one window runs exactly the path that ran before
this feature existed — one upload of the clip as-is, one run.

STATUS: the single-window path is validated on the GPU (5–6 Sep 2026); the
chained path is exercised against the fake service with real files and is
WAITING FOR GPU VALIDATION for what the model does at a seam.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    AdapterResult,
    ProgressCallback,
    cancellable,
)
from worker.comfy.client import ComfyError, evict_comfy_vram
from worker.comfy.ltx_graphs import (
    GraphError,
    ReplacementEdits,
    character_frames_for_seconds,
    compile_character_replacement,
    oriented_canvas,
)
from worker.comfy.ltx_prompts import (
    CHARACTER_REPLACEMENT_EXPOSURE,
    CHARACTER_REPLACEMENT_EXPOSURE_STYLISED,
    CHARACTER_REPLACEMENT_SKIN,
    character_replacement_prompt,
    looks_stylised,
    negative_for,
)
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.longform import GENERATE_FROM, GENERATE_TO, StageReporter
from worker.longform.progress import band_for
from worker.media import (
    FfmpegError,
    MediaInfo,
    OutputExpectation,
    concat_segments,
    extract_final_frame,
    ffmpeg,
    ffmpeg_stdout,
    probe_media,
    verify_output,
)
from worker.media.skin import SkinStats, anchor_skin, skin_target, source_gate
from worker.media.skin_hold import (
    HoldTarget,
    hold_apply_args,
    measure_window,
    plan_hold,
    source_skin_level,
)
from worker.media.upscale import is_4k, upscale_clip
from worker.providers.ltx_comfy import LtxComfyService

logger = get_logger(__name__)

WORKFLOW_ID = "character-replacement"

#: Below this the graph's own frame formula gives fewer than two latent
#: frames of motion to follow; the customer is told to upload a longer clip.
_MIN_SOURCE_SECONDS = 1.0

#: Frames a chained window repeats at its start — its reference picture,
#: rendered again at index 0 by the graph. Dropped at the seam.
SEAM_OVERLAP_FRAMES = 1

#: Every part of a chain goes through these encoder settings so the join is
#: a stream copy and no window differs from its neighbours in grain — held
#: or not. (The quality was the encoder's default until the hold shipped;
#: at that default a held window would have been visibly cleaner than an
#: unheld one in the same delivery.)
PART_ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]

#: Wall-clock allowance per frame for one pass of the skin hold. Measured on
#: the node (ffmpeg 6.1, Threadripper 9965WX): 0.04 s per frame to measure
#: and 0.08 s to apply at half canvas; this is an order of magnitude above
#: that, so only a pass that has genuinely hung hits it.
HOLD_SECONDS_PER_FRAME = 3.0

#: The first window's frames the colour anchor is read from: after the
#: graph's four-frame handoff from the photo, one second of its own
#: rendering of the customer's picture in the source's framing.
ANCHOR_FRAMES = (4, 27)

#: Bounds on the seed correction. A seed that is a little darker and flatter
#: than the anchor (the measured drift: a few units of mean luminance and
#: 10-30 units of highlight per window) is brought back fully; a seed that
#: differs more than this is a real change of framing or content, and only
#: this much of the difference is taken back.
ANCHOR_GAIN_RANGE = (0.80, 1.30)
ANCHOR_LUMA_OFFSET_LIMIT = 40.0
ANCHOR_CHROMA_OFFSET_LIMIT = 16.0
#: Below this luminance spread a frame has no usable contrast to scale.
ANCHOR_MIN_SPREAD = 8.0


@dataclass(frozen=True)
class Window:
    """One run of the graph over one stretch of the source."""

    index: int
    seconds: int
    """Whole seconds the graph is asked for (`Set Length (seconds)`)."""
    frames: int
    """What the graph renders for that: `round((fps·s − 1)/8)·8 + 1`."""
    start_frame: int
    """The first source frame (on the 24 fps grid) this window loads."""

    @property
    def kept_frames(self) -> int:
        return self.frames if self.index == 0 else self.frames - SEAM_OVERLAP_FRAMES


def plan_windows(total_seconds: int, window_seconds: int) -> list[Window]:
    """Whole-second windows, as even as possible, chained on a shared frame.

    Even rather than greedy (the rule `plan_segments` follows and explains):
    25 s at a 10 s ceiling is 9 + 8 + 8, never 10 + 10 + 5. Window k starts
    on the LAST frame of window k−1 — that frame is window k's reference
    picture and is rendered again at its index 0 — so consecutive windows
    overlap by exactly one frame and the delivered timeline is the source's
    own 24 fps timeline, frame for frame.
    """
    if total_seconds < 1 or window_seconds < 1:
        raise ValueError("windows need at least one whole second")
    count = math.ceil(total_seconds / window_seconds)
    base, extra = divmod(total_seconds, count)
    windows: list[Window] = []
    start = 0
    for index in range(count):
        seconds = base + (1 if index < extra else 0)
        frames = character_frames_for_seconds(seconds, settings.ltx_comfy_frame_rate)
        windows.append(Window(index=index, seconds=seconds, frames=frames, start_frame=start))
        start += frames - SEAM_OVERLAP_FRAMES
    return windows


def delivered_frames(windows: list[Window]) -> int:
    return sum(window.kept_frames for window in windows)


@dataclass(frozen=True)
class ColourAnchor:
    """Luminance level, luminance spread and chroma means of some frames.

    Limited-range YUV as `signalstats` reports it: `y_low`/`y_high` are its
    10th and 90th percentiles, so the spread is the picture's contrast
    without its extremes.
    """

    y_mean: float
    y_low: float
    y_high: float
    u_mean: float
    v_mean: float

    @property
    def spread(self) -> float:
        return self.y_high - self.y_low


@dataclass(frozen=True)
class WindowRecord:
    index: int
    seconds: int
    frames: int
    start_frame: int
    reference: str
    wall_seconds: float
    seed_correction: dict[str, float] | None = None
    """How the seed this window was given was brought back to the anchor
    (gain and offsets, and the seed's luminance before and after); None for
    window 0, for `photo` mode, and with anchoring off."""
    skin_correction: dict[str, object] | None = None
    """The skin-region re-anchoring of this window's seed (`worker.media.skin`):
    what was measured, whether it was applied and why not; None when the
    switch is off or there was no seed."""
    skin_hold: dict[str, object] | None = None
    """The per-frame skin hold on this window's delivered frames
    (`worker.media.skin_hold`): frames held, the offsets, refusals by
    reason, or why it could not run; None with the switch off."""


@dataclass
class ChainMetadata:
    job_id: str
    fps: int
    window_seconds: int
    reference_mode: str
    source: dict[str, object]
    windows: list[WindowRecord] = field(default_factory=list)
    seams: list[float] = field(default_factory=list)
    """Timestamps in the final timeline where one window hands over to the
    next (the shared frame)."""
    overlap_frames_per_seam: int = SEAM_OVERLAP_FRAMES
    audio: str = "the source's own track, laid over the whole result"
    promised_seconds: float = 0.0
    measured_seconds: float | None = None
    status: str = "WAITING FOR GPU VALIDATION (seam behaviour)"
    anchor: dict[str, float] | None = None
    """The first window's own rendering just after its handoff — what every
    later seed is matched to."""
    skin_clause: bool = False
    """Whether the hands clause was in every window's prompt."""
    skin_target: dict[str, float] | None = None
    """The first window's own skin level and chroma inside the source
    silhouette — what every later seed's dark skin is lifted back to."""
    skin_hold: bool = False
    stylised: bool = False
    """Whether this job was treated as a DRAWN character: the style terms
    dropped from the negative prompt and every skin stage stood down. Recorded
    because it is a heuristic over the prompt unless a deployment set
    `subject_style`, and a wrong guess has to be visible in the artefact
    rather than only in a log line."""
    """Whether every window's delivered frames went through the skin hold."""
    skin_source_y: float | None = None
    """The source performer's own skin luminance over the target frames —
    the level the per-frame hold target follows."""

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


class CharacterReplacementAdapter:
    name = "character_replacement"

    def __init__(self, service: LtxComfyService | None = None) -> None:
        self._service = service

    def supports(self, workflow_id: str) -> bool:
        return workflow_id == WORKFLOW_ID

    def service(self) -> LtxComfyService:
        if self._service is None:
            self._service = LtxComfyService()
        return self._service

    # ── Delivery ─────────────────────────────────────────────────────────

    @staticmethod
    def delivers_4k(job: AdapterJob) -> bool:
        """Whether the joined video is enlarged to 4K before it is uploaded.

        `execution.delivery` on the job, else the deployment's
        `character_replacement_delivery`. Anything unrecognised is native, so
        a typo cannot silently quadruple every file."""
        return is_4k(job.execution.get("delivery") or settings.character_replacement_delivery)

    @staticmethod
    def four_k_target(width: int, height: int) -> tuple[int, int]:
        """The 4K frame for a canvas of this shape.

        Computed from the delivered frame rather than looked up by ratio
        label: Character Replacement takes whatever shape the customer's
        source is, and a table would have to guess. 2160 on the SHORT side is
        what makes 16:9 land on 3840x2160 and 9:16 on 2160x3840 — the same
        frames Text to Video HD delivers. Both sides are forced even, which
        H.264 requires.
        """
        if width <= 0 or height <= 0:
            return (3840, 2160)
        scale = 2160 / min(width, height)
        return (
            max(2, int(round(width * scale)) // 2 * 2),
            max(2, int(round(height * scale)) // 2 * 2),
        )

    async def _deliver(
        self, job: AdapterJob, reporter: StageReporter, output: Path, info: MediaInfo
    ) -> AdapterResult:
        """The finished file, optionally at 4K, as an `AdapterResult`.

        One place, called by both paths, so the single-window job and the
        chained one cannot deliver differently. The upscale runs LAST — after
        the join and after the source audio is laid over — which is what
        keeps it one encode with one audio stream and no seam.
        """
        if self.delivers_4k(job):
            target = self.four_k_target(info.width, info.height)
            await reporter.generating(GENERATE_TO - 1, "Upscaling your video to 4K…")
            try:
                output = await upscale_clip(
                    output,
                    job.workspace / "output_4k.mp4",
                    target,
                    nvenc_timeout=settings.ltx_comfy_transfer_timeout,
                    cpu_timeout=settings.ltx_comfy_generation_timeout,
                    run=lambda awaitable: cancellable(job, awaitable),
                    log_extra={"job_id": job.job_id, "workflow_id": job.workflow_id},
                )
            except FfmpegError as exc:
                raise AdapterError(
                    "The finished video could not be upscaled to 4K.",
                    internal_detail=str(exc)[-600:],
                ) from exc
            info = await probe_media(output)
        await reporter.uploading()
        return AdapterResult(
            path=output,
            content_type="video/mp4",
            kind="video",
            duration_seconds=info.duration_seconds,
            width=info.width,
            height=info.height,
        )

    # ── Entry ────────────────────────────────────────────────────────────

    async def run(self, job: AdapterJob, on_progress: ProgressCallback) -> AdapterResult:
        if not self.supports(job.workflow_id):
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=(
                    f"character_replacement does not serve '{job.workflow_id}'; "
                    "check this deployment's execution.runtime"
                ),
                retriable=False,
            )
        reporter = StageReporter(on_progress)
        await reporter.preparing("Setting up your video…")
        await evict_comfy_vram(exclude=settings.ltx_comfy_base_url)

        source = job.input_for("source_video")
        reference = job.input_for("reference_image")
        if source is None:
            raise AdapterError(
                "Please add the video to work from.",
                internal_detail="character-replacement without source_video",
                retriable=False,
            )
        if reference is None:
            raise AdapterError(
                "Please add a picture of the new character.",
                internal_detail="character-replacement without reference_image",
                retriable=False,
            )

        await reporter.probing("Reading your video…")
        info = await self._probe(source.require_path())
        width, height = oriented_canvas(
            tuple(settings.character_replacement_canvas),
            source_width=info.width,
            source_height=info.height,
        )

        windows = self.windows_for(info, job)
        try:
            if len(windows) == 1:
                # A source within one window: the path that ran before the
                # chain existed, unchanged — the clip uploaded as-is, one run.
                return await self._run_single(
                    job, reporter, source, reference, info, windows[0].seconds, width, height
                )
            return await self._run_chain(
                job, reporter, source, reference, info, windows, width, height
            )
        finally:
            # A chain leaves ComfyUI holding one prompt's worth of memory per
            # window, and the container's limit is what killed the server four
            # times in two days — once mid-render of the client's own job.
            if settings.ltx_comfy_free_after_job or (
                len(windows) > 1 and settings.character_replacement_free_after_chain
            ):
                await self.service().free_memory()

    # ── One window: the source as uploaded ───────────────────────────────

    async def _run_single(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        source: AdapterInput,
        reference: AdapterInput,
        info: MediaInfo,
        seconds: int,
        width: int,
        height: int,
    ) -> AdapterResult:
        service = self.service()
        video_name = await self._upload_source(job, source, info)
        image_name = await self._upload_still(job, reference.require_path(), "reference")

        edits = ReplacementEdits(
            positive=character_replacement_prompt(
                job.prompt, exposure=self.exposure_clause(job)
            ),
            negative=negative_for(
                WORKFLOW_ID, job.execution, drop_style=self.is_stylised(job)
            ),
            video=video_name,
            image=image_name,
            seconds=seconds,
            width=width,
            height=height,
            seed_base=self.seed_base(job),
            filename_prefix=f"zolexai/{job.job_id}/output",
            lora_strength=self.ripple_strength(job),
        )
        # The graph renders round((fps·s − 1)/8)·8 + 1 frames, capped by the
        # frames the resampled source actually has; the sample (8 s) is 193
        # frames = 8.04 s. Audio is the source's own, passed through.
        fps = settings.ltx_comfy_frame_rate
        planned_frames = min(
            character_frames_for_seconds(seconds, fps),
            int(math.floor((info.duration_seconds or seconds) * fps)),
        )
        output = job.workspace / "output.mp4"
        result_info, _wall = await self._render(
            job,
            service,
            edits,
            output,
            reporter,
            band=(GENERATE_FROM, GENERATE_TO),
            section=None,
            expected_seconds=planned_frames / fps,
            log_extra={"source_seconds": info.duration_seconds, "canvas": [width, height]},
        )
        return await self._deliver(job, reporter, output, result_info)

    # ── Several windows: the whole source ────────────────────────────────

    async def _run_chain(
        self,
        job: AdapterJob,
        reporter: StageReporter,
        source: AdapterInput,
        reference: AdapterInput,
        info: MediaInfo,
        windows: list[Window],
        width: int,
        height: int,
    ) -> AdapterResult:
        service = self.service()
        fps = settings.ltx_comfy_frame_rate
        staged = source.require_path()
        mode = self.reference_mode(job)
        metadata = ChainMetadata(
            job_id=job.job_id,
            fps=fps,
            window_seconds=max(window.seconds for window in windows),
            reference_mode=mode,
            source={
                "name": staged.name,
                "seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "has_audio": info.has_audio,
            },
            promised_seconds=delivered_frames(windows) / fps,
        )
        logger.info(
            "character_replacement_chain",
            extra={
                "job_id": job.job_id,
                "source_seconds": round(info.duration_seconds or 0.0, 3),
                "windows": [window.seconds for window in windows],
                "frames": [window.frames for window in windows],
                "stylised": metadata.stylised,
                "promised_seconds": round(metadata.promised_seconds, 3),
                "reference_mode": mode,
                "canvas": [width, height],
            },
        )

        photo_name = await self._upload_still(job, reference.require_path(), "reference")
        image_name = photo_name
        seed = self.seed_base(job)
        parts: list[Path] = []
        total = len(windows)
        anchoring = mode == "previous_frame" and self.anchors_reference(job)
        anchor: ColourAnchor | None = None
        pending_correction: dict[str, float] | None = None
        skin_clause = CHARACTER_REPLACEMENT_SKIN if self.chain_skin_clause(job) else None
        metadata.skin_clause = skin_clause is not None
        skin_anchoring = anchoring and self.anchors_skin(job)
        holding = self.holds_skin(job)
        metadata.skin_hold = holding
        metadata.stylised = self.is_stylised(job)
        skin: SkinStats | None = None
        source_y: float | None = None
        pending_skin: dict[str, object] | None = None
        clips: list[Path] = []
        prepared: list[bool] = []
        lora_strength = self.ripple_strength(job)

        for window in windows:
            job.raise_if_cancelled()
            clip = await self._cut_window(job, staged, info, window)
            clips.append(clip)
            video_name = await self._upload_clip(clip)
            edits = ReplacementEdits(
                positive=character_replacement_prompt(
                    job.prompt, skin=skin_clause, exposure=self.exposure_clause(job)
                ),
                negative=negative_for(
                WORKFLOW_ID, job.execution, drop_style=self.is_stylised(job)
            ),
                video=video_name,
                image=image_name,
                seconds=window.seconds,
                width=width,
                height=height,
                seed_base=None if seed is None else seed + window.index,
                filename_prefix=f"zolexai/{job.job_id}/window{window.index:02d}",
                lora_strength=lora_strength,
            )
            output = job.workspace / f"window-{window.index:04d}.mp4"
            start_seconds = window.start_frame / fps
            _info, wall = await self._render(
                job,
                service,
                edits,
                output,
                reporter,
                band=band_for(window.index, total),
                section=(
                    window.index + 1,
                    total,
                    start_seconds,
                    start_seconds + window.frames / fps,
                ),
                expected_seconds=window.frames / fps,
                log_extra={"window": window.index, "of": total, "start_frame": window.start_frame},
            )
            if window.index == 0:
                if anchoring:
                    anchor = await self._measure_colour(job, output, ANCHOR_FRAMES)
                    metadata.anchor = asdict(anchor)
                if skin_anchoring or holding:
                    skin = await self._skin_target(job, output, clip, width, height)
                    metadata.skin_target = asdict(skin) if skin is not None else None
                if holding and skin is not None:
                    source_y = await self._source_skin_level(job, clip, width, height)
                    metadata.skin_source_y = source_y
            # The hold runs on the rendered window before its last frame
            # becomes the next seed, so the seam frame and the seed carry
            # the same skin. A held window is already its own prepared part.
            held: Path | None = None
            hold_record: dict[str, object] | None = None
            if holding and skin is not None and source_y is not None:
                held, hold_record = await self._hold_window(
                    job, output, clip, window, HoldTarget.of(skin, source_y), width, height
                )
            parts.append(held if held is not None else output)
            prepared.append(held is not None)
            metadata.windows.append(
                WindowRecord(
                    index=window.index,
                    seconds=window.seconds,
                    frames=window.frames,
                    start_frame=window.start_frame,
                    reference=image_name,
                    wall_seconds=round(wall, 1),
                    seed_correction=pending_correction,
                    skin_correction=pending_skin,
                    skin_hold=hold_record,
                )
            )
            pending_correction = None
            pending_skin = None
            if window.index + 1 < total:
                metadata.seams.append(round((window.start_frame + window.frames - 1) / fps, 4))
                if mode == "previous_frame":
                    frame = await self._final_frame(
                        job, held if held is not None else output, window.index + 1
                    )
                    if anchor is not None:
                        frame, pending_correction = await self._anchor_seed(
                            job, frame, anchor, window.index + 1
                        )
                    if held is not None:
                        pending_skin = {
                            "applied": False,
                            "reason": "seed taken from the held frames",
                        }
                    elif skin is not None and skin_anchoring:
                        # The seed is the last frame of THIS window; the
                        # source frame it corresponds to is the last frame
                        # of this window's cut clip.
                        frame, pending_skin = await self._anchor_skin(
                            job, frame, clip, window, skin, width, height
                        )
                    image_name = await self._upload_still(
                        job, frame, f"reference{window.index + 1:02d}"
                    )

        await reporter.stitching()
        assembled = job.workspace / "assembled.mp4"
        trimmed = [
            part if ready else await self._prepare_part(job, part, window)
            for part, window, ready in zip(parts, windows, prepared, strict=True)
        ]
        try:
            await cancellable(job, concat_segments(trimmed, assembled))
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"window assembly failed: {exc}",
            ) from exc

        await reporter.muxing("Adding your video's sound…")
        output = job.workspace / "output.mp4"
        await self._lay_source_audio(
            job, assembled, staged, info, output, metadata.promised_seconds
        )
        try:
            result_info = await verify_output(
                output,
                OutputExpectation(
                    expect_video=True,
                    expect_audio=True,
                    expected_seconds=metadata.promised_seconds,
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "The finished video failed its check.", internal_detail=str(exc)
            ) from exc
        metadata.measured_seconds = result_info.duration_seconds
        metadata.write(job.workspace / "character-replacement.json")
        logger.info(
            "character_replacement_chain_finished",
            extra={
                "job_id": job.job_id,
                "windows": total,
                "seams": metadata.seams,
                "promised_seconds": round(metadata.promised_seconds, 3),
                "measured_seconds": result_info.duration_seconds,
                "wall_seconds": round(sum(w.wall_seconds for w in metadata.windows), 1),
            },
        )
        return await self._deliver(job, reporter, output, result_info)

    # ── One graph run ────────────────────────────────────────────────────

    async def _render(
        self,
        job: AdapterJob,
        service: LtxComfyService,
        edits: ReplacementEdits,
        output: Path,
        reporter: StageReporter,
        *,
        band: tuple[int, int],
        section: tuple[int, int, float, float] | None,
        expected_seconds: float,
        log_extra: dict[str, object],
    ) -> tuple[MediaInfo, float]:
        """Compiles, submits, waits, collects and verifies one run of the graph."""
        try:
            api = compile_character_replacement(service.load("character_replacement"), edits)
        except (GraphError, ComfyError) as exc:
            raise AdapterError(
                "This tool is temporarily unavailable.",
                internal_detail=f"graph compile failed: {exc}",
                retriable=False,
            ) from exc

        expected_wall = max(
            1.0, edits.seconds * settings.character_replacement_expected_wall_per_output_second
        )
        low, high = band

        async def announce(progress: int, message: str) -> None:
            if section is not None:
                index, total, start, end = section
                await reporter.section(index, total, progress, start_seconds=start, end_seconds=end)
            else:
                await reporter.generating(progress, message)

        await announce(low, "Starting your video…")

        async def tick(elapsed: float) -> None:
            fraction = min(1.0, elapsed / (expected_wall * 1.2))
            await announce(low + int((high - low - 1) * fraction), "Generating your video…")

        started = time.monotonic()
        prompt_id = ""
        try:
            prompt_id = await service.generate(api, client_id=f"zolex-{job.job_id}")
            logger.info(
                "character_replacement_submitted",
                extra={
                    "job_id": job.job_id,
                    "prompt_id": prompt_id,
                    "seconds": edits.seconds,
                    "nodes": len(api),
                    **log_extra,
                },
            )
            remaining = job.seconds_remaining
            timeout = settings.ltx_comfy_generation_timeout
            if remaining is not None:
                timeout = max(1.0, min(timeout, remaining))
            history = await service.progress(job, prompt_id, timeout_seconds=timeout, on_tick=tick)
            await service.collect(history, output)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc
        wall = time.monotonic() - started

        try:
            result_info = await verify_output(
                output,
                OutputExpectation(
                    expect_video=True,
                    expect_audio=True,
                    expected_seconds=expected_seconds,
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "The finished video failed its check.", internal_detail=str(exc)
            ) from exc

        logger.info(
            "character_replacement_finished",
            extra={
                "job_id": job.job_id,
                "prompt_id": prompt_id,
                "wall_seconds": round(wall, 1),
                "duration_seconds": result_info.duration_seconds,
                "width": result_info.width,
                "height": result_info.height,
                **log_extra,
            },
        )
        return result_info, wall

    # ── Readings ─────────────────────────────────────────────────────────

    @staticmethod
    def window_seconds(info: MediaInfo, job: AdapterJob) -> int:
        """Whole seconds of the source ONE run of the graph may follow.

        The source's own length, floored to whole seconds (the graph's
        length input is an integer), capped by the per-window ceiling. A
        longer source is not cut to this any more — it is chained
        (`windows_for`); this is the size of one link.
        """
        seconds = info.duration_seconds or 0.0
        if seconds < _MIN_SOURCE_SECONDS:
            raise AdapterError(
                "Please upload a video at least one second long.",
                internal_detail=f"source is {seconds:.2f}s",
                retriable=False,
            )
        cap = job.execution_int("max_seconds", int(settings.character_replacement_max_seconds))
        return min(int(math.floor(seconds)), max(1, cap))

    @classmethod
    def windows_for(cls, info: MediaInfo, job: AdapterJob) -> list[Window]:
        """The chain a source needs: one window within the ceiling, more beyond it.

        The total is the source's whole seconds, capped by the deployment's
        total ceiling; anything above the cap is not followed, and that is
        logged rather than silent.
        """
        per_window = cls.window_seconds(info, job)
        seconds = info.duration_seconds or 0.0
        total_cap = job.execution_int(
            "max_total_seconds", int(settings.character_replacement_max_total_seconds)
        )
        total = min(int(math.floor(seconds)), max(per_window, total_cap))
        if total < seconds - 1e-6:
            logger.info(
                "character_replacement_window",
                extra={
                    "job_id": job.job_id,
                    "source_seconds": round(seconds, 3),
                    "followed_seconds": total,
                    "window_seconds": per_window,
                    "total_cap_seconds": total_cap,
                },
            )
        chain = cls.chain_window_seconds(job)
        if chain is not None and total > per_window:
            # Only a source that chains today; a single-window source never
            # sees this setting.
            return plan_windows(total, min(chain, per_window))
        return plan_windows(total, per_window)

    @staticmethod
    def is_stylised(job: AdapterJob) -> bool:
        """Whether the new character is drawn rather than photographed.

        `execution.subject_style` decides it outright — "cartoon", "stylised"
        or "animated" for drawn, "human", "real" or "photoreal" for not — and
        with nothing set the prompt is read (`looks_stylised`).

        What it changes is every stage built on a photographed human, and
        those are exactly the stages the client's 8 Sep 2026 Simpsons render
        went through: the negative prompt asks the model to avoid "cartoon,
        illustration, anime" while the positive asks for one, and the skin
        machinery keys on the human skin chroma range, which a bright drawn
        face sits close enough to for the gate to fire and pull it toward a
        human skin level. A dark grey face is what that looks like.

        (Their report also blamed ReActor and a per-frame SAM mask. Those run
        in the client's own MULTI4 graph and have never run here — this
        deployment's graph carries no face-swap branch at all, which is also
        why our render is not the eleven-minute one they measured.)
        """
        raw = str(job.execution.get("subject_style") or "").strip().lower()
        if raw in ("cartoon", "stylised", "stylized", "animated", "drawn", "nonhuman"):
            return True
        if raw in ("human", "real", "photoreal", "person"):
            return False
        return looks_stylised(job.prompt)

    @staticmethod
    def exposure_clause(job: AdapterJob) -> str | None:
        """The client's lighting lock, or None when this deployment is without it.

        On by default because the client asked for it directly, and unlike
        every other prompt lever here it is switchable per job so an A/B needs
        no redeploy. It applies to single-window jobs as well as chained ones:
        the darkening was measured WITHIN a window, so a clause that ran only
        across seams would leave the short jobs carrying the fault.
        """
        raw = job.execution.get("exposure_clause")
        if raw is None:
            raw = settings.character_replacement_exposure_clause
        if str(raw).strip().lower() in ("false", "no", "off", "0"):
            return None
        if CharacterReplacementAdapter.is_stylised(job):
            return CHARACTER_REPLACEMENT_EXPOSURE_STYLISED
        return CHARACTER_REPLACEMENT_EXPOSURE

    @staticmethod
    def chain_skin_clause(job: AdapterJob) -> bool:
        if CharacterReplacementAdapter.is_stylised(job):
            # Every one of these reads or writes HUMAN skin. See `is_stylised`.
            return False
        raw = job.execution.get("chain_skin_clause")
        if raw is None:
            return bool(settings.character_replacement_chain_skin_clause)
        return str(raw).strip().lower() not in ("false", "no", "off", "0")

    @staticmethod
    def chain_window_seconds(job: AdapterJob) -> int | None:
        raw = job.execution.get("chain_window_seconds")
        if raw is None:
            raw = settings.character_replacement_chain_window_seconds
        try:
            value = int(str(raw).strip()) if raw is not None and str(raw).strip() else None
        except ValueError:
            return None
        return value if value is not None and value >= 1 else None

    @staticmethod
    def holds_skin(job: AdapterJob) -> bool:
        if CharacterReplacementAdapter.is_stylised(job):
            # Every one of these reads or writes HUMAN skin. See `is_stylised`.
            return False
        raw = job.execution.get("skin_hold")
        if raw is None:
            return bool(settings.character_replacement_skin_hold)
        return str(raw).strip().lower() not in ("false", "no", "off", "0")

    @staticmethod
    def ripple_strength(job: AdapterJob) -> float | None:
        raw = job.execution.get("ripple_strength")
        if raw is None:
            raw = settings.character_replacement_ripple_strength
        if raw is None or not str(raw).strip():
            return None
        try:
            value = float(str(raw).strip())
        except ValueError:
            return None
        return value if value > 0 else None

    @staticmethod
    def anchors_skin(job: AdapterJob) -> bool:
        if CharacterReplacementAdapter.is_stylised(job):
            # Every one of these reads or writes HUMAN skin. See `is_stylised`.
            return False
        raw = job.execution.get("skin_anchor")
        if raw is None:
            return bool(settings.character_replacement_skin_anchor)
        return str(raw).strip().lower() not in ("false", "no", "off", "0")

    @staticmethod
    def anchors_reference(job: AdapterJob) -> bool:
        raw = job.execution.get("anchor_reference")
        if raw is None:
            return bool(settings.character_replacement_anchor_reference)
        return str(raw).strip().lower() not in ("false", "no", "off", "0")

    @staticmethod
    def reference_mode(job: AdapterJob) -> str:
        raw = (
            str(
                job.execution.get("chain_reference")
                or settings.character_replacement_chain_reference
            )
            .strip()
            .lower()
        )
        return raw if raw in ("previous_frame", "photo") else "previous_frame"

    @staticmethod
    def seed_base(job: AdapterJob) -> int | None:
        """A customer seed wins; otherwise None — the graph's own fixed seeds.

        The ZIP's character graph carries fixed seeds and the client asked
        for the tool to behave exactly like the ZIP (6 Sep 2026). So a job
        without a seed runs those seeds, and "Regenerate" with the same
        inputs returns the same result — the seed control is the way to a
        different take.
        """
        raw = job.parameters.get("seed")
        try:
            if raw is not None and str(raw).strip() != "":
                return abs(int(raw)) % (2**48)
        except (TypeError, ValueError):
            pass
        return None

    async def _probe(self, path: Path) -> MediaInfo:
        try:
            info = await probe_media(path)
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"probe failed: {exc}",
                retriable=False,
            ) from exc
        if not info.has_video or not info.duration_seconds:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"source probe: {info!r}",
                retriable=False,
            )
        return info

    # ── Files in and out ─────────────────────────────────────────────────

    async def _upload_source(self, job: AdapterJob, source: AdapterInput, info: MediaInfo) -> str:
        """The clip as uploaded — plus a silent track when it has none.

        The graph passes the source's audio through to the output; a source
        with no audio stream would hand the combiner nothing. Silence is the
        honest stand-in and costs a stream copy.
        """
        staged = source.require_path()
        name = f"zolex_{job.job_id}_source.mp4"
        local = job.workspace / name
        try:
            if info.has_audio:
                await ffmpeg(
                    ["-i", str(staged), "-c", "copy", "-movflags", "+faststart", str(local), "-y"]
                )
            else:
                await ffmpeg(
                    [
                        "-i",
                        str(staged),
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=48000:cl=stereo",
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-shortest",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        str(local),
                        "-y",
                    ]
                )
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"source repack failed: {exc}",
                retriable=False,
            ) from exc
        return await self._upload_clip(local)

    async def _cut_window(
        self, job: AdapterJob, staged: Path, info: MediaInfo, window: Window
    ) -> Path:
        """One window of the source as its own clip, on the graph's grid.

        Exactly `window.frames` frames at 24 fps from `start_frame` — the
        rate the graph's loader would resample to anyway — with the
        source's sound for that stretch (silence when it has none). A
        source that runs out inside the last window has its final frame
        held for the few frames the graph's formula overshoots by.
        """
        fps = settings.ltx_comfy_frame_rate
        start = window.start_frame / fps
        length = window.frames / fps
        dest = job.workspace / f"zolex_{job.job_id}_window{window.index:02d}.mp4"
        args = ["-ss", f"{start:.6f}", "-i", str(staged)]
        if not info.has_audio:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        args += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0" if info.has_audio else "1:a:0",
            "-t",
            f"{length:.6f}",
            "-filter:v",
            f"fps={fps},tpad=stop_mode=clone:stop_duration={length + 1.0:.3f},format=yuv420p",
            "-frames:v",
            str(window.frames),
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(dest),
            "-y",
        ]
        try:
            await cancellable(job, ffmpeg(args))
        except FfmpegError as exc:
            raise AdapterError(
                "That video could not be read. Please try another.",
                internal_detail=f"window {window.index} could not be cut: {exc}",
                retriable=False,
            ) from exc
        return dest

    async def _upload_clip(self, local: Path) -> str:
        try:
            return await self.service().upload(local, name=local.name)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc

    async def _upload_still(self, job: AdapterJob, staged: Path, tag: str) -> str:
        name = f"zolex_{job.job_id}_{tag}.png"
        local = job.workspace / name
        try:
            await ffmpeg(["-i", str(staged), "-frames:v", "1", str(local), "-y"])
        except FfmpegError as exc:
            raise AdapterError(
                "That picture could not be read. Please try another.",
                internal_detail=f"reference normalise failed for {tag}: {exc}",
                retriable=False,
            ) from exc
        try:
            return await self.service().upload(local, name=name)
        except ComfyError as exc:
            raise AdapterError(
                exc.user_message, internal_detail=exc.internal_detail, retriable=exc.retriable
            ) from exc

    # ── Colour anchoring of chained seeds ────────────────────────────────

    async def _measure_colour(
        self, job: AdapterJob, path: Path, frames: tuple[int, int] | None
    ) -> ColourAnchor:
        """`signalstats` over `frames` (inclusive, None = the whole file), averaged."""
        select = f"select='between(n,{frames[0]},{frames[1]})'," if frames else ""
        args = [
            "-i",
            str(path),
            "-vf",
            f"{select}signalstats,metadata=print:file=-",
            "-f",
            "null",
            "-",
        ]
        try:
            report = (await ffmpeg_stdout(args)).decode("utf-8", "replace")
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"colour measurement failed on {path.name}: {exc}",
            ) from exc
        sums: dict[str, list[float]] = {"YAVG": [], "YLOW": [], "YHIGH": [], "UAVG": [], "VAVG": []}
        for line in report.splitlines():
            for key in sums:
                marker = f"lavfi.signalstats.{key}="
                if marker in line:
                    try:
                        sums[key].append(float(line.split(marker, 1)[1].strip()))
                    except ValueError:
                        pass
        if not sums["YAVG"]:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"no colour statistics came back for {path.name}",
            )
        mean = {key: sum(values) / len(values) for key, values in sums.items() if values}
        return ColourAnchor(
            y_mean=mean["YAVG"],
            y_low=mean.get("YLOW", mean["YAVG"]),
            y_high=mean.get("YHIGH", mean["YAVG"]),
            u_mean=mean.get("UAVG", 128.0),
            v_mean=mean.get("VAVG", 128.0),
        )

    async def _anchor_seed(
        self, job: AdapterJob, frame: Path, anchor: ColourAnchor, index: int
    ) -> tuple[Path, dict[str, float]]:
        """Brings a seed frame's look back to the anchor before it is reused.

        Luminance: a gain that restores the anchor's spread (10th-90th
        percentile) and an offset that restores its level; chroma: offsets
        to the anchor's means. All bounded (`ANCHOR_*`). A seed already at
        the anchor passes through untouched.
        """
        seed = await self._measure_colour(job, frame, None)
        gain = 1.0
        if seed.spread >= ANCHOR_MIN_SPREAD and anchor.spread >= ANCHOR_MIN_SPREAD:
            gain = anchor.spread / seed.spread
        gain = min(ANCHOR_GAIN_RANGE[1], max(ANCHOR_GAIN_RANGE[0], gain))
        y_offset = anchor.y_mean - gain * seed.y_mean
        y_offset = min(ANCHOR_LUMA_OFFSET_LIMIT, max(-ANCHOR_LUMA_OFFSET_LIMIT, y_offset))
        u_offset = min(
            ANCHOR_CHROMA_OFFSET_LIMIT,
            max(-ANCHOR_CHROMA_OFFSET_LIMIT, anchor.u_mean - seed.u_mean),
        )
        v_offset = min(
            ANCHOR_CHROMA_OFFSET_LIMIT,
            max(-ANCHOR_CHROMA_OFFSET_LIMIT, anchor.v_mean - seed.v_mean),
        )
        correction = {
            "gain": round(gain, 4),
            "y_offset": round(y_offset, 2),
            "u_offset": round(u_offset, 2),
            "v_offset": round(v_offset, 2),
            "seed_y_mean": round(seed.y_mean, 2),
            "seed_y_high": round(seed.y_high, 2),
            "anchor_y_mean": round(anchor.y_mean, 2),
            "anchor_y_high": round(anchor.y_high, 2),
        }
        negligible = (
            abs(gain - 1.0) < 0.01
            and abs(y_offset) < 1.0
            and abs(u_offset) < 1.0
            and abs(v_offset) < 1.0
        )
        logger.info(
            "character_replacement_anchor",
            extra={"job_id": job.job_id, "window": index, "applied": not negligible, **correction},
        )
        if negligible:
            return frame, correction
        dest = frame.with_name(f"{frame.stem}-anchored.png")
        lut = (
            f"lutyuv=y='clip(val*{gain:.5f}+{y_offset:.3f},0,255)'"
            f":u='clip(val+{u_offset:.3f},0,255)'"
            f":v='clip(val+{v_offset:.3f},0,255)'"
        )
        try:
            await cancellable(
                job,
                ffmpeg(
                    [
                        "-i",
                        str(frame),
                        "-vf",
                        f"format=yuv444p,{lut},format=rgb24",
                        "-frames:v",
                        "1",
                        str(dest),
                        "-y",
                    ]
                ),
            )
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"seed anchoring failed for window {index}: {exc}",
            ) from exc
        return dest, correction

    # ── Skin-region re-anchoring (worker.media.skin) ─────────────────────

    async def _skin_target(
        self, job: AdapterJob, rendered: Path, clip: Path, width: int, height: int
    ) -> SkinStats | None:
        try:
            target = await cancellable(
                job,
                skin_target(rendered, clip, frames=ANCHOR_FRAMES, width=width, height=height),
            )
        except FfmpegError as exc:
            logger.warning(
                "character_replacement_skin_target_failed",
                extra={"job_id": job.job_id, "detail": str(exc)[:300]},
            )
            return None
        logger.info(
            "character_replacement_skin_target",
            extra={"job_id": job.job_id, **asdict(target)},
        )
        if target.cover <= 1e-4:
            return None
        return target

    async def _anchor_skin(
        self,
        job: AdapterJob,
        frame: Path,
        clip: Path,
        window: Window,
        target: SkinStats,
        width: int,
        height: int,
    ) -> tuple[Path, dict[str, object]]:
        """The skin-region lift on a seed, gated by the source's silhouette at
        the seam. A failure inside ffmpeg is logged and the seed passes
        through: this is containment, never a reason to fail the job."""
        index = window.index + 1
        try:
            gate = await cancellable(
                job,
                source_gate(
                    clip,
                    window.frames - 1,
                    job.workspace / f"reference{index:02d}-gate.png",
                    width=width,
                    height=height,
                ),
            )
            corrected, record = await cancellable(
                job,
                anchor_skin(
                    frame,
                    gate,
                    frame.with_name(f"reference{index:02d}-skin.png"),
                    target=target,
                    width=width,
                    work_dir=job.workspace,
                    tag=f"reference{index:02d}",
                ),
            )
        except FfmpegError as exc:
            logger.warning(
                "character_replacement_skin_failed",
                extra={"job_id": job.job_id, "window": index, "detail": str(exc)[:300]},
            )
            return frame, {"applied": False, "reason": f"ffmpeg: {str(exc)[:120]}"}
        logger.info(
            "character_replacement_skin",
            extra={"job_id": job.job_id, "window": index, **asdict(record)},
        )
        return corrected, asdict(record)

    async def _final_frame(self, job: AdapterJob, part: Path, index: int) -> Path:
        try:
            return await cancellable(
                job, extract_final_frame(part, job.workspace / f"reference{index:02d}.png")
            )
        except FfmpegError as exc:
            # A GENERATED window being unreadable is a generation flake, not
            # a bad upload — a retry can genuinely produce a readable one.
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"window {index - 1} produced an unreadable file: {exc}",
            ) from exc

    async def _source_skin_level(
        self, job: AdapterJob, clip: Path, width: int, height: int
    ) -> float | None:
        try:
            level = await cancellable(
                job,
                source_skin_level(clip, frames=ANCHOR_FRAMES, width=width, height=height),
            )
        except (FfmpegError, TimeoutError) as exc:
            logger.warning(
                "character_replacement_skin_source_failed",
                extra={"job_id": job.job_id, "detail": str(exc)[:300]},
            )
            return None
        logger.info(
            "character_replacement_skin_source",
            extra={"job_id": job.job_id, "y_mean": round(level, 2)},
        )
        return level

    async def _prepare_part(self, job: AdapterJob, part: Path, window: Window) -> Path:
        """Drops the seam frame, keeps exactly the frames this window contributes.

        Every part goes through the same encoder settings so the join is a
        stream copy; sound is left out here because the source's own track
        is laid over the whole result afterwards.
        """
        dest = job.workspace / f"part-{window.index:04d}.mp4"
        skip = 0 if window.index == 0 else SEAM_OVERLAP_FRAMES
        args = [
            "-i",
            str(part),
            "-an",
            "-vf",
            f"trim=start_frame={skip},setpts=PTS-STARTPTS",
            "-frames:v",
            str(window.kept_frames),
            "-fps_mode",
            "cfr",
            *PART_ENCODE,
            str(dest),
            "-y",
        ]
        try:
            await cancellable(job, ffmpeg(args))
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"window {window.index} could not be prepared: {exc}",
            ) from exc
        return dest

    async def _hold_window(
        self,
        job: AdapterJob,
        output: Path,
        clip: Path,
        window: Window,
        target: HoldTarget,
        width: int,
        height: int,
    ) -> tuple[Path | None, dict[str, object]]:
        """The skin hold on one rendered window (`worker.media.skin_hold`).

        Returns the window's prepared part (seam frame dropped, held frames)
        and the record; (None, record) when nothing was lifted or the hold
        could not run — the window is then prepared as always. Containment,
        never a reason to fail the job.
        """
        started = time.monotonic()
        dest = job.workspace / f"part-{window.index:04d}.mp4"
        skip = 0 if window.index == 0 else SEAM_OVERLAP_FRAMES
        # Measured on the node: 8 s of measuring and 16 s of applying per
        # 193-frame window. The budget is generous against that, and a pass
        # that overruns it is contained like any other failure rather than
        # taking the finished renders down with it.
        budget = max(600.0, HOLD_SECONDS_PER_FRAME * window.frames)
        try:
            readings, planes = await cancellable(
                job,
                measure_window(
                    output,
                    clip,
                    target,
                    width=width,
                    height=height,
                    work_dir=job.workspace,
                    tag=f"window{window.index:02d}",
                    timeout=budget,
                ),
            )
            measured = time.monotonic() - started
            plan = plan_hold(readings, target, first_window=window.index == 0)
            record: dict[str, object] = {**plan.summary(), "measured_frames": len(readings)}
            applied = False
            if len(readings) != window.frames:
                record.update(
                    applied=False,
                    reason=f"measured {len(readings)} frames, expected {window.frames}",
                )
            elif not plan.active:
                record.update(applied=False, reason="nothing to hold")
            else:
                await cancellable(
                    job,
                    ffmpeg(
                        hold_apply_args(
                            output,
                            planes,
                            plan,
                            dest,
                            width=width,
                            height=height,
                            skip=skip,
                            kept_frames=window.kept_frames,
                            encode=PART_ENCODE,
                        ),
                        timeout=budget,
                    ),
                )
                record["reason"] = "applied"
                applied = True
        except (FfmpegError, TimeoutError) as exc:
            logger.warning(
                "character_replacement_skin_hold_failed",
                extra={"job_id": job.job_id, "window": window.index, "detail": str(exc)[:300]},
            )
            return None, {"applied": False, "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}
        record["measure_seconds"] = round(measured, 1)
        record["wall_seconds"] = round(time.monotonic() - started, 1)
        record["target_y"] = round(target.y, 2)
        record["source_y_ref"] = round(target.source_y, 2)
        logger.info(
            "character_replacement_skin_hold",
            extra={"job_id": job.job_id, "window": window.index, **record},
        )
        return (dest if applied else None), record

    async def _lay_source_audio(
        self,
        job: AdapterJob,
        picture: Path,
        source: Path,
        info: MediaInfo,
        dest: Path,
        seconds: float,
    ) -> Path:
        """The source's own soundtrack over the assembled picture.

        The picture decides the length; sound that runs short (a held tail
        the source did not cover) is padded with silence, a source with no
        sound gets a silent track, so every result carries one audio stream
        as the single-window path's does.
        """
        args = ["-i", str(picture)]
        if info.has_audio:
            args += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-af", "apad"]
        else:
            args += [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        args += [
            "-t",
            f"{seconds:.6f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(dest),
            "-y",
        ]
        try:
            await cancellable(job, ffmpeg(args))
        except FfmpegError as exc:
            raise AdapterError(
                "This generation could not be completed. Please try again.",
                internal_detail=f"soundtrack could not be laid over the result: {exc}",
            ) from exc
        return dest
