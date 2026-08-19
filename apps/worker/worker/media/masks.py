"""Person masks, and the hybrid control signals built from them.

`control.py` turns footage into an edge map, which is what makes a restyle a
restyle: edges carry the subject's outline and the camera's geometry, and none
of their colour, skin or material. That is exactly right when the whole frame
should be re-imagined — and exactly wrong when the customer expects the PERSON
to survive.

Measured 2026-08-18, and it is not a subtle effect: transform a clip of a
falconer into a blizzard and the man who comes back has a different face and a
different skin tone. He is a plausible man matching the prompt, because an
outline is all the model was ever given of him. Nothing in the prompt asked for
a new person; the pipeline simply had no way to know which one to keep.

**The fix is regional, not global.** A matte says where the person is; the
control signal then carries their real pixels INSIDE it while staying an edge
map outside, and the pipeline's native attention mask tells the model to follow
that region harder than the rest. The model reproduces the person from their
own appearance, relights them for the new scene, and re-imagines everything
around them.

Verified on the RTX PRO 6000, an 8-second window at 1024x576: face, skin tone,
hair, beard and the bird's real plumage all survived while the hillside became
a dusk blizzard — at **61 seconds** against the edge-only path's ~54. Person
lock costs essentially nothing.

The same three pieces invert into person REPLACEMENT (the scene's pixels kept
outside the matte, the region inside it free to become someone else), which is
why `build_hybrid_control` takes a side rather than assuming one. Replacement
additionally requires recorded consent — a product gate, not this module's job.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg

logger = get_logger(__name__)

#: How far the matte is grown before feathering, in dilation passes.
#:
#: A mask that hugs the subject tightly is the wrong error to make. A halo of
#: real pixels a few pixels wide around a coat or a shoulder is invisible in
#: the result; a sliver of MISSING shoulder is a hard seam between a real
#: person and a generated one, and it lands exactly where the eye is already
#: looking. Grow first, then blur.
DILATION_PASSES = 2

#: Feather radius in pixels, applied after dilation. Wide enough that the
#: boundary is a gradient rather than a cut, narrow enough that the person's
#: own edge stays theirs.
FEATHER_RADIUS = 12

#: What the attention mask says about everything that is not the subject.
#:
#: The pipeline reads this clip as a per-region conditioning weight: white
#: follows the control signal fully, black ignores it. The subject is white —
#: their real pixels are the entire point — and the background sits at mid-grey
#: so the edge map still steers the scene's geometry while the prompt owns its
#: look. Dropping the background toward black restyles harder and starts losing
#: the camera; raising it toward white re-imposes the source everywhere, which
#: is the behaviour this module exists to avoid.
BACKGROUND_ATTENTION = 0.5


async def build_person_matte(
    source: Path,
    dest: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    width: int,
    height: int,
    fps: float,
    frames: int,
    timeout: float = 1800.0,
) -> Path:
    """A soft, temporally smoothed matte of the people in one window.

    Runs OUTSIDE this process. Matting is model work — this worker has no torch
    and should not grow one to get a mask — so it is invoked in the same GPU
    environment that runs the pipelines, exactly the way every other model this
    adapter uses is invoked.

    The alignment contract from `control.py` applies unchanged and for the same
    reason: a matte of a different length, grid or frame rate than the pass is
    not a weaker signal, it is a misaligned one, and it would protect the wrong
    part of the picture on every frame of the render.
    """
    if frames < 1:
        raise ValueError(f"a matte needs at least one frame, got {frames}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *settings.person_matte_argv,
        "--source", str(source),
        "--dest", str(dest),
        "--start-seconds", f"{max(0.0, start_seconds):.3f}",
        "--duration-seconds", f"{max(0.0, duration_seconds):.3f}",
        "--width", str(width),
        "--height", str(height),
        "--fps", f"{fps:g}",
        "--frames", str(frames),
        "--dilation-passes", str(DILATION_PASSES),
        "--feather-radius", str(FEATHER_RADIUS),
    ]

    logger.info(
        "person_matte_started",
        extra={
            "frames": frames,
            "grid": [width, height],
            "start_seconds": round(start_seconds, 3),
        },
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(settings.ltx_repo_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        dest.unlink(missing_ok=True)
        raise FfmpegError(f"person matting timed out after {timeout:.0f}s") from None

    if process.returncode != 0:
        dest.unlink(missing_ok=True)
        tail = (stdout or b"").decode("utf-8", "replace").strip().splitlines()[-12:]
        raise FfmpegError(
            f"person matting exited {process.returncode}: {' | '.join(tail)}"
        )
    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FfmpegError(f"person matting produced no matte for {source.name}")
    return dest


async def extract_source_window(
    source: Path,
    dest: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    width: int,
    height: int,
    fps: float,
    frames: int,
    timeout: float = 900.0,
) -> Path:
    """The source's own pixels for one window, on the render's exact grid.

    Deliberately the same scale/crop/fps/pad/`-frames:v` recipe `control.py`
    uses, because this clip and the edge clip are merged frame for frame: any
    difference in how the two are built becomes a misregistration between a
    person's real pixels and the outline they are meant to fill.
    """
    if frames < 1:
        raise ValueError(f"a source window needs at least one frame, got {frames}")

    pad_seconds = frames / max(fps, 1e-6)
    filters = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={fps:g},"
        f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f},"
        "format=yuv420p"
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        await ffmpeg(
            [
                "-ss", f"{max(0.0, start_seconds):.3f}",
                "-t", f"{max(0.0, duration_seconds):.3f}",
                "-i", str(source),
                "-filter:v", filters,
                "-frames:v", str(frames),
                "-fps_mode", "cfr",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                str(dest),
            ],
            timeout=timeout,
        )
    except FfmpegError:
        dest.unlink(missing_ok=True)
        raise

    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FfmpegError(
            f"no source window could be built from {source.name} at "
            f"{start_seconds:.2f}s"
        )
    return dest


async def build_hybrid_control(
    edges: Path,
    footage: Path,
    matte: Path,
    dest: Path,
    *,
    frames: int,
    invert: bool = False,
    timeout: float = 900.0,
) -> Path:
    """Edges everywhere, real pixels where the matte says so.

    `invert=False` keeps the PERSON's own pixels — person lock. `invert=True`
    keeps everything except them, which is the signal a replacement needs: the
    scene stays itself while the masked region is free to become someone new.
    One function, because the two differ by exactly which side is protected and
    a second implementation would be a second thing to keep correct.

    All three inputs must already share the pass's grid, frame rate and frame
    count. `maskedmerge` combines them frame for frame and cannot correct a
    misalignment it cannot see.
    """
    if frames < 1:
        raise ValueError(f"a control clip needs at least one frame, got {frames}")

    # `alphamerge` + `overlay` rather than `maskedmerge`: the latter matches
    # planes, so on yuv420p a black-and-white matte leaves chroma at neutral
    # and blends colour at half strength across the whole frame — the person's
    # real skin tone would arrive desaturated, which is most of the bug this
    # exists to fix. Turning the matte into an alpha channel and compositing
    # keeps colour intact and makes the feathered edge a true gradient.
    matte_chain = "[2:v]format=gray,negate[m]" if invert else "[2:v]format=gray[m]"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        await ffmpeg(
            [
                "-i", str(edges),
                "-i", str(footage),
                "-i", str(matte),
                "-filter_complex",
                "[0:v]format=yuv420p[base];"
                "[1:v]format=yuva420p[real];"
                + matte_chain
                + ";[real][m]alphamerge[fg];"
                "[base][fg]overlay=format=yuv420,format=yuv420p",
                "-frames:v", str(frames),
                "-fps_mode", "cfr",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                str(dest),
            ],
            timeout=timeout,
        )
    except FfmpegError:
        dest.unlink(missing_ok=True)
        raise

    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FfmpegError("no hybrid control signal could be built")
    return dest


async def build_attention_mask(
    matte: Path,
    dest: Path,
    *,
    frames: int,
    background: float = BACKGROUND_ATTENTION,
    subject: float = 1.0,
    timeout: float = 900.0,
) -> Path:
    """The matte rescaled into the pipeline's per-region conditioning weights.

    With the defaults — person lock — the subject stays at full strength and
    everything else is lifted to `background`, so the model follows the
    person's real pixels hard while the scene keeps enough of the edge map to
    hold its geometry.

    The two weights are independent because identity REPLACEMENT needs them
    the other way round: `background=1.0, subject=<lowered>` keeps the edge
    map's full grip on the scene and the camera while loosening it exactly
    where the person is, so their pose still tracks the footage but their
    facial geometry stops being re-imposed — which is what lets a reference
    image supply the identity instead.
    """
    if frames < 1:
        raise ValueError(f"an attention mask needs at least one frame, got {frames}")

    floor = round(min(1.0, max(0.0, background)) * 255)
    peak = round(min(1.0, max(0.0, subject)) * 255)
    gain = (peak - floor) / 255

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        await ffmpeg(
            [
                "-i", str(matte),
                "-filter:v",
                f"format=gray,lutyuv=y={floor}+val*({gain:.6f}),format=yuv420p",
                "-frames:v", str(frames),
                "-fps_mode", "cfr",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                str(dest),
            ],
            timeout=timeout,
        )
    except FfmpegError:
        dest.unlink(missing_ok=True)
        raise

    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FfmpegError("no attention mask could be built")
    return dest
