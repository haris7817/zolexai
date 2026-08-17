"""Control signals: turning footage into something a structure-conditioned
model will actually follow.

IC-LoRA Union Control does not consume ordinary RGB as its reference. It
expects an **aligned control video** — an edge map, a depth pass or a pose
skeleton — covering exactly the window, grid and frame count the render is
about to produce. Feed it the raw source and it behaves like a weak style hint;
feed it edges and the generated shot keeps the source's geometry while the
prompt supplies every pixel of the look.

Edges are the signal implemented here, deliberately:

  * they need no model, no weights and no additional licence — ffmpeg's
    `edgedetect` is the whole dependency, and it is already a hard requirement
    of this worker;
  * they carry exactly the information a restyle must preserve (subject
    outline, placement, camera geometry, scene layout) and none of what it must
    discard (colour, lighting, material, time of day), which is why a daylight
    desert plate can come back as a rain-soaked neon street with the subject's
    pose untouched;
  * depth and pose are strictly better for some jobs and both need a model
    (DepthCrafter, DWPose). They are a later signal behind the same seam, not a
    reason to delay the one that works today.

**Alignment is the contract.** The control clip is produced at the render's own
grid, frame rate and frame count, from the render's own window of the source.
A control video that is a different length than the pass is not a weaker
signal — it is a misaligned one, and it desynchronises the output from the
footage it is supposed to be tracking.
"""

from __future__ import annotations

from pathlib import Path

from worker.media.ffmpeg import FfmpegError, ffmpeg

#: Canny hysteresis thresholds, in ffmpeg's 0..1 scale.
#:
#: Low enough to keep a subject's silhouette and the scene's structural lines
#: through soft focus and compression noise, high enough that film grain and
#: sky gradients do not become a field of speckle the model dutifully renders
#: as texture. Overridable per workflow because the right values depend on the
#: footage, which is a judgement to be made against real uploads.
DEFAULT_EDGE_LOW = 0.1
DEFAULT_EDGE_HIGH = 0.4


async def extract_edge_control(
    source: Path,
    dest: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    width: int,
    height: int,
    fps: float,
    frames: int,
    low: float = DEFAULT_EDGE_LOW,
    high: float = DEFAULT_EDGE_HIGH,
    timeout: float = 900.0,
) -> Path:
    """Writes an edge-map clip of one window of `source`, exactly `frames` long.

    The filter order matters and is not interchangeable: scale and crop to the
    render's grid FIRST, resample to its frame rate SECOND, and detect edges
    LAST. Detecting before scaling would find edges at the source's resolution
    and then resample them into grey mush; resampling after detection would
    blend adjacent edge maps into frames that never existed.

    `-frames:v` pins the count rather than trusting `-t` to land on it. A window
    whose duration lands a rounding error short of a frame boundary otherwise
    yields one frame fewer than the pass will render, and the model is handed a
    control track that runs out before the shot does.
    """
    if frames < 1:
        raise ValueError(f"a control clip needs at least one frame, got {frames}")

    # A window at the very end of the source can run out of material early, and
    # a source shorter than its own declared duration can run out at any point.
    # Cloning the last picture keeps the control track exactly as long as the
    # pass; `-frames:v` below cuts the padding back off, so this costs nothing
    # when the material was there. Padding by the FULL window length rather than
    # a fixed second is what makes that true for a 20-second pass as well as a
    # 2-second one — the earlier fixed pad silently returned short clips.
    pad_seconds = frames / max(fps, 1e-6)

    filters = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={fps:g},"
        f"edgedetect=low={low:g}:high={high:g},"
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
                # Constant rate: the count above is only a frame count if every
                # frame is one tick apart. A variable-rate source otherwise
                # produces the right NUMBER of frames spread over the wrong
                # amount of time.
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
            f"no control signal could be built from {source.name} at "
            f"{start_seconds:.2f}s"
        )
    return dest
