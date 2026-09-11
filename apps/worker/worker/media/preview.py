"""A 1080p web preview beside the master, so a page never loads 8K.

Client instruction, 11 Sep 2026, after measuring a delivered file that took
too long to start playing:

    Do NOT play the actual 8K master on your webpage. [...] The user sees the
    1080p version instantly, but when they press Download 8K, they receive
    the real 8K file.

They called this the most important change of the set, and the arithmetic
agrees. An 8K master is 300-400 MB; a browser asked to play it downloads a
large part of that before the first frame, and on a page the customer is
almost never watching it fullscreen on an 8K panel. The preview is ~8 MB and
visually identical inside a player.

## What makes playback start immediately

Two things, and only the first is about size.

**`+faststart`.** An MP4 written in one pass puts its `moov` atom — the index
the player needs before it can decode anything — at the END of the file. The
client measured exactly that on a delivered file. `-movflags +faststart` runs
a second pass that moves it to the front, so a player can start on the first
bytes it receives. It costs no re-encode. Every file this worker delivers now
carries it, master and preview alike.

**Size and frame rate.** 1080p at ~5 Mbps, H.264 (which every browser
decodes in hardware) at 24 fps.

## Why the frame rate is pinned

The client measured a delivered file reporting 120 fps timing and ~60 fps
average. Nothing in this pipeline generates above 24 — the graph writes 24
and `LTX_COMFY_FRAME_RATE` is 24 — so that file's timing came from somewhere
after us. Pinning `-r` on both outputs makes the container say what the
pictures actually are, and costs nothing when the input is already 24.
"""

from __future__ import annotations

from pathlib import Path

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg
from worker.media.upscale import headroom

logger = get_logger(__name__)

#: The preview's long edge. Portrait keeps its shape: the scale expression
#: below fixes the SHORT side at 1080 and lets the long side follow, so a
#: 4320x7680 master previews at 1080x1920 rather than being squeezed.
PREVIEW_SHORT_SIDE = 1080


def preview_dimensions(width: int, height: int) -> tuple[int, int]:
    """The preview frame for a master of this shape — 1080 on the short side.

    Both sides even, because yuv420p subsamples chroma 2x2 and libx264 refuses
    an odd size outright.
    """
    if width <= 0 or height <= 0:
        return (1920, 1080)
    scale = PREVIEW_SHORT_SIDE / min(width, height)
    if scale >= 1.0:
        # Already at or below 1080 on the short side: a preview would be an
        # upscale, which is pure waste. The caller skips it.
        return (width, height)

    def even(value: float) -> int:
        return max(2, int(round(value * scale)) // 2 * 2)

    return even(width), even(height)


def is_worth_previewing(width: int, height: int) -> bool:
    """Whether a master is big enough that a preview earns its encode.

    A 1080p delivery is already the preview. Making a second copy of it would
    cost an encode and a second upload to hand the player the same file.
    """
    return min(width, height) > PREVIEW_SHORT_SIDE


async def write_preview(
    master: Path,
    out: Path,
    *,
    width: int,
    height: int,
    fps: int,
    nvenc_timeout: float,
    cpu_timeout: float,
    run: object = None,
    log_extra: dict | None = None,
) -> Path | None:
    """A 1080p H.264 fast-start copy of `master`, or None if not worth one.

    Returns None rather than raising on failure. A preview is an optimisation
    of how the page feels; the master is the product, and a job that rendered
    for minutes must not fail because its convenience copy did not encode.
    """
    if not is_worth_previewing(width, height):
        return None
    target_w, target_h = preview_dimensions(width, height)
    extra = {**(log_extra or {}), "preview": f"{target_w}x{target_h}"}
    rate = settings.ltx_hd_preview_bitrate or "5M"
    ceiling, buffer = headroom(rate)
    common = [
        "-i", str(master),
        "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        # H.264 even though the master is HEVC: every browser decodes H.264 in
        # hardware, and this file exists to start playing instantly.
        "-profile:v", "high",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
    ]

    async def _run(awaitable):
        return await (run(awaitable) if run is not None else awaitable)

    try:
        await _run(
            ffmpeg(
                [*common, "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                 "-rc", "vbr", "-cq", "21", "-b:v", rate,
                 "-maxrate", ceiling, "-bufsize", buffer, str(out)],
                timeout=nvenc_timeout,
            )
        )
    except FfmpegError as exc:
        logger.warning("preview_nvenc_unavailable", extra={**extra, "detail": str(exc)[-300:]})
        try:
            await _run(
                ffmpeg(
                    [*common, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(out)],
                    timeout=cpu_timeout,
                )
            )
        except FfmpegError as second:
            logger.warning("preview_failed", extra={**extra, "detail": str(second)[-300:]})
            return None
    logger.info("preview_written", extra={**extra, "bytes": out.stat().st_size})
    return out


async def ensure_faststart(clip: Path, *, timeout: float) -> Path:
    """`clip` with its `moov` atom at the front, without re-encoding it.

    The client measured a delivered file whose index sat at the END, which is
    what makes a browser download most of a video before it can show any of
    it. Most files through this worker already come from an encode that wrote
    `+faststart`; this is for the path that does not encode at all — a 1080p
    delivery with stabilisation off, which is handed over exactly as ComfyUI
    saved it.

    Stream copy only, so it costs a file rewrite and nothing else. On failure
    the original is returned: a video that starts slowly beats no video.
    """
    out = clip.with_name(f"{clip.stem}-fs.mp4")
    try:
        await ffmpeg(
            ["-i", str(clip), "-c", "copy", "-movflags", "+faststart", str(out)],
            timeout=timeout,
        )
    except FfmpegError as exc:
        logger.warning("faststart_failed", extra={"detail": str(exc)[-300:]})
        return clip
    return out


__all__ = [
    "PREVIEW_SHORT_SIDE",
    "ensure_faststart",
    "is_worth_previewing",
    "preview_dimensions",
    "write_preview",
]
