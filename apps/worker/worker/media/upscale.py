"""One finished clip, enlarged to a delivery frame — ffmpeg only.

Lifted out of `worker/adapters/ltx_hd.py` on 8 Sep 2026 when the client asked
for the same 4K finish on Character Replacement ("let's use the same approach
upscale to 4k after video is done"). It is the same code, moved rather than
rewritten, so the two tools cannot drift into two different 4K.

Why ffmpeg rather than the graph's own `ImageScale`: a 4K frame batch of 721
frames is ~72 GB as a tensor inside ComfyUI plus a CPU encode, where ffmpeg
streams it and NVENC does it in seconds - measured 2.3 s for a 10 s clip.
H.264 rather than HEVC because browsers play it; the file is ~2.5x larger for
that.

The soundtrack is copied through untouched (`-c:a copy`), which is the whole
reason this runs after the join rather than per section: one encode, one
audio stream, no seam.
"""

from __future__ import annotations

from pathlib import Path

from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError, ffmpeg

logger = get_logger(__name__)

#: The 4K frame per aspect ratio (client request, 8 Sep 2026).
DELIVERY_4K: dict[str, tuple[int, int]] = {
    "16:9": (3840, 2160),
    "9:16": (2160, 3840),
    # 2160x2160, not 3840: the square frame's 4K is the vertical one's
    # short side, which is what Text to Video HD has always delivered.
    "1:1": (2160, 2160),
}


def is_4k(raw: object) -> bool:
    """Whether a delivery setting names 4K. Anything unrecognised is not,
    so a typo cannot silently quadruple every file."""
    return str(raw or "").strip().lower() in ("4k", "2160p", "uhd")


async def upscale_clip(
    clip: Path,
    out: Path,
    target: tuple[int, int],
    *,
    nvenc_timeout: float,
    cpu_timeout: float,
    run: object = None,
    log_extra: dict | None = None,
) -> Path:
    """`clip` scaled to cover `target`, centre-cropped to it exactly.

    `run` wraps each ffmpeg call — the adapters pass `cancellable` bound to
    their job, so an abandoned job kills the encoder rather than finishing it.
    """
    width, height = target
    scale = (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    common = ["-i", str(clip), "-vf", scale, "-pix_fmt", "yuv420p", "-c:a", "copy",
              "-movflags", "+faststart"]

    async def _run(awaitable):
        return await (run(awaitable) if run is not None else awaitable)

    try:
        await _run(
            ffmpeg(
                [*common, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", str(out)],
                timeout=nvenc_timeout,
            )
        )
    except FfmpegError as exc:
        # No NVENC on this box: the CPU encoder, slower and identical.
        logger.warning(
            "upscale_nvenc_unavailable", extra={**(log_extra or {}), "detail": str(exc)[-300:]}
        )
        await _run(
            ffmpeg(
                [*common, "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out)],
                timeout=cpu_timeout,
            )
        )
    logger.info(
        "upscaled", extra={**(log_extra or {}), "target": f"{width}x{height}"}
    )
    return out


__all__ = ["DELIVERY_4K", "is_4k", "upscale_clip"]
