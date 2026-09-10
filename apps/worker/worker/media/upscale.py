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

**10 Sep 2026 — the client's review added two things to this one pass.**

*Temporal stabilisation.* Their note: "the main missing protections are
temporal deflicker, color stabilization and explicit Rec.709 output. The
negative prompt alone cannot fix this because the distilled workflow uses
CFG 1.0." They offered two insertion points — inside the Decode subgraph
between `5573 ImageScale` and `4849 CreateVideo`, or after ComfyUI saves the
file and before the backend returns it. This takes the second, because it is
where this module already runs: the graph stays byte-identical to the one
they sent (its sha256 is pinned by a test), and no ComfyUI custom node has to
exist on the box. The deflicker is prepended to the same filter chain, so
stabilising and resizing are ONE encode rather than two — a second pass would
be a second generation loss for nothing.

*Rec.709.* The tags are written on every clip this module touches. Untagged
H.264 is guessed at by players, and a 480p-class source enlarged to a big
frame is exactly the case where a decoder guessing BT.601 shifts the skin
tones. `-color_range tv` matches the limited-range yuv420p already written.

**Codec by frame size.** NVENC has no H.264 encoder above 4096x4096 — a
hardware cap, not a setting — so an 8K frame cannot go through it and the CPU
fallback would take longer than the render did. A target with a side over
4096 therefore encodes HEVC on NVENC (`hvc1`-tagged so QuickTime and Safari
play it); anything at or under 4096 stays H.264 exactly as before.
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

#: The 8K frame per aspect ratio (client request, 10 Sep 2026): "replace the
#: 4K destination with 8K so it scales directly from 480p to 8K". Exactly
#: twice `DELIVERY_4K` on each side, the square following the rule it already
#: did — the vertical frame's short side.
DELIVERY_8K: dict[str, tuple[int, int]] = {
    "16:9": (7680, 4320),
    "9:16": (4320, 7680),
    "1:1": (4320, 4320),
}

#: Above this side length NVENC has no H.264 encoder. See the module docstring.
_H264_MAX_SIDE = 4096

#: Written on every clip this module encodes — and it takes BOTH of these.
#:
#: The encoder flags alone are what the client's snippet has, and measured on
#: this pipeline 10 Sep 2026 they land only `colorspace` and `color_range`:
#: ffprobe reports no `color_primaries` or `color_transfer` at all, because
#: the frames reaching the encoder carry "unspecified" and the flags do not
#: retag them. `setparams` tags the frames themselves, and then all four
#: survive into the file. Both are kept: the filter fixes the frames, the
#: flags state the same thing to the encoder, and neither is redundant with
#: the other in practice.
_SETPARAMS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
_REC709 = [
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-color_range", "tv",
]

#: The client's temporal stabiliser, verbatim from their 10 Sep 2026 note: a
#: 5-frame arithmetic-mean deflicker. It levels each frame's average
#: luminance against its neighbours, which is what the pulsing exposure of a
#: CFG-1.0 distilled render actually is.
DEFLICKER = "deflicker=size=5:mode=am"


def is_4k(raw: object) -> bool:
    """Whether a delivery setting names 4K. Anything unrecognised is not,
    so a typo cannot silently quadruple every file."""
    return str(raw or "").strip().lower() in ("4k", "2160p", "uhd")


def is_8k(raw: object) -> bool:
    """Whether a delivery setting names 8K. Same rule as `is_4k`, and for a
    stronger reason: an 8K frame is sixteen times a 1080p one."""
    return str(raw or "").strip().lower() in ("8k", "4320p", "uhd8k")


def _headroom(bitrate: str) -> tuple[str, str]:
    """`-maxrate` and `-bufsize` for a target rate: 1.5x and 2x it.

    A hard cap at the target would starve the opening seconds of a shot,
    which is where an enlarged 480p frame needs the bits most. A string this
    cannot parse falls back to the target itself rather than raising — the
    finishing pass is not the place to fail a rendered job over a config
    string.
    """
    digits = "".join(c for c in bitrate if c.isdigit())
    if not digits:
        return bitrate, bitrate
    suffix = bitrate[len(digits) :]
    value = int(digits)
    return f"{int(value * 1.5)}{suffix}", f"{value * 2}{suffix}"


def _video_codec(target: tuple[int, int], bitrate: str | None) -> tuple[list[str], list[str]]:
    """(NVENC arguments, CPU-fallback arguments) for this frame and rate.

    A bitrate turns both encoders from constant-quality into capped VBR,
    which is how the client's "approximately 100 Mbps" is honoured: `-cq`
    alone lands wherever the content puts it, which at 4K measured 35-45 Mbps
    on their files and at 8K would be neither predictable nor theirs to
    choose.
    """
    hevc = max(target) > _H264_MAX_SIDE
    nvenc = ["-c:v", "hevc_nvenc", "-tag:v", "hvc1"] if hevc else ["-c:v", "h264_nvenc"]
    nvenc += ["-preset", "p5", "-tune", "hq"]
    cpu = ["-c:v", "libx265", "-tag:v", "hvc1"] if hevc else ["-c:v", "libx264"]
    cpu += ["-preset", "fast"]
    if bitrate is None:
        return [*nvenc, "-cq", "19"], [*cpu, "-crf", "18"]
    ceiling, buffer = _headroom(bitrate)
    return (
        [*nvenc, "-rc", "vbr", "-cq", "0", "-b:v", bitrate,
         "-maxrate", ceiling, "-bufsize", buffer],
        [*cpu, "-b:v", bitrate, "-maxrate", ceiling, "-bufsize", buffer],
    )


async def upscale_clip(
    clip: Path,
    out: Path,
    target: tuple[int, int],
    *,
    nvenc_timeout: float,
    cpu_timeout: float,
    run: object = None,
    log_extra: dict | None = None,
    deflicker: bool = False,
    bitrate: str | None = None,
) -> Path:
    """`clip` scaled to cover `target`, centre-cropped to it exactly.

    `run` wraps each ffmpeg call — the adapters pass `cancellable` bound to
    their job, so an abandoned job kills the encoder rather than finishing it.

    `deflicker` prepends the client's temporal stabiliser to the same filter
    chain. It runs BEFORE the resize on purpose: the flicker is in the
    generated frame, so correcting it at the generated size costs a fraction
    of the same filter on an enlarged one, and the enlargement then carries
    corrected pixels instead of magnifying the pulse.

    **The deflicker averages luminance across neighbouring frames, so it must
    never span a hard cut** — a cut is two different scenes, and their average
    is a flash on both sides of it. This function has no notion of a cut and
    treats what it is given as one continuous shot; a caller with cuts
    stabilises each shot before the join. Every caller today qualifies: Text
    to Video renders a single pass, and Character Replacement chains windows
    of one continuous source.
    """
    width, height = target
    chain = [DEFLICKER] if deflicker else []
    chain += [
        f"scale=w={width}:h={height}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={width}:{height}",
        _SETPARAMS,
    ]
    nvenc_codec, cpu_codec = _video_codec(target, bitrate)
    common = ["-i", str(clip), "-vf", ",".join(chain), "-pix_fmt", "yuv420p",
              *_REC709, "-c:a", "copy", "-movflags", "+faststart"]

    async def _run(awaitable):
        return await (run(awaitable) if run is not None else awaitable)

    try:
        await _run(ffmpeg([*common, *nvenc_codec, str(out)], timeout=nvenc_timeout))
    except FfmpegError as exc:
        # No NVENC on this box: the CPU encoder, slower and identical.
        logger.warning(
            "upscale_nvenc_unavailable", extra={**(log_extra or {}), "detail": str(exc)[-300:]}
        )
        await _run(ffmpeg([*common, *cpu_codec, str(out)], timeout=cpu_timeout))
    logger.info(
        "upscaled",
        extra={
            **(log_extra or {}),
            "target": f"{width}x{height}",
            "deflicker": deflicker,
            "bitrate": bitrate or "cq",
        },
    )
    return out


__all__ = [
    "DEFLICKER",
    "DELIVERY_4K",
    "DELIVERY_8K",
    "is_4k",
    "is_8k",
    "upscale_clip",
]
