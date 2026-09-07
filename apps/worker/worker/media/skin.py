"""Skin-region re-anchoring of a chained seed frame — ffmpeg only.

Why this exists (7 Sep 2026). In chained Character Replacement the face
holds the photo's skin tone but the HANDS darken inside every window, and
each seed (the last frame of the previous window) carries that into the
next one. A whole-frame luminance match cannot see it: the room dominates
the statistics and its beige/gold chroma sits inside the classic skin
range. What does separate the character from the room is that the output
follows the source's composition frame for frame — so the source
performer's own skin silhouette at the seam instant, keyed tightly and
dilated, is a spatial gate that contains the character's face and hands
and excludes the room.

Inside that gate, the drift is "a whole region is dark", not shading. So
the correction fires on the LOCAL skin-mean luminance (a box-blurred
skin-weighted average) sitting well below the first window's own skin
level, never on ordinary shading, and it is an additive offset per plane
toward that level — the drift measured on the client's clip was additive
(hands ~65 units darker with their contrast range unchanged) with a
chroma shift toward brown.

Everything here is a pure function of files: the adapter decides when to
call it and records what it did. No numpy, no Pillow — the worker has
neither; `geq`, `boxblur`, `blend`, `alphamerge` and `signalstats` do the
work, and every number is read in the same (limited-range) space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.media.ffmpeg import FfmpegError, ffmpeg, ffmpeg_stdout

# ── Keys (geq expressions over X, Y) ────────────────────────────────────────

#: Pixel skin key on the SEED — wide, soft edges: warm chroma that is neither
#: black nor blown out. Values 0..1.
KEY_WIDE = (
    "clip((cb(X,Y)-84)/8,0,1)*clip((130-cb(X,Y))/8,0,1)"
    "*clip((cr(X,Y)-130)/8,0,1)*clip((184-cr(X,Y))/8,0,1)"
    "*clip((lum(X,Y)-24)/12,0,1)*clip((250-lum(X,Y))/12,0,1)"
)

#: Silhouette key on the SOURCE frame — tight (Cr >= 142), which on the
#: measured clip passes 10-19 % of the frame and under 0.5 % of its upper
#: half: the performer's skin, not their room.
KEY_TIGHT = (
    "clip((cb(X,Y)-88)/8,0,1)*clip((126-cb(X,Y))/8,0,1)"
    "*clip((cr(X,Y)-142)/8,0,1)*clip((182-cr(X,Y))/8,0,1)"
    "*clip((lum(X,Y)-30)/20,0,1)"
)

#: Frame × mask through an alpha plane: Y·a/255 and chroma scaled about 128,
#: so a plane's mean divided by the mask's mean is the masked mean. (A
#: two-input `blend` would first convert the limited-range frame to the
#: mask's full range and read 7 units high — measured.)
PRODUCT_EXPR = (
    "geq=lum='lum(X,Y)*alpha(X,Y)/255':cb='(cb(X,Y)-128)*alpha(X,Y)/255+128'"
    ":cr='(cr(X,Y)-128)*alpha(X,Y)/255+128':a='alpha(X,Y)',format=yuv444p"
)

#: The ramp: weight 0 where the local skin mean is within `RAMP_DEAD` of the
#: target, 1 where it is `RAMP_FULL` or more below it.
RAMP_DEAD = 10.0
RAMP_FULL = 34.0

#: Bounds on the offsets applied inside the mask.
LUMA_OFFSET_LIMIT = 80.0
CHROMA_OFFSET_LIMIT = 20.0

#: Guards. The gate must cover a plausible share of the frame (the source
#: key found the performer, not nothing and not the whole room); at least
#: this share of the keyed skin must be dark as a whole; offsets below the
#: threshold are noise.
GATE_COVER_RANGE = (0.02, 0.40)
MIN_DARK_FRACTION = 0.30
MIN_OFFSET = 2.0


def _radii(width: int) -> tuple[int, int, int]:
    """Gate dilation, gate feather and local-mean radius, from the canvas width."""
    return max(2, round(0.025 * width)), max(1, round(0.011 * width)), max(4, round(0.04 * width))


@dataclass(frozen=True)
class SkinStats:
    """Masked means over some frames: luminance and chroma of the keyed skin,
    and how much of the frame the mask covered."""

    y_mean: float
    cb_mean: float
    cr_mean: float
    cover: float


@dataclass(frozen=True)
class SkinCorrection:
    applied: bool
    reason: str
    gate_cover: float
    cover_keyed: float
    cover_ramped: float
    dark_fraction: float
    seed_y: float
    seed_cb: float
    seed_cr: float
    target_y: float
    target_cb: float
    target_cr: float
    y_offset: float
    cb_offset: float
    cr_offset: float


def _parse_stats(report: str) -> dict[str, float]:
    sums: dict[str, list[float]] = {"YAVG": [], "UAVG": [], "VAVG": []}
    for line in report.splitlines():
        for key in sums:
            marker = f"lavfi.signalstats.{key}="
            if marker in line:
                try:
                    sums[key].append(float(line.split(marker, 1)[1].strip()))
                except ValueError:
                    pass
    if not sums["YAVG"]:
        raise FfmpegError("no signalstats came back")
    return {key: sum(values) / len(values) for key, values in sums.items() if values}


async def _stats(args: list[str]) -> dict[str, float]:
    report = (await ffmpeg_stdout(args)).decode("utf-8", "replace")
    return _parse_stats(report)


def _gate_chain(width: int, height: int) -> str:
    """Source frame → dilated, feathered silhouette (gray)."""
    dilate, feather, _ = _radii(width)
    return (
        f"scale={width}:{height}:flags=bicubic,format=yuv444p,boxblur=1:1:5:1,"
        f"geq=lum='255*{KEY_TIGHT}':cb=128:cr=128,boxblur={dilate}:2,"
        f"lutyuv=y='clip((val-90)*4,0,255)',boxblur={feather}:1,format=gray"
    )


async def source_gate(
    clip: Path, frame_index: int, dest: Path, *, width: int, height: int
) -> Path:
    """The source performer's skin silhouette at `frame_index` of `clip`, on
    the canvas, dilated and feathered — a gray PNG."""
    await ffmpeg(
        [
            "-i",
            str(clip),
            "-vf",
            f"select='eq(n,{frame_index})',{_gate_chain(width, height)}",
            "-frames:v",
            "1",
            str(dest),
            "-y",
        ]
    )
    return dest


async def skin_target(
    rendered: Path,
    clip: Path,
    *,
    frames: tuple[int, int],
    width: int,
    height: int,
) -> SkinStats:
    """The keyed skin's luminance and chroma in `rendered` over `frames`,
    gated by the source silhouette of the aligned `clip` frames — the look
    the first window gave the character, measured where the character is."""
    select = f"select='between(n,{frames[0]},{frames[1]})'"
    count = frames[1] - frames[0] + 1
    gate = f"[1:v]{select},{_gate_chain(width, height)}[g]"
    keyed = (
        f"[0:v]{select},format=yuv444p,boxblur=1:1:5:1,split[a][b];"
        f"[a]geq=lum='255*{KEY_WIDE}':cb=128:cr=128,format=gray[k];"
        f"[k][g]blend=all_mode=multiply,format=gray[m]"
    )
    product = await _stats(
        [
            "-i",
            str(rendered),
            "-i",
            str(clip),
            "-filter_complex",
            f"{gate};{keyed};[b][m]alphamerge,format=yuva444p,{PRODUCT_EXPR},"
            f"signalstats,metadata=print:file=-",
            "-frames:v",
            str(count),
            "-f",
            "null",
            "-",
        ]
    )
    mask = await _stats(
        [
            "-i",
            str(rendered),
            "-i",
            str(clip),
            "-filter_complex",
            f"{gate};[0:v]{select},format=yuv444p,boxblur=1:1:5:1,"
            f"geq=lum='255*{KEY_WIDE}':cb=128:cr=128,format=gray[k];"
            f"[k][g]blend=all_mode=multiply,format=gray,signalstats,metadata=print:file=-",
            "-frames:v",
            str(count),
            "-f",
            "null",
            "-",
        ]
    )
    cover = mask["YAVG"] / 255.0
    if cover <= 1e-4:
        return SkinStats(y_mean=0.0, cb_mean=128.0, cr_mean=128.0, cover=0.0)
    return SkinStats(
        y_mean=product["YAVG"] / cover,
        cb_mean=(product.get("UAVG", 128.0) - 128.0) / cover + 128.0,
        cr_mean=(product.get("VAVG", 128.0) - 128.0) / cover + 128.0,
        cover=cover,
    )


async def _mask_mean(mask: Path) -> float:
    stats = await _stats(
        ["-i", str(mask), "-vf", "format=gray,signalstats,metadata=print:file=-", "-f", "null", "-"]
    )
    return stats["YAVG"] / 255.0


async def _masked_stats(image: Path, mask: Path) -> SkinStats:
    product = await _stats(
        [
            "-i",
            str(image),
            "-i",
            str(mask),
            "-filter_complex",
            f"[1]format=gray[m];[0]format=yuv444p[a];[a][m]alphamerge,format=yuva444p,"
            f"{PRODUCT_EXPR},signalstats,metadata=print:file=-",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    cover = await _mask_mean(mask)
    if cover <= 1e-4:
        return SkinStats(y_mean=0.0, cb_mean=128.0, cr_mean=128.0, cover=0.0)
    return SkinStats(
        y_mean=product["YAVG"] / cover,
        cb_mean=(product.get("UAVG", 128.0) - 128.0) / cover + 128.0,
        cr_mean=(product.get("VAVG", 128.0) - 128.0) / cover + 128.0,
        cover=cover,
    )


async def anchor_skin(
    seed: Path,
    gate: Path,
    dest: Path,
    *,
    target: SkinStats,
    width: int,
    work_dir: Path,
    tag: str,
) -> tuple[Path, SkinCorrection]:
    """Lifts the seed's dark skin regions (inside the gate) back to the target.

    Returns the frame to use (`dest` when applied, `seed` otherwise) and the
    record of what was measured and done.
    """
    _, _, local = _radii(width)
    keyed_mask = work_dir / f"{tag}-skin-keyed.png"
    ramped_mask = work_dir / f"{tag}-skin-mask.png"

    # Keyed × gate: where skin-like pixels of the character are at all.
    await ffmpeg(
        [
            "-i",
            str(seed),
            "-i",
            str(gate),
            "-filter_complex",
            f"[0]format=yuv444p,boxblur=1:1:5:1,geq=lum='255*{KEY_WIDE}':cb=128:cr=128,"
            f"format=gray[k];[1]format=gray[g];[k][g]blend=all_mode=multiply,format=gray[m]",
            "-map",
            "[m]",
            "-frames:v",
            "1",
            str(keyed_mask),
            "-y",
        ]
    )
    # The ramp on the LOCAL skin-mean luminance: Y·K in luma and 255·K in
    # cb, both box-blurred, their ratio is the skin-weighted local mean; the
    # ramp fires where that mean sits RAMP_DEAD..RAMP_FULL below the target.
    ramp = (
        f"255*clip(({target.y_mean:.3f}-{RAMP_DEAD}-lum(X,Y))/{RAMP_FULL - RAMP_DEAD},0,1)"
    )
    await ffmpeg(
        [
            "-i",
            str(seed),
            "-i",
            str(keyed_mask),
            "-filter_complex",
            f"[0]format=yuv444p,boxblur=1:1:5:1,geq=lum='lum(X,Y)*{KEY_WIDE}':cb='255*{KEY_WIDE}':cr=128,"
            f"boxblur={local}:1:{local}:1,"
            f"geq=lum='if(gt(cb(X,Y),6),lum(X,Y)*255/cb(X,Y),255)':cb=128:cr=128,"
            f"geq=lum='{ramp}':cb=128:cr=128,format=gray[r];"
            f"[1]format=gray[k];[r][k]blend=all_mode=multiply,boxblur=5:2,format=gray[m]",
            "-map",
            "[m]",
            "-frames:v",
            "1",
            str(ramped_mask),
            "-y",
        ]
    )

    gate_cover = await _mask_mean(gate)
    cover_keyed = await _mask_mean(keyed_mask)
    ramped = await _masked_stats(seed, ramped_mask)
    dark_fraction = ramped.cover / cover_keyed if cover_keyed > 1e-6 else 0.0

    y_offset = max(-LUMA_OFFSET_LIMIT, min(LUMA_OFFSET_LIMIT, target.y_mean - ramped.y_mean))
    cb_offset = max(-CHROMA_OFFSET_LIMIT, min(CHROMA_OFFSET_LIMIT, target.cb_mean - ramped.cb_mean))
    cr_offset = max(-CHROMA_OFFSET_LIMIT, min(CHROMA_OFFSET_LIMIT, target.cr_mean - ramped.cr_mean))

    def record(applied: bool, reason: str) -> SkinCorrection:
        return SkinCorrection(
            applied=applied,
            reason=reason,
            gate_cover=round(gate_cover, 4),
            cover_keyed=round(cover_keyed, 4),
            cover_ramped=round(ramped.cover, 4),
            dark_fraction=round(dark_fraction, 3),
            seed_y=round(ramped.y_mean, 2),
            seed_cb=round(ramped.cb_mean, 2),
            seed_cr=round(ramped.cr_mean, 2),
            target_y=round(target.y_mean, 2),
            target_cb=round(target.cb_mean, 2),
            target_cr=round(target.cr_mean, 2),
            y_offset=round(y_offset, 2) if applied else 0.0,
            cb_offset=round(cb_offset, 2) if applied else 0.0,
            cr_offset=round(cr_offset, 2) if applied else 0.0,
        )

    if not (GATE_COVER_RANGE[0] <= gate_cover <= GATE_COVER_RANGE[1]):
        return seed, record(False, "gate outside range")
    if target.cover <= 1e-4:
        return seed, record(False, "no target skin")
    if dark_fraction < MIN_DARK_FRACTION:
        return seed, record(False, "skin not dark as a whole")
    if abs(y_offset) < MIN_OFFSET and abs(cb_offset) < MIN_OFFSET and abs(cr_offset) < MIN_OFFSET:
        return seed, record(False, "offsets negligible")
    if y_offset < 0:
        # Only the darker direction is shipped: a region that is BRIGHTER than
        # the target inside the ramp cannot happen (the ramp fires below it),
        # so a negative offset means the numbers disagree — do nothing.
        return seed, record(False, "offset direction disagrees with the ramp")

    # The mask rides as an alpha plane and the offsets are applied in one
    # exact expression per plane. (`maskedmerge` was tried first: with a
    # full-range white mask it lands at 94 % of the offset, so it scales the
    # mask by a range it was never in.)
    await ffmpeg(
        [
            "-i",
            str(seed),
            "-i",
            str(ramped_mask),
            "-filter_complex",
            f"[1]format=gray[m];[0]format=yuv444p[a];[a][m]alphamerge,format=yuva444p,"
            f"geq=lum='clip(lum(X,Y)+({y_offset:.3f})*alpha(X,Y)/255,0,255)'"
            f":cb='clip(cb(X,Y)+({cb_offset:.3f})*alpha(X,Y)/255,0,255)'"
            f":cr='clip(cr(X,Y)+({cr_offset:.3f})*alpha(X,Y)/255,0,255)'"
            f":a='alpha(X,Y)',format=rgb24[o]",
            "-map",
            "[o]",
            "-frames:v",
            "1",
            str(dest),
            "-y",
        ]
    )
    return dest, record(True, "applied")
