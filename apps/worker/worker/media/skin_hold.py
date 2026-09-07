"""Per-frame skin hold for chained character replacement (7 Sep 2026).

The seed-side re-anchoring (`worker.media.skin`) puts every chained seed
back at the first window's skin level, but the skin slides again INSIDE the
next window: measured on the client's clip, from Y ≈ 136 in the first second
to ≈ 100 by the last frame of an 8 s window, and with a prompt that does not
describe the character the seams then read as brightness steps. This module
holds the DELIVERED frames: in every frame of a chained window, skin inside
the source performer's silhouette whose LOCAL mean sits below the first
window's own skin level is lifted back toward it — each pixel by its own
deficit, capped — so a lit face gets nothing by construction and a dark hand
gets exactly what it lacks, whether or not anything else is dark in that
frame. (A per-frame scalar through a mask was tried first and refuted on a
synthetic pair: shaded face pixels swung 27 Y with the hands in and out of
the mask; the per-pixel deficit gave the same result with and without them.)

The seed for the next window is taken from the HELD frames, so the seam
frame and the seed carry the same skin; the seed pass's skin step is then
skipped for that window. The GPU graph, its parameters and the renders are
untouched; only delivered pixels inside the mask move.

ffmpeg only (no numpy/PIL/cv2 on the worker). Two passes per window:

* the MEASURE pass builds the local-skin-mean plane and the keyed×gate plane
  once at half the canvas (soft masks; 8 s against 28 s at full size on the
  node), writes both losslessly, and reads six per-frame statistics;
* the APPLY pass turns those planes into the per-pixel lift with the
  per-frame target and cap (piecewise-linear in the frame number, smoothed
  over a second), upscales it and adds it through an alpha plane, in the
  same encode that drops the seam frame.

Every number is read in one range: new mask planes are plane copies
(`extractplanes=y`; `format=gray` was measured to range-expand on ffmpeg 9,
100 → 98 and 12 → 0), and the frame meets the lift only through
`alphamerge` (a two-input `blend` on the frame reads 7 units high;
`maskedmerge` lands at 94 % of a full mask — both measured earlier).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.media.ffmpeg import FfmpegError, ffmpeg
from worker.media.skin import (
    CHROMA_OFFSET_LIMIT,
    KEY_TIGHT,
    KEY_WIDE,
    LUMA_OFFSET_LIMIT,
    MIN_OFFSET,
    PRODUCT_EXPR,
    RAMP_DEAD,
    RAMP_FULL,
    SkinStats,
    _radii,
    _stats,
)

#: Frames on each side that the per-frame target and cap are averaged over:
#: lighting is slow, and a frame the gate vetoes becomes a one-second dip,
#: never a step.
SMOOTH_RADIUS = 24

#: The apply expressions are piecewise linear between knots this far apart.
KNOT_STEP = 12

#: Below this share of the frame the corrected region is noise.
MIN_RAMPED_COVER = 0.002

#: The window is held only when the average lift across the keyed silhouette
#: reaches this. Measured on a synthetic window that does not drift at all:
#: the feathered boundary between skin and room reads 3.4, so a bar below
#: this would "hold" a window whose only dark skin is the edge of the mask.
HOLD_ACTIVATION = 6.0

#: The gate (the source performer's silhouette) must cover a plausible share
#: of the frame: the weight is 1 between the floor and `GATE_SOFT_FROM`, and
#: fades to 0 at `GATE_SOFT_TO` (a close-up that fills the frame).
GATE_FLOOR = 0.02
GATE_SOFT_FROM = 0.40
GATE_SOFT_TO = 0.50

#: The target follows the SOURCE performer's own skin level under the very
#: pixels being lifted, within this band: a real shadow the source shows
#: there stays a shadow; a wider swing is more likely blur or key loss than
#: lighting.
SOURCE_RATIO_RANGE = (0.8, 1.15)

#: The masks are built at the canvas divided by this.
ANALYSIS_DIVISOR = 2


@dataclass(frozen=True)
class FrameReading:
    """One frame's numbers: covers as fractions of the frame, the masked
    means of the rendered skin under the ramp, the source performer's skin
    luminance under the same ramp, and the mean lift the nominal target
    would give inside the keyed silhouette."""

    gate: float
    keyed: float
    ramped: float
    y: float
    cb: float
    cr: float
    source_y: float
    lift: float


@dataclass(frozen=True)
class HoldTarget:
    """What the skin is held to: the first window's own skin level (the
    seed pass's target) and the source's skin level over the same frames."""

    y: float
    cb: float
    cr: float
    source_y: float

    @classmethod
    def of(cls, skin: SkinStats, source_y: float) -> HoldTarget:
        return cls(y=skin.y_mean, cb=skin.cb_mean, cr=skin.cr_mean, source_y=source_y)


@dataclass(frozen=True)
class HoldPlan:
    frames: int
    target: tuple[float, ...]
    """The per-frame target luminance (the window target × the smoothed
    source ratio)."""
    cap: tuple[float, ...]
    """The per-frame cap on the lift (`LUMA_OFFSET_LIMIT` × the smoothed
    gate weight; 0 on the first frame of the first window)."""
    r_cb: float
    r_cr: float
    """Chroma offset per unit of luma lift, one pair per window."""
    held_frames: int
    """Frames whose nominal lift inside the silhouette is not negligible."""
    refused: dict[str, int]
    mean_lift: float
    max_lift: float
    y_first: float
    y_last: float

    @property
    def active(self) -> bool:
        return self.max_lift >= HOLD_ACTIVATION and max(self.cap, default=0.0) > 0.0

    def summary(self) -> dict[str, object]:
        return {
            "applied": self.active,
            "frames": self.frames,
            "held_frames": self.held_frames,
            "mean_lift": round(self.mean_lift, 2),
            "max_lift": round(self.max_lift, 2),
            "y_first": round(self.y_first, 2),
            "y_last": round(self.y_last, 2),
            "target_min": round(min(self.target), 2) if self.target else 0.0,
            "target_max": round(max(self.target), 2) if self.target else 0.0,
            "cap_min": round(min(self.cap), 2) if self.cap else 0.0,
            "r_cb": round(self.r_cb, 3),
            "r_cr": round(self.r_cr, 3),
            "refused": dict(self.refused),
        }


def analysis_size(width: int, height: int) -> tuple[int, int]:
    """The mask canvas: the frame divided by `ANALYSIS_DIVISOR`, even."""
    return (
        max(64, (width // ANALYSIS_DIVISOR) // 2 * 2),
        max(64, (height // ANALYSIS_DIVISOR) // 2 * 2),
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_series(text: str) -> list[dict[str, float]]:
    """`metadata=print` output → one dict of signalstats values per frame."""
    frames: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in text.splitlines():
        if line.startswith("frame:"):
            current = {}
            frames.append(current)
            continue
        if current is None or "lavfi.signalstats." not in line:
            continue
        key, _, value = line.partition("=")
        try:
            current[key.strip().rsplit(".", 1)[-1]] = float(value.strip())
        except ValueError:
            continue
    return frames


def _gate_chain(aw: int, ah: int) -> str:
    """Source frame → dilated, feathered silhouette, as a plane copy."""
    dilate, feather, _ = _radii(aw)
    return (
        f"scale={aw}:{ah}:flags=bicubic,format=yuv444p,boxblur=1:1:5:1,"
        f"geq=lum='255*{KEY_TIGHT}':cb=128:cr=128,boxblur={dilate}:2,"
        f"lutyuv=y='clip((val-90)*4,0,255)',boxblur={feather}:1,extractplanes=y"
    )


def _local_chain(local: int) -> str:
    """Blurred rendered frame → the local skin-mean luminance plane: Y·K and
    255·K box-blurred, their ratio; 255 (no lift) where there is no skin."""
    return (
        f"geq=lum='lum(X,Y)*{KEY_WIDE}':cb='255*{KEY_WIDE}':cr=128,"
        f"boxblur={local}:1:{local}:1,"
        f"geq=lum='if(gt(cb(X,Y),6),lum(X,Y)*255/cb(X,Y),255)':cb=128:cr=128,extractplanes=y"
    )


def _lift_expr(target: str, cap: str) -> str:
    """The per-pixel lift from the local mean `lum(X,Y)`: the deficit to the
    target, capped, weighted by the ramp that is 0 within `RAMP_DEAD` of the
    target and 1 from `RAMP_FULL` below it."""
    return (
        f"clip(min({target}-lum(X,Y),{cap})"
        f"*clip(({target}-{RAMP_DEAD}-lum(X,Y))/{RAMP_FULL - RAMP_DEAD},0,1),0,255)"
    )


def _measure_graph(aw: int, ah: int, target_y: float, names: dict[str, str]) -> str:
    """Inputs: 0 = the rendered window, 1 = the aligned cut of the source.

    Labels out: six statistics streams, the local-mean plane `[locout]` and
    the keyed×gate plane `[kgout]`.
    """
    _, _, local = _radii(aw)
    ramp = f"255*clip(({target_y:.3f}-{RAMP_DEAD}-lum(X,Y))/{RAMP_FULL - RAMP_DEAD},0,1)"
    lift = _lift_expr(f"{target_y:.3f}", f"{LUMA_OFFSET_LIMIT:.3f}")
    return (
        f"[1:v]split=2[s0][s1];"
        f"[s0]{_gate_chain(aw, ah)},split=2[g][gk];"
        f"[s1]scale={aw}:{ah}:flags=bicubic,format=yuv444p[sv];"
        f"[0:v]scale={aw}:{ah}:flags=bicubic,format=yuv444p,split=2[c][b0];"
        f"[b0]boxblur=1:1:5:1,split=2[a][b];"
        f"[a]geq=lum='255*{KEY_WIDE}':cb=128:cr=128,extractplanes=y[k0];"
        f"[k0][gk]blend=all_mode=multiply,split=4[kg][kg2][kg3][kgout];"
        f"[b]{_local_chain(local)},split=3[loc][loc2][locout];"
        f"[loc]geq=lum='{ramp}'[r0];[r0][kg]blend=all_mode=multiply,split=3[m][m2][m3];"
        f"[loc2]geq=lum='{lift}'[d0];[d0][kg2]blend=all_mode=multiply,"
        f"signalstats,metadata=print:file={names['lift']}[o6];"
        f"[c][m]alphamerge,format=yuva444p,{PRODUCT_EXPR},signalstats,"
        f"metadata=print:file={names['product']}[o1];"
        f"[m2]signalstats,metadata=print:file={names['ramped']}[o2];"
        f"[kg3]signalstats,metadata=print:file={names['keyed']}[o3];"
        f"[g]signalstats,metadata=print:file={names['gate']}[o4];"
        f"[sv][m3]alphamerge,format=yuva444p,{PRODUCT_EXPR},signalstats,"
        f"metadata=print:file={names['source']}[o5]"
    )


async def source_skin_level(
    clip: Path, *, frames: tuple[int, int], width: int, height: int
) -> float:
    """The SOURCE performer's skin luminance under their own silhouette over
    `frames` of `clip` — the reference the per-frame target follows."""
    aw, ah = analysis_size(width, height)
    select = f"select='between(n,{frames[0]},{frames[1]})'"
    count = frames[1] - frames[0] + 1
    product = await _stats(
        [
            "-i",
            str(clip),
            "-filter_complex",
            f"[0:v]{select},split=2[s0][s1];[s0]{_gate_chain(aw, ah)}[g];"
            f"[s1]scale={aw}:{ah}:flags=bicubic,format=yuv444p[sv];"
            f"[sv][g]alphamerge,format=yuva444p,{PRODUCT_EXPR},signalstats,"
            f"metadata=print:file=-",
            "-frames:v",
            str(count),
            "-f",
            "null",
            "-",
        ]
    )
    gate = await _stats(
        [
            "-i",
            str(clip),
            "-vf",
            f"{select},{_gate_chain(aw, ah)},signalstats,metadata=print:file=-",
            "-frames:v",
            str(count),
            "-f",
            "null",
            "-",
        ]
    )
    cover = gate["YAVG"] / 255.0
    return product["YAVG"] / cover if cover > 1e-4 else 0.0


@dataclass(frozen=True)
class Planes:
    local: Path
    keyed: Path


async def measure_window(
    part: Path,
    clip: Path,
    target: HoldTarget,
    *,
    width: int,
    height: int,
    work_dir: Path,
    tag: str,
) -> tuple[list[FrameReading], Planes]:
    """The measure pass: per-frame readings and the two mask planes.

    The statistics are written to files named relative to `work_dir` (a
    path with a drive letter cannot ride inside a filter option), the planes
    as raw gray frames in NUT containers.
    """
    aw, ah = analysis_size(width, height)
    keys = ("product", "ramped", "keyed", "gate", "source", "lift")
    names = {key: f"hold-{tag}-{key}.txt" for key in keys}
    planes = Planes(
        local=work_dir / f"hold-{tag}-local.nut", keyed=work_dir / f"hold-{tag}-keyed.nut"
    )
    outputs: list[str] = []
    for label in ("o1", "o2", "o3", "o4", "o5", "o6"):
        outputs += ["-map", f"[{label}]", "-f", "null", "-"]
    await ffmpeg(
        [
            "-i",
            str(part),
            "-i",
            str(clip),
            "-filter_complex",
            _measure_graph(aw, ah, target.y, names),
            *outputs,
            "-map",
            "[locout]",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-f",
            "nut",
            str(planes.local),
            "-map",
            "[kgout]",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-f",
            "nut",
            str(planes.keyed),
        ],
        cwd=work_dir,
    )
    series = {
        key: _parse_series((work_dir / name).read_text(encoding="utf-8", errors="replace"))
        for key, name in names.items()
    }
    counts = {key: len(value) for key, value in series.items()}
    if len(set(counts.values())) != 1 or counts["product"] == 0:
        raise FfmpegError(f"hold statistics disagree on the frame count: {counts}")

    readings: list[FrameReading] = []
    for index in range(counts["product"]):
        gate = series["gate"][index].get("YAVG", 0.0) / 255.0
        keyed = series["keyed"][index].get("YAVG", 0.0) / 255.0
        ramped = series["ramped"][index].get("YAVG", 0.0) / 255.0
        product = series["product"][index]
        if ramped > 1e-6:
            y = product.get("YAVG", 0.0) / ramped
            cb = (product.get("UAVG", 128.0) - 128.0) / ramped + 128.0
            cr = (product.get("VAVG", 128.0) - 128.0) / ramped + 128.0
        else:
            y, cb, cr = target.y, target.cb, target.cr
        source = series["source"][index].get("YAVG", 0.0)
        source_y = source / ramped if ramped > 1e-6 else 0.0
        lift = series["lift"][index].get("YAVG", 0.0)
        lift = lift / keyed if keyed > 1e-6 else 0.0
        readings.append(
            FrameReading(
                gate=gate,
                keyed=keyed,
                ramped=ramped,
                y=y,
                cb=cb,
                cr=cr,
                source_y=source_y,
                lift=lift,
            )
        )
    return readings, planes


def _moving_average(values: list[float], radius: int) -> list[float]:
    out: list[float] = []
    count = len(values)
    for index in range(count):
        window = values[max(0, index - radius) : min(count, index + radius + 1)]
        out.append(sum(window) / len(window))
    return out


def plan_hold(readings: list[FrameReading], target: HoldTarget, *, first_window: bool) -> HoldPlan:
    """The per-frame target and cap from the readings.

    The target follows the source performer's skin under the corrected
    region (ratio clamped, smoothed); the cap is the luma limit weighted by
    the gate's plausibility (smoothed). Neither decides which pixels move —
    the per-pixel deficit does that in the apply pass. In the first window
    frame 0 is the customer's own picture and gets a cap of 0.
    """
    count = len(readings)
    ratios: list[float] = []
    weights: list[float] = []
    refused: dict[str, int] = {}
    for reading in readings:
        if reading.gate < GATE_FLOOR:
            weight = 0.0
            refused["gate too small"] = refused.get("gate too small", 0) + 1
        elif reading.gate > GATE_SOFT_FROM:
            weight = _clip(
                (GATE_SOFT_TO - reading.gate) / (GATE_SOFT_TO - GATE_SOFT_FROM), 0.0, 1.0
            )
            if weight <= 0.0:
                refused["gate too large"] = refused.get("gate too large", 0) + 1
        else:
            weight = 1.0
        weights.append(weight)
        ratio = 1.0
        if reading.ramped >= MIN_RAMPED_COVER and target.source_y > 1.0 and reading.source_y > 0.0:
            ratio = _clip(reading.source_y / target.source_y, *SOURCE_RATIO_RANGE)
        ratios.append(ratio)

    smooth_ratio = _moving_average(ratios, SMOOTH_RADIUS)
    smooth_weight = _moving_average(weights, SMOOTH_RADIUS)
    targets = [target.y * ratio for ratio in smooth_ratio]
    caps = [LUMA_OFFSET_LIMIT * weight for weight in smooth_weight]
    if first_window and caps:
        caps[0] = 0.0

    # Chroma per unit of luma, from the frames that have a corrected region.
    numerator_cb = numerator_cr = denominator = 0.0
    for reading, frame_target in zip(readings, targets, strict=True):
        if reading.ramped < MIN_RAMPED_COVER:
            continue
        deficit = _clip(frame_target - reading.y, 0.0, LUMA_OFFSET_LIMIT)
        if deficit < MIN_OFFSET:
            continue
        numerator_cb += deficit * _clip(
            target.cb - reading.cb, -CHROMA_OFFSET_LIMIT, CHROMA_OFFSET_LIMIT
        )
        numerator_cr += deficit * _clip(
            target.cr - reading.cr, -CHROMA_OFFSET_LIMIT, CHROMA_OFFSET_LIMIT
        )
        denominator += deficit * deficit
    bound = CHROMA_OFFSET_LIMIT / LUMA_OFFSET_LIMIT
    r_cb = _clip(numerator_cb / denominator, -bound, bound) if denominator > 0 else 0.0
    r_cr = _clip(numerator_cr / denominator, -bound, bound) if denominator > 0 else 0.0

    lifts = [
        reading.lift * (cap / LUMA_OFFSET_LIMIT)
        for reading, cap in zip(readings, caps, strict=True)
    ]
    held = [lift for lift in lifts if lift >= MIN_OFFSET / 4]
    return HoldPlan(
        frames=count,
        target=tuple(round(value, 3) for value in targets),
        cap=tuple(round(value, 3) for value in caps),
        r_cb=r_cb,
        r_cr=r_cr,
        held_frames=len(held),
        refused=refused,
        mean_lift=sum(held) / len(held) if held else 0.0,
        max_lift=max(lifts, default=0.0),
        y_first=readings[0].y if readings else 0.0,
        y_last=readings[-1].y if readings else 0.0,
    )


def piecewise(values: tuple[float, ...]) -> str:
    """`values` as a piecewise-linear expression in the frame number `N`,
    with knots every `KNOT_STEP` frames and on the last frame."""
    count = len(values)
    if count == 0:
        return "0"
    indices = list(range(0, count, KNOT_STEP))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    knots = [(index, values[index]) for index in indices]
    expression = f"{knots[-1][1]:.3f}"
    for (n0, v0), (n1, v1) in reversed(list(zip(knots, knots[1:], strict=False))):
        expression = f"if(lt(N,{n1}),{v0:.3f}+({v1:.3f}-{v0:.3f})*(N-{n0})/{n1 - n0},{expression})"
    return expression


def hold_apply_args(
    part: Path,
    planes: Planes,
    plan: HoldPlan,
    dest: Path,
    *,
    width: int,
    height: int,
    skip: int,
    kept_frames: int,
    encode: list[str],
) -> list[str]:
    """The apply pass as ffmpeg arguments (the caller runs them cancellably).

    Inputs: 0 = the rendered window, 1 = the local-mean plane, 2 = the
    keyed×gate plane. The lift is computed per pixel on the half-size local
    plane with the per-frame target and cap, weighted by the keyed
    silhouette, softened, upscaled, and added through an alpha plane. The
    seam frame is dropped AFTER the lift so `N` is the part's own index.
    """
    lift = _lift_expr(f"({piecewise(plan.target)})", f"({piecewise(plan.cap)})")
    graph = (
        f"[1:v]geq=lum='{lift}'[d];"
        f"[d][2:v]blend=all_mode=multiply,boxblur=5:2,"
        f"scale={width}:{height}:flags=bilinear[lift];"
        f"[0:v]format=yuv444p[raw];[raw][lift]alphamerge,format=yuva444p,"
        f"geq=lum='clip(lum(X,Y)+alpha(X,Y),0,255)'"
        f":cb='clip(cb(X,Y)+({plan.r_cb:.4f})*alpha(X,Y),0,255)'"
        f":cr='clip(cr(X,Y)+({plan.r_cr:.4f})*alpha(X,Y),0,255)'"
        f":a='alpha(X,Y)',format=yuv420p,trim=start_frame={skip},setpts=PTS-STARTPTS[o]"
    )
    return [
        "-i",
        str(part),
        "-i",
        str(planes.local),
        "-i",
        str(planes.keyed),
        "-filter_complex",
        graph,
        "-map",
        "[o]",
        "-an",
        "-frames:v",
        str(kept_frames),
        "-fps_mode",
        "cfr",
        *encode,
        str(dest),
    ]
