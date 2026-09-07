"""The per-frame skin hold (`worker.media.skin_hold`) on synthetic clips
whose every region is known.

A grey scene, a light FACE rectangle that never changes, a HANDS rectangle
in the same skin chroma that darkens frame by frame, a source clip whose
skin silhouette covers both, and a dark skin-coloured patch OUTSIDE the
silhouette. The hold must bring the hands back toward the first frames'
skin level, leave the face and the outside patch and the room where they
are, and — the reason the design changed — give the face the same answer
whether or not the hands are in the frame. Real ffmpeg, no fake service.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import needs_ffmpeg
from worker.media.ffmpeg import ffmpeg, ffmpeg_stdout
from worker.media.skin import skin_target
from worker.media.skin_hold import (
    GATE_SOFT_TO,
    LUMA_OFFSET_LIMIT,
    HoldTarget,
    hold_apply_args,
    measure_window,
    piecewise,
    plan_hold,
    source_skin_level,
)

W, H = 368, 640
FPS = 24
FRAMES = 49
GREY = "0x5a5a5a"
LIGHT_SKIN = (225, 180, 150)
SOURCE_SKIN = "0xc89678"
FACE = (120, 100, 130, 150)
HANDS = (80, 380, 210, 180)
SHADED = (128, 250, 60, 40)  # a strip of face that is already a little dark
OUTSIDE = (8, 8, 40, 50)
SILHOUETTE_FACE = (105, 85, 160, 180)
SILHOUETTE_HANDS = (65, 365, 240, 210)
ENCODE = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p"]


def _rgb(scale: float) -> str:
    return "0x" + "".join(f"{max(0, min(255, round(c * scale))):02x}" for c in LIGHT_SKIN)


def _box(box: tuple[int, int, int, int], colour: str) -> str:
    x, y, w, h = box
    return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={colour}@1:t=fill"


async def _clip(
    dest: Path, boxes: list[tuple[tuple[int, int, int, int], str]], frames: int = FRAMES
) -> Path:
    chain = ",".join(_box(box, colour) for box, colour in boxes) or "null"
    await ffmpeg(
        [
            "-f", "lavfi", "-i", f"color=c={GREY}:s={W}x{H}:r={FPS}:d={frames / FPS + 1:.3f}",
            "-vf", chain, "-frames:v", str(frames), *ENCODE, str(dest), "-y",
        ]
    )
    return dest


async def _drifting(dest: Path, *, hands: bool = True, frames: int = FRAMES) -> Path:
    """The rendered window: the face holds, the hands darken to 55 % over
    the clip (`T` is the frame's timestamp, so the boxes move per frame)."""
    fade = f"(1-0.45*min(T*{FPS}/{frames - 1},1))"
    chain = [
        f"drawbox=x={FACE[0]}:y={FACE[1]}:w={FACE[2]}:h={FACE[3]}:color={_rgb(1.0)}@1:t=fill",
        f"drawbox=x={SHADED[0]}:y={SHADED[1]}:w={SHADED[2]}:h={SHADED[3]}:color={_rgb(0.86)}@1:t=fill",
        f"drawbox=x={OUTSIDE[0]}:y={OUTSIDE[1]}:w={OUTSIDE[2]}:h={OUTSIDE[3]}:color={_rgb(0.5)}@1:t=fill",
    ]
    if hands:
        chain.append(
            f"geq=r='if(between(X,{HANDS[0]},{HANDS[0] + HANDS[2]})*"
            f"between(Y,{HANDS[1]},{HANDS[1] + HANDS[3]}),{LIGHT_SKIN[0]}*{fade},r(X,Y))'"
            f":g='if(between(X,{HANDS[0]},{HANDS[0] + HANDS[2]})*"
            f"between(Y,{HANDS[1]},{HANDS[1] + HANDS[3]}),{LIGHT_SKIN[1]}*{fade},g(X,Y))'"
            f":b='if(between(X,{HANDS[0]},{HANDS[0] + HANDS[2]})*"
            f"between(Y,{HANDS[1]},{HANDS[1] + HANDS[3]}),{LIGHT_SKIN[2]}*{fade},b(X,Y))'"
        )
    await ffmpeg(
        [
            "-f", "lavfi", "-i", f"color=c={GREY}:s={W}x{H}:r={FPS}:d={frames / FPS + 1:.3f}",
            "-vf", ",".join(chain), "-frames:v", str(frames), *ENCODE, str(dest), "-y",
        ]
    )
    return dest


async def _source(dest: Path, frames: int = FRAMES, *, darken_hands: bool = False) -> Path:
    boxes = [(SILHOUETTE_FACE, SOURCE_SKIN), (SILHOUETTE_HANDS, SOURCE_SKIN)]
    if darken_hands:
        boxes[1] = (SILHOUETTE_HANDS, "0x9a745c")  # the SOURCE's own hands are in shadow
    return await _clip(dest, boxes, frames)


async def _mean_y(video: Path, box: tuple[int, int, int, int], frame: int) -> float:
    x, y, w, h = box
    inset = 8
    report = (
        await ffmpeg_stdout(
            [
                "-i", str(video), "-vf",
                f"select='eq(n,{frame})',"
                f"crop={w - 2 * inset}:{h - 2 * inset}:{x + inset}:{y + inset},"
                "format=yuv444p,signalstats,metadata=print:file=-",
                "-frames:v", "1", "-f", "null", "-",
            ]
        )
    ).decode()
    return float(
        next(line.split("=")[1] for line in report.splitlines() if "signalstats.YAVG=" in line)
    )


async def _run_hold(tmp_path: Path, *, hands: bool = True, darken_source_hands: bool = False):
    rendered = await _drifting(tmp_path / f"window-{hands}-{darken_source_hands}.mp4", hands=hands)
    clip = await _source(
        tmp_path / f"clip-{darken_source_hands}.mp4", darken_hands=darken_source_hands
    )
    skin = await skin_target(rendered, clip, frames=(0, 5), width=W, height=H)
    source_y = await source_skin_level(clip, frames=(0, 5), width=W, height=H)
    target = HoldTarget.of(skin, source_y)
    readings, planes = await measure_window(
        rendered,
        clip,
        target,
        width=W,
        height=H,
        work_dir=tmp_path,
        tag=f"t{hands}{darken_source_hands}",
    )
    plan = plan_hold(readings, target, first_window=False)
    dest = tmp_path / f"held-{hands}-{darken_source_hands}.mp4"
    if plan.active:
        await ffmpeg(
            hold_apply_args(
                rendered, planes, plan, dest, width=W, height=H, skip=0,
                kept_frames=len(readings), encode=ENCODE,
            )
        )
    return rendered, clip, target, readings, plan, dest


@needs_ffmpeg
async def test_the_darkening_hands_are_brought_back_and_nothing_else_moves(tmp_path: Path) -> None:
    rendered, _clip_path, target, readings, plan, dest = await _run_hold(tmp_path)
    assert plan.active, plan.summary()
    last = FRAMES - 1

    before_hands = await _mean_y(rendered, HANDS, last)
    after_hands = await _mean_y(dest, HANDS, last)
    first_hands = await _mean_y(rendered, HANDS, 0)
    # The hands lost about 45 % over the window; the hold puts most of it back.
    assert before_hands < first_hands - 30
    assert after_hands > before_hands + 20
    assert after_hands <= first_hands + 6

    # The lit face, the room and a dark patch the silhouette never reaches
    # are where they were.
    for box in (FACE, OUTSIDE):
        before = await _mean_y(rendered, box, last)
        after = await _mean_y(dest, box, last)
        assert abs(after - before) < 3, (box, before, after)
    room = (300, 20, 60, 60)
    assert abs(await _mean_y(dest, room, last) - await _mean_y(rendered, room, last)) < 2


@needs_ffmpeg
async def test_the_face_gets_the_same_answer_with_and_without_the_hands(tmp_path: Path) -> None:
    """Why the correction is per pixel: a per-frame scalar through the mask
    lifted this strip by the HANDS' deficit whenever the hands were in the
    frame (measured: 27 Y). Each pixel's own deficit cannot do that."""
    *_, with_hands = await _run_hold(tmp_path)
    rendered_without = await _drifting(tmp_path / "window-nohands.mp4", hands=False)
    clip = await _source(tmp_path / "clip-plain.mp4")
    skin = await skin_target(rendered_without, clip, frames=(0, 5), width=W, height=H)
    source_y = await source_skin_level(clip, frames=(0, 5), width=W, height=H)
    target = HoldTarget.of(skin, source_y)
    readings, planes = await measure_window(
        rendered_without, clip, target, width=W, height=H, work_dir=tmp_path, tag="nohands"
    )
    plan = plan_hold(readings, target, first_window=False)
    without_hands = tmp_path / "held-nohands.mp4"
    if plan.active:
        await ffmpeg(
            hold_apply_args(
                rendered_without, planes, plan, without_hands, width=W, height=H, skip=0,
                kept_frames=len(readings), encode=ENCODE,
            )
        )
    else:
        without_hands = rendered_without

    last = FRAMES - 1
    strip_with = await _mean_y(with_hands, SHADED, last)
    strip_without = await _mean_y(without_hands, SHADED, last)
    assert abs(strip_with - strip_without) < 6, (strip_with, strip_without)


@needs_ffmpeg
async def test_a_window_that_does_not_drift_is_left_alone(tmp_path: Path) -> None:
    rendered = await _clip(
        tmp_path / "steady.mp4",
        [(FACE, _rgb(1.0)), (HANDS, _rgb(1.0)), (OUTSIDE, _rgb(0.5))],
    )
    clip = await _source(tmp_path / "clip-steady.mp4")
    skin = await skin_target(rendered, clip, frames=(0, 5), width=W, height=H)
    source_y = await source_skin_level(clip, frames=(0, 5), width=W, height=H)
    target = HoldTarget.of(skin, source_y)
    readings, _planes = await measure_window(
        rendered, clip, target, width=W, height=H, work_dir=tmp_path, tag="steady"
    )
    plan = plan_hold(readings, target, first_window=False)
    assert not plan.active, plan.summary()


@needs_ffmpeg
async def test_the_first_frame_of_the_first_window_is_the_customers_picture(tmp_path: Path) -> None:
    rendered = await _drifting(tmp_path / "window0.mp4")
    clip = await _source(tmp_path / "clip0.mp4")
    skin = await skin_target(rendered, clip, frames=(0, 5), width=W, height=H)
    source_y = await source_skin_level(clip, frames=(0, 5), width=W, height=H)
    target = HoldTarget.of(skin, source_y)
    readings, planes = await measure_window(
        rendered, clip, target, width=W, height=H, work_dir=tmp_path, tag="w0"
    )
    plan = plan_hold(readings, target, first_window=True)
    assert plan.cap[0] == 0.0
    dest = tmp_path / "held0.mp4"
    await ffmpeg(
        hold_apply_args(
            rendered, planes, plan, dest, width=W, height=H, skip=0,
            kept_frames=len(readings), encode=ENCODE,
        )
    )
    for box in (FACE, HANDS, OUTSIDE):
        assert abs(await _mean_y(dest, box, 0) - await _mean_y(rendered, box, 0)) < 2, box


@needs_ffmpeg
async def test_a_shadow_the_source_shows_is_not_corrected_away(tmp_path: Path) -> None:
    """The target follows the source performer's own skin under the very
    pixels being lifted, so a genuine shadow in the source stays."""
    _r, _c, _t, _readings, bright_plan, _dest = await _run_hold(tmp_path)
    _r2, _c2, _t2, _readings2, shadow_plan, _dest2 = await _run_hold(
        tmp_path, darken_source_hands=True
    )
    assert min(shadow_plan.target) < min(bright_plan.target) - 3, (
        shadow_plan.summary(),
        bright_plan.summary(),
    )


@needs_ffmpeg
async def test_a_silhouette_that_fills_the_frame_switches_the_hold_off(tmp_path: Path) -> None:
    rendered = await _drifting(tmp_path / "window-close.mp4")
    clip = await _clip(tmp_path / "clip-close.mp4", [((0, 0, W, H), SOURCE_SKIN)])
    skin = await skin_target(rendered, clip, frames=(0, 5), width=W, height=H)
    source_y = await source_skin_level(clip, frames=(0, 5), width=W, height=H)
    readings, _planes = await measure_window(
        rendered,
        clip,
        HoldTarget.of(skin, source_y),
        width=W,
        height=H,
        work_dir=tmp_path,
        tag="close",
    )
    assert readings[0].gate > GATE_SOFT_TO
    plan = plan_hold(readings, HoldTarget.of(skin, source_y), first_window=False)
    assert not plan.active, plan.summary()
    assert plan.refused, plan.refused


def test_the_piecewise_expression_interpolates_between_knots() -> None:
    values = tuple(float(index) for index in range(25))
    expression = piecewise(values)
    assert expression.startswith("if(lt(N,12),")
    # Every knot appears as itself, and the tail is the last value.
    assert "0.000+(12.000-0.000)*(N-0)/12" in expression
    assert expression.endswith("12.000+(24.000-12.000)*(N-12)/12,24.000))")
    assert piecewise(()) == "0"
    assert piecewise((7.5,)) == "7.500"


def test_the_plan_summary_carries_what_an_ab_needs() -> None:
    from worker.media.skin_hold import FrameReading

    readings = [
        FrameReading(
            gate=0.1, keyed=0.08, ramped=0.03, y=100.0, cb=115.0, cr=140.0,
            source_y=120.0, lift=30.0,
        )
        for _ in range(30)
    ]
    target = HoldTarget(y=136.0, cb=110.0, cr=150.0, source_y=120.0)
    plan = plan_hold(readings, target, first_window=False)
    summary = plan.summary()
    assert summary["applied"] is True
    assert summary["frames"] == 30
    assert summary["held_frames"] == 30
    assert summary["mean_lift"] == pytest.approx(30.0, abs=0.01)
    assert summary["target_max"] == pytest.approx(136.0, abs=0.01)
    assert summary["cap_min"] == pytest.approx(LUMA_OFFSET_LIMIT, abs=0.01)
    assert summary["refused"] == {}
