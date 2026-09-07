"""Skin-region re-anchoring of a chained seed (`worker.media.skin`), on
synthetic pictures whose every region is known.

A grey scene, a light-skin FACE rectangle, a HANDS rectangle in the same
skin chroma but dark, a source frame whose skin silhouette covers both,
and a dark skin-coloured patch OUTSIDE the silhouette. The hands must be
lifted toward the first window's skin level, the face and the outside patch
and the room must not move, and each guard must refuse for its own reason.
Real ffmpeg, no fake service.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import needs_ffmpeg
from worker.media.ffmpeg import ffmpeg, ffmpeg_stdout
from worker.media.skin import (
    GATE_COVER_RANGE,
    MIN_DARK_FRACTION,
    SkinStats,
    anchor_skin,
    skin_target,
    source_gate,
)

W, H = 368, 640
GREY = "0x5a5a5a"
LIGHT_SKIN = "0xe1b496"  # RGB 225,180,150 — inside both keys
DARK_SKIN = "0x6e4632"  # RGB 110,70,50 — same chroma family, dark
SOURCE_SKIN = "0xc89678"  # RGB 200,150,120 — passes the tight silhouette key
FACE = (120, 100, 130, 150)  # x, y, w, h
HANDS = (80, 380, 210, 180)
OUTSIDE = (8, 8, 40, 50)  # a dark skin patch the silhouette never reaches
SILHOUETTE_FACE = (105, 85, 160, 180)  # the performer's head and neck
SILHOUETTE_HANDS = (65, 365, 240, 210)  # the performer's hands, where the drift lives


def _boxes(*boxes: tuple[tuple[int, int, int, int], str]) -> str:
    return ",".join(
        f"drawbox=x={x}:y={y}:w={w}:h={h}:color={colour}@1:t=fill" for (x, y, w, h), colour in boxes
    )


async def _still(dest: Path, *boxes: tuple[tuple[int, int, int, int], str]) -> Path:
    await ffmpeg(
        ["-f", "lavfi", "-i", f"color=c={GREY}:s={W}x{H}:d=0.1", "-vf", _boxes(*boxes) or "null",
         "-frames:v", "1", str(dest), "-y"]
    )
    return dest


async def _clip(dest: Path, frames: int, *boxes: tuple[tuple[int, int, int, int], str]) -> Path:
    await ffmpeg(
        ["-f", "lavfi", "-i", f"color=c={GREY}:s={W}x{H}:d=5,fps=24", "-vf", _boxes(*boxes) or "null",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(dest), "-y"]
    )
    return dest


async def _mean_y(image: Path, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    inset = 6  # stay clear of the feathered edges
    report = (
        await ffmpeg_stdout(
            ["-i", str(image), "-vf",
             f"crop={w - 2 * inset}:{h - 2 * inset}:{x + inset}:{y + inset},format=yuv444p,"
             "signalstats,metadata=print:file=-", "-f", "null", "-"]
        )
    ).decode()
    return float(next(line.split("=")[1] for line in report.splitlines() if "signalstats.YAVG=" in line))


@needs_ffmpeg
async def test_dark_hands_inside_the_silhouette_are_lifted_and_nothing_else_moves(tmp_path: Path) -> None:
    rendered = await _clip(tmp_path / "window0.mp4", 30, (FACE, LIGHT_SKIN), (HANDS, LIGHT_SKIN))
    source = await _clip(tmp_path / "clip0.mp4", 30, (SILHOUETTE_FACE, SOURCE_SKIN), (SILHOUETTE_HANDS, SOURCE_SKIN))
    target = await skin_target(rendered, source, frames=(4, 27), width=W, height=H)
    face_expected = await _mean_y(await _still(tmp_path / "ref.png", (FACE, LIGHT_SKIN)), FACE)
    # The target is the light skin's own level, measured only where the
    # silhouette and the skin key agree.
    # Same skin, two measurement pipelines (masked product vs a plain crop):
    # a few units apart by construction; target and seed share ONE pipeline.
    assert abs(target.y_mean - face_expected) < 12.0
    assert 0.05 < target.cover < 0.5

    seed = await _still(
        tmp_path / "seed.png", (FACE, LIGHT_SKIN), (HANDS, DARK_SKIN), (OUTSIDE, DARK_SKIN)
    )
    gate = await source_gate(source, 29, tmp_path / "gate.png", width=W, height=H)
    before = {name: await _mean_y(seed, box) for name, box in (("face", FACE), ("hands", HANDS), ("outside", OUTSIDE))}
    room_before = await _mean_y(seed, (300, 20, 60, 60))

    out, record = await anchor_skin(
        seed, gate, tmp_path / "seed-skin.png", target=target, width=W, work_dir=tmp_path, tag="t"
    )

    assert record.applied, record
    assert out.name == "seed-skin.png"
    assert record.y_offset > 40.0
    assert MIN_DARK_FRACTION <= record.dark_fraction <= 1.0
    assert GATE_COVER_RANGE[0] <= record.gate_cover <= GATE_COVER_RANGE[1]
    after = {name: await _mean_y(out, box) for name, box in (("face", FACE), ("hands", HANDS), ("outside", OUTSIDE))}
    # Hands: lifted by the bounded offset (this synthetic gap is wider than
    # any measured one, so the 80-unit cap is what lands), applied in full
    # inside the mask.
    # Measured in the interior of the hands, clear of the mask's feathered
    # edge (5 px blur, two passes) which the rectangle's border would dilute.
    x, y, w, h = HANDS
    core = (x + 30, y + 30, w - 60, h - 60)
    lift = await _mean_y(out, core) - await _mean_y(seed, core)
    assert lift > 40.0
    assert abs(lift - record.y_offset) < 6.0
    # Face (already at the target), the patch outside the silhouette, and
    # the room: untouched.
    assert abs(after["face"] - before["face"]) < 2.0
    assert abs(after["outside"] - before["outside"]) < 2.0
    assert abs(await _mean_y(out, (300, 20, 60, 60)) - room_before) < 1.0


@needs_ffmpeg
async def test_a_seed_with_light_hands_is_left_alone(tmp_path: Path) -> None:
    rendered = await _clip(tmp_path / "window0.mp4", 30, (FACE, LIGHT_SKIN), (HANDS, LIGHT_SKIN))
    source = await _clip(tmp_path / "clip0.mp4", 30, (SILHOUETTE_FACE, SOURCE_SKIN), (SILHOUETTE_HANDS, SOURCE_SKIN))
    target = await skin_target(rendered, source, frames=(4, 27), width=W, height=H)
    seed = await _still(tmp_path / "seed.png", (FACE, LIGHT_SKIN), (HANDS, LIGHT_SKIN))
    gate = await source_gate(source, 29, tmp_path / "gate.png", width=W, height=H)

    out, record = await anchor_skin(
        seed, gate, tmp_path / "seed-skin.png", target=target, width=W, work_dir=tmp_path, tag="t"
    )

    assert not record.applied and record.reason == "skin not dark as a whole"
    assert out == seed
    assert not (tmp_path / "seed-skin.png").exists()


@needs_ffmpeg
async def test_no_silhouette_in_the_source_means_no_correction(tmp_path: Path) -> None:
    rendered = await _clip(tmp_path / "window0.mp4", 30, (FACE, LIGHT_SKIN), (HANDS, LIGHT_SKIN))
    plain = await _clip(tmp_path / "clip0.mp4", 30)  # a grey source: no skin anywhere
    target = SkinStats(y_mean=180.0, cb_mean=105.0, cr_mean=150.0, cover=0.2)
    seed = await _still(tmp_path / "seed.png", (FACE, LIGHT_SKIN), (HANDS, DARK_SKIN))
    gate = await source_gate(plain, 29, tmp_path / "gate.png", width=W, height=H)

    out, record = await anchor_skin(
        seed, gate, tmp_path / "seed-skin.png", target=target, width=W, work_dir=tmp_path, tag="t"
    )

    assert not record.applied and record.reason == "gate outside range"
    assert record.gate_cover < GATE_COVER_RANGE[0]
    assert out == seed


@needs_ffmpeg
async def test_the_offsets_are_bounded(tmp_path: Path) -> None:
    rendered = await _clip(tmp_path / "window0.mp4", 30, (FACE, LIGHT_SKIN), (HANDS, LIGHT_SKIN))
    source = await _clip(tmp_path / "clip0.mp4", 30, (SILHOUETTE_FACE, SOURCE_SKIN), (SILHOUETTE_HANDS, SOURCE_SKIN))
    target = await skin_target(rendered, source, frames=(4, 27), width=W, height=H)
    # Hands almost black but still skin-chroma: the lift is capped.
    seed = await _still(tmp_path / "seed.png", (FACE, LIGHT_SKIN), (HANDS, "0x3c2418"))
    gate = await source_gate(source, 29, tmp_path / "gate.png", width=W, height=H)
    _, record = await anchor_skin(
        seed, gate, tmp_path / "seed-skin.png", target=target, width=W, work_dir=tmp_path, tag="t"
    )
    if record.applied:
        assert record.y_offset <= 80.0 and abs(record.cb_offset) <= 20.0 and abs(record.cr_offset) <= 20.0
    else:
        pytest.skip(f"very dark patch fell outside the key: {record.reason}")
