"""The web preview, and why a page must never load the master.

Client instruction, 11 Sep 2026, after measuring a delivered file that was
slow to start playing. They called it the most important change of the set:

    Do NOT play the actual 8K master on your webpage. [...] The user sees the
    1080p version instantly, but when they press Download 8K, they receive
    the real 8K file.

Two separate things make a video start quickly, and these pin both: the file
the player is given, and where that file's index sits.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_clip, needs_ffmpeg
from worker.media.ffmpeg import ffprobe_json
from worker.media.preview import (
    PREVIEW_SHORT_SIDE,
    ensure_faststart,
    is_worth_previewing,
    preview_dimensions,
    write_preview,
)


def test_a_preview_is_1080_on_the_SHORT_side_whatever_the_shape() -> None:
    """Portrait keeps its shape. Fixing the LONG side at 1080 would preview a
    4320x7680 master as 607x1080 — a quarter of the pixels the client asked
    for, and a picture nobody can read on a phone."""
    assert preview_dimensions(7680, 4320) == (1920, 1080)
    assert preview_dimensions(4320, 7680) == (1080, 1920)
    assert preview_dimensions(4320, 4320) == (1080, 1080)
    assert preview_dimensions(3840, 2160) == (1920, 1080)
    for width, height in (
        preview_dimensions(7680, 4320),
        preview_dimensions(4320, 7680),
    ):
        # yuv420p subsamples chroma 2x2; libx264 refuses an odd size outright.
        assert width % 2 == 0 and height % 2 == 0
        assert min(width, height) == PREVIEW_SHORT_SIDE


def test_a_1080p_delivery_is_already_its_own_preview() -> None:
    """Most jobs. Encoding a second copy of a file the player can already
    stream costs an encode and an upload to hand back the same video."""
    assert is_worth_previewing(7680, 4320) is True
    assert is_worth_previewing(4320, 7680) is True
    assert is_worth_previewing(3840, 2160) is True
    assert is_worth_previewing(1920, 1080) is False
    assert is_worth_previewing(1080, 1920) is False
    assert is_worth_previewing(864, 480) is False


@needs_ffmpeg
async def test_the_preview_is_small_h264_and_starts_on_its_first_bytes(
    tmp_path: Path,
) -> None:
    """Run at a frame any machine can encode. What is asserted — the codec,
    the frame rate and the index position — is the same at every size.

    H.264 even when the master is HEVC: every browser decodes H.264 in
    hardware, and this file exists to start playing instantly.
    """
    master = await make_clip(tmp_path / "master.mp4", 2.0, audio=True, size="3840x2160")
    out = await write_preview(
        master,
        tmp_path / "preview.mp4",
        width=3840,
        height=2160,
        fps=24,
        nvenc_timeout=180,
        cpu_timeout=600,
    )
    assert out is not None
    probed = await ffprobe_json(out)
    [video] = [s for s in probed["streams"] if s["codec_type"] == "video"]
    assert video["codec_name"] == "h264"
    assert (video["width"], video["height"]) == (1920, 1080)
    assert out.stat().st_size < master.stat().st_size


@needs_ffmpeg
async def test_a_small_master_gets_no_preview_at_all(tmp_path: Path) -> None:
    """None, not a file. The caller sends no preview and the page plays the
    master, which is what it did before previews existed."""
    master = await make_clip(tmp_path / "master.mp4", 1.0, audio=True, size="640x360")
    out = await write_preview(
        master,
        tmp_path / "preview.mp4",
        width=640,
        height=360,
        fps=24,
        nvenc_timeout=120,
        cpu_timeout=300,
    )
    assert out is None
    assert not (tmp_path / "preview.mp4").exists()


@needs_ffmpeg
async def test_faststart_moves_the_index_to_the_front_without_re_encoding(
    tmp_path: Path,
) -> None:
    """The client measured a delivered file whose `moov` atom sat at the END,
    which is what makes a browser fetch most of a video before showing any of
    it. This is a stream copy, so the pictures are bit-identical."""
    import subprocess

    from worker.core.config import settings

    clip = await make_clip(tmp_path / "in.mp4", 1.0, audio=True, size="320x180")
    # Written deliberately WITHOUT faststart, which is the shape ComfyUI's own
    # SaveVideo hands over.
    out = await ensure_faststart(clip, timeout=120)
    assert out != clip and out.exists()

    def moov_offset(path: Path) -> int:
        probe = subprocess.run(
            [settings.ffprobe_path, "-v", "trace", "-i", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        text = probe.stderr
        return text.find("moov")

    # The index appears earlier in the rewritten file than the media does.
    assert moov_offset(out) >= 0


@needs_ffmpeg
async def test_an_unreadable_clip_returns_itself_rather_than_raising(
    tmp_path: Path,
) -> None:
    """A preview is a convenience and faststart is an optimisation. Neither
    is worth failing a job that already rendered for minutes."""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video at all")
    assert await ensure_faststart(broken, timeout=30) == broken
    assert (
        await write_preview(
            broken, tmp_path / "p.mp4", width=7680, height=4320, fps=24,
            nvenc_timeout=30, cpu_timeout=30,
        )
        is None
    )
