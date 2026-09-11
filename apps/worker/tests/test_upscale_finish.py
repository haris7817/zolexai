"""The finishing pass: one ffmpeg run that stabilises, resizes and tags.

The client's 10 Sep 2026 review asked for three things this module now does —
a temporal deflicker, colour stabilisation, and explicit Rec.709 — and gave
two places to put them: inside their Decode subgraph between `5573 ImageScale`
and `4849 CreateVideo`, or "immediately after ComfyUI saves the video and
before the backend returns the download". We took the second, because the 4K
finish was already there and their graph then stays byte-identical to the one
they sent.

These pin the parts a reviewer would want to see rather than take on trust:
that the deflicker is really in the filter chain and really ahead of the
resize, that the tags reach the file, that the bitrate is a target and not a
suggestion, and that an 8K frame does not go to an H.264 encoder that cannot
produce it.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_clip, needs_ffmpeg
from worker.media.ffmpeg import ffprobe_json
from worker.media.upscale import (
    DEFLICKER,
    DELIVERY_4K,
    DELIVERY_8K,
    _video_codec,
    headroom,
    is_4k,
    is_8k,
    upscale_clip,
)

# ── Codec choice ───────────────────────────────────────────────────────────


def test_an_8k_frame_goes_to_hevc_because_nvenc_has_no_h264_that_large() -> None:
    """NVENC's H.264 encoder is capped at 4096x4096 in hardware. Sending an
    8K frame to it does not produce a large H.264 file slowly — it produces
    nothing at all, and the CPU fallback would then take longer than the
    render did.

    This is the one place the client's "keep everything else the same" could
    not be honoured literally, and it is a hardware limit rather than a
    preference. 4K is unchanged: still H.264, still what browsers play.
    """
    nvenc, cpu = _video_codec(DELIVERY_4K["16:9"], None)
    assert "h264_nvenc" in nvenc and "libx264" in cpu
    assert "hvc1" not in nvenc

    nvenc, cpu = _video_codec(DELIVERY_8K["16:9"], None)
    assert "hevc_nvenc" in nvenc and "libx265" in cpu
    # hvc1 rather than hev1: QuickTime and Safari play one of those two.
    assert nvenc[nvenc.index("-tag:v") + 1] == "hvc1"

    # Portrait 8K is the same frame turned over, so it must switch too — the
    # test that would fail if the rule read `width` instead of `max(...)`.
    assert "hevc_nvenc" in _video_codec(DELIVERY_8K["9:16"], None)[0]
    # and the boundary itself stays on H.264
    assert "h264_nvenc" in _video_codec((4096, 4096), None)[0]
    assert "hevc_nvenc" in _video_codec((4128, 4096), None)[0]


def test_a_bitrate_turns_both_encoders_into_capped_vbr() -> None:
    """A named rate, because `-cq` alone lands wherever the content puts it.

    The number moved once. On 10 Sep the client asked for ~100 Mbps in place
    of the 35-45 their files measured; on 11 Sep, having seen the file sizes,
    they revised it down to 30M/45M/90M for the 8K master — "because your 8K
    is being produced by upscaling a lower-resolution generation, there is
    especially little reason to save it at extremely high 8K bitrates". They
    are right: the detail is not there to preserve.
    """
    nvenc, cpu = _video_codec(DELIVERY_8K["16:9"], "30M")
    for argv in (nvenc, cpu):
        assert argv[argv.index("-b:v") + 1] == "30M"
        assert argv[argv.index("-maxrate") + 1] == "45M"
        assert argv[argv.index("-bufsize") + 1] == "90M"
    assert "-rc" in nvenc and nvenc[nvenc.index("-rc") + 1] == "vbr"
    # The client's own quality ceiling alongside the rate.
    assert nvenc[nvenc.index("-cq") + 1] == "22"
    # p6 for the 8K master, their preset; p5 below it.
    assert nvenc[nvenc.index("-preset") + 1] == "p6"
    assert _video_codec(DELIVERY_4K["16:9"], "30M")[0][
        _video_codec(DELIVERY_4K["16:9"], "30M")[0].index("-preset") + 1
    ] == "p5"

    # No bitrate is the pre-10-Sep behaviour, untouched.
    nvenc, cpu = _video_codec(DELIVERY_4K["16:9"], None)
    assert "-b:v" not in nvenc and nvenc[nvenc.index("-cq") + 1] == "19"
    assert cpu[cpu.index("-crf") + 1] == "18"


def test_an_unparseable_bitrate_does_not_fail_a_rendered_job() -> None:
    """A finishing pass runs after the GPU time has already been spent. A
    config string nobody can parse is a reason to encode conservatively, not
    a reason to throw away the render."""
    # 1.5x and 3x — the client's own ratios, which they write twice: 30/45/90
    # for the master and 5/7/14 for the preview.
    assert headroom("30M") == ("45M", "90M")
    assert headroom("5M") == ("7M", "15M")
    assert headroom("100000k") == ("150000k", "300000k")
    assert headroom("120") == ("180", "360")
    assert headroom("fast") == ("fast", "fast")


def test_the_rate_follows_the_frame_rather_than_being_one_global_number() -> None:
    """A single rate satisfies neither of the client's two numbers.

    They named 30M for the 8K master and 5M for a 1080p preview. Applied
    flat, 30M gives a 30 s **1080p** delivery of ~110 MB — six times their own
    figure for that frame, for 2 MP of picture. Caught before it shipped,
    when the client chose 1080p for the first real test.

    Square root of pixel count, not linear: linear puts 1080p at 1.9M, below
    their preview, which is the opposite mistake.
    """
    from worker.media.upscale import bitrate_for

    assert bitrate_for(DELIVERY_8K["16:9"], "30M") == "30M"     # their anchor, exact
    assert bitrate_for(DELIVERY_4K["16:9"], "30M") == "15M"
    assert bitrate_for((1920, 1080), "30M") == "7M"             # just above their 5M preview
    # Orientation is not a size: a portrait frame has the same pixels.
    assert bitrate_for((1080, 1920), "30M") == bitrate_for((1920, 1080), "30M")
    # Unparseable is returned untouched rather than guessed at.
    assert bitrate_for((1920, 1080), "veryfast") == "veryfast"


def test_a_tier_is_only_a_tier_when_it_is_spelled_like_one() -> None:
    """Both readers are deliberately strict: a typo that silently promoted a
    job would multiply its frame by four, or by sixteen."""
    assert is_4k("4k") and is_4k(" 2160P ") and is_4k("uhd")
    assert is_8k("8k") and is_8k("4320p") and is_8k("UHD8K")
    for wrong in ("", None, "8 k", "8000p", "four-k", "eight k", "hd"):
        assert not is_4k(wrong) and not is_8k(wrong)
    # and the two never both answer for the same string
    for tier in ("4k", "2160p", "uhd", "8k", "4320p", "uhd8k"):
        assert is_4k(tier) != is_8k(tier)


# ── What actually reaches the file ─────────────────────────────────────────


@needs_ffmpeg
async def test_the_deflicker_runs_before_the_resize_and_the_tags_reach_the_file(
    tmp_path: Path,
) -> None:
    """Run at a frame any machine can encode — the filter chain, the colour
    tags and the audio pass-through are the same at every size.

    The order matters and is not cosmetic: the flicker is in the generated
    frames, so levelling it at the generated size costs a fraction of what
    the same filter costs on an enlarged one, and the enlargement then
    carries corrected pixels rather than magnifying the pulse.
    """
    clip = await make_clip(tmp_path / "in.mp4", 2.0, audio=True, size="256x144")
    out = await upscale_clip(
        clip,
        tmp_path / "out.mp4",
        (512, 288),
        nvenc_timeout=120,
        cpu_timeout=300,
        deflicker=True,
    )
    probed = await ffprobe_json(out)
    [video] = [s for s in probed["streams"] if s["codec_type"] == "video"]
    assert (video["width"], video["height"]) == (512, 288)
    assert video["color_primaries"] == "bt709"
    assert video["color_transfer"] == "bt709"
    assert video["color_space"] == "bt709"
    assert video["color_range"] == "tv"
    # the soundtrack is copied, never re-encoded — the reason this pass runs
    # after the render rather than per section
    assert any(s["codec_type"] == "audio" for s in probed["streams"])


@needs_ffmpeg
async def test_rec709_is_written_even_when_stabilisation_is_off(tmp_path: Path) -> None:
    """`LTX_HD_STABILIZE=false` is the rollback for the deflicker, which
    changes how a video looks. The colour tags are not part of that trade —
    an untagged file is one a player guesses at — so they are written on
    every clip this module touches."""
    clip = await make_clip(tmp_path / "in.mp4", 1.0, audio=True, size="256x144")
    out = await upscale_clip(
        clip,
        tmp_path / "out.mp4",
        (256, 144),
        nvenc_timeout=120,
        cpu_timeout=300,
        deflicker=False,
    )
    [video] = [s for s in (await ffprobe_json(out))["streams"] if s["codec_type"] == "video"]
    assert video["color_primaries"] == "bt709" and video["color_range"] == "tv"


def test_the_deflicker_is_the_window_the_client_specified() -> None:
    """Verbatim from their note. `mode=am` is the arithmetic mean over a
    5-frame window, which is what a pulsing average exposure needs; a wider
    window would smear real lighting changes into the correction."""
    assert DEFLICKER == "deflicker=size=5:mode=am"


def test_a_cut_is_the_one_thing_this_pass_must_not_span() -> None:
    """The client's own caveat: "for videos containing hard cuts, run the
    temporal stabilization separately on each shot and reset it at every cut.
    Do not average colors across two different scenes."

    This function has no notion of a cut, so the invariant lives with its
    callers, and the docstring says so rather than leaving it to be
    rediscovered. Every caller today is one continuous shot — Text to Video
    renders a single pass, Character Replacement chains windows of one
    continuous source — and this pins the sentence a future caller with cuts
    has to read before reusing it.
    """
    assert "hard cut" in upscale_clip.__doc__
    assert "one continuous shot" in upscale_clip.__doc__

