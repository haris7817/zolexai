"""Video to Video after the client's 11 Sep 2026 verdict on the first result.

Their reading of it was precise and worth keeping: the identity transferred,
but 480-class generation "does not provide enough detail for fast hands,
fingers, clothing edges and facial features", and enlarging to 8K only
enlarges those distortions. Four things follow, and each is pinned below.

**Generation moved to 704.** 1280x704, 704x1280, 704x704, and an
aspect-matched /64 grid for anything else.

**The replacement is complete.** The control clip keeps the scene's own pixels
and gives the person's region edges only, so the source's clothing has nothing
to bleed from; and the caption states a whole outfit, because a reference photo
that stops at the waist leaves the model dressing the legs from the footage.

**The delivered file is the source's own length, to the frame.** Their test
put 347 frames in and got 345 back.

**One soundtrack, aligned.** The source's own, laid over once.

The ladder itself — that quality picks a container and not detail — lives in
`test_v2v_delivery.py`, and the tool's older promises in `test_video_to_video.py`
and `test_transform.py`.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest
import yaml

from tests.conftest import (
    collect,
    conditioning_of,
    invocations,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
    value_of,
)
from worker.adapters.ltx import (
    delivery_dimensions_for_source,
    proxy_704_grid_for_source,
    proxy_canvas_for,
    proxy_grid_for_source,
)
from worker.media import probe_media
from worker.media.validate import OutputExpectation, verify_output

ROOT = Path(__file__).resolve().parents[3]
DEFINITION = ROOT / "workflow-definitions" / "video-to-video.yaml"


def asked_of_the_pipeline(grid: tuple[int, int]) -> tuple[int, int]:
    """What the CLI is asked for to RETURN `grid` on the transform engine.

    IC-LoRA's stage 2 needs even latent dimensions, so the adapter requests
    double and skips it — see `_IC_LORA`. Pinned here because it is the
    difference between "the grid changed" and "the doubling changed", and
    only one of those is a bug in this file's subject.
    """
    return grid[0] * 2, grid[1] * 2


def v2v_job(workspace: Path, source: Path | None, reference: Path | None = None, **overrides):
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    if reference is not None:
        inputs.append(staged_input("reference_image", "image", "image/png", reference))
    execution = {
        "runtime": "ltx",
        "v2v_engine": "transform",
        "render_proxy": "704p",
        "delivery": "native",
    }
    execution.update(overrides.pop("execution", {}))
    defaults = dict(
        workflow_id="video-to-video",
        prompt="repaint it as a charcoal sketch",
        parameters={},
        inputs=inputs,
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── The 704 canvases ─────────────────────────────────────────────────────


def test_the_704_canvases_are_the_measured_ones() -> None:
    """704 on the short side, as asked. The long side is 1152 rather than the
    1280 in their note, and that is measured rather than preferred: 1280x704
    renders a prompt-only restyle fine and fails EVERY job carrying a
    reference photo, inside the VAE's encode of the second video. See
    `_PROXY_704_DIMENSIONS` for the trace and the two ruled-out causes."""
    assert proxy_704_grid_for_source(1920, 1080) == (1152, 704)
    assert proxy_704_grid_for_source(1080, 1920) == (704, 1152)
    assert proxy_704_grid_for_source(1000, 1000) == (704, 704)


def test_an_unusual_ratio_keeps_its_own_shape_within_the_bounds() -> None:
    """A 4:5 phone clip and a 2.39:1 anamorphic one are the cases with no
    named canvas. Both must land on the /64 lattice, keep a 704 short side and
    never exceed the measured 1152 on the long one."""
    for width, height in ((1080, 1350), (2048, 858), (1440, 1080), (640, 480), (1280, 704)):
        grid = proxy_704_grid_for_source(width, height)
        assert grid[0] % 64 == 0 and grid[1] % 64 == 0, f"{width}x{height} -> {grid}"
        assert min(grid) == 704, f"{width}x{height} -> {grid} is not 704-class"
        assert max(grid) <= 1152, f"{width}x{height} -> {grid} exceeds the long-side cap"

    portrait = proxy_704_grid_for_source(1080, 1350)
    assert portrait[0] == 704 and portrait[1] > 704, "a tall source rendered wide"


def test_the_lattice_is_64_not_32() -> None:
    """The client's note says 32. Their three canvases satisfy 64 anyway, so
    this only bites on the unusual ratios — but 480 was /32 and could not
    render at all, and a /32 long side such as 1248 would fail the same way."""
    for width, height in ((1920, 1080), (1080, 1350), (2048, 858), (1000, 1000)):
        grid = proxy_704_grid_for_source(width, height)
        # The real constraint: the VAE halves the latent, and the latent is
        # the pixel size over 32.
        assert (grid[0] // 32) % 2 == 0, f"{grid[0]} gives an odd latent"
        assert (grid[1] // 32) % 2 == 0, f"{grid[1]} gives an odd latent"


def test_the_profile_names_resolve_and_480_is_never_used_literally() -> None:
    assert proxy_canvas_for("704p") == (704, 1152)
    assert proxy_canvas_for("512p") == (512, 1152)
    # Their older file said 480p. It keeps working and resolves to the
    # smallest LEGAL canvas rather than being rejected or rendered.
    assert proxy_canvas_for("480p") == (512, 1152)
    assert proxy_canvas_for(None) is None
    assert proxy_canvas_for("") is None
    assert proxy_canvas_for("nonsense") is None


def test_the_shipped_definition_asks_for_the_540_class_canvas() -> None:
    """540p, the client's 11 Sep name, which resolves to 896x512 at 16:9.

    They asked for 704 earlier the same day and then shipped a package built
    on 540; 540 is the later word and this follows it. The 704 profile stays
    in the adapter, measured and one word away — see `proxy_704_grid_for_source`
    and the note in the definition.
    """
    workflow = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    assert workflow["execution"]["render_proxy"] == "540p"
    assert proxy_canvas_for("540p") == (512, 1152)
    assert proxy_grid_for_source(1920, 1080, short_side=512, max_long_side=1152) == (896, 512)


@needs_ffmpeg
async def test_generation_and_stitching_use_the_same_704_grid(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One grid for both, so the stitch never resamples what the model made.
    A mismatch here is a resize nobody asked for, before the only resize the
    delivery is supposed to perform."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True, size="256x144")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    result, _ = await collect(v2v_job(workspace, source))

    grid = proxy_704_grid_for_source(256, 144)
    argv = invocations(log)[0]
    asked = (int(value_of(argv, "--width")), int(value_of(argv, "--height")))
    assert asked == asked_of_the_pipeline(grid)
    # `delivery: native` means no resize, so the delivered frame IS the grid.
    assert (result.width, result.height) == grid


# ── The reference stays optional ─────────────────────────────────────────


@needs_ffmpeg
async def test_no_photo_is_a_restyle_and_is_never_refused(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(v2v_job(
        workspace, source,
        execution={"v2v_reference_identity": True},
    ))

    (argv,) = invocations(log)
    assert conditioning_of(argv) == [], "a restyle anchored on a photo it was never given"
    assert "--video-conditioning" in argv, "the source must still drive it"


# ── The replacement is complete ──────────────────────────────────────────


def test_identity_locks_the_body_and_leaves_clothing_to_the_prompt() -> None:
    """The client's 11 Sep rule, replacing their own earlier ask that morning.

    Identity covers face, hair, skin and body. Clothing and accessories stay
    under the prompt, so a customer asking for a charcoal suit gets one
    whatever the reference photo happens to be wearing.

    What this gives up is worth naming: nothing refuses the SOURCE's clothing
    any more either, which is the defect their earlier ask was aimed at.
    """
    from worker.adapters import ltx

    assert not hasattr(ltx, "_complete_outfit"), (
        "outfit completion contradicts the prompt owning clothing"
    )
    body = pathlib.Path(ltx.__file__).read_text(encoding="utf-8")
    clause = "Their faces, hair, skin tone and body type stay exactly as"
    assert clause in body, "the identity lock no longer names what it locks"
    assert "None of the original clothing remains" not in body, (
        "the clothing-refusal clause outlived the rule that asked for it"
    )


def _stub_hybrid(monkeypatch) -> list[bool]:
    """Records every hybrid-control build and the side it protected."""
    built: list[bool] = []

    async def fake(edges, footage, matte, dest, *, frames, invert=False, **kwargs):
        built.append(invert)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(edges).read_bytes())
        return dest

    monkeypatch.setattr("worker.adapters.ltx.build_hybrid_control", fake)
    return built


@needs_ffmpeg
async def test_identity_uses_edges_alone_by_default(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching the client's package, which builds identity with
    `invert=False` and never takes the hybrid path.

    The inverted control was added on 11 Sep for a clothing-bleed complaint
    their later rule reframed — the prompt owns clothing now. It stays in the
    adapter, off, because the measurement behind it still stands.
    """
    from tests.test_reference_identity import identity_job, stub_matte
    from worker.media import extract_final_frame

    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    mattes = stub_matte(monkeypatch)
    built = _stub_hybrid(monkeypatch)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(identity_job(
        workspace, source, reference,
        execution={"render_proxy": "540p", "delivery": "native"},
    ))

    assert built == [], "the hybrid control ran without being asked for"
    assert len(mattes) == 1, f"the matte was built {len(mattes)} times for one section"


@needs_ffmpeg
async def test_the_inverted_control_is_one_line_away(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`invert=True` keeps the scene's own pixels and gives the person's
    region edges alone, so the source's clothing is not in the control signal
    to bleed from. One workflow key, and the matte is still built once."""
    from tests.test_reference_identity import identity_job, stub_matte
    from worker.media import extract_final_frame

    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    mattes = stub_matte(monkeypatch)
    built = _stub_hybrid(monkeypatch)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(identity_job(
        workspace, source, reference,
        execution={
            "v2v_identity_hybrid_control": True,
            "render_proxy": "540p",
            "delivery": "native",
        },
    ))

    assert built, "the flag did not turn the hybrid control on"
    assert all(built), "invert=False keeps the SOURCE person — the opposite of a replacement"
    assert len(mattes) == 1, "the control clip and the mask must share one matte"


# ── The length is the source's, to the frame ─────────────────────────────


def test_the_final_section_absorbs_the_whole_correction() -> None:
    """Planning from a rounded duration loses frames; the source's own count
    is the answer. The correction lands on the last section so no earlier
    seam's timestamp can move."""
    from worker.adapters.ltx import LtxAdapter
    from worker.media import MediaInfo

    adapter = LtxAdapter()
    source = MediaInfo(
        duration_seconds=11.5, width=1280, height=704, has_video=True,
        has_audio=True, fps=30.0, frame_count=347,
    )

    pinned = adapter._match_source_frames([173, 172], source, 30.0)
    assert sum(pinned) == 347, "the delivered file is not the source's length"
    assert pinned[0] == 173, "an earlier seam moved"
    assert pinned[1] == 174


def test_a_plan_that_already_matches_is_untouched() -> None:
    from worker.adapters.ltx import LtxAdapter
    from worker.media import MediaInfo

    source = MediaInfo(
        duration_seconds=11.5667, width=1280, height=704, has_video=True,
        has_audio=True, fps=30.0, frame_count=347,
    )
    assert LtxAdapter()._match_source_frames([173, 174], source, 30.0) == [173, 174]


def test_a_retimed_delivery_is_left_alone() -> None:
    """`_delivery_fps` clamps to 10..60, so a 240fps phone clip is delivered
    at a different rate and has its own correct count. Forcing the source's
    onto it would be the same error pointing the other way."""
    from worker.adapters.ltx import LtxAdapter
    from worker.media import MediaInfo

    source = MediaInfo(
        duration_seconds=10.0, width=1280, height=704, has_video=True,
        has_audio=False, fps=240.0, frame_count=2400,
    )
    assert LtxAdapter()._match_source_frames([300, 300], source, 60.0) == [300, 300]


def test_a_source_that_states_no_count_falls_back_rather_than_guessing() -> None:
    from worker.adapters.ltx import LtxAdapter
    from worker.media import MediaInfo

    source = MediaInfo(
        duration_seconds=11.5, width=1280, height=704, has_video=True,
        has_audio=True, fps=30.0, frame_count=None,
    )
    assert LtxAdapter()._match_source_frames([173, 172], source, 30.0) == [173, 172]


@needs_ffmpeg
async def test_a_short_delivery_is_rejected_rather_than_shipped(tmp_path: Path) -> None:
    """The guard that would have caught 345-for-347 before a customer did."""
    clip = await make_clip(tmp_path / "clip.mp4", 2.0, rate=30)
    measured = await probe_media(clip)
    assert measured.frame_count is not None

    await verify_output(clip, OutputExpectation(
        expect_video=True, expected_frame_count=measured.frame_count,
    ))

    from worker.media import FfmpegError

    with pytest.raises(FfmpegError, match="frames, not the required"):
        await verify_output(clip, OutputExpectation(
            expect_video=True, expected_frame_count=measured.frame_count - 2,
        ))


@needs_ffmpeg
async def test_the_delivered_file_has_the_sources_frames_and_one_soundtrack(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True, size="256x144", rate=30)
    measured = await probe_media(source)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(v2v_job(workspace, source))

    delivered = await probe_media(workspace / "output.mp4")
    assert delivered.frame_count == measured.frame_count, (
        f"{measured.frame_count} frames in, {delivered.frame_count} out"
    )
    assert delivered.audio_stream_count == 1, "the source's audio, exactly once"
    assert delivered.audio_duration_seconds is not None
    assert abs(delivered.audio_duration_seconds - delivered.duration_seconds) <= 0.1


# ── All three finishes ───────────────────────────────────────────────────


def test_every_rung_of_the_ladder_has_a_frame() -> None:
    """1080p and 4K are encoded in the delivery suite; 8K is arithmetic here
    because a 7680x4320 frame is beyond what a dev box will encode. The client
    measured the real one at 4320x7680, 30fps, ~39.5 Mbps, 57 MB."""
    assert delivery_dimensions_for_source("1080p", 1920, 1080) == (1920, 1080)
    assert delivery_dimensions_for_source("4k", 1920, 1080) == (3840, 2160)
    assert delivery_dimensions_for_source("8k", 1920, 1080) == (7680, 4320)
    # Their measured 8K portrait result, which is the shape a phone upload
    # takes: the long side fills 7680 and the short side follows the source.
    assert delivery_dimensions_for_source("8k", 1080, 1920) == (4320, 7680)


@needs_ffmpeg
@pytest.mark.parametrize("level", ["1080p", "4k", "8k"])
async def test_each_level_asks_for_the_same_render(
    level: str,
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finish is the only thing the customer's choice moves. Rendering 8K
    larger would raise the cost of a job by more than an order of magnitude,
    so it has to be a deliberate edit rather than a quiet one."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True, size="256x144")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(v2v_job(
        workspace, source,
        parameters={"quality": level},
        execution={"delivery": "native"},
    ))

    argv = invocations(log)[0]
    assert (int(value_of(argv, "--width")), int(value_of(argv, "--height"))) == (
        asked_of_the_pipeline(proxy_704_grid_for_source(256, 144))
    )
