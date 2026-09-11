"""Video to Video: generate on a proxy grid, deliver at a chosen size.

The client's 10 Sep 2026 package replaced the Fast/Best pair with a delivery
ladder — 1080p, 4K, 8K — and put generation on a fixed small grid underneath
all three. Their grid was 480-class and could not render at all; see
`test_every_proxy_side_is_divisible_by_64`. Two things follow, both pinned
here.

**The quality control buys a container, not detail.** Every level renders the
same picture; only the final resize differs. `test_every_quality_level_renders
_the_same_picture` states that in the one place a future change would have to
edit it deliberately, because the alternative — quietly making 8K render
larger — would change the cost of a job by an order of magnitude while
looking like a bug fix.

**The source keeps its own shape.** V2V has no aspect selector, so a 4:5 phone
clip must not be cropped to 16:9 on the way in (the proxy grid) or on the way
out (the delivery frame). Both are computed from the upload.

The tool's older promises are unchanged and are asserted in
`test_video_to_video.py` and `test_transform.py`: the result is the source's
length, and the source's own audio goes back on exactly once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.conftest import (
    collect,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
)
from worker.adapters.base import AdapterError
from worker.adapters.ltx import (
    LtxAdapter,
    delivery_dimensions_for_source,
    output_dimensions,
    proxy_grid_for_source,
)
from worker.media import probe_media

ROOT = Path(__file__).resolve().parents[3]


def v2v_job(workspace: Path, source: Path | None, reference: Path | None = None, **overrides):
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    if reference is not None:
        inputs.append(staged_input("reference_image", "image", "image/png", reference))
    defaults = dict(
        workflow_id="video-to-video",
        prompt="repaint it as a charcoal sketch",
        parameters={},
        inputs=inputs,
        execution={"runtime": "ltx", "v2v_engine": "transform"},
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── The proxy grid ───────────────────────────────────────────────────────


def test_every_proxy_side_is_divisible_by_64() -> None:
    """The constraint that killed the client's own grid on the GPU.

    IC-LoRA encodes the reference video through the VAE, which halves the
    latent, and the latent is the pixel size over 32. A side of 480 gives an
    odd latent of 15 and the encoder dies on it — measured 10 Sep 2026:

        Input tensor shape: torch.Size([1, 1024, 50, 15, 26])
        Shape mismatch, can't divide axis of length 15 in chunks of 2

    So /64 is not a preference here, and a future "make it cheaper" change
    that reaches for 480 again fails this test rather than every job.
    """
    for width, height in (
        (1920, 1080), (1080, 1920), (1000, 1000), (1080, 1350),
        (1280, 704), (2048, 858), (640, 480), (1440, 1080),
    ):
        grid = proxy_grid_for_source(width, height)
        assert grid[0] % 64 == 0, f"{width}x{height} -> {grid[0]} is not /64"
        assert grid[1] % 64 == 0, f"{width}x{height} -> {grid[1]} is not /64"
        assert min(grid) == 512, f"{width}x{height} -> {grid} is not 512-class"


def test_the_proxy_grid_follows_the_sources_shape_rather_than_a_product_ratio() -> None:
    """A 4:5 phone clip is the case that matters: cropping it to 16:9 on the
    way in would throw away the top and bottom of every frame before the model
    ever saw them."""
    assert proxy_grid_for_source(1920, 1080) == (896, 512)
    assert proxy_grid_for_source(1080, 1920) == (512, 896)

    portrait = proxy_grid_for_source(1080, 1350)
    assert portrait[0] == 512 and portrait[1] > 512
    # As close to 4:5 as the /64 lattice reaches at this short side.
    assert abs(portrait[0] / portrait[1] - 0.8) < 0.06

    square = proxy_grid_for_source(1000, 1000)
    assert square == (512, 512)


def test_an_unprobeable_source_still_gets_a_grid() -> None:
    """The widest legal landscape shape of the profile, which is what the
    named table says for 16:9. A source nobody could measure is far more
    likely to be landscape than square."""
    assert proxy_grid_for_source(None, None) == (1152, 512)
    assert proxy_grid_for_source(0, 0) == (1152, 512)


# ── The delivery frame ───────────────────────────────────────────────────


def test_each_level_fills_its_own_box() -> None:
    assert delivery_dimensions_for_source("1080p", 1920, 1080) == (1920, 1080)
    assert delivery_dimensions_for_source("4k", 1920, 1080) == (3840, 2160)
    assert delivery_dimensions_for_source("8k", 1920, 1080) == (7680, 4320)
    assert delivery_dimensions_for_source("4k", 1080, 1920) == (2160, 3840)


def test_delivery_preserves_the_sources_aspect_instead_of_cropping_it() -> None:
    """The 4:5 case again, at the other end of the pipeline."""
    width, height = delivery_dimensions_for_source("4k", 1080, 1350)
    assert abs(width / height - 0.8) < 0.01, f"{width}x{height} is not 4:5"
    # Fits inside the 4K box rather than overflowing it.
    assert max(width, height) <= 3840 and min(width, height) <= 2160


def test_every_delivery_frame_is_even() -> None:
    """yuv420p subsamples chroma 2x2 and libx264 refuses odd sizes."""
    for profile in ("1080p", "4k", "8k"):
        for shape in ((1920, 1080), (1080, 1350), (1440, 1080), (999, 501)):
            width, height = delivery_dimensions_for_source(profile, *shape)
            assert width % 2 == 0 and height % 2 == 0, f"{profile} {shape} -> {width}x{height}"


def test_native_is_exactly_the_previous_behaviour() -> None:
    """The default. A workflow that names no delivery size must be untouched
    by any of this — the tool shipped that way for months."""
    for shape in ((1920, 1080), (640, 480), (1080, 1350)):
        assert delivery_dimensions_for_source("native", *shape) == output_dimensions(*shape)


def test_an_unknown_profile_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="unknown delivery profile"):
        delivery_dimensions_for_source("1440p", 1920, 1080)


@needs_ffmpeg
async def test_an_unknown_delivery_setting_fails_before_any_gpu_time(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.conftest import invocations

    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    job = v2v_job(
        workspace, source,
        execution={"runtime": "ltx", "v2v_engine": "transform", "delivery": "1440p"},
    )
    with pytest.raises(AdapterError) as failure:
        await collect(job)
    assert "delivery setting is unavailable" in failure.value.user_message
    assert failure.value.retriable is False
    assert invocations(log) == [], "the model ran before the setting was checked"


# ── One tool, two paths ──────────────────────────────────────────────────


def test_identity_needs_the_photo_not_just_the_flag(workspace: Path) -> None:
    """The client's 9 Sep package made the reference mandatory and their
    10 Sep one made it optional again. The flag can therefore be
    unconditionally true in the workflow, and the UPLOAD decides."""
    adapter = LtxAdapter()
    on = {"runtime": "ltx", "v2v_engine": "transform", "v2v_reference_identity": True}

    job = v2v_job(workspace, None, execution=on)
    assert adapter._uses_reference_identity(job, None) is False
    assert adapter._uses_reference_identity(job, workspace / "photo.png") is True

    off = {**on, "v2v_reference_identity": False}
    job = v2v_job(workspace, None, execution=off)
    assert adapter._uses_reference_identity(job, workspace / "photo.png") is False


# ── What the customer's quality choice actually changes ──────────────────


@needs_ffmpeg
async def test_every_quality_level_renders_the_same_picture(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The honest statement of what the ladder sells.

    Each level asks the model for exactly the same frame size; only the final
    resize differs. If this ever fails because a level started rendering
    larger, that is a real product change — the cost of an 8K job would rise
    by more than an order of magnitude — and it must be made deliberately
    rather than discovered here.
    """
    from tests.conftest import invocations, value_of

    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    asked: dict[str, tuple[str, str]] = {}

    for level in ("1080p", "4k", "8k"):
        log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))
        job = v2v_job(
            workspace, source,
            parameters={"quality": level},
            execution={
                "runtime": "ltx",
                "v2v_engine": "transform",
                "render_proxy": "480p",
                "delivery": "native",
            },
        )
        await collect(job)
        argv = invocations(log)[0]
        asked[level] = (value_of(argv, "--width"), value_of(argv, "--height"))
        log.unlink()

    assert len(set(asked.values())) == 1, f"the levels rendered different sizes: {asked}"


@needs_ffmpeg
async def test_the_proxy_is_opt_in_and_off_means_the_measured_grid(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`render_proxy` is the one line that decides where quality is spent, so
    removing it has to put generation back exactly where it was."""
    from tests.conftest import invocations, value_of

    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)

    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))
    await collect(v2v_job(
        workspace, source,
        execution={"runtime": "ltx", "v2v_engine": "transform", "render_proxy": "480p"},
    ))
    proxy = int(value_of(invocations(log)[0], "--height"))
    log.unlink()

    await collect(v2v_job(workspace, source))
    plain = int(value_of(invocations(log)[0], "--height"))

    assert proxy != plain, "the proxy grid was not used"
    assert proxy < plain, "the proxy must be the SMALLER grid"


@needs_ffmpeg
async def test_the_delivered_file_is_the_requested_size_with_its_audio(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1080p rather than 4K deliberately: this asserts the whole path — proxy
    render, stitch, audio, one resize — on a box that may not have the memory
    to encode a 4K frame. The size arithmetic above covers the larger rungs.

    The source is 16:9 so the delivered frame is the full 1920x1080. A 4:3
    upload would correctly come back 1440x1080, which is the point of
    `test_delivery_preserves_the_sources_aspect_instead_of_cropping_it`.
    """
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True, size="256x144")
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    result, _ = await collect(v2v_job(
        workspace, source,
        parameters={"quality": "1080p"},
        execution={
            "runtime": "ltx",
            "v2v_engine": "transform",
            "render_proxy": "480p",
            "delivery": "1080p",
        },
    ))

    assert (result.width, result.height) == (1920, 1080)
    info = await probe_media(workspace / "output.mp4")
    assert info.has_audio, "the customer's own soundtrack is still on it"
    assert info.duration_seconds == pytest.approx(2.0, abs=1.0)


@needs_ffmpeg
async def test_sound_off_delivers_a_silent_video(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`settings.sound` is a public control, so a job that carries it false
    must not quietly keep the audio."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 2.0))

    await collect(v2v_job(
        workspace, source,
        parameters={"sound": False},
        execution={"runtime": "ltx", "v2v_engine": "transform", "delivery": "native"},
    ))

    info = await probe_media(workspace / "output.mp4")
    assert not info.has_audio


# ── The shipped contract ─────────────────────────────────────────────────


def test_the_definition_offers_the_ladder_and_keeps_the_photo_optional() -> None:
    workflow = yaml.safe_load(
        (ROOT / "workflow-definitions" / "video-to-video.yaml").read_text(encoding="utf-8")
    )
    assert workflow["supported_quality_levels"] == ["1080p", "4k", "8k"]
    assert workflow["settings"]["quality"] is True
    assert workflow["duration_mode"] == "source"

    roles = {item["role"]: item for item in workflow["inputs"]}
    assert roles["source_video"]["required"] is True
    assert roles["reference_image"]["required"] is False
    help_text = roles["reference_image"]["help"].lower()
    assert "empty" in help_text, "the prompt-only path is the default and must be described"
    # "works best when your video shows one person" is fine and still true;
    # what must not survive is the old promise that keyed identity to a
    # quality level the form no longer offers.
    assert "on best" not in help_text, "the copy still names the retired quality level"

    execution = workflow["execution"]
    assert execution["v2v_engine"] == "transform"
    assert execution["v2v_reference_identity"] is True
    assert execution["render_proxy"] == "540p"
    assert execution["execution_by_quality"] == {
        "1080p": {"delivery": "1080p"},
        "4k": {"delivery": "4k"},
        "8k": {"delivery": "8k"},
    }


def test_quality_changes_the_finish_and_nothing_else() -> None:
    """Read through the worker's own resolver, so this pins the behaviour the
    adapter will actually see rather than the shape of the YAML."""
    from worker.workflows.resolver import _execution_for

    workflow = yaml.safe_load(
        (ROOT / "workflow-definitions" / "video-to-video.yaml").read_text(encoding="utf-8")
    )
    base = workflow["execution"]

    for level in ("1080p", "4k", "8k"):
        execution = _execution_for({"execution": base, "parameters": {"quality": level}})
        assert execution["delivery"] == level
        assert execution["render_proxy"] == "540p", "the render must not follow the finish"
        assert execution["v2v_engine"] == "transform"
