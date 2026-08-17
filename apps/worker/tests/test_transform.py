"""Video to Video, `transform` engine: restyle by control signal.

The restyle engine next door shows the model stills of the source. That keeps
the shot and weakens the transformation, because a photograph carries the
source's colour, light and material along with its geometry — and the client's
report is exactly that: "V2V preserves the source strongly but the requested
restyling is too weak".

This engine separates the two signals. An edge map carries geometry and nothing
else, so the prompt owns the look with nothing to fight. Verified on the GPU on
17 Aug 2026: the LTX-2.3 Union Control IC-LoRA loads against the installed 2.5
distilled checkpoint, and a daylight desert plate came back as a rain-soaked
neon street with the subject's pose, the car's position and the road's
perspective unchanged.

What these tests defend is the part that is easy to get quietly wrong. A
control clip that is a different length, grid or window than the pass it feeds
is not a weaker signal — it is a misaligned one, and misalignment produces
output that tracks nothing while still looking like a plausible video. So the
assertions are about correspondence: one control clip per pass, built from that
pass's own window, at that pass's own frame count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
from worker.adapters.ltx import _V2V_CONTINUITY_STRENGTH, grid_for_source
from worker.media import probe_media


def transform_job(workspace: Path, source: Path | None, reference: Path | None = None, **overrides):
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    if reference is not None:
        inputs.append(staged_input("reference_image", "image", "image/png", reference))

    execution = {"runtime": "ltx", "v2v_engine": "transform"}
    execution.update(overrides.pop("execution", {}))
    defaults = dict(
        workflow_id="video-to-video",
        prompt="a neon-lit cyberpunk street at night, heavy rain",
        # `duration_mode: source` — no duration is sent for this workflow.
        parameters={"aspect_ratio": "16:9", "quality": "High"},
        inputs=inputs,
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


def control_of(argv: list[str]) -> tuple[str, float] | None:
    """The `--video-conditioning PATH STRENGTH` pair, if the pass carried one."""
    if "--video-conditioning" not in argv:
        return None
    index = argv.index("--video-conditioning")
    return argv[index + 1], float(argv[index + 2])


# ── The engine is opt-in and nothing else changed shape ──────────────────


@needs_ffmpeg
async def test_the_default_engine_is_still_the_still_conditioned_restyle(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A video-to-video job that asks for nothing gets exactly what it got
    before: stills, the distilled entry point, no LoRA, no control clip.

    The transform engine is a new capability, not a silent replacement of the
    one customers are already using.
    """
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    job = make_job(
        workspace,
        workflow_id="video-to-video",
        prompt="charcoal sketch",
        parameters={"aspect_ratio": "16:9"},
        inputs=[staged_input("source_video", "video", "video/mp4", source)],
    )
    await collect(job)

    argv = invocations(log)[0]
    assert control_of(argv) is None
    assert "--lora" not in argv
    assert "--offload" not in argv
    # Stills from the source are the restyle's whole mechanism.
    assert len(conditioning_of(argv)) > 1


# ── The control signal reaches the model ─────────────────────────────────


@needs_ffmpeg
async def test_the_transform_engine_conditions_on_a_control_video(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--video-conditioning` plus the Union Control LoRA, on the IC-LoRA entry
    point. All four have to be present together: the flag without the adapter
    is ignored, and the adapter without the flag is a no-op."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    control = control_of(argv)
    assert control is not None, "the transform engine rendered without a control signal"
    path, strength = control
    assert Path(path).exists()
    assert strength == 1.0
    assert "union-control" in value_of(argv, "--lora")


@needs_ffmpeg
async def test_a_lora_pass_drops_quantization_and_offloads_to_cpu(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule that makes the difference between a render and a crash.

    LoRA + FP8/NVFP4 fusion reaches Triton kernels that do not exist for these
    shapes. The reference engine forces quantization off whenever a LoRA is
    loaded and fits the unquantized model with CPU offload instead; every
    unexplained "resolution ceiling" this project chased turned out to be this
    clash. A `--quantization` flag reappearing next to a `--lora` flag is
    therefore a bug, not a tuning choice.
    """
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    assert "--lora" in argv
    assert "--quantization" not in argv
    assert value_of(argv, "--offload") == "cpu"


@needs_ffmpeg
async def test_the_transform_pass_asks_for_double_and_stops_at_stage_one(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape rule that cost a failed render on 17 Aug 2026.

    Stage 2 upscales by halving a latent grid and needs both latent dimensions
    even. The product's 16:9 grid is 1024x576 and 576/64 = 9 is odd, so the
    two-stage path dies in a VAE rearrange. Stage 1 renders at half the request,
    so asking for 2x and stopping there delivers the target grid exactly.

    If `--skip-stage-2` ever goes missing while the doubling stays, every
    transform job silently returns a video at twice the intended size; if the
    doubling goes missing while the skip stays, every one returns half.
    """
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    grid = grid_for_source(160, 120)
    assert (int(value_of(argv, "--width")), int(value_of(argv, "--height"))) == (
        grid[0] * 2,
        grid[1] * 2,
    )
    assert "--skip-stage-2" in argv


@needs_ffmpeg
async def test_a_control_only_pass_still_counts_as_conditioned(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass carrying a control video but no `--image` must use the CONDITIONED
    shape tables. This is a production failure, reproduced.

    A 14.976s upload asks for 359 frames. With the conditioned tables it lands
    on the measured-safe 360; without them the lattice snaps it to 361, which
    this decoder is documented to die on — `_CONDITIONED_BANDS` exists because
    three video-to-video jobs died on exactly 361 on 16 Aug 2026.

    The old restyle never reached that state because it always passed source
    stills. The transform engine drops them on purpose, so its first pass
    carries no image at all unless the customer supplied a reference — and the
    common case is that they did not. Job 2f4a22b9 died this way, at 361, on
    the first real production job after the engine shipped.
    """
    # 16:9 so the source lands on a MEASURED grid with a 60s ceiling and the
    # whole 14.976s is one pass — the shape the production job had. A grid with
    # no measured ceiling would be split into two passes and never reach 359.
    source = await make_clip(workspace / "source.mp4", 14.976, size="160x90")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 14.976))

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    assert conditioning_of(argv) == [], "this test is meaningless if a still crept in"
    assert control_of(argv) is not None
    assert int(value_of(argv, "--num-frames")) == 360


# ── Alignment: the control clip must match the pass that consumes it ─────


@needs_ffmpeg
async def test_each_pass_gets_a_control_clip_of_its_own_window(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three passes, three DIFFERENT control clips.

    One clip reused across passes would restyle the whole video against the
    opening seconds of the source — which still produces a video, and is
    precisely the failure a "did it render?" assertion cannot see.
    """
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 1.0))

    await collect(
        transform_job(workspace, source, execution={"max_segment_seconds": 1})
    )

    controls = [control_of(argv) for argv in invocations(log)]
    assert len(controls) >= 3
    assert all(c is not None for c in controls)
    paths = [c[0] for c in controls]
    assert len(set(paths)) == len(paths), "passes shared a control clip"


@needs_ffmpeg
async def test_the_control_clip_matches_the_frame_count_actually_rendered(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not the REQUESTED count — the one the shape tables settled on.

    `safe_frame_count` substitutes a measured-safe count for one the decoder
    cannot handle, and the render is trimmed back afterwards. A control clip
    built against the requested count is short by exactly that difference, and
    the tail of every nudged pass drifts off the source.
    """
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    rendered_frames = int(value_of(argv, "--num-frames"))
    control_path, _ = control_of(argv)
    control = await probe_media(Path(control_path))
    assert control.frame_count == rendered_frames


@needs_ffmpeg
async def test_the_control_clip_is_built_at_the_generation_grid(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A portrait source produces a portrait control clip.

    The grid follows the SOURCE's aspect, not the requested one, and a control
    signal at a different aspect would be centre-cropped against footage it no
    longer describes.

    The clip is built at the DELIVERED grid, which is half of what the argv asks
    for (see the doubling rule above) — verified on the GPU, where a 1024x576
    control against a 2048x1152 request rendered clean.
    """
    source = await make_clip(workspace / "source.mp4", 2.0, size="120x160")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    control_path, _ = control_of(argv)
    control = await probe_media(Path(control_path))
    grid = grid_for_source(120, 160)
    assert (control.width, control.height) == grid
    assert control.width < control.height, "a portrait source produced a landscape control"


# ── Structure comes from the control clip; stills stop competing ─────────


@needs_ffmpeg
async def test_source_stills_do_not_compete_with_the_control_signal(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At most ONE image, and only ever at frame zero.

    The restyle's spread of source stills is what pulls the output back toward
    the source's look. The control clip already states where everything is for
    the whole window, so keeping the stills would reassert exactly the thing
    this engine exists to discard — the transformation would be weak again, for
    the same reason as before.
    """
    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))

    await collect(transform_job(workspace, source))

    images = conditioning_of(invocations(log)[0])
    assert len(images) <= 1
    assert all(frame == 0 for _, frame, _ in images)


@needs_ffmpeg
async def test_later_passes_open_on_the_previous_passs_final_frame(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam. Without it each pass starts a differently-styled video, which
    on a chained source is visible as a hard change of look mid-result."""
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 1.0))

    await collect(
        transform_job(workspace, source, execution={"max_segment_seconds": 1})
    )

    later = conditioning_of(invocations(log)[1])
    assert len(later) == 1
    path, frame, strength = later[0]
    assert frame == 0
    assert strength == _V2V_CONTINUITY_STRENGTH
    assert "condition" in Path(path).name


# ── Everything the restyle promises, this promises too ───────────────────


@needs_ffmpeg
async def test_the_result_is_the_length_of_the_source(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's automatic-duration rule is a property of the workflow, not
    of whichever engine happens to serve it."""
    source = await make_clip(workspace / "source.mp4", 2.5)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.5))

    result, _ = await collect(transform_job(workspace, source))

    measured = await probe_media(result.path)
    assert measured.duration_seconds == pytest.approx(2.5, abs=1.0)


@needs_ffmpeg
async def test_the_sources_own_audio_survives_the_transform(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One audio stream, from the upload — never the model's invented one."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    result, _ = await collect(transform_job(workspace, source))

    measured = await probe_media(result.path)
    assert measured.has_audio
    assert measured.audio_stream_count == 1


@needs_ffmpeg
async def test_a_silent_source_stays_silent(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No invented soundtrack on a source that never had one."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=False)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    result, _ = await collect(transform_job(workspace, source))

    measured = await probe_media(result.path)
    assert not measured.has_audio


@needs_ffmpeg
async def test_a_missing_union_control_adapter_fails_before_any_gpu_time(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node without the adapter must say so, once, and not render.

    Without this the pass runs, the flag is silently ineffective, and the
    customer is charged for a restyle that did nothing — the most expensive
    possible way to be missing a file.
    """
    from worker.adapters.base import AdapterError
    from worker.adapters.ltx import _OPTIONAL_MODEL_FILES

    source = await make_clip(workspace / "source.mp4", 2.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))
    (fake_models / _OPTIONAL_MODEL_FILES["union_control_lora"]).unlink()

    with pytest.raises(AdapterError) as raised:
        await collect(transform_job(workspace, source))

    assert raised.value.retriable is False
    assert "union_control_lora" in raised.value.internal_detail
    assert invocations(log) == []
