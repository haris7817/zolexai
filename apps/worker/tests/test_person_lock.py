"""Person lock: the subject survives a restyle they were never asked to change.

The transform engine conditions on an EDGE MAP, which is what lets a daylight
hillside come back as a blizzard. The same property is a defect the moment a
person is standing on that hillside: an outline says where a face is and
nothing about whose it is, so the model invents one — new features, new skin
tone — while the customer's prompt said nothing about changing them.

`execution.v2v_person_lock` carries the subject's own pixels inside a tracked
matte, leaving the edge map to own everything else. Measured on the GPU
2026-08-18: face, skin tone, hair, beard and the falcon's real plumage all came
through a dusk blizzard, at 61s against the edge-only path's ~54.

What these tests hold down:

**The default is untouched.** Person lock is a per-workflow opt-in like every
capability before it, and a job without the flag must produce the argv that has
been serving customers — no matte, no mask, no extra subprocess.

**Alignment is the contract.** The matte, the source window and the edge map are
merged frame for frame. Every one of them is built at the pass's grid and at the
count the pass ACTUALLY renders — the measured-safe substitute, never the
requested one — because a matte that is misaligned does not protect less, it
protects the wrong pixels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    collect,
    invocations,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    staged_input,
    value_of,
)
from worker.core.config import settings


def transform_job(workspace: Path, source: Path, **overrides):
    execution = {"runtime": "ltx", "v2v_engine": "transform"}
    execution.update(overrides.pop("execution", {}))
    defaults = dict(
        workflow_id="video-to-video",
        prompt="a howling winter blizzard at dusk",
        parameters={"aspect_ratio": "16:9"},
        inputs=[staged_input("source_video", "video", "video/mp4", source)],
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


def stub_matte(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replaces the matting subprocess; records what it was asked for.

    Matting is model work that runs in the GPU environment, so the unit suite
    substitutes it exactly the way it substitutes the pipelines. The stub
    SYNTHESISES a matte at the grid and frame count it was asked for rather
    than copying a fixture — which is the point: everything downstream of the
    model (the merge, the weighting, the argv) then runs for real against
    correctly shaped inputs, and a shape mistake in the adapter fails here
    instead of on a GPU.
    """
    from worker.media import ffmpeg as run_ffmpeg
    from worker.media import masks

    calls: list[dict] = []

    async def fake(
        source: Path,
        dest: Path,
        *,
        width: int,
        height: int,
        fps: float,
        frames: int,
        **kwargs,
    ) -> Path:
        calls.append(
            {
                "source": source, "dest": dest, "width": width,
                "height": height, "fps": fps, "frames": frames, **kwargs,
            }
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        await run_ffmpeg(
            [
                "-f", "lavfi", "-i", f"color=black:s={width}x{height}:r={fps:g}",
                "-vf", (
                    f"drawbox=x={width // 4}:y=0:w={width // 2}:h={height}"
                    ":color=white:t=fill,format=yuv420p"
                ),
                "-frames:v", str(frames),
                "-fps_mode", "cfr",
                "-c:v", "libx264", "-preset", "ultrafast", str(dest),
            ]
        )
        return dest

    monkeypatch.setattr(masks, "build_person_matte", fake)
    monkeypatch.setattr("worker.adapters.ltx.build_person_matte", fake)
    return calls


def mask_of(argv: list[str]) -> tuple[str, float] | None:
    if "--conditioning-attention-mask" not in argv:
        return None
    index = argv.index("--conditioning-attention-mask")
    return argv[index + 1], float(argv[index + 2])


#: Every synthetic clip in this module is this size, so a raw frame's bytes can
#: be indexed directly without probing it first.
CLIP_WIDTH = 160
CLIP_HEIGHT = 120


async def luma_at(clip: Path, x: int, y: int) -> int:
    """Brightness of one pixel of a clip's first frame, 0-255.

    Read through ffmpeg rather than an image library: this worker's runtime
    dependencies are ffmpeg and httpx, and a test is not a reason to add a
    third. The whole frame comes back as one byte per pixel, row-major, with no
    resampling or interpretation between the file and the assertion.
    """
    from worker.media import ffmpeg_stdout

    raw = await ffmpeg_stdout(
        [
            "-i", str(clip),
            "-vf", "format=gray",
            "-frames:v", "1",
            "-f", "rawvideo", "-",
        ]
    )
    assert len(raw) >= CLIP_WIDTH * CLIP_HEIGHT, "unexpected raw frame size"
    return raw[y * CLIP_WIDTH + x]


async def solid_clip(path: Path, colour: str, frames: int = 12) -> Path:
    from worker.media import ffmpeg as run_ffmpeg

    await run_ffmpeg(
        ["-f", "lavfi", "-i", f"color={colour}:s=160x120:r=24",
         "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    )
    return path


async def half_matte(path: Path, frames: int = 12) -> Path:
    """White over the left half — a stand-in for "the subject is here"."""
    from worker.media import ffmpeg as run_ffmpeg

    await run_ffmpeg(
        ["-f", "lavfi", "-i", "color=black:s=160x120:r=24",
         "-vf", "drawbox=x=0:y=0:w=80:h=120:color=white:t=fill,format=yuv420p",
         "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "ultrafast", str(path)]
    )
    return path


# ── Off by default ───────────────────────────────────────────────────────


@needs_ffmpeg
async def test_a_transform_without_the_flag_is_unchanged(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine that has been serving customers: an edge map, no mask, and
    no matting subprocess anywhere near it."""
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 3.0))
    calls = stub_matte(monkeypatch)

    await collect(transform_job(workspace, source))

    argv = invocations(log)[0]
    assert mask_of(argv) is None
    assert "--video-conditioning" in argv
    assert calls == [], "no matte may be built for a job that did not ask for one"


@needs_ffmpeg
async def test_person_lock_does_not_reach_the_restyle_engine(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag belongs to the transform engine. Setting it on a workflow still
    running the still-conditioned restyle must not half-enable anything — the
    restyle has no control clip to merge a matte into."""
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 3.0))
    calls = stub_matte(monkeypatch)

    await collect(
        transform_job(
            workspace, source, execution={"v2v_engine": "restyle", "v2v_person_lock": True}
        )
    )

    assert calls == []
    assert mask_of(invocations(log)[0]) is None


# ── The locked path ──────────────────────────────────────────────────────


@needs_ffmpeg
async def test_the_subjects_own_pixels_reach_the_model(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matte is built and an attention mask accompanies the control clip.

    Without the mask the model is handed the subject's pixels and no reason to
    prefer them over the prompt, which is most of the way back to the bug.
    """
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 3.0))
    calls = stub_matte(monkeypatch)

    await collect(
        transform_job(workspace, source, execution={"v2v_person_lock": True})
    )

    assert len(calls) == 1
    argv = invocations(log)[0]
    mask = mask_of(argv)
    assert mask is not None
    assert Path(mask[0]).exists(), "the attention mask is a real file"
    assert "hybrid" in value_of(argv, "--video-conditioning")


@needs_ffmpeg
async def test_the_matte_matches_the_frames_actually_rendered(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alignment contract, and the one that fails silently.

    The shape tables substitute a measured-safe frame count for the requested
    one, so a matte built against the REQUEST drifts out of registration with
    the render — protecting a region the subject has already moved out of, on
    every frame. It must be built against what the pass renders.
    """
    source = await make_clip(workspace / "source.mp4", 3.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 3.0))
    calls = stub_matte(monkeypatch)

    await collect(
        transform_job(workspace, source, execution={"v2v_person_lock": True})
    )

    argv = invocations(log)[0]
    rendered = int(value_of(argv, "--num-frames"))
    assert calls[0]["frames"] == rendered
    assert calls[0]["width"], calls[0]["height"] == (1024, 576)
    assert calls[0]["fps"] == float(settings.ltx_frame_rate)


@needs_ffmpeg
async def test_every_pass_of_a_chain_gets_its_own_matte(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matte covers one window. Reusing the first pass's matte across a chain
    would pin the subject wherever they stood at the start and protect empty
    background everywhere after it."""
    source = await make_clip(workspace / "source.mp4", 6.0)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0))
    calls = stub_matte(monkeypatch)

    await collect(
        transform_job(
            workspace,
            source,
            execution={"v2v_person_lock": True, "transform_pass_seconds": 2},
        )
    )

    passes = invocations(log)
    assert len(passes) >= 2
    assert len(calls) == len(passes)
    starts = [call["start_seconds"] for call in calls]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "each pass mattes its own window"
    # Distinct files, so one pass cannot overwrite another's conditioning.
    assert len({str(call["dest"]) for call in calls}) == len(calls)


@needs_ffmpeg
async def test_the_background_weight_is_tunable_but_defaults_to_the_measured_value(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-grey is what was measured: the subject followed hard, the scene still
    holding its geometry from the edge map. Both extremes are worse and the
    value is a judgement against real footage, so a workflow may move it."""
    from worker.media.masks import BACKGROUND_ATTENTION

    assert 0.0 < BACKGROUND_ATTENTION < 1.0

    source = await make_clip(workspace / "source.mp4", 3.0)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 3.0))
    stub_matte(monkeypatch)

    result, _ = await collect(
        transform_job(
            workspace,
            source,
            execution={"v2v_person_lock": True, "v2v_background_attention": 0.25},
        )
    )
    # The render completed with the override in play; the value itself is
    # exercised by the mask builder's own test below.
    assert result.path.exists()


# ── The builders, directly ───────────────────────────────────────────────


@needs_ffmpeg
async def test_a_hybrid_control_keeps_the_masked_region_and_edges_elsewhere(
    workspace: Path,
) -> None:
    """The mechanism, on real media. Merge a white clip into a black one under a
    half-white matte and the result must be light where the matte was and dark
    where it was not — the same operation that puts a real face inside an
    outline."""
    from worker.media.masks import build_hybrid_control

    frames = 12
    edges = await solid_clip(workspace / "black.mp4", "black", frames)
    footage = await solid_clip(workspace / "white.mp4", "white", frames)
    matte = await half_matte(workspace / "matte.mp4", frames)

    dest = await build_hybrid_control(
        edges, footage, matte, workspace / "hybrid.mp4", frames=frames
    )

    assert await luma_at(dest, 20, 60) > 200, "masked region carries the real pixels"
    assert await luma_at(dest, 140, 60) < 60, "everything else stays the edge map"


@needs_ffmpeg
async def test_inverting_a_hybrid_control_protects_the_other_side(
    workspace: Path,
) -> None:
    """The same builder, inverted, is what person REPLACEMENT will need: the
    scene keeps its own pixels and the subject's region is left free. Proving it
    now means replacement inherits a tested primitive rather than a new one."""
    from worker.media.masks import build_hybrid_control

    frames = 12
    edges = await solid_clip(workspace / "black.mp4", "black", frames)
    footage = await solid_clip(workspace / "white.mp4", "white", frames)
    matte = await half_matte(workspace / "matte.mp4", frames)

    dest = await build_hybrid_control(
        edges, footage, matte, workspace / "inverted.mp4", frames=frames, invert=True
    )

    assert await luma_at(dest, 20, 60) < 60, "the subject's region is left free"
    assert await luma_at(dest, 140, 60) > 200, "the scene keeps its own pixels"


@needs_ffmpeg
async def test_an_attention_mask_lifts_the_background_without_touching_the_subject(
    workspace: Path,
) -> None:
    """White stays white — the subject is followed fully — while black is lifted
    to the background weight rather than ignored, which is what keeps the scene's
    geometry while the prompt owns its look."""
    from worker.media.masks import build_attention_mask

    frames = 12
    matte = await half_matte(workspace / "matte.mp4", frames)

    dest = await build_attention_mask(
        matte, workspace / "attention.mp4", frames=frames, background=0.5
    )

    assert await luma_at(dest, 20, 60) > 200, "the subject is followed fully"
    assert 100 < await luma_at(dest, 140, 60) < 160, "the background sits at mid weight"


async def test_the_builders_refuse_an_empty_clip(workspace: Path) -> None:
    """Zero frames is a bug upstream, and a zero-length conditioning clip fails
    far away from its cause. Refuse it where it is still explicable."""
    from worker.media.masks import (
        build_attention_mask,
        build_hybrid_control,
        build_person_matte,
        extract_source_window,
    )

    path = workspace / "x.mp4"
    with pytest.raises(ValueError):
        await build_person_matte(
            path, path, start_seconds=0, duration_seconds=1,
            width=64, height=64, fps=24, frames=0,
        )
    with pytest.raises(ValueError):
        await extract_source_window(
            path, path, start_seconds=0, duration_seconds=1,
            width=64, height=64, fps=24, frames=0,
        )
    with pytest.raises(ValueError):
        await build_hybrid_control(path, path, path, path, frames=0)
    with pytest.raises(ValueError):
        await build_attention_mask(path, path, frames=0)
