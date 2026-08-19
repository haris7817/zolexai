"""Reference identity: the person follows the picture, the footage keeps the motion.

`execution.v2v_reference_identity` turns the optional reference image from a
one-shot look hint into the answer to "who is in this video". The mechanics
are three levers that only work together, and each test pins one of them:

**The reference conditions every pass.** The M1 contract showed it once, on
the first pass, at 0.3 — which is why a 30-second result drifted back to the
source person by its second section. Identity mode anchors the opening on the
reference and then re-shows it at an interior frame of every later pass,
exactly the mechanism image-to-video already uses to keep its subject.

**The edge map lets go of the person.** Canny edges of a face ARE its
geometry — jawline, hairline, features — and a control signal that redraws
the original face outline 24 times a second beats any reference strength. A
person matte turns into an attention mask weighting the person's region BELOW
the scene, so pose still tracks while appearance is freed.

**Nothing silently pretends.** Identity without the transform engine, or
combined with person lock (its exact opposite), or with matting broken, is a
refusal — never a job that delivers the source person while claiming the
reference replaced them.

Default behaviour without the flag is pinned untouched by the existing
video-to-video and transform suites.
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
)
from worker.adapters.base import AdapterError
from worker.media import FfmpegError, extract_final_frame, probe_media


def identity_job(workspace: Path, source: Path, reference: Path | None, **overrides):
    execution = {
        "runtime": "ltx",
        "v2v_engine": "transform",
        "v2v_reference_identity": True,
        # Off by default IN THE TESTS (the product default is on): most of
        # this suite is about conditioning and masking, and the describer is
        # a subprocess these tests stub explicitly when it is the subject.
        "v2v_identity_describe_reference": False,
    }
    execution.update(overrides.pop("execution", {}))
    inputs = [staged_input("source_video", "video", "video/mp4", source)]
    if reference is not None:
        inputs.append(staged_input("reference_image", "image", "image/png", reference))
    defaults = dict(
        workflow_id="video-to-video",
        prompt="keep the performance and the camera, use the person from the reference",
        parameters={"aspect_ratio": "16:9"},
        inputs=inputs,
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


def stub_matte(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Synthesises a matte at whatever shape it is asked for, recording the ask.

    Matting is model work that runs in the GPU environment; the suite
    substitutes it the way it substitutes the pipelines, and everything
    downstream — the weighting, the argv — runs for real against correctly
    shaped inputs.
    """
    from worker.media import ffmpeg as run_ffmpeg
    from worker.media import masks

    calls: list[dict] = []

    async def fake(
        source: Path, dest: Path, *, width: int, height: int,
        fps: float, frames: int, **kwargs,
    ) -> Path:
        calls.append(
            {"source": source, "width": width, "height": height,
             "fps": fps, "frames": frames}
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


def record_attention(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Passes build_attention_mask through for real, recording its weights."""
    from worker.media import masks

    calls: list[dict] = []
    real = masks.build_attention_mask

    async def recording(matte: Path, dest: Path, **kwargs) -> Path:
        calls.append(dict(kwargs))
        return await real(matte, dest, **kwargs)

    monkeypatch.setattr("worker.adapters.ltx.build_attention_mask", recording)
    return calls


def mask_of(argv: list[str]) -> tuple[str, float] | None:
    if "--conditioning-attention-mask" not in argv:
        return None
    index = argv.index("--conditioning-attention-mask")
    return argv[index + 1], float(argv[index + 2])


# ── The reference persists through every pass ────────────────────────────


@needs_ffmpeg
async def test_later_passes_carry_the_seam_and_never_the_raw_photo(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interior photo-anchor is OFF by default, on production evidence:
    the first customer identity job got the reference PHOTOGRAPH cut into
    their dance video at exactly the anchor's timestamp — at 0.35, and again
    at I2V's "safe" 0.2, because unlike I2V the photo's composition is alien
    to the footage. Identity persistence across passes is carried by the
    continuity frame (which shows the REPLACED person) and the describer's
    caption; the raw photo conditions the opening pass only."""
    source = await make_clip(workspace / "source.mp4", 3.7)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={"transform_pass_seconds": 1},
    )
    await collect(job)

    passes = invocations(log)
    assert len(passes) == 4

    first = conditioning_of(passes[0])
    assert first == [(str(reference), 0, 0.65)], (
        "the opening is anchored on the reference, and on nothing else"
    )

    for argv in passes[1:]:
        items = conditioning_of(argv)
        assert str(reference) not in [path for path, _, _ in items], (
            "the raw photo mid-pass is the flash defect, not an identity aid"
        )
        seam = items[0]
        assert seam[1] == 0, "frame 0 stays the seam's"


@needs_ffmpeg
async def test_the_interior_anchor_is_still_a_knob_for_footage_that_drifts(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 3.7)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={
            "transform_pass_seconds": 1,
            "v2v_identity_refresh_strength": 0.2,
        },
    )
    await collect(job)

    for argv in invocations(log)[1:]:
        refresh = next(
            item for item in conditioning_of(argv) if item[0] == str(reference)
        )
        assert refresh[1] > 0, "away from frame 0 — it must never fight the seam"
        assert refresh[2] == pytest.approx(0.2)


@needs_ffmpeg
async def test_the_identity_strengths_are_tunable_per_workflow(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The right values are a GPU judgement; the sweep script sets them
    through the same private keys every other conditioning dial uses."""
    source = await make_clip(workspace / "source.mp4", 3.7)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={
            "transform_pass_seconds": 1,
            "v2v_identity_anchor_strength": 0.8,
            "v2v_identity_refresh_strength": 0.5,
        },
    )
    await collect(job)

    passes = invocations(log)
    assert conditioning_of(passes[0])[0][2] == pytest.approx(0.8)
    refresh = next(
        item for item in conditioning_of(passes[1]) if item[0] == str(reference)
    )
    assert refresh[2] == pytest.approx(0.5)


# ── The edge map lets go of the person ───────────────────────────────────


@needs_ffmpeg
async def test_identity_softens_the_control_grip_over_the_person(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every pass mattes its own window and weights the person BELOW the
    scene — the mirror of person lock. The scene keeps the edge map's full
    grip (camera, layout); the person's region is where the reference is
    allowed to win."""
    source = await make_clip(workspace / "source.mp4", 3.7)
    reference = await extract_final_frame(source, workspace / "reference.png")
    mattes = stub_matte(monkeypatch)
    weights = record_attention(monkeypatch)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={"transform_pass_seconds": 1},
    )
    await collect(job)

    passes = invocations(log)
    assert len(mattes) == len(passes), "one matte per pass — identity is not a first-pass event"
    for argv in passes:
        assert "--video-conditioning" in argv, "the control signal still carries the motion"
        masked = mask_of(argv)
        assert masked is not None, "every pass weights the person's region"
        assert masked[1] == pytest.approx(1.0)

    for call, matte_call in zip(weights, mattes, strict=True):
        assert call["background"] == pytest.approx(1.0)
        assert call["subject"] == pytest.approx(0.5)
        assert call["frames"] == matte_call["frames"], (
            "a mask of a different length weights the wrong pixels"
        )


@needs_ffmpeg
async def test_the_matte_matches_the_frames_actually_rendered(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    mattes = stub_matte(monkeypatch)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    await collect(identity_job(workspace, source, reference))

    (argv,) = invocations(log)
    rendered = int(argv[argv.index("--num-frames") + 1])
    assert [call["frames"] for call in mattes] == [rendered]


# ── The worker describes the reference itself ────────────────────────────


def stub_describer(monkeypatch: pytest.MonkeyPatch, facts: str) -> list[Path]:
    """Replaces the vision subprocess; records which image it was shown."""
    calls: list[Path] = []

    async def fake(image_path) -> str:
        calls.append(Path(image_path))
        return facts

    monkeypatch.setattr("worker.adapters.ltx.reference_person_facts", fake)
    return calls


@needs_ffmpeg
async def test_the_worker_describes_the_reference_into_every_pass_prompt(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model reads the reference as pixels and the prompt as words, and
    identity needs both to agree. The first production identity job proved
    what happens when they don't: a prompt of meta-instructions that named no
    visible attribute rendered neither the source person nor the reference.
    So the worker looks at the photo and appends what it sees — after the
    user's own text, which stays verbatim and first."""
    source = await make_clip(workspace / "source.mp4", 3.7)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    shown = stub_describer(
        monkeypatch, "an adult woman with long dark hair, black leather jacket"
    )
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={
            "transform_pass_seconds": 1,
            "v2v_identity_describe_reference": True,
        },
    )
    await collect(job)

    assert shown == [reference], "described once, and it is the REFERENCE it describes"
    for argv in invocations(log):
        prompt = argv[argv.index("--prompt") + 1]
        assert prompt.startswith(
            "keep the performance and the camera, use the person from the reference"
        ), "the user's text survives verbatim, first"
        assert "long dark hair, black leather jacket" in prompt
        assert "The same person, with the same face, hair and clothing" in prompt
        # No photograph vocabulary: "image" and "photographed" in a prompt are
        # CONTENT to the model, and content is rendered — a posed photo shot
        # cut into the customer's video.
        assert "photographed" not in prompt and "reference image" not in prompt


@needs_ffmpeg
async def test_a_failed_description_leaves_the_prompt_untouched(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The description is reinforcement, not a dependency: "" means the
    prompt goes to the model exactly as typed, and the job proceeds."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    stub_describer(monkeypatch, "")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={"v2v_identity_describe_reference": True},
    )
    await collect(job)

    (argv,) = invocations(log)
    assert argv[argv.index("--prompt") + 1] == job.prompt


@needs_ffmpeg
async def test_the_describer_is_optional_and_off_means_off(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    shown = stub_describer(monkeypatch, "should never be asked")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(
        workspace, source, reference,
        execution={"v2v_identity_describe_reference": False},
    )
    await collect(job)

    assert shown == []
    (argv,) = invocations(log)
    assert argv[argv.index("--prompt") + 1] == job.prompt


@needs_ffmpeg
async def test_without_identity_mode_nothing_is_described(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain transform with a look-hint reference must not pay a vision
    subprocess it gets nothing from."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    shown = stub_describer(monkeypatch, "should never be asked")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(workspace, source, reference)
    job.execution.pop("v2v_reference_identity")
    await collect(job)

    assert shown == []
    (argv,) = invocations(log)
    assert argv[argv.index("--prompt") + 1] == job.prompt


# ── Nothing silently pretends ────────────────────────────────────────────


@needs_ffmpeg
async def test_identity_without_a_reference_is_an_ordinary_transform(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag names a capability, not a demand: with nothing to replace the
    person WITH, the job renders exactly as if the flag were absent — no
    matte, no mask, no invented conditioning."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    mattes = stub_matte(monkeypatch)
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    await collect(identity_job(workspace, source, reference=None))

    (argv,) = invocations(log)
    assert mattes == []
    assert mask_of(argv) is None
    assert conditioning_of(argv) == []
    assert "--video-conditioning" in argv


@needs_ffmpeg
async def test_identity_and_person_lock_refuse_to_run_together(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One flag preserves the source person, the other replaces them. A
    workflow carrying both is a configuration bug, not a preference order."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    with pytest.raises(AdapterError) as raised:
        await collect(
            identity_job(
                workspace, source, reference,
                execution={"v2v_person_lock": True},
            )
        )
    assert raised.value.retriable is False


@needs_ffmpeg
async def test_identity_on_the_still_engine_is_refused_not_ignored(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The still-conditioned restyle cannot replace a person. Falling back to
    it would deliver the source person while the workflow claims replacement —
    the silent-success failure mode this feature must never ship."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    job = identity_job(workspace, source, reference)
    job.execution.pop("v2v_engine")

    with pytest.raises(AdapterError) as raised:
        await collect(job)
    assert raised.value.retriable is False


@needs_ffmpeg
async def test_a_matting_failure_fails_the_job_not_the_promise(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = await make_clip(workspace / "source.mp4", 2.0)
    reference = await extract_final_frame(source, workspace / "reference.png")
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    async def broken(*args, **kwargs):
        raise FfmpegError("matting model unavailable")

    monkeypatch.setattr("worker.adapters.ltx.build_person_matte", broken)

    with pytest.raises((AdapterError, FfmpegError)):
        await collect(identity_job(workspace, source, reference))
    assert not (workspace / "output.mp4").exists()


# ── The workflow's standing promises survive the mode ────────────────────


@needs_ffmpeg
async def test_identity_keeps_the_sources_length_and_audio(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacement changes who is on screen and nothing else the workflow
    promises: the result is still the source's length, and the source's own
    audio still survives, attached exactly once."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    reference = await extract_final_frame(source, workspace / "reference.png")
    stub_matte(monkeypatch)
    render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0))

    await collect(identity_job(workspace, source, reference))

    info = await probe_media(workspace / "output.mp4")
    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)
    assert info.has_audio is True
    assert info.audio_stream_count == 1
