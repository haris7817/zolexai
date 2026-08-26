"""The guided quality tier: `execution.generation_engine: guided`.

The distilled entry point that serves every default text- and image-to-video
runs unguided by construction — no CFG, no negative prompt, no step count.
That is a property of distillation, not a setting anyone forgot, and it is why
adherence complaints cannot be tuned away on the default tier.

`generation_engine: guided` swaps in the guided two-stage pipeline
(`ltx_pipelines.ti2vid_two_stages`): dev transformer + distilled LoRA, which
the pipeline requires together, so the LoRA/quantization rule applies —
unquantized, CPU offload. Measured 17 Aug 2026 on the RTX PRO 6000 at
1024x576, 121 frames: 146s against the distilled tier's 34s for the same
prompt and seed, with video and audio both present in the output.

Two things this suite exists to keep true:

**The default is byte-identical to what has served production.** The tier is a
cost and behaviour change, so it is reachable only through the execution flag,
exactly like the transform engine and audio conditioning before it.

**Unmeasured frame counts never reach this decoder.** 241 frames at 1024x576 —
a count the distilled tables call safe and the audio tier renders happily — is
a REPRODUCED illegal-memory-access failure on this pipeline. Its passes land
on the one measured count (121) and its pass ceiling stays at that landing's
length until someone measures a longer one.
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
from worker.adapters.ltx import _GUIDED_PASS_SECONDS, LtxAdapter
from worker.core.config import settings


def guided_job(workspace: Path, **overrides):
    execution = {"runtime": "ltx", "generation_engine": "guided"}
    execution.update(overrides.pop("execution", {}))
    defaults = dict(
        parameters={"duration": "2s", "aspect_ratio": "16:9"},
        execution=execution,
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── Off by default; the existing product is unchanged ────────────────────


@needs_ffmpeg
async def test_a_plain_generation_still_runs_the_distilled_tier(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No flag, no change: quantized distilled transformer, no LoRA, no
    offload — the argv that has served every production text-to-video."""
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(make_job(workspace))

    argv = invocations(log)[0]
    assert "--distilled-lora" not in argv
    assert "--offload" not in argv
    assert value_of(argv, "--quantization") == settings.ltx_quantization
    assert "distilled-transformer" in value_of(argv, "--transformer-path")


@needs_ffmpeg
async def test_an_unrecognised_engine_value_stays_on_the_default_tier(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the one documented value switches tiers — the same contract as
    `v2v_engine`. A typo in a workflow lands on the proven default, and the
    argv makes the routing visible in any log that records it."""
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(guided_job(workspace, execution={"generation_engine": "ultra"}))

    argv = invocations(log)[0]
    assert "--distilled-lora" not in argv
    assert value_of(argv, "--quantization") == settings.ltx_quantization


# ── The guided tier's own configuration ──────────────────────────────────


@needs_ffmpeg
async def test_the_guided_tier_runs_the_guided_pipeline_unquantized(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dev transformer + distilled LoRA (the pipeline requires both), and the
    LoRA/quantization rule applies: no `--quantization`, `--offload cpu`.
    A quantization flag reappearing here is the crash the audio tier already
    paid for, not a speed-up."""
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(guided_job(workspace))

    argv = invocations(log)[0]
    assert "--quantization" not in argv
    assert value_of(argv, "--offload") == "cpu"
    assert "distilled-lora" in value_of(argv, "--distilled-lora")
    assert "dev-transformer" in value_of(argv, "--transformer-path")


@needs_ffmpeg
async def test_a_node_with_headroom_can_keep_the_weights_resident(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ltx_unquantized_offload: none` — the 27 Aug music-video speed lever:
    on the 96 GB node (with lazy ComfyUI eviction making room) the audio
    tier stops streaming the 22B transformer from host RAM, measured 23-30%
    faster. Quantization stays off either way."""
    from worker.core.config import settings

    monkeypatch.setattr(settings, "ltx_unquantized_offload", "none")
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(guided_job(workspace))

    argv = invocations(log)[0]
    assert "--offload" not in argv
    assert "--quantization" not in argv


@needs_ffmpeg
async def test_every_guided_pass_lands_on_the_measured_frame_count(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """121 is the one count proven on this pipeline's decode path; 241 is a
    reproduced FAIL. Whatever length a pass plans, the render lands on the
    measured count and the extra frames are trimmed after, never delivered."""
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    result, _ = await collect(guided_job(workspace))

    argv = invocations(log)[0]
    assert value_of(argv, "--num-frames") == "121"
    assert result.duration_seconds == pytest.approx(2.0, abs=1.0)


@needs_ffmpeg
async def test_a_long_guided_generation_chains_at_the_measured_ceiling(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 10s request is two guided passes, not one 241-frame pass — the
    distilled tier's 60s grid ceiling must not leak in here, because past the
    largest measured landing sits a decoder crash, not slower output."""
    fixture = await make_clip(tmp_path / "r.mp4", 5.0, audio=True)
    log = render_stub(tmp_path, monkeypatch, fixture)

    result, _ = await collect(
        guided_job(workspace, parameters={"duration": "10s", "aspect_ratio": "16:9"})
    )

    passes = invocations(log)
    assert len(passes) == 2
    assert all(value_of(argv, "--num-frames") == "121" for argv in passes)
    assert result.duration_seconds == pytest.approx(10.0, abs=1.0)


@needs_ffmpeg
async def test_guided_image_to_video_keeps_its_conditioning_stills(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine changes the pipeline, not the conditioning contract: the
    uploaded still anchors frame zero of the first pass at full strength,
    exactly as it does on the distilled tier."""
    still = tmp_path / "portrait.png"
    await make_clip(tmp_path / "still-src.mp4", 0.2)
    # A one-frame PNG lifted from a real clip keeps the fixture honest.
    from worker.media import ffmpeg as run_ffmpeg

    await run_ffmpeg(["-i", str(tmp_path / "still-src.mp4"), "-frames:v", "1", str(still)])
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(
        guided_job(
            workspace,
            workflow_id="image-to-video",
            inputs=[staged_input("source_image", "image", "image/png", still)],
        )
    )

    argv = invocations(log)[0]
    assert conditioning_of(argv) == [(str(still), 0, 1.0)]
    assert "dev-transformer" in value_of(argv, "--transformer-path")


# ── The pass ceiling is this tier's own measurement ──────────────────────


def test_the_guided_pass_ceiling_is_the_measured_landing(workspace: Path) -> None:
    """5s is 121 frames minus the rounding — the one measured cell. The
    distilled tier's 60s and the audio tier's 20s are other pipelines'
    measurements and neither transfers."""
    adapter = LtxAdapter()
    job = make_job(workspace, execution={"runtime": "ltx", "generation_engine": "guided"})

    assert adapter._guided_pass_seconds(job) == _GUIDED_PASS_SECONDS == 5.0
    assert adapter._guided_pass_seconds(job) < adapter._audio_pass_seconds(job)
    assert adapter._guided_pass_seconds(job) < adapter._per_pass_seconds(job, (1024, 576))


def test_a_workflow_can_lower_the_guided_ceiling_but_the_brake_still_clamps(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`execution.guided_pass_seconds` is where a future measurement lands,
    and `settings.ltx_max_seconds` — the one mid-incident lever that pulls
    every shape down without a deploy — keeps working on this tier too."""
    adapter = LtxAdapter()
    lowered = make_job(
        workspace, execution={"runtime": "ltx", "guided_pass_seconds": 2}
    )
    assert adapter._guided_pass_seconds(lowered) == 2.0

    monkeypatch.setattr(settings, "ltx_max_seconds", 3)
    raised = make_job(
        workspace, execution={"runtime": "ltx", "guided_pass_seconds": 300}
    )
    assert adapter._guided_pass_seconds(raised) == 3.0


@needs_ffmpeg
async def test_a_missing_dev_checkpoint_fails_before_any_gpu_time(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node without the guided tier's weights refuses the job up front and
    keeps serving every other workflow — per path, not per process, the same
    contract the audio tier already holds."""
    from worker.adapters.base import AdapterError
    from worker.adapters.ltx import _OPTIONAL_MODEL_FILES

    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))
    (fake_models / _OPTIONAL_MODEL_FILES["transformer_dev"]).unlink()

    with pytest.raises(AdapterError) as raised:
        await collect(guided_job(workspace))

    assert raised.value.retriable is False
    assert "transformer_dev" in raised.value.internal_detail
    assert invocations(log) == []


# ── Guidance knobs: only where guiders exist ─────────────────────────────


@needs_ffmpeg
async def test_the_guided_tier_carries_negative_prompt_and_cfg_when_asked(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one place a negative prompt exists in this model family is the
    guided tier — the distilled and ic_lora entry points have no guiders at
    all, which is why "add negative prompting" is a tier decision, not a
    flag. Unset, the pipeline's own defaults remain the measured baseline."""
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(guided_job(
        workspace,
        execution={
            "negative_prompt": "deformed limbs, flickering, subject vanishing",
            "guidance_scale": 4.5,
        },
    ))

    argv = invocations(log)[0]
    assert value_of(argv, "--negative-prompt") == (
        "deformed limbs, flickering, subject vanishing"
    )
    assert value_of(argv, "--video-cfg-guidance-scale") == "4.5"


@needs_ffmpeg
async def test_guidance_keys_never_reach_a_pipeline_without_guiders(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workflow that sets the keys while running the distilled tier gets
    the flags dropped, not forwarded — an unknown argument is a crashed
    render, and 'the model ignored my negative prompt' must never become
    'the job failed'."""
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(make_job(
        workspace,
        execution={
            "runtime": "ltx",
            "negative_prompt": "deformed limbs",
            "guidance_scale": 4.5,
        },
    ))

    argv = invocations(log)[0]
    assert "--negative-prompt" not in argv
    assert "--video-cfg-guidance-scale" not in argv


@needs_ffmpeg
async def test_a_blank_negative_prompt_is_not_sent(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = render_stub(tmp_path, monkeypatch, await make_clip(tmp_path / "r.mp4", 2.0, audio=True))

    await collect(guided_job(workspace, execution={"negative_prompt": "   "}))

    argv = invocations(log)[0]
    assert "--negative-prompt" not in argv
