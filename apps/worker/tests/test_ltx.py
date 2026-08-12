"""LTX adapter tests — everything except the model, proven without a GPU.

The adapter's GPU-specific surface is one seam: `_launcher()`, the argv prefix
that reaches the LTX environment. These tests substitute a plain Python stub
there and exercise every real code path around it — command construction,
subprocess supervision, cancellation kill, progress parsing, output
verification. The GPU-node smoke test then only has to answer one question:
does the model behave the same as the stub.

Tests that touch real media need ffmpeg and skip without it, same as the
harness suite.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    GenerationAdapter,
    JobCancelled,
    JobTimedOut,
)
from worker.adapters.ltx import _MARKERS, LtxAdapter, match_marker
from worker.core.config import settings
from worker.media import ffmpeg, tools_available

needs_ffmpeg = pytest.mark.skipif(
    not tools_available(), reason="ffmpeg/ffprobe not installed"
)


def make_job(workspace: Path, **overrides) -> AdapterJob:
    defaults = dict(
        job_id="00000000-0000-0000-0000-0000000000ff",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="a cinematic drone shot over a coastline",
        parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        inputs=[],
        execution={"runtime": "ltx"},
        output_content_type="video/mp4",
        workspace=workspace,
    )
    return AdapterJob(**{**defaults, **overrides})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "job"
    path.mkdir()
    return path


@pytest.fixture
def fake_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A weights directory that passes the existence check without 85 GB."""
    root = tmp_path / "models"
    from worker.adapters.ltx import _MODEL_FILES

    for relative in _MODEL_FILES.values():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(settings, "ltx_model_dir", root)
    return root


@pytest.fixture
def stub_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory that exists, standing in for the LTX repo."""
    repo = tmp_path / "ltx-repo"
    repo.mkdir()
    monkeypatch.setattr(settings, "ltx_repo_dir", repo)
    return repo


def stub_launcher(monkeypatch: pytest.MonkeyPatch, script: Path) -> None:
    """Routes `_launcher()` to a local Python script; every real flag from
    `_command()` still lands in the stub's argv."""
    monkeypatch.setattr(LtxAdapter, "_launcher", lambda self: [sys.executable, str(script)])


async def collect(job: AdapterJob):
    reported: list[tuple[str, int, str]] = []

    async def on_progress(status: str, progress: int, message: str) -> None:
        reported.append((status, progress, message))

    result = await LtxAdapter().run(job, on_progress)
    return result, reported


# ── The contract ─────────────────────────────────────────────────────────


def test_the_ltx_adapter_satisfies_the_protocol() -> None:
    assert isinstance(LtxAdapter(), GenerationAdapter)


# ── Command construction (pure, no GPU, no ffmpeg) ───────────────────────


def test_the_command_carries_every_flag_the_benchmark_needed(workspace: Path) -> None:
    """The exact flag set that produced video on the RTX 5090 — a missing VAE
    path was a hard startup error there, so the full set is pinned here."""
    cmd = LtxAdapter()._command(make_job(workspace), 2.0, workspace / "output.mp4")

    for flag in (
        "--transformer-path", "--text-encoder-path", "--video-vae-path",
        "--audio-vae-path", "--duration-head-path", "--spatial-upsampler-path",
        "--quantization", "--prompt", "--num-frames", "--height", "--width",
        "--frame-rate", "--seed", "--output-path",
    ):
        assert flag in cmd, f"missing {flag}"

    def value_of(flag: str) -> str:
        return cmd[cmd.index(flag) + 1]

    assert value_of("--prompt") == "a cinematic drone shot over a coastline"
    assert value_of("--num-frames") == "48"  # 2s at 24 fps
    assert value_of("--quantization") == "nvfp4-prequant"
    assert "nvfp4" in value_of("--transformer-path")
    assert value_of("--output-path") == str(workspace / "output.mp4")


@pytest.mark.parametrize("aspect", ["16:9", "9:16", "1:1", "4:5", None])
def test_every_dimension_is_divisible_by_64(aspect, workspace: Path) -> None:
    """LTX's two-stage pipeline rejects anything else outright — 480x848 was
    refused on the GPU before a single frame was generated."""
    job = make_job(workspace, parameters={"duration": "2s", "aspect_ratio": aspect})
    cmd = LtxAdapter()._command(job, 2.0, workspace / "output.mp4")

    width = int(cmd[cmd.index("--width") + 1])
    height = int(cmd[cmd.index("--height") + 1])
    assert width % 64 == 0 and height % 64 == 0


def test_seeds_differ_between_jobs_and_repeat_within_one(workspace: Path) -> None:
    """The pipeline's default seed is fixed: without this, two users with the
    same prompt receive the same video. A retried job, though, should
    reproduce rather than surprise."""
    def seed_for(job_id: str) -> str:
        job = make_job(workspace, job_id=job_id)
        cmd = LtxAdapter()._command(job, 2.0, workspace / "output.mp4")
        return cmd[cmd.index("--seed") + 1]

    assert seed_for("job-a") == seed_for("job-a")
    assert seed_for("job-a") != seed_for("job-b")


# ── Guardrails (no GPU, no ffmpeg) ───────────────────────────────────────


async def test_an_audio_job_is_refused_loudly(workspace: Path, fake_models: Path) -> None:
    """Benchmarked: the distilled entry point cannot emit audio-only output.
    Routing music here must fail once with the reason, not burn retries."""
    job = make_job(
        workspace,
        workflow_id="music",
        execution={"runtime": "ltx", "output_kind": "audio"},
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert raised.value.retriable is False
    assert "libx264" in raised.value.internal_detail


async def test_a_video_conditioned_job_is_refused_not_silently_ignored(
    workspace: Path, fake_models: Path
) -> None:
    """The dangerous failure is quiet: a video-to-video job whose source is
    ignored would return unrelated footage that *looks* like success."""
    job = make_job(
        workspace,
        workflow_id="video-to-video",
        inputs=[
            AdapterInput(
                role="source_video",
                kind="video",
                content_type="video/mp4",
                download_url="https://storage.test/signed",
                path=workspace / "source.mp4",
            )
        ],
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert raised.value.retriable is False
    assert "source_video" in raised.value.internal_detail


async def test_a_request_beyond_the_measured_ceiling_is_refused(
    workspace: Path, fake_models: Path
) -> None:
    """60s hard-OOMed at 29.6/31.4 GiB on the RTX 5090. Until segmentation is
    enabled for this runtime, longer requests must not reach the GPU."""
    job = make_job(workspace, parameters={"duration": "60s", "aspect_ratio": "16:9"})
    with pytest.raises(AdapterError) as raised:
        await collect(job)

    assert raised.value.retriable is False
    assert "ceiling" in raised.value.internal_detail


async def test_missing_weights_fail_before_any_subprocess(workspace: Path) -> None:
    """A node without the models is a deployment mistake; the log should name
    the missing files and the customer should see only generic copy."""
    with pytest.raises(AdapterError) as raised:
        await collect(make_job(workspace))

    assert raised.value.retriable is False
    assert "safetensors" in raised.value.internal_detail
    assert ".safetensors" not in raised.value.user_message


# ── Image-to-video conditioning ──────────────────────────────────────────


def make_i2v_job(workspace: Path, image_path: Path | None, **overrides) -> AdapterJob:
    return make_job(
        workspace,
        workflow_id="image-to-video",
        parameters={"duration": "5s", "aspect_ratio": "16:9", "quality": "High"},
        inputs=[
            AdapterInput(
                role="source_image",
                kind="image",
                content_type="image/png",
                download_url="https://storage.test/signed",
                path=image_path,
            )
        ],
        **overrides,
    )


def test_the_adapter_declares_i2v_support() -> None:
    adapter = LtxAdapter()
    assert adapter.supports("text-to-video")
    assert adapter.supports("image-to-video")
    assert not adapter.supports("music")


def test_the_command_pins_the_still_as_frame_zero_at_full_strength(
    workspace: Path,
) -> None:
    """The upstream contract is positional: `--image PATH FRAME_IDX STRENGTH`.
    A wrong order silently conditions the wrong frame, so the exact argv
    sequence is pinned, not just the flag's presence."""
    still = workspace / "still.png"
    cmd = LtxAdapter()._command(
        make_i2v_job(workspace, still), 5.0, workspace / "output.mp4",
        conditioning_image=still,
    )

    at = cmd.index("--image")
    assert cmd[at : at + 4] == ["--image", str(still), "0", "1.0"]


def test_a_text_to_video_command_carries_no_image_flag(workspace: Path) -> None:
    cmd = LtxAdapter()._command(make_job(workspace), 2.0, workspace / "output.mp4")
    assert "--image" not in cmd


async def test_an_unstaged_image_is_an_internal_error_not_a_render(
    workspace: Path, fake_models: Path
) -> None:
    """`path=None` means the runner's staging step was skipped — a platform
    bug, and it must surface before any subprocess launches."""
    with pytest.raises(AdapterError) as raised:
        await collect(make_i2v_job(workspace, image_path=None))

    assert raised.value.retriable is False
    assert "not staged" in raised.value.internal_detail


@needs_ffmpeg
async def test_an_unreadable_image_fails_without_burning_retries_or_gpu_time(
    workspace: Path, fake_models: Path
) -> None:
    """A corrupt upload is corrupt on all three attempts; the probe turns a
    wasted render into an immediate, actionable answer."""
    junk = workspace / "still.png"
    junk.write_bytes(b"definitely not an image")

    with pytest.raises(AdapterError) as raised:
        await collect(make_i2v_job(workspace, junk))

    assert raised.value.retriable is False
    assert "ffprobe" not in raised.value.user_message.lower()
    assert "image" in raised.value.user_message.lower()


@needs_ffmpeg
async def test_the_full_i2v_run_conditions_on_the_staged_still(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The i2v twin of the t2v end-to-end: a real still in, every real flag
    through the seam, a real verified MP4 back."""
    still = workspace / "still.png"
    await ffmpeg(
        ["-f", "lavfi", "-i", "testsrc2=size=896x512:rate=1", "-frames:v", "1", str(still)]
    )
    fixture = tmp_path / "render.mp4"
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=896x512:rate=24",
            "-t", "5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(fixture),
        ]
    )

    script = tmp_path / "render.py"
    script.write_text(
        "import shutil, sys\n"
        "args = sys.argv[1:]\n"
        "at = args.index('--image')\n"
        "image_args = args[at + 1 : at + 4]\n"
        "out = args[args.index('--output-path') + 1]\n"
        # `args.index('--image')` above is the assertion: a t2v-shaped argv
        # (no conditioning) crashes the stub, failing the run. Only a command
        # that carried the staged still through the seam completes.
        "print(f'CONDITIONED_ON {image_args}', flush=True)\n"
        f"shutil.copyfile({str(fixture)!r}, out)\n"
        "print(f'INFO:...:Video saved to {out}', flush=True)\n"
    )
    stub_launcher(monkeypatch, script)

    result, reported = await collect(make_i2v_job(workspace, still))

    assert result.content_type == "video/mp4"
    assert result.duration_seconds == pytest.approx(5.0, abs=0.75)
    assert result.path.parent == workspace
    statuses = [status for status, _, _ in reported]
    assert statuses[0] == "preparing" and statuses[-1] == "uploading"


# ── Progress parsing (pure) ──────────────────────────────────────────────


def test_markers_walk_forward_through_the_real_log_sequence() -> None:
    """The exact INFO lines observed on the GPU, in order — including the
    twice-occurring denoising line mapping to two different steps."""
    lines = [
        "INFO:ltx_pipelines.utils.blocks:Building text encoder from /x",
        "INFO:ltx_pipelines.utils.blocks:Running denoising loop (8 steps, 448x256 ...)",
        "INFO:ltx_pipelines.utils.blocks:Building video encoder + spatial upsampler from /y",
        "INFO:ltx_pipelines.utils.blocks:Running denoising loop (3 steps, 896x512 ...)",
        "INFO:ltx_pipelines.utils.blocks:Building video decoder from /z",
        "INFO:ltx_pipelines.utils.media_io.encode:Video saved to /tmp/out.mp4",
    ]
    start, seen = 0, []
    for line in lines:
        matched = match_marker(line, start)
        if matched is not None:
            seen.append(_MARKERS[matched][1])
            start = matched + 1

    assert seen == [20, 40, 55, 70, 80, 85]
    assert seen == sorted(seen), "progress must never move backwards"


def test_a_skipped_milestone_does_not_wedge_the_sequence() -> None:
    """Variant pipelines log different stages; matching is forward-scanning so
    an absent line just means that step's progress is skipped."""
    matched = match_marker("INFO:...:Building video decoder from /z", 0)
    assert matched is not None
    assert _MARKERS[matched][1] == 80


def test_marker_progress_stays_inside_the_generating_band() -> None:
    """The API ranks statuses strictly forward; all pipeline milestones must
    fit between `preparing` (10) and `post_processing` (90)."""
    values = [progress for _, progress, _ in _MARKERS]
    assert values == sorted(values)
    assert all(15 <= value <= 85 for value in values)


# ── Supervision (subprocess stubs, no GPU, no ffmpeg) ────────────────────


async def test_a_failing_pipeline_surfaces_its_output_tail(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\n"
        "print('torch.OutOfMemoryError: CUDA out of memory')\n"
        "sys.exit(3)\n"
    )
    stub_launcher(monkeypatch, script)

    with pytest.raises(AdapterError) as raised:
        await collect(make_job(workspace))

    assert raised.value.retriable is True
    assert "exited 3" in raised.value.internal_detail
    assert "CUDA out of memory" in raised.value.internal_detail
    # The customer never sees provider vocabulary.
    assert "cuda" not in raised.value.user_message.lower()


async def test_cancellation_kills_the_render_process(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole reason supervision exists: a cancelled job must stop holding
    the GPU within seconds, not when the render happens to finish."""
    sentinel = tmp_path / "survived.txt"
    script = tmp_path / "slow.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "print('started', flush=True)\n"
        "time.sleep(4)\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('the kill missed')\n"
    )
    stub_launcher(monkeypatch, script)

    cancelled = asyncio.Event()
    cancelled.set()
    job = make_job(workspace, _cancelled=cancelled)

    began = time.monotonic()
    with pytest.raises(JobCancelled):
        await collect(job)
    assert time.monotonic() - began < 3, "cancellation must not wait for the render"

    # A killed child can never write the sentinel; a survivor would within 4s.
    await asyncio.sleep(5)
    assert not sentinel.exists(), "the subprocess outlived its job"


async def test_an_expired_budget_kills_the_render_the_same_way(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    stub_launcher(monkeypatch, script)

    job = make_job(workspace, _deadline_monotonic=0.0)
    began = time.monotonic()
    with pytest.raises(JobTimedOut):
        await collect(job)
    assert time.monotonic() - began < 5


async def test_a_missing_ltx_environment_is_a_configuration_error(
    workspace: Path, fake_models: Path, stub_repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`uv` absent / wrong repo dir — a mis-deployed node must fail its first
    job with the launch failure named, not retry into the same wall."""
    monkeypatch.setattr(
        LtxAdapter, "_launcher", lambda self: ["definitely-not-a-real-binary-7817"]
    )
    with pytest.raises(AdapterError) as raised:
        await collect(make_job(workspace))

    assert raised.value.retriable is False
    assert "could not launch" in raised.value.internal_detail


# ── End to end against a stub render (needs ffmpeg) ──────────────────────


@needs_ffmpeg
async def test_the_full_run_produces_a_verified_result(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Everything the runner sees from a successful LTX job, with only the
    model swapped out: real flags in, real MP4 out, measured metadata back."""
    fixture = tmp_path / "render.mp4"
    await ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=896x512:rate=24",
            "-t", "2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(fixture),
        ]
    )

    script = tmp_path / "render.py"
    script.write_text(
        "import shutil, sys\n"
        "args = sys.argv[1:]\n"
        "out = args[args.index('--output-path') + 1]\n"
        "print('INFO:...:Building text encoder from /x', flush=True)\n"
        "print('INFO:...:Running denoising loop (8 steps)', flush=True)\n"
        "print('INFO:...:Building video decoder from /z', flush=True)\n"
        f"shutil.copyfile({str(fixture)!r}, out)\n"
        "print(f'INFO:...:Video saved to {out}', flush=True)\n"
    )
    stub_launcher(monkeypatch, script)

    result, reported = await collect(make_job(workspace))

    assert result.content_type == "video/mp4"
    assert result.kind == "video"
    # Measured from the file, not echoed from the request.
    assert result.duration_seconds == pytest.approx(2.0, abs=0.75)
    assert (result.width, result.height) == (896, 512)
    assert result.path.parent == workspace

    order = [status for status, _, _ in reported]
    assert order == sorted(
        order, key=["preparing", "generating", "post_processing", "uploading"].index
    )
    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)
    assert progress[-1] < 100
