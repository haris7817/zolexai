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
import math
import os
import time
from pathlib import Path

import pytest

# Job builders, the model substitute, the ffmpeg guard and the synthetic media
# fixtures are shared with the video-to-video, music-video and long-form suites.
from tests.conftest import (  # noqa: E402  (local package, after worker imports)
    collect,
    conditioning_of,
    invocations,
    make_clip,
    make_job,
    needs_ffmpeg,
    render_stub,
    stub_launcher,
)
from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    GenerationAdapter,
    JobCancelled,
    JobTimedOut,
    parse_duration_seconds,
)
from worker.adapters.ltx import (
    _DIMENSIONS,
    _GRID_CEILINGS,
    _MARKERS,
    ConditioningFrame,
    LtxAdapter,
    conforming_frames,
    grid_for_source,
    match_marker,
    output_dimensions,
    safe_frame_count,
)
from worker.core.config import settings
from worker.media import ffmpeg, plan_segments, probe_media

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


# ── The user's prompt reaches the model (client revision, 13 Aug 2026) ───


@pytest.mark.parametrize(
    "prompt",
    [
        "Two cars racing on a Los Angeles road",
        "3 dancers, then 2 more join",  # digits must survive as digits
        "a woman says \"don't stop\" — then turns, 50% lit",  # quotes, dash, %
        "prompt with\nan embedded newline",
        "café naïve 日本語 emoji 🎬",  # non-ASCII must not be mangled
        "  leading and trailing space  ",
        "word " * 400,  # far longer than any UI field suggests
    ],
)
def test_the_users_prompt_reaches_the_model_verbatim(prompt: str, workspace: Path) -> None:
    """The client's report was "I put something and it shows something else".

    This pins the half we control: whatever reaches the adapter is handed to
    the model as ONE argv element, byte for byte — no truncation, no quoting,
    no reflowing, and nothing prepended or appended. If a future "prompt
    improvement" ever rewrites user text, this fails loudly.
    """
    job = make_job(workspace, prompt=prompt)
    cmd = LtxAdapter()._command(job, 2.0, workspace / "output.mp4")

    assert cmd.count("--prompt") == 1
    assert cmd[cmd.index("--prompt") + 1] == prompt

    # And nothing else in the command smuggles a second prompt-ish value.
    assert not any(
        isinstance(part, str) and part.startswith("--negative-prompt") for part in cmd
    )


def test_the_prompt_is_never_rewritten_between_the_claim_and_the_command(
    workspace: Path,
) -> None:
    """The transport hop the worker owns: claim payload → AdapterJob → argv."""
    from worker.workflows.resolver import build_adapter_job

    typed = "Two cars racing on a Los Angeles road, low angle, golden hour"
    job = build_adapter_job(
        {
            "job_id": "abc",
            "workflow_id": "text-to-video",
            "workflow_version": "1",
            "prompt": typed,
            "parameters": {"duration": "5s", "aspect_ratio": "16:9"},
            "inputs": [],
            "execution": {"runtime": "ltx"},
            "output_content_type": "video/mp4",
        },
        workspace=workspace,
    )
    assert job.prompt == typed

    cmd = LtxAdapter()._command(job, 5.0, workspace / "output.mp4")
    assert cmd[cmd.index("--prompt") + 1] == typed


def test_prompt_enhancement_is_off_unless_a_workflow_asks_for_it(workspace: Path) -> None:
    """LTX's enhancer rewrites the prompt, which is exactly what a user who
    typed something specific does not want by default. It is the only
    adherence lever the distilled entry point offers, so it stays available —
    per workflow, deliberately, never implicitly."""
    plain = LtxAdapter()._command(make_job(workspace), 2.0, workspace / "output.mp4")
    assert "--enhance-prompt" not in plain

    opted_in = LtxAdapter()._command(
        make_job(workspace, execution={"runtime": "ltx", "enhance_prompt": True}),
        2.0,
        workspace / "output.mp4",
    )
    assert "--enhance-prompt" in opted_in
    # The gemma4 encode root cannot generate, so the bare flag alone is a
    # pipeline ValueError — the instruct root must ride along.
    assert (
        opted_in[opted_in.index("--prompt-enhancer-gemma-root") + 1]
        == str(settings.director_gemma_root)
    )
    # Even then the user's own words are still what is sent.
    assert opted_in[opted_in.index("--prompt") + 1] == make_job(workspace).prompt


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


def test_a_user_seed_reaches_each_section_deterministically(workspace: Path) -> None:
    job = make_job(workspace, parameters={"duration": "4s", "seed": 1234})
    adapter = LtxAdapter()
    assert adapter._seed_for_step(job, 0) == 1234
    assert adapter._seed_for_step(job, 1) == 1235


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


async def test_a_source_conditioned_job_never_reaches_the_plain_generation_path(
    workspace: Path, fake_models: Path
) -> None:
    """The dangerous failure is quiet: a source-conditioned job whose source is
    dropped returns unrelated footage that *looks* like success.

    Dispatch is on the workflow id, so this can only happen through a routing
    mistake — a workflow that carries a source but is not one of the handled
    ones. That must fail loudly rather than silently becoming text-to-video.
    """
    job = make_job(
        workspace,
        workflow_id="text-to-video",  # the plain path…
        inputs=[                      # …carrying an input it cannot honour
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


@pytest.mark.parametrize(
    ("workflow_id", "duration", "passes"),
    [
        ("text-to-video", "5s", 1),
        ("text-to-video", "30s", 1),
        ("image-to-video", "30s", 1),
        ("image-to-video", "60s", 2),
        ("extend-video", "60s", 2),
        # The long end of the extension ladder (client ask #1, 17 Aug 2026) —
        # minutes of continuation are still nothing but conforming passes.
        ("extend-video", "2m", 4),
        ("extend-video", "5m", 10),
    ],
)
def test_no_public_duration_can_become_an_oversized_gpu_pass(
    workflow_id: str, duration: str, passes: int
) -> None:
    """The safety property behind offering 60s at all.

    A 60s render hard-OOMed at 29.6/31.4 GiB on the RTX 5090, so every length
    the product offers must decompose into passes that each stay inside the
    measured ceiling. This asserts both halves: the count, and that no single
    pass exceeds it.
    """
    seconds = parse_duration_seconds(duration)
    assert seconds is not None
    segments = plan_segments(seconds, max_segment_seconds=30)

    assert len(segments) == passes
    assert all(segment.duration_seconds <= 30 for segment in segments)
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(seconds)


def test_a_workflow_cannot_raise_the_per_pass_ceiling_above_the_benchmark(
    workspace: Path,
) -> None:
    """`execution.max_segment_seconds` may make passes SMALLER. A typo that
    made them bigger would put the kernel failure back on the table, so the
    measured value is a hard clamp rather than a default."""
    adapter = LtxAdapter()
    assert adapter._per_pass_seconds(make_job(workspace)) == 60.0
    assert (
        adapter._per_pass_seconds(
            make_job(workspace, execution={"runtime": "ltx", "max_segment_seconds": 10})
        )
        == 10.0
    )
    assert (
        adapter._per_pass_seconds(
            make_job(workspace, execution={"runtime": "ltx", "max_segment_seconds": 600})
        )
        == 60.0
    )


def test_the_pass_ceiling_is_a_property_of_the_grid_not_the_product(
    workspace: Path,
) -> None:
    """The whole point of measuring per shape.

    896x512 holds FEWER pixels than 1024x576 and dies at 60s where 1024x576
    survives, so no size-derived rule can produce these two numbers. Both come
    from running them (16 Aug 2026). A global ceiling had to satisfy the worst
    shape, which cost every 60s render five seams it did not need.
    """
    adapter = LtxAdapter()
    job = make_job(workspace)
    assert adapter._per_pass_seconds(job, (1024, 576)) == 60.0
    assert adapter._per_pass_seconds(job, (896, 512)) == 30.0


class TestSafeFrameCount:
    """Frame counts the decoder can actually decode: MEASUREMENT OUTRANKS THEORY.

    The 8k+1 lattice (stated in the model's own source; every unconditioned
    crash on record was non-conforming; 737 verified by a live customer job)
    is real — and it is NOT SUFFICIENT for conditioned passes. The proof cost
    a production night, 17 Aug 2026: emptying the conditioned bands on the
    lattice theory turned the matrix-proven 720 into 721, and seven customer
    image-to-video jobs died on it. 720 passes, 721 fails — one frame apart,
    the mirror image of 1440 FAIL / 1441 PASS.

    So the order is: measured-safe conditioned counts pass through EXACTLY;
    then the lattice; then the measured bands, whose landings are used exactly
    (1528 is measured-pass and off-lattice — "1529 must be fine, it conforms"
    is the same reasoning that produced 721).
    """

    def test_the_production_regression_720_conditioned_passes_through(self) -> None:
        """The 17 Aug incident, pinned. 30s image-to-video is 720 frames
        conditioned; the matrix proved 720 and production proved 721 crashes.
        Any change that stops returning exactly 720 here reintroduces the
        outage."""
        for grid in ((1024, 576), (576, 1024), (768, 768)):
            assert safe_frame_count(grid, 720, conditioned=True) == 720

    def test_all_measured_safe_conditioned_counts_are_untouched(self) -> None:
        for frames in (120, 240, 360, 720, 1289, 1385, 1441, 1528):
            assert safe_frame_count((1024, 576), frames, conditioned=True) == frames

    def test_the_production_regression_359_conditioned_lands_on_360(self) -> None:
        """The second incident, pinned. A 14.976s upload is 359 frames; the
        lattice carries it to 361, and 361 conditioned crashed three
        video-to-video jobs on 16 Aug 2026. The 15.018s file beside it asked for
        360, passed through, and worked — so the fix is to land on the
        measurement rather than reach over it."""
        for grid in ((1024, 576), (576, 1024), (768, 768)):
            assert safe_frame_count(grid, 359, conditioned=True) == 360

    def test_361_conditioned_is_never_emitted(self) -> None:
        """It is measured-FAIL, so no request may produce it — including a
        request for exactly 361, which the lattice would otherwise pass
        straight through."""
        for frames in range(353, 366):
            for grid in ((1024, 576), (576, 1024), (768, 768)):
                assert safe_frame_count(grid, frames, conditioned=True) != 361

    def test_721_conditioned_is_never_emitted_either(self) -> None:
        """The first incident's count, guarded the same way — 719 snaps to 721
        on the lattice, which is what made a 29.96s source a crash."""
        for frames in range(714, 726):
            for grid in ((1024, 576), (576, 1024), (768, 768)):
                assert safe_frame_count(grid, frames, conditioned=True) != 721

    def test_no_conditioned_count_measured_to_fail_is_ever_emitted(self) -> None:
        """The whole invariant, swept. Every count this table records as a
        conditioned failure must be unreachable from any request."""
        measured_fail = (361, 721, 1381, 1437, 1440, 1464)
        for frames in range(1, 2000):
            out = safe_frame_count((1024, 576), frames, conditioned=True)
            assert out not in measured_fail, f"{frames} -> {out}, a measured failure"

    def test_the_production_conditioned_passes_are_reproduced_exactly(self) -> None:
        """Counts the GPU actually decoded on 16 Aug, conditioned, in customer
        jobs. Whatever the table says, these must keep coming out unchanged —
        they are the only conditioned evidence between 121 and 1289."""
        for frames in (81, 121, 360, 720, 1441):
            assert safe_frame_count((1024, 576), frames, conditioned=True) == frames

    def test_the_measured_failures_all_become_measured_passes(self) -> None:
        """Every conditioned count measured to FAIL maps to one measured to
        PASS — never to an interpolation."""
        for bad, good in (
            (721, 1289), (1381, 1385), (1437, 1441), (1440, 1441), (1464, 1528)
        ):
            assert safe_frame_count((1024, 576), bad, conditioned=True) == good

    def test_conditioned_results_are_always_measured_or_conforming(self) -> None:
        """No path may emit a count that is neither on the lattice nor in the
        measured-safe set."""
        from worker.adapters.ltx import _MEASURED_SAFE_CONDITIONED

        for frames in range(1, 2000, 7):
            out = safe_frame_count((1024, 576), frames, conditioned=True)
            assert out % 8 == 1 or out in _MEASURED_SAFE_CONDITIONED, (
                f"{frames} -> {out} is neither measured nor conforming"
            )

    def test_unconditioned_results_stay_on_the_lattice_or_measured_landings(self) -> None:
        for frames in range(1, 2000, 7):
            for grid in ((1024, 576), (576, 1024), (768, 768), (896, 640)):
                out = safe_frame_count(grid, frames, conditioned=False)
                assert out % 8 == 1 or out in (248, 736), (
                    f"{grid} {frames} -> {out}"
                )

    def test_the_lattice_overshoot_is_never_more_than_a_third_of_a_second(self) -> None:
        for frames in range(1, 3000):
            assert 0 <= conforming_frames(frames) - frames <= 7

    def test_no_result_is_ever_below_the_request(self) -> None:
        """Trimming down is exact; padding up is impossible — a landing below
        the request would deliver a short video."""
        for frames in range(1, 3000):
            for conditioned in (True, False):
                assert safe_frame_count(
                    (1024, 576), frames, conditioned=conditioned
                ) >= frames

    def test_unconditioned_bands_still_apply_with_exact_landings(self) -> None:
        """Landings are measurements and are used exactly: 736 and 248 were
        matrix-proven; snapping them would trade evidence for theory."""
        assert safe_frame_count((1024, 576), 240, conditioned=False) == 248
        assert safe_frame_count((1024, 576), 720, conditioned=False) == 736

    def test_an_unmeasured_grid_still_gets_a_conforming_count(self) -> None:
        """No band to consult, but the lattice is a property of the MODEL, not
        of a grid — so it applies to shapes nobody has probed."""
        assert safe_frame_count((896, 640), 240, conditioned=False) % 8 == 1


def test_an_unmeasured_grid_gets_the_pessimistic_ceiling(workspace: Path) -> None:
    """A shape absent from the table has never been executed.

    Because the failure is a set of bad shapes rather than a size threshold,
    an unrun grid cannot be assumed safe by being small — 768x960 fails while
    the larger 1024x576 passes. Unknown therefore means the last value proven
    across every shape, not an interpolation.
    """
    adapter = LtxAdapter()
    assert adapter._per_pass_seconds(make_job(workspace), (704, 704)) == 10.0


@needs_ffmpeg
async def test_a_long_text_to_video_is_chained_rather_than_refused(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client asked for 60s on Image to Video; the mechanism is shared, so
    this proves it on the plainest workflow: a request beyond one pass becomes
    several renders chained off each other's final frames, assembled into one
    file of the requested length."""
    fixture = await make_clip(tmp_path / "render.mp4", 2.0, audio=True)
    log = extension_stub(tmp_path, monkeypatch, fixture, image_optional=True)

    job = make_job(
        workspace,
        parameters={"duration": "4s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "max_segment_seconds": 2},
    )
    result, reported = await collect(job)

    conditioned_on = log.read_text().splitlines()
    assert conditioned_on[0] == "NONE", "the first pass of a text prompt conditions on nothing"
    assert conditioned_on[1] == str(workspace / "segment-condition-0001.png")

    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)
    messages = [message for status, _, message in reported if status == "generating"]
    assert "Generating section 1 of 2…" in messages
    assert "Generating section 2 of 2…" in messages
    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)


@needs_ffmpeg
async def test_each_longform_command_gets_only_its_assigned_dialogue(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = await make_clip(tmp_path / "render.mp4", 2.0, audio=True)
    log = render_stub(tmp_path, monkeypatch, fixture)
    prompt = """Persistent: same woman and same robot
Section 1: MAYA says first line
Section 2: ROBOT says second line"""
    job = make_job(
        workspace,
        prompt=prompt,
        parameters={"duration": "4s", "aspect_ratio": "16:9"},
        execution={"runtime": "ltx", "max_segment_seconds": 2},
    )

    await collect(job)
    prompts = [call[call.index("--prompt") + 1] for call in invocations(log)]

    assert "MAYA says first line" in prompts[0]
    assert "MAYA says first line" not in prompts[1]
    assert "ROBOT says second line" not in prompts[0]
    assert "ROBOT says second line" in prompts[1]


@needs_ffmpeg
async def test_a_silent_model_pass_cannot_be_marked_completed(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    silent = await make_clip(tmp_path / "silent.mp4", 2.0)
    render_stub(tmp_path, monkeypatch, silent)

    with pytest.raises(AdapterError) as raised:
        await collect(make_job(workspace))

    assert "no audio stream" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_long_image_to_video_conditions_the_first_pass_on_the_still(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's actual request: Image to Video at a length beyond one
    pass. The uploaded still must still be frame one."""
    still = workspace / "still.png"
    await ffmpeg(
        ["-f", "lavfi", "-i", "testsrc2=size=896x512:rate=1", "-frames:v", "1", str(still)]
    )
    fixture = await make_clip(tmp_path / "render.mp4", 2.0, audio=True)
    log = render_stub(tmp_path, monkeypatch, fixture)

    # Test-scale passes are 48 frames — not a count anyone has measured with
    # a second conditioning image, so the anchor gate would (correctly) drop
    # it. Admit the test count so this test keeps proving the ANCHOR path;
    # the drop path has its own test.
    monkeypatch.setattr(
        "worker.adapters.ltx._TWO_IMAGE_SAFE_FRAMES", frozenset({48, 120, 240, 360})
    )
    job = make_i2v_job(
        workspace,
        still,
        parameters={"duration": "4s", "aspect_ratio": "16:9", "quality": "High"},
        execution={"runtime": "ltx", "max_segment_seconds": 2},
    )
    result, _ = await collect(job)

    calls = invocations(log)
    assert conditioning_of(calls[0]) == [(str(still), 0, 1.0)]
    second = conditioning_of(calls[1])
    assert second[0] == (str(workspace / "segment-condition-0001.png"), 0, 1.0)
    assert second[1][0] == str(still), "the original identity anchor was dropped"
    assert second[1][1] > 0, "the identity anchor reset the continuation at frame zero"
    assert second[1][2] == pytest.approx(0.2)
    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)


async def test_missing_weights_fail_before_any_subprocess(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node without the models is a deployment mistake; the log should name
    the missing files and the customer should see only generic copy.

    The empty root is set explicitly rather than relied upon. This test used to
    pass only because a development machine happens to have no weights — on the
    GPU node, where they exist, it failed. A test whose result depends on which
    machine runs it is not testing what it claims to.
    """
    monkeypatch.setattr(settings, "ltx_model_dir", tmp_path / "no-models-here")

    with pytest.raises(AdapterError) as raised:
        await collect(make_job(workspace))

    assert raised.value.retriable is False
    assert "safetensors" in raised.value.internal_detail
    assert ".safetensors" not in raised.value.user_message


# ── Image-to-video conditioning ──────────────────────────────────────────


def make_i2v_job(workspace: Path, image_path: Path | None, **overrides) -> AdapterJob:
    defaults = dict(
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
    )
    return make_job(workspace, **{**defaults, **overrides})


def test_the_adapter_declares_its_workflow_support() -> None:
    """Every video workflow, and no audio one.

    Music stays off this runtime for a measured reason, not a scheduling one:
    the distilled entry point cannot write an audio-only file at all.
    """
    adapter = LtxAdapter()
    for workflow in (
        "text-to-video", "image-to-video", "extend-video", "video-to-video", "music-video"
    ):
        assert adapter.supports(workflow), workflow
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
        conditioning=[ConditioningFrame(still, 0, 1.0)],
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
    # WITH audio. The model emits a soundtrack on every pass, and since the
    # long-form hardening a silent i2v/t2v pass fails validation deliberately —
    # the client reported missing audio as a defect. A silent fixture is
    # therefore not a smaller version of a real render, it is an invalid one.
    fixture = await make_clip(tmp_path / "render.mp4", 5, audio=True, size="896x512")

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


# ── Video extension ──────────────────────────────────────────────────────


def make_extension_job(workspace: Path, source: Path | None, **overrides) -> AdapterJob:
    defaults = dict(
        workflow_id="extend-video",
        parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        inputs=[
            AdapterInput(
                role="source_video",
                kind="video",
                content_type="video/mp4",
                download_url="https://storage.test/signed",
                path=source,
            )
        ],
    )
    return make_job(workspace, **{**defaults, **overrides})


def extension_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: Path,
    *,
    sleep: float = 0.0,
    image_optional: bool = False,
) -> Path:
    """A render stub that records what it was conditioned on.

    The invocation log is how chaining is proven: each line is one render's
    `--image` argument, so the test can assert that segment 2 was conditioned
    on segment 1's extracted final frame rather than on the source's.

    `image_optional` logs "NONE" instead of failing when a pass carries no
    conditioning at all — which is exactly what the first pass of a
    text-to-video chain must look like.
    """
    log = tmp_path / "invocations.log"
    script = tmp_path / "render.py"
    lookup = (
        "img = args[args.index('--image') + 1] if '--image' in args else 'NONE'\n"
        if image_optional
        else "img = args[args.index('--image') + 1]\n"
    )
    script.write_text(
        "import pathlib, shutil, sys, time\n"
        "args = sys.argv[1:]\n"
        + lookup
        + "out = args[args.index('--output-path') + 1]\n"
        f"with pathlib.Path({str(log)!r}).open('a') as f:\n"
        "    f.write(img + '\\n')\n"
        "print('INFO:...:Running denoising loop (8 steps)', flush=True)\n"
        + (f"time.sleep({sleep})\n" if sleep else "")
        + f"shutil.copyfile({str(fixture)!r}, out)\n"
        "print(f'INFO:...:Video saved to {out}', flush=True)\n"
    )
    stub_launcher(monkeypatch, script)
    return log


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((1920, 1080), (1024, 576)),
        ((1080, 1920), (576, 1024)),
        ((720, 720), (768, 768)),
        # 4:3 has no exact grid inside the budget — 1024x768 is 786k px against
        # a 590k ceiling — so it takes the largest shape within the aspect
        # tolerance instead.
        ((160, 120), (896, 640)),
        ((None, None), (1024, 576)),
    ],
)
def test_the_generation_grid_follows_the_source_aspect(source, expected) -> None:
    """The I2V benchmark's hard lesson: a grid that fights the conditioning
    image's aspect keeps the style and replaces the subject. For an extension
    that means a different video after the seam."""
    assert grid_for_source(*source) == expected


def test_a_source_aspect_keeps_its_shape_even_without_a_measured_ceiling() -> None:
    """Aspect fidelity outranks pass length.

    A 4:3 upload has no measured grid. Snapping it to the nearest measured one
    would be a 16:9 render of a 4:3 source, and a mismatched aspect is what made
    the model keep the style and replace the subject. So it renders at its true
    shape and is chained at the pessimistic ceiling instead — a seam is
    recoverable, a swapped subject is not.
    """
    grid = grid_for_source(160, 120)
    # The same 0.08 log-aspect tolerance `grid_for_source` selects within: close
    # enough that the crop is invisible, which is the point of allowing it.
    assert abs(math.log(grid[0] / grid[1]) - math.log(4 / 3)) <= 0.08
    assert grid not in _GRID_CEILINGS, "test is meaningless if this becomes measured"

    adapter = LtxAdapter()
    ceiling = adapter._per_pass_seconds(make_job(Path(".")), grid)
    assert ceiling == 10.0, "an unmeasured shape must be chained, not gambled on"


def test_every_reachable_grid_is_legal_for_the_model() -> None:
    for width, height in [(w, h) for w in (100, 640, 1280, 3840) for h in (90, 480, 2160)]:
        gw, gh = grid_for_source(width, height)
        assert gw % 64 == 0 and gh % 64 == 0
        assert gw * gh <= 1024 * 576, "beyond the measured budget"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((3840, 2160), (1920, 1080)),  # 4K capped to full HD
        ((1080, 1920), (1080, 1920)),  # portrait full HD is already legal
        ((161, 121), (160, 120)),      # odd sizes rounded even for yuv420p
        ((None, None), (1024, 576)),
    ],
)
def test_output_resolution_is_the_sources_capped_and_even(source, expected) -> None:
    """A user's 1080p clip must not come back at the generation grid's 512p —
    the original footage sets the delivery resolution."""
    assert output_dimensions(*source) == expected


@pytest.mark.parametrize("duration", ["5s", "10s", "15s", "30s", "60s"])
def test_every_public_duration_is_a_single_pass_on_a_measured_grid(duration) -> None:
    """The client contract (5/10/15/30/60) against the GPU fact.

    Before NATTEN the per-pass ceiling was 10s, so a 60s render was six passes
    with five seams — and every boundary symptom the client reported (action
    replay, identity drift, the visible pause) can only occur AT a boundary.
    Measured 16 Aug on the RTX PRO 6000, every current grid sustains 60s in one
    pass, so the product's whole duration range is now seamless.

    This is the regression guard for that: if a grid's ceiling is ever lowered
    below 60, this fails and says exactly which duration started chaining again.
    """
    seconds = float(duration.rstrip("s"))
    adapter = LtxAdapter()
    job = make_job(Path("."))
    for aspect, grid in _DIMENSIONS.items():
        # The grid is passed explicitly, so the job's own aspect is irrelevant —
        # this asks each shape directly what it was measured at.
        ceiling = adapter._per_pass_seconds(job, grid)
        assert len(plan_segments(seconds, max_segment_seconds=ceiling)) == 1, (
            f"{aspect} {grid} chains at {duration} (ceiling {ceiling}s)"
        )


def test_the_extension_command_pins_dimensions_and_seed(workspace: Path) -> None:
    still = workspace / "condition.png"
    cmd = LtxAdapter()._command(
        make_extension_job(workspace, workspace / "src.mp4"),
        2.0,
        workspace / "part.mp4",
        conditioning=[ConditioningFrame(still, 0, 1.0)],
        dimensions=(768, 576),
        seed=7,
    )
    assert cmd[cmd.index("--width") + 1] == "768"
    assert cmd[cmd.index("--height") + 1] == "576"
    assert cmd[cmd.index("--seed") + 1] == "7"
    at = cmd.index("--image")
    assert cmd[at : at + 4] == ["--image", str(still), "0", "1.0"]


async def test_an_unstaged_source_video_is_an_internal_error(
    workspace: Path, fake_models: Path
) -> None:
    with pytest.raises(AdapterError) as raised:
        await collect(make_extension_job(workspace, source=None))

    assert raised.value.retriable is False
    assert "not staged" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_corrupt_source_fails_without_burning_retries(
    workspace: Path, fake_models: Path
) -> None:
    junk = workspace / "source.mp4"
    junk.write_bytes(b"not a video")

    with pytest.raises(AdapterError) as raised:
        await collect(make_extension_job(workspace, junk))

    assert raised.value.retriable is False
    assert "video" in raised.value.user_message.lower()


@needs_ffmpeg
async def test_an_extension_source_is_not_held_to_the_render_source_ceiling(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client ask #1 ("make video extension unlimited"), the half that was
    actually broken: the second extension's SOURCE is the first extension's
    output, so holding extensions to `ltx_max_source_seconds` capped the chain
    at 5½ minutes total. Extension renders only the continuation — the source
    is re-encoded, never re-rendered — so it carries its own, looser ceiling.

    The render ceiling is pinned BELOW this test's source; if the extension
    path ever consults it again, this refuses instead of rendering.
    """
    monkeypatch.setattr(settings, "ltx_max_source_seconds", 1.0)
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    fixture = await make_clip(tmp_path / "render.mp4", 2.0)
    extension_stub(tmp_path, monkeypatch, fixture)

    result, _ = await collect(make_extension_job(workspace, source))

    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)


@needs_ffmpeg
async def test_an_extension_source_beyond_its_own_ceiling_is_refused_before_compute(
    workspace: Path, fake_models: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The looser ceiling is still a ceiling — an unbounded upload is ffmpeg
    time and disk on a card other customers are queued for. Same contract as
    the music-video refusal: immediate, named lengths, not retriable."""
    monkeypatch.setattr(settings, "ltx_max_extend_source_seconds", 1.0)
    source = await make_clip(workspace / "source.mp4", 2.0)

    with pytest.raises(AdapterError) as raised:
        await collect(make_extension_job(workspace, source))

    assert raised.value.retriable is False
    assert "trim" in raised.value.user_message.lower()
    assert "source ceiling" in raised.value.internal_detail


@needs_ffmpeg
async def test_a_single_pass_extension_stitches_source_plus_continuation(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core promise: 2s source + 2s extension ≈ 4s output, delivered at
    the SOURCE's resolution with the source's audio preserved."""
    source = await make_clip(workspace / "source.mp4", 2.0, audio=True)
    fixture = await make_clip(tmp_path / "render.mp4", 2.0)
    log = extension_stub(tmp_path, monkeypatch, fixture)

    result, reported = await collect(make_extension_job(workspace, source))

    assert result.duration_seconds == pytest.approx(4.0, abs=1.0)
    assert (result.width, result.height) == (160, 120), "delivery at source resolution"
    info = await probe_media(result.path)
    assert info.has_audio is True, "the original's audio must not be destroyed"

    conditioned_on = log.read_text().splitlines()
    assert conditioned_on == [str(workspace / "seed-frame.png")]
    # A single pass never mentions sections (harness rule).
    assert all("section" not in message.lower() for _, _, message in reported)

    statuses = [status for status, _, _ in reported]
    assert statuses == sorted(
        statuses, key=["preparing", "generating", "post_processing", "uploading"].index
    )


@needs_ffmpeg
async def test_a_long_extension_chains_segments_off_each_others_final_frames(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 60s architecture in miniature: a request beyond one pass becomes
    several renders, each conditioned on the PREVIOUS segment's last frame,
    assembled into one file of the planned total length."""
    source = await make_clip(workspace / "source.mp4", 2.0)  # silent on purpose
    fixture = await make_clip(tmp_path / "render.mp4", 2.0)
    log = extension_stub(tmp_path, monkeypatch, fixture)

    job = make_extension_job(
        workspace,
        source,
        parameters={"duration": "4s", "aspect_ratio": "16:9", "quality": "High"},
        execution={"runtime": "ltx", "max_segment_seconds": 2},
    )
    result, reported = await collect(job)

    conditioned_on = log.read_text().splitlines()
    assert conditioned_on == [
        str(workspace / "seed-frame.png"),  # the source's final frame
        str(workspace / "continuation-condition-0001.png"),  # segment 1's final frame
    ]
    assert result.duration_seconds == pytest.approx(6.0, abs=1.5)

    info = await probe_media(result.path)
    assert info.has_audio is False, "a silent source stays silent — no invented audio"

    messages = [message for status, _, message in reported if status == "generating"]
    assert "Generating section 1 of 2…" in messages
    assert "Generating section 2 of 2…" in messages
    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress), "chained segments must not restart the bar"


@needs_ffmpeg
async def test_a_wrong_length_continuation_fails_verification(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render that quietly produced the wrong length must fail the job, not
    ship as "the extension feels short"."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    wrong = await make_clip(tmp_path / "render.mp4", 5.0)  # asked for 2s
    extension_stub(tmp_path, monkeypatch, wrong)

    with pytest.raises(AdapterError) as raised:
        await collect(make_extension_job(workspace, source))

    assert "differs from planned" in raised.value.internal_detail


@needs_ffmpeg
async def test_cancellation_stops_the_chain_before_the_next_segment(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled two-segment job must not start segment two — every further
    render would be paid GPU time on a job nobody wants."""
    source = await make_clip(workspace / "source.mp4", 2.0)
    fixture = await make_clip(tmp_path / "render.mp4", 2.0)
    log = extension_stub(tmp_path, monkeypatch, fixture, sleep=15.0)

    cancelled = asyncio.Event()

    async def cancel_on_first_render(
        status: str, progress: int, message: str, _details=None
    ) -> None:
        if status == "generating":
            cancelled.set()

    job = make_extension_job(
        workspace,
        source,
        parameters={"duration": "4s", "aspect_ratio": "16:9", "quality": "High"},
        execution={"runtime": "ltx", "max_segment_seconds": 2},
        _cancelled=cancelled,
    )

    began = time.monotonic()
    with pytest.raises(JobCancelled):
        await LtxAdapter().run(job, cancel_on_first_render)
    assert time.monotonic() - began < 10, "cancellation must not wait out the render"
    assert len(log.read_text().splitlines()) == 1, "segment two must never start"


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

    # Out of memory depends on what else held the card, so another attempt can
    # genuinely succeed. This is the case retries exist for.
    assert raised.value.retriable is True
    assert "exited 3" in raised.value.internal_detail
    assert "CUDA out of memory" in raised.value.internal_detail
    # The customer never sees provider vocabulary.
    assert "cuda" not in raised.value.user_message.lower()


@pytest.mark.parametrize(
    "crash",
    [
        "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling "
        "`cublasGemmStridedBatchedEx(handle, opa, opb, (int)m, (int)n, (int)k, ...)`",
        "RuntimeError: CUDA error: no kernel image is available for execution",
        "RuntimeError: CUDA error: invalid argument",
    ],
)
async def test_a_shape_crash_is_not_retried(
    crash: str, workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some pipeline failures are properties of the input, not of the moment.

    The VAE's batched GEMM casts its dimensions to int32 and fails on
    particular frame counts — the same count crashes on every attempt. On
    2026-08-16 a music-video job failed twice with exactly this, six minutes
    in each time, spending twelve minutes of a card that other customers were
    queued behind to reach the identical crash.

    Distinct from out-of-memory above, which really does depend on what else
    was running.
    """
    script = tmp_path / "fail.py"
    script.write_text(f"import sys\nprint({crash!r})\nsys.exit(1)\n")
    stub_launcher(monkeypatch, script)

    with pytest.raises(AdapterError) as raised:
        await collect(make_job(workspace))

    assert raised.value.retriable is False, "the same input crashes the same way"
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


@pytest.mark.skipif(
    not hasattr(os, "killpg"),
    reason="process groups are POSIX; the worker runs on Linux and the stub "
    "launcher on Windows spawns nothing to orphan",
)
async def test_cancellation_kills_what_the_render_itself_spawned(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killing the handle is not killing the render, and production proved it.

    The launcher is `uv run python -m ltx_pipelines.distilled`: the handle is
    uv, and every byte of VRAM belongs to the python process uv started. On
    2026-08-16 a job lost its lease, cleanup signalled the handle and logged
    `ltx_kill_timeout`, the worker claimed the next job in the same second,
    and that job died with 43 MB free of a 102 GB card — the "killed" render
    was still holding ~56 GB.

    So this mirrors the real shape: a parent that spawns a survivor and then
    waits. Only a signal to the process GROUP reaches the survivor.
    """
    sentinel = tmp_path / "grandchild-survived.txt"

    # Two files rather than a `-c` one-liner: the survivor's body has to
    # contain a quoted path, and nesting that inside a quoted argument inside
    # a written script is how the first version of this test came to spawn
    # nothing at all and pass against the very code it was written to catch.
    survivor = tmp_path / "survivor.py"
    survivor.write_text(
        "import pathlib, time\n"
        "time.sleep(4)\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('the kill missed the child')\n"
    )

    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(survivor)!r}])\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"
    )
    stub_launcher(monkeypatch, script)

    # Cancelled AFTER the render is up, not before. Pre-setting the event —
    # which the sibling tests do, correctly, for their own purpose — kills the
    # process on the read loop's first pass, before it has spawned anything,
    # so there is no orphan to miss and the test passes against any code at
    # all. That is how the first version of this test came to pass against the
    # pre-fix adapter.
    cancelled = asyncio.Event()
    job = make_job(workspace, _cancelled=cancelled)

    async def cancel_once_it_is_running() -> None:
        await asyncio.sleep(1.5)
        cancelled.set()

    began = time.monotonic()
    with pytest.raises(JobCancelled):
        await asyncio.gather(collect(job), cancel_once_it_is_running())
    assert time.monotonic() - began < 8, "cancellation must not wait for the render"

    # The survivor writes at +4s. Waiting past that is the whole test: an
    # orphan is invisible until it does something, and on the GPU the thing it
    # does is hold the card until the next job fails.
    await asyncio.sleep(5)
    assert not sentinel.exists(), (
        "a process the render spawned outlived the job — on the GPU this is "
        "the orphan that OOMs whatever runs next"
    )


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
        LtxAdapter,
        "_launcher",
        lambda self, module=None: ["definitely-not-a-real-binary-7817"],
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
    # WITH audio — see the i2v twin. A silent pass is rejected on purpose.
    fixture = await make_clip(tmp_path / "render.mp4", 2, audio=True, size="896x512")

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
    # The fixture is 896x512; the result is the 16:9 render grid. The gap is
    # the point — sections are re-encoded to one explicit dimension before
    # concat, so a pass that came back the wrong size cannot reach the customer
    # as a resolution change mid-video. Read from `_DIMENSIONS` so that raising
    # a grid never silently strands this assertion on the old value again.
    assert (result.width, result.height) == _DIMENSIONS["16:9"]
    assert result.path.parent == workspace

    order = [status for status, _, _ in reported]
    assert order == sorted(
        order, key=["preparing", "generating", "post_processing", "uploading"].index
    )
    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)
    assert progress[-1] < 100


@needs_ffmpeg
async def test_image_to_video_restates_who_is_on_screen_in_every_later_section(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The uploaded still conditions pixels; the prompt never mentioned them.

    Across a chain the text prior is unopposed, and what it produces is a
    generic — younger — person. Reported 28 Aug 2026 as a reference image
    whose AGE the result did not follow, which is exactly what a prompt with
    no age in it does. Music video's answer applies unchanged here: caption
    the picture once, and tell every later section who the person is.
    """
    from worker.media import extract_final_frame

    clip = await make_clip(workspace / "seed.mp4", 1.0)
    still = await extract_final_frame(clip, workspace / "still.png")

    seen: list[Path] = []

    async def fake_describer(image_path) -> str:
        seen.append(Path(image_path))
        return "a 61-year-old man with short grey hair and a white beard, in a navy coat"

    monkeypatch.setattr("worker.adapters.ltx.reference_person_facts", fake_describer)
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0, audio=True)
    )

    job = make_i2v_job(
        workspace,
        still,
        parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    )
    await collect(job)

    prompts = [argv[argv.index("--prompt") + 1] for argv in invocations(log)]
    assert len(prompts) > 1, "this test needs a chained render to say anything"
    assert seen == [still], "described once per video, and it is the UPLOAD it describes"
    assert "61-year-old" not in prompts[0], (
        "section 1 opens on the still itself — its pixels are the description"
    )
    for prompt in prompts[1:]:
        assert "61-year-old man with short grey hair" in prompt


@needs_ffmpeg
async def test_image_to_video_identity_description_is_never_a_dependency(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"" is a real answer: no checkpoint, a text-only one, a timeout, garbage.

    The prompt then reads exactly as it did before this existed, which is why
    the behaviour needs no flag to be safe.
    """
    from worker.media import extract_final_frame

    clip = await make_clip(workspace / "seed.mp4", 1.0)
    still = await extract_final_frame(clip, workspace / "still.png")

    async def silent(image_path) -> str:
        return ""

    monkeypatch.setattr("worker.adapters.ltx.reference_person_facts", silent)
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0, audio=True)
    )

    job = make_i2v_job(
        workspace,
        still,
        parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    )
    await collect(job)

    for argv in invocations(log):
        prompt = argv[argv.index("--prompt") + 1]
        assert "The same person stays on screen" not in prompt


@needs_ffmpeg
async def test_generated_audio_is_told_what_language_to_speak(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime writes its own audio from this same prompt.

    There is no separate audio prompt and no language flag, so a prompt that
    says nothing about language leaves the model to choose — and it chooses
    badly. Client, 28 Aug 2026: "everything I hit sound in images comes with a
    language that is not english". Director mode never showed this because its
    plan puts quoted dialogue in the prompt and the language rides in the
    words; the standard path said nothing about sound at all.
    """
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0, audio=True)
    )
    await collect(make_job(
        workspace,
        parameters={"duration": "2s", "aspect_ratio": "16:9", "quality": "High"},
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    ))

    for argv in invocations(log):
        prompt = argv[argv.index("--prompt") + 1]
        assert "If anyone speaks or sings, they do so in English." in prompt


@needs_ffmpeg
async def test_the_language_sentence_is_conditional_and_the_choice_is_the_customers(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two properties that matter more than the default itself.

    It is phrased "if anyone speaks" because a prompt that says "the woman
    speaks English" invents a speaking woman in a silent scene — a text
    encoder cannot hear an instruction as optional, and naming a noun is how
    you summon it. And a customer who picks a language gets that language.
    """
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0, audio=True)
    )
    await collect(make_job(
        workspace,
        parameters={
            "duration": "2s", "aspect_ratio": "16:9",
            "quality": "High", "dialogue_language": "spanish",
        },
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    ))

    prompt = invocations(log)[0][invocations(log)[0].index("--prompt") + 1]
    assert "If anyone speaks or sings, they do so in Spanish." in prompt
    assert prompt.startswith("If anyone") is False, "the user's own text stays first"


@needs_ffmpeg
async def test_a_muted_result_is_not_given_a_language_sentence(
    workspace: Path, fake_models: Path, stub_repo: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sound: false` strips the audio after rendering, so the sentence would
    buy nothing — and it is not free, because every word moves the picture."""
    log = render_stub(
        tmp_path, monkeypatch, await make_clip(tmp_path / "render.mp4", 1.0, audio=True)
    )
    await collect(make_job(
        workspace,
        parameters={
            "duration": "2s", "aspect_ratio": "16:9",
            "quality": "High", "sound": False,
        },
        execution={"runtime": "ltx", "max_segment_seconds": 1},
    ))

    for argv in invocations(log):
        assert "speaks or sings" not in argv[argv.index("--prompt") + 1]
