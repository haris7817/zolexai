"""Worker unit tests — the adapter seam, the mock runtime, and error handling.

These deliberately do not need an API, a database or object storage. The
worker's whole design is that it depends on none of those directly, and a test
suite that had to spin them up would be evidence the boundary had leaked.
"""

from __future__ import annotations

import asyncio
import struct
import zlib
from pathlib import Path

import pytest

from worker.adapters.base import (
    AdapterError,
    AdapterInput,
    AdapterJob,
    GenerationAdapter,
    JobCancelled,
    JobTimedOut,
    parse_duration_seconds,
)
from worker.adapters.harness import HarnessAdapter
from worker.adapters.mock import STAGES, MockAdapter
from worker.adapters.registry import available_runtimes, get_adapter
from worker.workflows.resolver import build_adapter_job, resolve_adapter


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Every job owns a directory; adapters write nothing outside it."""
    path = tmp_path / "job"
    path.mkdir()
    return path


def make_job(workspace: Path | None = None, **overrides) -> AdapterJob:
    defaults = dict(
        job_id="00000000-0000-0000-0000-000000000001",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="A cinematic drone shot",
        parameters={"duration": "10s", "aspect_ratio": "16:9", "quality": "High"},
        inputs=[],
        execution={"runtime": "mock"},
        output_content_type="image/png",
        workspace=workspace or Path(),
    )
    return AdapterJob(**{**defaults, **overrides})


# ── The provider abstraction (directive §12) ─────────────────────────────


def test_the_mock_adapter_satisfies_the_protocol() -> None:
    """The contract M2's real adapter must also satisfy.

    If this ever fails, the seam has changed shape and a provider swap is no
    longer a drop-in.
    """
    assert isinstance(MockAdapter(), GenerationAdapter)


def test_the_harness_adapter_satisfies_the_protocol() -> None:
    """The harness drives the same seam as a provider, which is the only reason
    testing against it says anything about the real thing."""
    assert isinstance(HarnessAdapter(), GenerationAdapter)


def test_routing_comes_from_the_workflow_definition() -> None:
    """Which adapter runs is declared in version-controlled YAML, not in code —
    so moving a workflow to a real provider in M2 is a config change."""
    assert resolve_adapter(make_job()).name == "mock"
    assert available_runtimes() == ["harness", "ltx", "mock"]


def test_an_unknown_runtime_fails_loudly_and_is_not_retried() -> None:
    """A silent fallback to mock would ship placeholder images while looking
    healthy — far more expensive to discover than a failed job."""
    with pytest.raises(AdapterError) as raised:
        get_adapter("does-not-exist")

    assert raised.value.retriable is False
    # The customer message names nothing internal.
    assert "does-not-exist" not in raised.value.user_message
    # The detail, which only reaches the log, does.
    assert "does-not-exist" in raised.value.internal_detail


def test_the_claim_payload_maps_onto_an_adapter_job() -> None:
    job = build_adapter_job(
        {
            "job_id": "abc",
            "workflow_id": "video-to-video",
            "workflow_version": "1",
            "prompt": "restyle it",
            "parameters": {"duration": "5s"},
            "inputs": [
                {
                    "role": "source_video",
                    "kind": "video",
                    "content_type": "video/mp4",
                    "download_url": "https://storage.example/signed",
                },
                {
                    "role": "reference_image",
                    "kind": "image",
                    "content_type": "image/png",
                    "download_url": "https://storage.example/signed2",
                },
            ],
            "execution": {"runtime": "mock"},
            "output_content_type": "image/png",
        }
    )

    assert job.workflow_id == "video-to-video"
    # Roles, not positions — which is what makes an OPTIONAL extra input
    # possible without changing this mapping (directive §14).
    assert job.input_for("reference_image") is not None
    assert job.input_for("missing_role") is None


# ── The mock runtime ─────────────────────────────────────────────────────


async def test_it_reports_every_lifecycle_stage_in_order(workspace: Path) -> None:
    reported: list[tuple[str, int]] = []

    async def on_progress(status: str, progress: int, _message: str) -> None:
        reported.append((status, progress))

    result = await MockAdapter().run(make_job(workspace), on_progress)

    assert [status for status, _ in reported] == [stage.status for stage in STAGES]
    # Progress is monotonic — a bar that jumps backwards reads as a fault.
    values = [progress for _, progress in reported]
    assert values == sorted(values)
    assert values[-1] < 100, "the adapter must not claim 100%; the API sets that on completion"
    assert result.content_type == "image/png"


async def test_it_produces_a_genuinely_valid_png(workspace: Path) -> None:
    """The output is uploaded under a signature that binds image/png, so it has
    to really be one — a placeholder that failed to decode would surface as a
    broken image in the result canvas."""

    async def noop(*_args) -> None:
        return None

    result = await MockAdapter().run(make_job(workspace), noop)

    # The result is a file inside the job's workspace, not bytes in memory.
    assert result.path.parent == workspace
    assert result.size_bytes == result.path.stat().st_size
    data = result.path.read_bytes()

    assert data[:8] == b"\x89PNG\r\n\x1a\n"

    # IHDR dimensions must match what the adapter reported.
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (result.width, result.height) == (960, 540)

    # Every chunk CRC must verify, which is what a decoder actually checks.
    offset = 8
    seen = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])[0]
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, f"bad CRC in {tag!r}"
        seen.append(tag)
        offset += 12 + length

    assert seen == [b"IHDR", b"IDAT", b"IEND"]


@pytest.mark.parametrize(
    ("aspect", "expected"),
    [("16:9", (960, 540)), ("9:16", (540, 960)), ("1:1", (720, 720)), (None, (960, 540))],
)
async def test_output_dimensions_follow_the_requested_aspect_ratio(
    aspect, expected, workspace: Path
) -> None:
    """Audio workflows send no aspect ratio at all, hence the None case."""

    async def noop(*_args) -> None:
        return None

    job = make_job(workspace, parameters={"duration": "10s", "aspect_ratio": aspect})
    result = await MockAdapter().run(job, noop)
    assert (result.width, result.height) == expected


async def test_consecutive_results_are_visually_distinct(tmp_path: Path) -> None:
    """Otherwise a grid of results is four identical tiles."""

    async def noop(*_args) -> None:
        return None

    first_space = tmp_path / "a"
    second_space = tmp_path / "b"
    first_space.mkdir()
    second_space.mkdir()

    first = await MockAdapter().run(make_job(first_space, job_id="job-1"), noop)
    second = await MockAdapter().run(make_job(second_space, job_id="job-2"), noop)
    assert first.path.read_bytes() != second.path.read_bytes()


async def test_the_requested_duration_is_carried_onto_the_asset(workspace: Path) -> None:
    async def noop(*_args) -> None:
        return None

    result = await MockAdapter().run(
        make_job(workspace, parameters={"duration": "30s", "aspect_ratio": "16:9"}), noop
    )
    assert result.duration_seconds == 30.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10s", 10.0),
        ("3m", 180.0),  # music durations are chosen in minutes (CR-009)
        ("1m", 60.0),
        ("10", 10.0),
        (None, None),  # automatic-duration workflows send no duration at all
        ("", None),
        ("abc", None),
        ("0s", None),
        ("-5s", None),
    ],
)
def test_duration_strings_parse_to_seconds(value, expected) -> None:
    """One parser for every adapter — "3m" quietly read as 3 seconds would ship
    a three-second song to someone who asked for three minutes."""
    assert parse_duration_seconds(value) == expected


# ── Cancellation and the time budget ─────────────────────────────────────


async def test_a_cancelled_job_stops_at_the_next_checkpoint(workspace: Path) -> None:
    """The whole point of cooperative cancellation: a job the user abandoned
    must stop consuming compute long before it would have finished."""
    cancelled = asyncio.Event()
    cancelled.set()

    async def noop(*_args) -> None:
        return None

    job = make_job(workspace, _cancelled=cancelled)
    with pytest.raises(JobCancelled):
        await MockAdapter().run(job, noop)


async def test_an_expired_budget_stops_the_job(workspace: Path) -> None:
    """A deadline already in the past is the same signal a long overrun gives."""

    async def noop(*_args) -> None:
        return None

    job = make_job(workspace, _deadline_monotonic=0.0)
    with pytest.raises(JobTimedOut):
        await MockAdapter().run(job, noop)


def test_a_job_without_cancellation_wiring_never_reports_cancelled() -> None:
    """Adapters are exercised directly in tests and tooling, where no runner has
    attached an event; that must not look like a cancellation."""
    job = make_job()
    assert job.is_cancelled is False
    assert job.seconds_remaining is None
    job.raise_if_cancelled()


def test_execution_tuning_values_fall_back_when_absent_or_malformed() -> None:
    """`execution` is `extra="allow"` so M2 can tune per workflow without a
    schema change — which also means the worker must tolerate junk in it."""
    def tuned(execution: dict) -> int:
        return make_job(execution=execution).execution_int("max_segment_seconds", 5)

    assert tuned({"max_segment_seconds": 12}) == 12
    assert tuned({"max_segment_seconds": "nope"}) == 5
    assert tuned({}) == 5


# ── Error reporting ──────────────────────────────────────────────────────


def test_adapter_errors_separate_the_two_audiences() -> None:
    """`user_message` is customer-safe; `internal_detail` never leaves the log."""
    error = AdapterError(
        "This generation could not be completed. Please try again.",
        internal_detail="torch.cuda.OutOfMemoryError on device 0",
        retriable=True,
    )
    assert "cuda" not in error.user_message.lower()
    assert "cuda" in error.internal_detail.lower()
    assert error.retriable is True


def test_adapter_input_carries_a_presigned_url_not_a_credential() -> None:
    """The worker holds no standing storage key — every file it can reach
    arrives as a URL scoped to one object for one job."""
    item = AdapterInput(
        role="source_video",
        kind="video",
        content_type="video/mp4",
        download_url="https://storage.example/bucket/key?X-Amz-Signature=abc",
    )
    assert item.download_url.startswith("https://")
    assert not hasattr(item, "access_key")
