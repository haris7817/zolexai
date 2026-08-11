"""Worker unit tests — the adapter seam, the mock runtime, and error handling.

These deliberately do not need an API, a database or object storage. The
worker's whole design is that it depends on none of those directly, and a test
suite that had to spin them up would be evidence the boundary had leaked.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from worker.adapters.base import AdapterError, AdapterInput, AdapterJob, GenerationAdapter
from worker.adapters.mock import STAGES, MockAdapter
from worker.adapters.registry import available_runtimes, get_adapter
from worker.workflows.resolver import build_adapter_job, resolve_adapter


def make_job(**overrides) -> AdapterJob:
    defaults = dict(
        job_id="00000000-0000-0000-0000-000000000001",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="A cinematic drone shot",
        parameters={"duration": "10s", "aspect_ratio": "16:9", "quality": "High"},
        inputs=[],
        execution={"runtime": "mock"},
        output_content_type="image/png",
    )
    return AdapterJob(**{**defaults, **overrides})


# ── The provider abstraction (directive §12) ─────────────────────────────


def test_the_mock_adapter_satisfies_the_protocol() -> None:
    """The contract M2's real adapter must also satisfy.

    If this ever fails, the seam has changed shape and a provider swap is no
    longer a drop-in.
    """
    assert isinstance(MockAdapter(), GenerationAdapter)


def test_routing_comes_from_the_workflow_definition() -> None:
    """Which adapter runs is declared in version-controlled YAML, not in code —
    so moving a workflow to a real provider in M2 is a config change."""
    assert resolve_adapter(make_job()).name == "mock"
    assert available_runtimes() == ["mock"]


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


async def test_it_reports_every_lifecycle_stage_in_order() -> None:
    reported: list[tuple[str, int]] = []

    async def on_progress(status: str, progress: int, _message: str) -> None:
        reported.append((status, progress))

    result = await MockAdapter().run(make_job(), on_progress)

    assert [status for status, _ in reported] == [stage.status for stage in STAGES]
    # Progress is monotonic — a bar that jumps backwards reads as a fault.
    values = [progress for _, progress in reported]
    assert values == sorted(values)
    assert values[-1] < 100, "the adapter must not claim 100%; the API sets that on completion"
    assert result.content_type == "image/png"


async def test_it_produces_a_genuinely_valid_png() -> None:
    """The output is uploaded under a signature that binds image/png, so it has
    to really be one — a placeholder that failed to decode would surface as a
    broken image in the result canvas."""

    async def noop(*_args) -> None:
        return None

    result = await MockAdapter().run(make_job(), noop)
    data = result.content

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
async def test_output_dimensions_follow_the_requested_aspect_ratio(aspect, expected) -> None:
    """Audio workflows send no aspect ratio at all, hence the None case."""

    async def noop(*_args) -> None:
        return None

    job = make_job(parameters={"duration": "10s", "aspect_ratio": aspect})
    result = await MockAdapter().run(job, noop)
    assert (result.width, result.height) == expected


async def test_consecutive_results_are_visually_distinct() -> None:
    """Otherwise a grid of results is four identical tiles."""

    async def noop(*_args) -> None:
        return None

    first = await MockAdapter().run(make_job(job_id="job-1"), noop)
    second = await MockAdapter().run(make_job(job_id="job-2"), noop)
    assert first.content != second.content


async def test_the_requested_duration_is_carried_onto_the_asset() -> None:
    async def noop(*_args) -> None:
        return None

    result = await MockAdapter().run(
        make_job(parameters={"duration": "30s", "aspect_ratio": "16:9"}), noop
    )
    assert result.duration_seconds == 30.0


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
