"""Extend Video as a repeatable chain, with optional first and last frames.

Client brief, 6 Sep 2026: an extension may carry a first frame and/or a
last frame (both optional, both images), and any result — an original or an
extension — must be extendable again without limit. "Unlimited" means
unlimited chained extension JOBS, never one infinite render: every step is
its own job with its own output, and the source is never rewritten.

The chain here is built the only way tests can build one: the mock
worker's protocol completes each job with a video output, which becomes the
next job's source. Nothing touches object storage.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from pathlib import Path

from app.core.config import settings as app_settings
from app.services.workflow_registry import ValidationFailed, load_registry
from tests.test_worker_protocol import claim, register

REGISTRY = load_registry(Path(app_settings.workflow_definitions_dir))


# ── The catalogue and the request contract ─────────────────────────────────


async def test_extend_offers_the_source_and_two_optional_stills(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/extend-video")).json()
    roles = {item["role"]: item for item in workflow["inputs"]}
    assert list(roles) == ["source_video", "first_frame", "last_frame"]
    assert roles["source_video"]["required"] is True
    assert roles["first_frame"]["required"] is False
    assert roles["first_frame"]["kind"] == "image"
    assert roles["first_frame"]["label"] == "FIRST FRAME"
    assert roles["last_frame"]["required"] is False
    assert roles["last_frame"]["kind"] == "image"
    assert roles["last_frame"]["label"] == "LAST FRAME"
    for role in ("first_frame", "last_frame"):
        assert "optional" in roles[role]["help"].lower()
    # The ladder and the rest of the contract did not move.
    assert workflow["supported_durations"] == ["5s", "10s", "15s", "30s"]
    assert workflow["capabilities"]["extend"] is True


def test_the_stills_are_optional_and_the_source_is_not() -> None:
    for roles in ({"source_video"}, {"source_video", "first_frame"},
                  {"source_video", "last_frame"}, {"source_video", "first_frame", "last_frame"}):
        REGISTRY.validate_request(
            workflow_id="extend-video",
            prompt="the walk continues",
            duration="10s",
            aspect_ratio="16:9",
            quality=None,
            input_roles=roles,
        )
    with pytest.raises(ValidationFailed) as raised:
        REGISTRY.validate_request(
            workflow_id="extend-video",
            prompt="the walk continues",
            duration="10s",
            aspect_ratio="16:9",
            quality=None,
            input_roles={"first_frame", "last_frame"},
        )
    [problem] = [p for p in raised.value.details["fields"] if p["field"] == "inputs"]
    assert problem["missing_roles"] == ["source_video"]
    with pytest.raises(ValidationFailed):
        REGISTRY.validate_request(
            workflow_id="extend-video",
            prompt="the walk continues",
            duration="10s",
            aspect_ratio="16:9",
            quality=None,
            input_roles={"source_video", "reference_image"},
        )


# ── The chain, through the worker protocol ─────────────────────────────────


async def _finish(
    client: AsyncClient, headers: dict, worker_id: str, *, kind: str = "video",
    duration_seconds: float | None = None,
) -> dict:
    """Claims the oldest queued job, completes it, returns its public view."""
    job = await claim(client, headers, worker_id)
    assert job is not None
    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/complete",
        headers=headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": kind,
            "output_content_type": "video/mp4" if kind == "video" else "image/png",
            "size_bytes": 4096,
            **({"duration_seconds": duration_seconds} if duration_seconds is not None else {}),
        },
    )
    assert ack.json()["accepted"] is True, ack.text
    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "completed"
    return public


async def _claim_and_fail(client: AsyncClient, headers: dict, worker_id: str) -> dict:
    job = await claim(client, headers, worker_id)
    assert job is not None
    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/fail",
        headers=headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "user_message": "The video service is not available right now.",
            "internal_detail": "ComfyUI: CUDA out of memory (test)",
            "retriable": False,
        },
    )
    assert ack.json()["accepted"] is True, ack.text
    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "failed"
    return public


def _output(public: dict) -> str:
    return public["outputs"][0]["asset_id"]


async def _extend(client: AsyncClient, source: str, *, duration: str = "30s", **stills: str) -> dict:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "extend-video",
            "prompt": "the shot continues",
            "parameters": {"duration": duration, "aspect_ratio": "16:9"},
            "inputs": {"source_video": source, **stills},
        },
    )
    assert response.status_code == 202, response.text
    return (await client.get(f"/api/v1/generations/{response.json()['job_id']}")).json()


async def _seed_video(client: AsyncClient, headers: dict, worker_id: str, seconds: float) -> dict:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "text-to-video",
            "prompt": "an original to extend",
            "parameters": {"duration": "30s", "aspect_ratio": "16:9"},
        },
    )
    assert response.status_code == 202, response.text
    return await _finish(client, headers, worker_id, duration_seconds=seconds)


async def _seed_image(client: AsyncClient, headers: dict, worker_id: str) -> str:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "text-to-video",
            "prompt": "a still to frame with",
            "parameters": {"duration": "5s", "aspect_ratio": "16:9"},
        },
    )
    assert response.status_code == 202, response.text
    return _output(await _finish(client, headers, worker_id, kind="image"))


async def test_an_extension_can_be_extended_again_without_limit(
    client: AsyncClient, worker_headers: dict
) -> None:
    """original → +30 → +30 → +30 → +30: every step accepted, every step a new
    job whose source is the previous OUTPUT, the original never rewritten, and
    the stored record counts the chain 1, 2, 3, 4 with no cap anywhere."""
    worker_id = await register(client, worker_headers)
    original = await _seed_video(client, worker_headers, worker_id, 30.041667)
    original_output = _output(original)

    source, source_seconds = original_output, 30.041667
    expected_parent = original["id"]
    for depth in (1, 2, 3, 4):
        job = await _extend(client, source)
        record = job["parameters"]["extension"]
        assert record["generation"] == depth
        assert record["parent_job_id"] == expected_parent
        assert record["source_seconds"] == pytest.approx(source_seconds)
        assert {i["role"]: i["asset_id"] for i in job["inputs"]} == {"source_video": source}
        assert job["parameters"]["duration"] == "30s"
        # The next step's source is THIS step's output, never the original.
        done = await _finish(
            client, worker_headers, worker_id, duration_seconds=source_seconds + 30.0
        )
        assert done["id"] == job["id"]
        assert _output(done) != source
        source, source_seconds = _output(done), source_seconds + 30.0
        expected_parent = done["id"]

    # The original is untouched: still one output, still the same asset.
    again = (await client.get(f"/api/v1/generations/{original['id']}")).json()
    assert [o["asset_id"] for o in again["outputs"]] == [original_output]
    # And that original can STILL be extended — a chain is not a lock.
    branch = await _extend(client, original_output)
    assert branch["parameters"]["extension"] == {
        "generation": 1,
        "parent_job_id": original["id"],
        "source_seconds": pytest.approx(30.041667),
    }


async def test_first_and_last_frames_travel_with_the_extension(
    client: AsyncClient, worker_headers: dict
) -> None:
    """Both stills are accepted, stored by role, and delivered to the worker as
    inputs with their own download URLs — beside the source video."""
    worker_id = await register(client, worker_headers)
    original = await _seed_video(client, worker_headers, worker_id, 10.041667)
    first = await _seed_image(client, worker_headers, worker_id)
    last = await _seed_image(client, worker_headers, worker_id)

    job = await _extend(client, _output(original), first_frame=first, last_frame=last)
    roles = {i["role"]: i for i in job["inputs"]}
    assert set(roles) == {"source_video", "first_frame", "last_frame"}
    assert roles["first_frame"]["kind"] == "image"
    assert roles["last_frame"]["kind"] == "image"

    claimed = await claim(client, worker_headers, worker_id)
    assert claimed is not None and claimed["job_id"] == job["id"]
    delivered = {i["role"]: i for i in claimed["inputs"]}
    assert set(delivered) == {"source_video", "first_frame", "last_frame"}
    assert delivered["first_frame"]["asset_id"] == first
    assert delivered["last_frame"]["asset_id"] == last
    for item in delivered.values():
        assert item["download_url"].startswith("http")

    # A still of the wrong kind is refused by role, before any job exists.
    wrong = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "extend-video",
            "prompt": "the shot continues",
            "parameters": {"duration": "10s", "aspect_ratio": "16:9"},
            "inputs": {"source_video": _output(original), "last_frame": _output(original)},
        },
    )
    assert wrong.status_code == 422
    fields = {f["field"] for f in wrong.json()["error"]["details"]["fields"]}
    assert fields == {"inputs.last_frame"}


async def test_a_failed_or_cancelled_extension_does_not_lock_the_chain(
    client: AsyncClient, worker_headers: dict
) -> None:
    """A step that fails (worker error) or is cancelled leaves the source as
    extendable as before: the next Extend of the same video is accepted, and
    the chain record counts only completed ancestors."""
    worker_id = await register(client, worker_headers)
    original = await _seed_video(client, worker_headers, worker_id, 30.041667)
    source = _output(original)

    failed_job = await _extend(client, source)
    failed = await _claim_and_fail(client, worker_headers, worker_id)
    assert failed["id"] == failed_job["id"]
    assert failed["outputs"] == []
    assert failed["error"]["message"] == "The video service is not available right now."

    cancelled_job = await _extend(client, source)
    cancelled = (
        await client.post(f"/api/v1/generations/{cancelled_job['id']}/cancel")
    ).json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["outputs"] == []

    # Same source, third attempt: accepted, generation 1 (nothing completed
    # in between), and it completes like any other.
    retry = await _extend(client, source)
    assert retry["parameters"]["extension"]["generation"] == 1
    assert retry["parameters"]["extension"]["parent_job_id"] == original["id"]
    done = await _finish(client, worker_headers, worker_id, duration_seconds=60.0)
    assert done["id"] == retry["id"]
    # …and its output is extendable in turn.
    deeper = await _extend(client, _output(done))
    assert deeper["parameters"]["extension"]["generation"] == 2
    assert deeper["parameters"]["extension"]["parent_job_id"] == done["id"]

