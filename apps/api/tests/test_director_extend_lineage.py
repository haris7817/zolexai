"""Director lineage on Extend — resolved at creation, from rows that existed.

The chain under test is built entirely from completed jobs (the mock runtime's
outputs are READY assets), so no object storage is involved:

    T2V (standard, mock)  → image output
        ↓ becomes the source_image of
    I2V, Director mode, Spanish  → video output
        ↓ becomes the source_video of
    Extend #1  → carries director_lineage + the ORIGINAL image as identity
        ↓ its video output becomes the source of
    Extend #2  → inherits the lineage, accumulates the seconds, keeps identity

And the guarantee on the other side: a standard job's output extends with
NOTHING added — no lineage key, no injected input, byte-identical behaviour.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.test_worker_protocol import claim, register

#: Director mode left Image to Video on 28 Aug 2026 and Text to Video on
#: 5 Sep 2026 (client decisions), so no public request can create the
#: Director ancestor this chain starts from. The lineage machinery stays in
#: the API and the worker for rollback; the proof waits with it.
pytestmark = pytest.mark.skip(
    reason="Director mode is unrouted since 5 Sep 2026; lineage code kept for rollback"
)


async def _complete(
    client: AsyncClient,
    headers: dict,
    worker_id: str,
    *,
    kind: str,
    content_type: str,
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
            "output_content_type": content_type,
            "size_bytes": 4096,
        },
    )
    assert ack.json()["accepted"] is True, ack.text
    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "completed"
    return public


def _primary_asset(public: dict) -> str:
    return public["outputs"][0]["asset_id"]


async def _seed_image(client: AsyncClient, headers: dict, worker_id: str) -> str:
    """A READY image asset, made the only way tests can make one: a job."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "text-to-video",
            "prompt": "a still frame to seed the chain",
            "parameters": {"duration": "10s", "aspect_ratio": "16:9"},
        },
    )
    assert response.status_code == 202, response.text
    done = await _complete(
        client, headers, worker_id, kind="image", content_type="image/png"
    )
    return _primary_asset(done)


async def test_extending_a_director_video_carries_its_whole_world(
    client: AsyncClient, worker_headers: dict
) -> None:
    worker_id = await register(client, worker_headers)
    image_asset = await _seed_image(client, worker_headers, worker_id)

    # The ancestor: Image to Video, Director mode, Spanish.
    created = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "image-to-video",
            "prompt": "La mujer y el robot hablan del futuro de la educación.",
            "parameters": {
                "duration": "30s",
                "aspect_ratio": "16:9",
                "prompt_mode": "director",
                "dialogue_language": "spanish",
            },
            "inputs": {"source_image": image_asset},
        },
    )
    assert created.status_code == 202, created.text
    parent = await _complete(
        client, worker_headers, worker_id, kind="video", content_type="video/mp4"
    )

    # Extend it. The request itself says nothing about Director mode — the
    # lineage is the server's own resolution of the source's ancestry.
    extended = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "extend-video",
            "prompt": "Se levantan y caminan juntos hacia casa.",
            "parameters": {"duration": "10s", "aspect_ratio": "16:9"},
            "inputs": {"source_video": _primary_asset(parent)},
        },
    )
    assert extended.status_code == 202, extended.text
    job = (
        await client.get(f"/api/v1/generations/{extended.json()['job_id']}")
    ).json()

    lineage = job["parameters"]["director_lineage"]
    assert lineage["prompt_mode"] == "director"
    assert lineage["dialogue_language"] == "spanish"
    assert lineage["idea"] == "La mujer y el robot hablan del futuro de la educación."
    assert lineage["prior_seconds"] == 30.0
    assert lineage["source_workflow"] == "image-to-video"
    assert lineage["identity_image_asset_id"] == image_asset

    # The ORIGINAL upload rides along as a server-attached input, so the
    # worker can keep it as the identity anchor.
    roles = {item["role"]: item["asset_id"] for item in job["inputs"]}
    assert roles["source_video"] == _primary_asset(parent)
    assert roles["identity_image"] == image_asset

    # Extend the extension: the lineage is inherited whole, the seconds
    # accumulate, and the identity image survives another generation.
    second = await _complete(
        client, worker_headers, worker_id, kind="video", content_type="video/mp4"
    )
    again = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "extend-video",
            "prompt": "Siguen caminando mientras cae la tarde.",
            "parameters": {"duration": "10s", "aspect_ratio": "16:9"},
            "inputs": {"source_video": _primary_asset(second)},
        },
    )
    assert again.status_code == 202, again.text
    job2 = (
        await client.get(f"/api/v1/generations/{again.json()['job_id']}")
    ).json()

    lineage2 = job2["parameters"]["director_lineage"]
    assert lineage2["dialogue_language"] == "spanish"
    assert lineage2["idea"] == lineage["idea"]
    assert lineage2["prior_seconds"] == 40.0  # 30s original + 10s first extension
    roles2 = {item["role"]: item["asset_id"] for item in job2["inputs"]}
    assert roles2["identity_image"] == image_asset


async def test_extending_a_standard_video_stores_nothing_extra(
    client: AsyncClient, worker_headers: dict
) -> None:
    """The other half of the contract: no ancestry, no change — the stored
    job is exactly what the request said, and no input is injected."""
    worker_id = await register(client, worker_headers)
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "text-to-video",
            "prompt": "a slow drone shot over a fjord",
            "parameters": {"duration": "30s", "aspect_ratio": "16:9"},
        },
    )
    assert response.status_code == 202
    parent = await _complete(
        client, worker_headers, worker_id, kind="video", content_type="video/mp4"
    )

    extended = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "extend-video",
            "prompt": "the shot keeps drifting up the valley",
            "parameters": {"duration": "10s", "aspect_ratio": "16:9"},
            "inputs": {"source_video": _primary_asset(parent)},
        },
    )
    assert extended.status_code == 202, extended.text
    job = (
        await client.get(f"/api/v1/generations/{extended.json()['job_id']}")
    ).json()

    assert "director_lineage" not in job["parameters"]
    assert [item["role"] for item in job["inputs"]] == ["source_video"]
