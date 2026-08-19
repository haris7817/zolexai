"""Generation API tests — creation, validation, idempotency, limits, cancel."""

from __future__ import annotations

import asyncio

from httpx import AsyncClient


async def test_create_returns_202_immediately(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    """Generation must never block the request (scalability rule #3)."""
    response = await client.post("/api/v1/generations", json=text_to_video_request)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["stage_label"] == "Queued"
    # The client is told where to subscribe rather than constructing the URL.
    assert body["events_url"].endswith(f"/generations/{body['job_id']}/events")


async def test_job_is_persisted_and_readable(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]

    job = (await client.get(f"/api/v1/generations/{job_id}")).json()
    assert job["status"] == "queued"
    assert job["workflow_name"] == "Text to Video"
    assert job["parameters"]["duration"] == "10s"
    assert job["is_terminal"] is False
    # A queued event exists, so an SSE client reconnecting has something to replay.
    assert job["last_event_seq"] == 1


# ── Validation ───────────────────────────────────────────────────────────


async def test_rejects_aspect_ratio_on_audio_workflow(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "music",
            "prompt": "synthwave, 120bpm",
            # Music durations are in minutes since M2; a valid one keeps this
            # test pointed at its actual subject — the aspect ratio.
            "parameters": {"duration": "1m", "aspect_ratio": "16:9"},
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "unsupported_parameter"
    assert error["details"]["fields"][0]["field"] == "aspect_ratio"


async def test_rejects_unsupported_duration_and_lists_the_valid_ones(
    client: AsyncClient,
) -> None:
    """A rejection must tell the client what WOULD work."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "image-to-video",
            "prompt": "move it",
            "parameters": {"duration": "999s", "aspect_ratio": "16:9"},
            "inputs": {"source_image": "00000000-0000-0000-0000-000000000000"},
        },
    )
    assert response.status_code == 422
    fields = {f["field"]: f for f in response.json()["error"]["details"]["fields"]}
    assert fields["duration"]["allowed"] == ["5s", "10s", "15s", "30s", "60s"]


async def test_accepts_director_mode_on_text_to_video(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    """Text to Video declares `settings.prompt_modes`, so the director mode and
    its language are stored with the job exactly as sent — the worker reads
    them from there."""
    request = {
        **text_to_video_request,
        "parameters": {
            **text_to_video_request["parameters"],
            "prompt_mode": "director",
            "dialogue_language": "spanish",
        },
    }
    response = await client.post("/api/v1/generations", json=request)
    assert response.status_code == 202

    job = (await client.get(f"/api/v1/generations/{response.json()['job_id']}")).json()
    assert job["parameters"]["prompt_mode"] == "director"
    assert job["parameters"]["dialogue_language"] == "spanish"


async def test_accepts_director_mode_on_image_to_video(client: AsyncClient) -> None:
    """Image to Video declares `settings.prompt_modes` too (source-anchored
    Director). With the mode and language supplied, the ONLY failing field is
    the fabricated asset — which is the registry saying yes to everything it
    judges."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "image-to-video",
            "prompt": "A woman and a robot on a bench discuss the future of education.",
            "parameters": {
                "duration": "10s",
                "aspect_ratio": "16:9",
                "prompt_mode": "director",
                "dialogue_language": "english",
            },
            "inputs": {"source_image": "00000000-0000-0000-0000-000000000000"},
        },
    )
    assert response.status_code == 422
    fields = {f["field"] for f in response.json()["error"]["details"]["fields"]}
    assert fields == {"inputs.source_image"}


async def test_rejects_prompt_mode_on_a_workflow_without_the_control(
    client: AsyncClient,
) -> None:
    """Same policy as lyrics: present-and-unsupported is reported, never
    silently dropped — Director mode must not leak past the workflows that
    declare it."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "extend-video",
            "prompt": "keep the shot going",
            "parameters": {
                "duration": "5s",
                "aspect_ratio": "16:9",
                "prompt_mode": "director",
            },
            "inputs": {"source_video": "00000000-0000-0000-0000-000000000000"},
        },
    )
    assert response.status_code == 422
    fields = {f["field"] for f in response.json()["error"]["details"]["fields"]}
    assert "prompt_mode" in fields


async def test_rejects_a_dialogue_language_outside_director_mode(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    request = {
        **text_to_video_request,
        "parameters": {
            **text_to_video_request["parameters"],
            "dialogue_language": "spanish",
        },
    }
    response = await client.post("/api/v1/generations", json=request)
    assert response.status_code == 422
    fields = {f["field"] for f in response.json()["error"]["details"]["fields"]}
    assert "dialogue_language" in fields


async def test_rejects_an_unknown_dialogue_language_and_lists_the_valid_ones(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    request = {
        **text_to_video_request,
        "parameters": {
            **text_to_video_request["parameters"],
            "prompt_mode": "director",
            "dialogue_language": "klingon",
        },
    }
    response = await client.post("/api/v1/generations", json=request)
    assert response.status_code == 422
    fields = {f["field"]: f for f in response.json()["error"]["details"]["fields"]}
    assert "auto" in fields["dialogue_language"]["allowed"]
    assert "spanish" in fields["dialogue_language"]["allowed"]


async def test_rejects_missing_required_input(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "image-to-video",
            "prompt": "animate this",
            "parameters": {"duration": "5s", "aspect_ratio": "16:9"},
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "missing_required_input"
    assert error["details"]["fields"][0]["missing_roles"] == ["source_image"]


async def test_rejects_an_asset_the_user_does_not_own(client: AsyncClient) -> None:
    """A fabricated asset id must not become a worker's download target.

    The message is deliberately identical to "not found": distinguishing them
    would confirm that another user's asset exists.
    """
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "image-to-video",
            "prompt": "animate this",
            "parameters": {"duration": "5s", "aspect_ratio": "16:9"},
            "inputs": {"source_image": "11111111-2222-3333-4444-555555555555"},
        },
    )
    assert response.status_code == 422
    assert "not found" in response.text.lower()


async def test_rejects_an_unknown_input_role(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "text-to-video",
            "prompt": "a scene",
            "parameters": {"duration": "5s", "aspect_ratio": "16:9"},
            "inputs": {"not_a_role": "11111111-2222-3333-4444-555555555555"},
        },
    )
    assert response.status_code == 422


# ── Idempotency (directive §24) ──────────────────────────────────────────


async def test_replaying_an_idempotency_key_returns_the_same_job(
    client: AsyncClient, text_to_video_request: dict, idempotency_key: str
) -> None:
    headers = {"Idempotency-Key": idempotency_key}

    first = await client.post("/api/v1/generations", json=text_to_video_request, headers=headers)
    second = await client.post("/api/v1/generations", json=text_to_video_request, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 200  # not 202 — nothing new was created
    assert first.json()["job_id"] == second.json()["job_id"]

    listing = (await client.get("/api/v1/generations")).json()
    assert len(listing["items"]) == 1, "a duplicate job was created"


async def test_concurrent_double_click_creates_one_job(
    client: AsyncClient, text_to_video_request: dict, idempotency_key: str
) -> None:
    """The race a database uniqueness check alone cannot win.

    Two simultaneous requests both pass a SELECT before either INSERT lands.
    Redis `SET NX` is what closes it.
    """
    headers = {"Idempotency-Key": idempotency_key}
    results = await asyncio.gather(
        *(
            client.post("/api/v1/generations", json=text_to_video_request, headers=headers)
            for _ in range(5)
        )
    )

    assert {r.status_code for r in results} <= {200, 202, 409}
    listing = (await client.get("/api/v1/generations")).json()
    assert len(listing["items"]) == 1


async def test_a_failed_request_releases_its_key(
    client: AsyncClient, text_to_video_request: dict, idempotency_key: str
) -> None:
    """A typo must not lock the key for a day."""
    headers = {"Idempotency-Key": idempotency_key}

    bad = dict(text_to_video_request, parameters={"duration": "999s"})
    assert (await client.post("/api/v1/generations", json=bad, headers=headers)).status_code == 422

    retry = await client.post("/api/v1/generations", json=text_to_video_request, headers=headers)
    assert retry.status_code == 202


async def test_different_keys_create_different_jobs(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    """Idempotency must not suppress a genuine second generation."""
    a = await client.post(
        "/api/v1/generations", json=text_to_video_request, headers={"Idempotency-Key": "a"}
    )
    b = await client.post(
        "/api/v1/generations", json=text_to_video_request, headers={"Idempotency-Key": "b"}
    )
    assert a.json()["job_id"] != b.json()["job_id"]


# ── Fair use (directive §18) ─────────────────────────────────────────────


async def test_one_user_cannot_occupy_every_worker(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    """The default concurrency limit is 3 with no worker draining the queue."""
    for _ in range(3):
        assert (
            await client.post("/api/v1/generations", json=text_to_video_request)
        ).status_code == 202

    refused = await client.post("/api/v1/generations", json=text_to_video_request)
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "concurrency_limit_reached"


async def test_cancelling_frees_a_concurrency_slot(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    ids = [
        (await client.post("/api/v1/generations", json=text_to_video_request)).json()["job_id"]
        for _ in range(3)
    ]
    assert (await client.post("/api/v1/generations", json=text_to_video_request)).status_code == 429

    cancelled = await client.post(f"/api/v1/generations/{ids[0]}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["is_terminal"] is True

    assert (await client.post("/api/v1/generations", json=text_to_video_request)).status_code == 202


async def test_cancelling_a_finished_job_conflicts(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    assert (await client.post(f"/api/v1/generations/{job_id}/cancel")).status_code == 200

    repeat = await client.post(f"/api/v1/generations/{job_id}/cancel")
    assert repeat.status_code == 409
    assert repeat.json()["error"]["code"] == "conflict"


# ── Pagination (directive §5) ────────────────────────────────────────────


async def test_history_pages_with_a_cursor_and_never_repeats(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    for index in range(5):
        created = await client.post(
            "/api/v1/generations", json=dict(text_to_video_request, prompt=f"probe {index}")
        )
        # Stay under the concurrency limit by cancelling as we go.
        if created.status_code == 202:
            await client.post(f"/api/v1/generations/{created.json()['job_id']}/cancel")

    first = (await client.get("/api/v1/generations?limit=2")).json()
    assert len(first["items"]) == 2
    assert first["has_more"] is True

    second = (
        await client.get(f"/api/v1/generations?limit=2&cursor={first['next_cursor']}")
    ).json()

    ids = {item["id"] for item in first["items"]}
    assert ids.isdisjoint({item["id"] for item in second["items"]}), "a page repeated a row"


async def test_limit_is_capped(client: AsyncClient) -> None:
    """Without a ceiling, one request could ask for an entire history table."""
    assert (await client.get("/api/v1/generations?limit=5000")).status_code == 422


async def test_malformed_cursor_is_rejected_cleanly(client: AsyncClient) -> None:
    response = await client.get("/api/v1/generations?cursor=not-a-cursor")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failed"
