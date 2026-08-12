"""Internal worker protocol — auth, claiming, leasing, retries, SSE.

These cover the properties the whole scaling story rests on. If any of them
regress, the system still *appears* to work with one worker and quietly breaks
with two.
"""

from __future__ import annotations

import asyncio

from httpx import AsyncClient
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

WORKFLOWS = [
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "extend-video",
    "music",
    "music-video",
]


async def register(client: AsyncClient, headers: dict, name: str = "test-worker") -> str:
    response = await client.post(
        "/api/v1/internal/workers/register",
        headers=headers,
        json={"name": name, "runtime": "mock", "workflows": WORKFLOWS, "max_concurrency": 2},
    )
    assert response.status_code == 201, response.text
    return response.json()["worker_id"]


async def claim(
    client: AsyncClient,
    headers: dict,
    worker_id: str,
    runtimes: list[str] | None = None,
) -> dict | None:
    payload: dict = {"worker_id": worker_id}
    if runtimes is not None:
        payload["runtimes"] = runtimes
    response = await client.post(
        "/api/v1/internal/jobs/claim", headers=headers, json=payload
    )
    assert response.status_code == 200, response.text
    return response.json()["job"]


# ── Authentication (directive §16, §17) ──────────────────────────────────


async def test_internal_endpoints_require_the_service_token(client: AsyncClient) -> None:
    for path, payload in [
        ("/api/v1/internal/workers/register", {"name": "x"}),
        ("/api/v1/internal/jobs/claim", {"worker_id": "00000000-0000-0000-0000-000000000000"}),
    ]:
        response = await client.post(path, json=payload)
        assert response.status_code == 401, f"{path} is reachable without a token"


async def test_a_wrong_token_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/internal/workers/register",
        headers={"X-Worker-Token": "not-the-token"},
        json={"name": "x"},
    )
    assert response.status_code == 401


async def test_internal_routes_are_absent_from_the_public_schema(client: AsyncClient) -> None:
    """The customer-facing OpenAPI document must not advertise the worker API."""
    schema = (await client.get("/openapi.json")).json()
    assert not [path for path in schema["paths"] if "internal" in path]


async def test_registration_rejects_an_unknown_workflow(
    client: AsyncClient, worker_headers: dict
) -> None:
    """A typo would otherwise leave a worker idle forever, looking like a
    capacity problem rather than a configuration one."""
    response = await client.post(
        "/api/v1/internal/workers/register",
        headers=worker_headers,
        json={"name": "typo-worker", "workflows": ["text-to-vidio"]},
    )
    assert response.status_code == 422
    assert "text-to-vidio" in response.text


async def test_re_registering_reuses_the_same_identity(
    client: AsyncClient, worker_headers: dict
) -> None:
    """A restarted container must not add a row per restart."""
    first = await register(client, worker_headers, "stable-name")
    second = await register(client, worker_headers, "stable-name")
    assert first == second


# ── Claiming ─────────────────────────────────────────────────────────────


async def test_claiming_an_empty_queue_is_not_an_error(
    client: AsyncClient, worker_headers: dict
) -> None:
    worker_id = await register(client, worker_headers)
    assert await claim(client, worker_headers, worker_id) is None


async def test_a_claim_carries_everything_the_worker_needs(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)

    job = await claim(client, worker_headers, worker_id)
    assert job is not None

    # Presigned output target — the worker never invents storage keys.
    assert job["output_upload_url"].startswith("http")
    assert job["output_upload_key"].startswith("users/")
    assert job["lease_token"]
    assert job["attempt"] == 1

    # The private execution block DOES cross this boundary — to an
    # authenticated worker on the private network, and nowhere else.
    assert job["execution"]["runtime"] == "mock"


async def test_two_workers_never_receive_the_same_job(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """`FOR UPDATE SKIP LOCKED` — the property the worker design rests on.

    Without it, N workers either serialise behind one lock or duplicate work.
    """
    await client.post("/api/v1/generations", json=text_to_video_request)

    a = await register(client, worker_headers, "worker-a")
    b = await register(client, worker_headers, "worker-b")

    first, second = await asyncio.gather(
        claim(client, worker_headers, a), claim(client, worker_headers, b)
    )

    claimed = [job for job in (first, second) if job is not None]
    assert len(claimed) == 1, "the same job was handed to two workers"


async def test_claims_are_fair_oldest_first(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    ids = []
    for index in range(3):
        response = await client.post(
            "/api/v1/generations", json=dict(text_to_video_request, prompt=f"job {index}")
        )
        ids.append(response.json()["job_id"])

    worker_id = await register(client, worker_headers)
    first = await claim(client, worker_headers, worker_id)
    assert first["job_id"] == ids[0], "claiming is not FIFO"


# ── Leasing and reporting ────────────────────────────────────────────────


async def test_progress_advances_the_job_and_is_visible_publicly(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/progress",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "status": "generating",
            "progress": 62,
            "message": "This usually takes a couple of minutes.",
        },
    )
    assert ack.json()["accepted"] is True

    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "generating"
    assert public["stage_label"] == "Generating"
    assert public["progress"] == 62


async def test_a_stale_lease_token_is_refused(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """The zombie-worker guard.

    A process that stalls past its lease and wakes up must not overwrite the
    state of whichever worker took the job over.
    """
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/progress",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": "00000000-0000-0000-0000-000000000000",
            "status": "generating",
            "progress": 50,
            "message": "",
        },
    )
    assert ack.json()["accepted"] is False
    assert "lease" in ack.json()["reason"]


async def test_progress_never_moves_backwards(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """An out-of-order report would make the bar jump left — which reads as a
    fault even on a perfectly healthy job."""
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    base = {"worker_id": worker_id, "lease_token": job["lease_token"], "message": ""}
    for status, progress in (("generating", 62), ("generating", 20)):
        await client.post(
            f"/api/v1/internal/jobs/{job['job_id']}/progress",
            headers=worker_headers,
            json={**base, "status": status, "progress": progress},
        )

    assert (await client.get(f"/api/v1/generations/{job['job_id']}")).json()["progress"] == 62


async def test_a_cancelled_job_rejects_further_worker_reports(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """Cancellation stops the work by invalidating the lease — the API never
    has to reach out to the worker, and could not."""
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    await client.post(f"/api/v1/generations/{job_id}/cancel")

    ack = await client.post(
        f"/api/v1/internal/jobs/{job_id}/progress",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "status": "generating",
            "progress": 80,
            "message": "",
        },
    )
    assert ack.json()["accepted"] is False


# ── Completion and failure ───────────────────────────────────────────────


async def test_completion_registers_an_asset_and_finishes_the_job(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/complete",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": "image",
            "output_content_type": "image/png",
            "size_bytes": 4096,
            "width": 960,
            "height": 540,
        },
    )
    assert ack.json()["accepted"] is True

    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "completed"
    assert public["progress"] == 100
    assert public["is_terminal"] is True
    assert len(public["outputs"]) == 1
    assert public["outputs"][0]["url"], "the result carries no presigned URL"

    # The output is in the media library.
    media = (await client.get("/api/v1/media")).json()
    assert any(item["source"] == "generated" for item in media["items"])


async def test_a_retriable_failure_requeues_and_the_attempt_count_rises(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/fail",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "user_message": "Something went wrong.",
            "internal_detail": "simulated",
            "retriable": True,
        },
    )

    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "queued", "a retriable failure did not requeue"

    again = await claim(client, worker_headers, worker_id)
    assert again["attempt"] == 2


async def test_retries_are_bounded(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """No infinite retries (directive §23). max_attempts is 3."""
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    worker_id = await register(client, worker_headers)

    for _ in range(4):
        job = await claim(client, worker_headers, worker_id)
        if job is None:
            break
        await client.post(
            f"/api/v1/internal/jobs/{job_id}/fail",
            headers=worker_headers,
            json={
                "worker_id": worker_id,
                "lease_token": job["lease_token"],
                "user_message": "Something went wrong.",
                "internal_detail": "simulated",
                "retriable": True,
            },
        )

    public = (await client.get(f"/api/v1/generations/{job_id}")).json()
    assert public["status"] == "failed"
    assert public["attempt_count"] <= 3
    assert await claim(client, worker_headers, worker_id) is None


async def test_worker_internals_never_reach_the_customer(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """A worker is trusted to generate, not to write customer-facing copy.

    Anything that looks like a traceback or a model detail is replaced wholesale
    rather than trimmed (directive §23, architecture rule #10).
    """
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/fail",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "user_message": 'Traceback: File "/opt/ltx/pipeline.py", CUDA out of memory',
            "internal_detail": "torch.cuda.OutOfMemoryError on device 0",
            "retriable": False,
        },
    )

    body = (await client.get(f"/api/v1/generations/{job['job_id']}")).text
    for leaked in ("Traceback", "ltx", "pipeline.py", "CUDA", "torch"):
        assert leaked.lower() not in body.lower(), f"'{leaked}' reached the customer"

    assert "could not be completed" in body


# ── Lease recovery ───────────────────────────────────────────────────────


async def test_an_expired_lease_is_requeued(
    client: AsyncClient,
    worker_headers: dict,
    text_to_video_request: dict,
    db: AsyncSession,
) -> None:
    """A worker that dies without reporting must not strand its job.

    The lease is expired directly here rather than by waiting two minutes — the
    behaviour under test is the reaper's, not the clock's.
    """
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    worker_id = await register(client, worker_headers)
    await claim(client, worker_headers, worker_id)

    await db.execute(
        sql_text(
            "UPDATE generation_jobs SET lease_expires_at = now() - interval '1 hour' "
            "WHERE id = :id"
        ),
        {"id": job_id},
    )
    await db.commit()

    reaped = await client.post("/api/v1/internal/maintenance/reap-leases", headers=worker_headers)
    assert reaped.json()["requeued"] == 1

    public = (await client.get(f"/api/v1/generations/{job_id}")).json()
    assert public["status"] == "queued"

    # And another worker can now pick it up.
    assert await claim(client, worker_headers, worker_id) is not None


# ── Runtime routing (M2) ─────────────────────────────────────────────────
#
# Which adapter runs a workflow is declared in its private `execution.runtime`.
# Nothing used to check that the claiming worker could actually run it, which
# was harmless while every definition said `mock`. The moment one says something
# else, a mock node claims that job, finds no adapter, and fails it with
# `retriable=False` — permanently dead, and the customer is simply told the tool
# is unavailable.


async def test_a_worker_never_claims_work_it_cannot_run(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """The regression guard for a mixed fleet.

    Every shipped workflow is routed to `mock`, so a node that serves only a
    GPU runtime must come away empty rather than taking the job and killing it.
    """
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)

    assert await claim(client, worker_headers, worker_id, runtimes=["ltx"]) is None

    # And the job is still queued for a node that can run it.
    assert await claim(client, worker_headers, worker_id, runtimes=["mock"]) is not None


async def test_declaring_several_runtimes_widens_what_a_node_may_claim(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)

    assert await claim(client, worker_headers, worker_id, runtimes=["ltx", "mock"]) is not None


async def test_a_worker_that_declares_nothing_keeps_the_old_behaviour(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """A pre-M2 worker cannot assert its runtimes. Starving it during a rolling
    upgrade would turn a deploy into an outage."""
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)

    assert await claim(client, worker_headers, worker_id) is not None


async def test_capability_is_re_asserted_on_every_claim_not_trusted_from_registration(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """A node restarted with a different runtime must not inherit what its
    previous incarnation recorded."""
    await client.post("/api/v1/generations", json=text_to_video_request)

    response = await client.post(
        "/api/v1/internal/workers/register",
        headers=worker_headers,
        json={
            "name": "was-a-mock-node",
            "runtime": "mock",
            "runtimes": ["mock"],
            "workflows": WORKFLOWS,
            "max_concurrency": 1,
        },
    )
    worker_id = response.json()["worker_id"]

    # Same row, now running a GPU build: the stale "mock" capability must not
    # let it claim mock-routed work.
    assert await claim(client, worker_headers, worker_id, runtimes=["ltx"]) is None


async def test_registration_records_runtimes_for_operators(
    client: AsyncClient, worker_headers: dict, db: AsyncSession
) -> None:
    """"Which nodes can run this workflow?" is the first question asked when a
    queue stops draining, and it should be answerable from the fleet table."""
    await client.post(
        "/api/v1/internal/workers/register",
        headers=worker_headers,
        json={
            "name": "gpu-node-1",
            "runtime": "ltx",
            "runtimes": ["ltx", "harness"],
            "workflows": WORKFLOWS,
            "max_concurrency": 1,
        },
    )

    recorded = (
        await db.execute(
            sql_text("SELECT capabilities FROM worker_nodes WHERE name = 'gpu-node-1'")
        )
    ).scalar_one()
    assert recorded["runtimes"] == ["ltx", "harness"]
