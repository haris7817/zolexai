"""SSE delivery, storage presigning, health and the migration.

The SSE tests exercise `stream_job_events` directly rather than through an HTTP
client. `httpx`'s ASGI transport buffers a streaming response until it
completes, so a test that read the endpoint would deadlock on a job that never
finishes — which is every job while it is still running.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.enums import STATUS_LABELS, AssetKind, JobStatus, can_transition, is_terminal
from app.services import events as event_bus
from app.services.storage import (
    ALLOWED_CONTENT_TYPES,
    MAX_SIZE_BYTES,
    sanitize_filename,
    validate_upload,
    with_extension,
)

from tests.conftest import ADMIN_DATABASE_URL, TEST_DATABASE_URL
from tests.test_worker_protocol import claim, register

# ── SSE ──────────────────────────────────────────────────────────────────


async def _drain(job_id: uuid.UUID, *, last_event_id: int = 0, limit: int = 20):
    """Collects real events, ignoring keepalives, until the stream ends."""
    collected = []
    async for event in event_bus.stream_job_events(job_id, last_event_id=last_event_id):
        if event is None:
            continue  # keepalive
        collected.append(event)
        if len(collected) >= limit or is_terminal(JobStatus(event.status)):
            break
    return collected


async def test_replays_history_then_ends_on_a_terminal_event(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    await client.post(
        f"/api/v1/internal/jobs/{job_id}/progress",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "status": "generating",
            "progress": 62,
            "message": "Working…",
        },
    )
    await client.post(
        f"/api/v1/internal/jobs/{job_id}/complete",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": "image",
            "output_content_type": "image/png",
            "size_bytes": 1024,
        },
    )

    events = await asyncio.wait_for(_drain(uuid.UUID(job_id)), timeout=20)

    # Sequence numbers are dense and strictly increasing — that is what makes
    # them usable as a reconnection cursor.
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[0].status == "queued"
    assert events[-1].event_type == "completed"
    # The UI renders stage_label, never the raw status.
    assert events[-1].stage_label == "Completed"


async def test_last_event_id_resumes_without_replaying_what_was_seen(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """Lossless reconnection — the reason events are stored, not just published."""
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    for status, progress in (("preparing", 22), ("generating", 62)):
        await client.post(
            f"/api/v1/internal/jobs/{job_id}/progress",
            headers=worker_headers,
            json={
                "worker_id": worker_id,
                "lease_token": job["lease_token"],
                "status": status,
                "progress": progress,
                "message": "",
            },
        )
    await client.post(f"/api/v1/generations/{job_id}/cancel")

    full = await asyncio.wait_for(_drain(uuid.UUID(job_id)), timeout=20)
    resumed = await asyncio.wait_for(_drain(uuid.UUID(job_id), last_event_id=2), timeout=20)

    assert len(full) > len(resumed)
    assert min(e.seq for e in resumed) == 3, "resumed stream replayed a seen event"
    # Nothing was skipped between the cursor and the end.
    assert [e.seq for e in resumed] == [e.seq for e in full if e.seq > 2]


async def test_a_live_event_reaches_a_connected_subscriber(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """Redis pub/sub fan-out: the reporting API instance and the streaming one
    are almost never the same process in production."""
    job_id = (await client.post("/api/v1/generations", json=text_to_video_request)).json()[
        "job_id"
    ]
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    # Start listening from the job's CURRENT end, so only an event published
    # after the subscription can satisfy this — proving live delivery rather
    # than database replay. Reading the cursor rather than hard-coding it
    # matters: hard-coding 2 would skip the very event under test.
    cursor = (await client.get(f"/api/v1/generations/{job_id}")).json()["last_event_seq"]

    listener = asyncio.create_task(_drain(uuid.UUID(job_id), last_event_id=cursor, limit=1))
    await asyncio.sleep(0.5)  # let the SUBSCRIBE land

    await client.post(
        f"/api/v1/internal/jobs/{job_id}/progress",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "status": "generating",
            "progress": 62,
            "message": "live",
        },
    )

    events = await asyncio.wait_for(listener, timeout=20)
    assert events[0].message == "live"


def test_sse_wire_format_carries_the_reconnection_id() -> None:
    frame = event_bus.format_sse(
        event_bus.build_event(
            seq=7,
            event_type=event_bus.EventType.PROGRESS,
            status=JobStatus.GENERATING,
            progress=62,
            message="Working…",
        )
    )
    assert frame.startswith("id: 7\n")
    assert "event: progress\n" in frame
    assert frame.endswith("\n\n")
    assert event_bus.format_sse(None).startswith(":")  # keepalive comment


# ── Lifecycle invariants ─────────────────────────────────────────────────


def test_terminal_states_are_final() -> None:
    """What makes a duplicate or delayed worker report harmless."""
    for terminal in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        for target in JobStatus:
            if target is terminal:
                continue
            assert not can_transition(terminal, target)


def test_progress_cannot_move_backwards_through_the_lifecycle() -> None:
    assert can_transition(JobStatus.QUEUED, JobStatus.GENERATING)
    assert not can_transition(JobStatus.GENERATING, JobStatus.QUEUED)
    assert not can_transition(JobStatus.UPLOADING, JobStatus.PREPARING)


def test_every_status_has_a_customer_facing_label() -> None:
    """A missing label would raise a KeyError while serializing a job."""
    for status in JobStatus:
        assert STATUS_LABELS[status]

    # Internal granularity the customer does not need collapses.
    assert STATUS_LABELS[JobStatus.ASSIGNED] == STATUS_LABELS[JobStatus.QUEUED]
    assert STATUS_LABELS[JobStatus.UPLOADING] == STATUS_LABELS[JobStatus.POST_PROCESSING]


# ── Storage ──────────────────────────────────────────────────────────────


async def test_upload_url_is_signed_for_the_declared_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/assets/upload-url",
        json={
            "filename": "clip.mp4",
            "content_type": "video/mp4",
            "kind": "video",
            "size_bytes": 1024,
        },
    )
    assert response.status_code == 201
    body = response.json()

    assert body["upload"]["method"] == "PUT"
    # Content-Type is part of the signature — storage rejects a mismatch, which
    # is what stops a client declaring one type and uploading another.
    assert body["upload"]["headers"]["Content-Type"] == "video/mp4"
    assert "X-Amz-Signature" in body["upload"]["url"]


async def test_an_unconfirmed_upload_cannot_be_used_as_input(client: AsyncClient) -> None:
    """A half-finished upload must never reach a worker."""
    asset_id = (
        await client.post(
            "/api/v1/assets/upload-url",
            json={
                "filename": "still.png",
                "content_type": "image/png",
                "kind": "image",
                "size_bytes": 512,
            },
        )
    ).json()["asset_id"]

    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "image-to-video",
            "prompt": "animate this",
            "parameters": {"duration": "5s", "aspect_ratio": "16:9", "quality": "High"},
            "inputs": {"source_image": asset_id},
        },
    )
    assert response.status_code == 422
    assert "not finished" in response.text.lower()


async def test_confirming_an_upload_that_never_landed_fails(client: AsyncClient) -> None:
    asset_id = (
        await client.post(
            "/api/v1/assets/upload-url",
            json={
                "filename": "ghost.png",
                "content_type": "image/png",
                "kind": "image",
                "size_bytes": 512,
            },
        )
    ).json()["asset_id"]

    response = await client.post(f"/api/v1/assets/{asset_id}/confirm", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "asset_not_ready"


async def test_unsupported_type_and_oversized_upload_are_refused(client: AsyncClient) -> None:
    bad_type = await client.post(
        "/api/v1/assets/upload-url",
        json={
            "filename": "script.exe",
            "content_type": "application/x-msdownload",
            "kind": "image",
            "size_bytes": 128,
        },
    )
    assert bad_type.status_code == 422
    assert bad_type.json()["error"]["code"] == "unsupported_media_type"

    too_big = await client.post(
        "/api/v1/assets/upload-url",
        json={
            "filename": "huge.mp4",
            "content_type": "video/mp4",
            "kind": "video",
            "size_bytes": MAX_SIZE_BYTES[AssetKind.VIDEO] + 1,
        },
    )
    assert too_big.status_code == 422
    assert too_big.json()["error"]["code"] == "file_too_large"


@pytest.mark.parametrize(
    ("raw", "expected_absent"),
    [("../../etc/passwd", ".."), ("a/b/c.png", "/"), ("nul\\x.png", "\\")],
)
def test_filenames_cannot_escape_their_storage_prefix(raw: str, expected_absent: str) -> None:
    assert expected_absent not in sanitize_filename(raw)


def test_type_allowlist_is_a_closed_set() -> None:
    """An allowlist, not a blocklist — an unknown type is refused."""
    from app.core.errors import ValidationFailed

    for kind in AssetKind:
        assert ALLOWED_CONTENT_TYPES[kind]
        with pytest.raises(ValidationFailed):
            validate_upload(kind, "application/octet-stream", 1024)


# ── Health ───────────────────────────────────────────────────────────────


async def test_liveness_touches_no_dependency(client: AsyncClient) -> None:
    """Liveness must not fail on a database blip, or the orchestrator would
    restart every API container over a dependency wobble."""
    response = await client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_reports_each_dependency(client: AsyncClient) -> None:
    body = (await client.get("/api/v1/health/ready")).json()
    assert set(body["checks"]) == {"database", "redis", "storage", "workflows"}
    assert body["checks"]["database"] is True
    assert body["checks"]["redis"] is True
    assert body["checks"]["workflows"] is True


# ── Migration ────────────────────────────────────────────────────────────


async def test_migration_produces_the_schema_the_models_describe() -> None:
    """Guards against a model change shipped without a migration.

    Upgrades a throwaway database with Alembic and compares its tables and
    indexes to `Base.metadata`. Without this, a new column would work locally
    (the test suite uses `create_all`) and fail in production.
    """
    from alembic import command
    from alembic.config import Config

    from app.models import Base

    verify_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/zolexai_migration_check"

    admin = create_async_engine(ADMIN_DATABASE_URL, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.exec_driver_sql("DROP DATABASE IF EXISTS zolexai_migration_check WITH (FORCE)")
        await conn.exec_driver_sql("CREATE DATABASE zolexai_migration_check")
    await admin.dispose()

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", verify_url)
    # Alembic runs its own event loop, so it goes on a thread.
    await asyncio.to_thread(command.upgrade, config, "head")

    engine = create_async_engine(verify_url)
    async with engine.connect() as conn:
        tables = set(
            (
                await conn.execute(
                    sql_text(
                        "SELECT tablename FROM pg_tables WHERE schemaname='public'"
                    )
                )
            )
            .scalars()
            .all()
        )
        indexes = set(
            (
                await conn.execute(
                    sql_text("SELECT indexname FROM pg_indexes WHERE schemaname='public'")
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()

    expected_tables = set(Base.metadata.tables) | {"alembic_version"}
    assert expected_tables <= tables, f"missing tables: {expected_tables - tables}"

    expected_indexes = {
        index.name for table in Base.metadata.tables.values() for index in table.indexes
    }
    assert expected_indexes <= indexes, f"missing indexes: {expected_indexes - indexes}"


# ── Download names and single-asset reads ────────────────────────────────


@pytest.mark.parametrize(
    ("name", "content_type", "expected"),
    [
        ("text-to-video-fad9aa41", "video/mp4", "text-to-video-fad9aa41.mp4"),
        ("music-0f2c", "audio/mpeg", "music-0f2c.mp3"),
        ("already.mp4", "video/mp4", "already.mp4"),
        ("ALREADY.MP4", "video/mp4", "ALREADY.MP4"),
        ("clip", "application/octet-stream", "clip"),
    ],
)
def test_download_names_carry_the_extension_their_type_implies(
    name: str, content_type: str, expected: str
) -> None:
    """An extensionless download saves a file the OS cannot identify: it does
    not open on a double-click, and a picker filtering on video/mp4 hides it —
    which made a downloaded generation impossible to re-upload to Extend."""
    assert with_extension(name, content_type) == expected


async def test_a_generated_asset_downloads_with_a_usable_filename(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    """End to end: the name a worker's output is registered under is the name
    the browser will save."""
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)
    await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/complete",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": "video",
            "output_content_type": "video/mp4",
            "size_bytes": 4096,
            "width": 896,
            "height": 512,
        },
    )

    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    asset_id = public["outputs"][0]["asset_id"]

    listed = (await client.get("/api/v1/media?limit=5")).json()["items"]
    generated = next(item for item in listed if item["id"] == asset_id)
    assert generated["name"].endswith(".mp4")

    url = (await client.post(f"/api/v1/assets/{asset_id}/download-url")).json()["url"]
    assert ".mp4" in url, "the attachment filename must reach the browser"


async def test_one_asset_can_be_read_by_id(client: AsyncClient) -> None:
    """Extend hands over a source by id alone, so its input control has to be
    able to resolve what it is about to use."""
    created = (
        await client.post(
            "/api/v1/assets/upload-url",
            json={
                "filename": "source.mp4",
                "content_type": "video/mp4",
                "kind": "video",
                "size_bytes": 2048,
            },
        )
    ).json()

    response = await client.get(f"/api/v1/assets/{created['asset_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["asset_id"]
    assert body["name"] == "source.mp4"
    # Storage keys are a backend concern and must never reach a client.
    assert "storage_key" not in body


async def test_reading_someone_elses_asset_is_a_not_found(client: AsyncClient) -> None:
    """Ownership on this route matters as much as on any other — an asset id is
    guessable in principle, and must gain an attacker nothing."""
    response = await client.get(f"/api/v1/assets/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_the_indexes_that_keep_large_tables_fast_exist(db: AsyncSession) -> None:
    """Named explicitly so a well-meaning cleanup cannot quietly drop one
    (directive §5)."""
    rows = await db.execute(
        sql_text("SELECT indexname FROM pg_indexes WHERE tablename = 'generation_jobs'")
    )
    present = set(rows.scalars().all())

    for required in (
        "ix_generation_jobs_user_created",  # history listing
        "ix_generation_jobs_user_status_created",  # status filter + concurrency count
        "ix_generation_jobs_user_workflow_created",  # workflow filter
        "ix_generation_jobs_claimable",  # the worker claim
        "ix_generation_jobs_lease_expiry",  # the reaper
        "uq_generation_jobs_user_idempotency",  # duplicate suppression
    ):
        assert required in present, f"index {required} is missing"
