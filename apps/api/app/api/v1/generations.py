"""Generation endpoints — create, list, read, cancel and stream progress."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Header, Query, Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession, Pagination, RedisClient, Registry
from app.core.config import settings
from app.core.enums import ErrorCode, JobStatus
from app.core.errors import Conflict
from app.core.logging import bind_context, get_logger
from app.schemas.common import Page
from app.schemas.generation import (
    GenerationAccepted,
    GenerationCreateRequest,
    GenerationJobPublic,
)
from app.services import events as event_bus
from app.services import idempotency
from app.services.generation import GenerationService

logger = get_logger(__name__)
router = APIRouter(prefix="/generations", tags=["generations"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenerationAccepted,
    summary="Submit a generation",
)
async def create_generation(
    payload: GenerationCreateRequest,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
    user: CurrentUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
) -> GenerationAccepted:
    """Accepts the request and returns immediately.

    **202, not 200.** Nothing has been generated when this responds — a job row
    exists and a worker will pick it up. An endpoint that waited for a video
    would hold a connection for minutes and cap throughput at whatever one
    instance can keep open (directive §6, scalability rule #3).

    Send `Idempotency-Key` to make a double-click or a network retry safe: the
    repeat returns the original job instead of starting a second one.
    """
    service = GenerationService(session, redis, registry)

    reserved = False
    if idempotency_key:
        holder = await idempotency.reserve(redis, user.id, idempotency_key)
        if holder is not None:
            if idempotency.is_pending(holder):
                # The first request is still in flight. 409 rather than a
                # duplicate job — the client should wait, not retry harder.
                raise Conflict(
                    "That request is already being processed.",
                    code=ErrorCode.IDEMPOTENCY_KEY_REUSED,
                )
            existing = await service.require_for_user(uuid.UUID(holder), user.id)
            response.status_code = status.HTTP_200_OK
            return _accepted(existing)
        reserved = True

    try:
        job = await service.create(user=user, request=payload, idempotency_key=idempotency_key)
    except Exception:
        # Free the key so a corrected retry is not blocked for a day.
        if reserved and idempotency_key:
            await idempotency.release(redis, user.id, idempotency_key)
        raise

    if idempotency_key:
        await idempotency.complete(redis, user.id, idempotency_key, job.id)

    bind_context(job_id=str(job.id))
    return _accepted(job)


def _accepted(job) -> GenerationAccepted:  # noqa: ANN001 — ORM model
    from app.core.enums import STATUS_LABELS

    status_value = JobStatus(job.status)
    return GenerationAccepted(
        job_id=job.id,
        status=status_value,
        stage_label=STATUS_LABELS[status_value],
        events_url=f"{settings.api_v1_prefix}/generations/{job.id}/events",
    )


@router.get("", response_model=Page[GenerationJobPublic], summary="Generation history")
async def list_generations(
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
    user: CurrentUser,
    page: Pagination,
    response: Response,
    status_filter: Annotated[list[JobStatus] | None, Query(alias="status")] = None,
    workflow_id: Annotated[str | None, Query(max_length=64)] = None,
) -> Page[GenerationJobPublic]:
    """One page of history, newest first.

    Cursor-paginated, never offset — see `schemas/common.py`. There is no
    endpoint anywhere that loads a full history table.
    """
    service = GenerationService(session, redis, registry)
    jobs, next_cursor, has_more = await service.repo.list_for_user(
        user.id, page=page, statuses=status_filter, workflow_id=workflow_id
    )

    # Items carry short-lived presigned URLs; a shared cache must not keep them.
    response.headers["Cache-Control"] = "no-store"

    return Page[GenerationJobPublic](
        items=[await service.to_public(job) for job in jobs],
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get("/{job_id}", response_model=GenerationJobPublic, summary="One generation")
async def get_generation(
    job_id: uuid.UUID,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
    user: CurrentUser,
    response: Response,
) -> GenerationJobPublic:
    bind_context(job_id=str(job_id))
    service = GenerationService(session, redis, registry)
    job = await service.require_for_user(job_id, user.id)
    response.headers["Cache-Control"] = "no-store"
    return await service.to_public(job)


@router.post("/{job_id}/cancel", response_model=GenerationJobPublic, summary="Cancel")
async def cancel_generation(
    job_id: uuid.UUID,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
    user: CurrentUser,
) -> GenerationJobPublic:
    bind_context(job_id=str(job_id))
    service = GenerationService(session, redis, registry)
    job = await service.require_for_user(job_id, user.id)
    cancelled = await service.cancel(job)
    return await service.to_public(cancelled)


@router.get("/{job_id}/events", summary="Progress stream (SSE)")
async def stream_generation_events(
    job_id: uuid.UUID,
    session: DbSession,
    redis: RedisClient,
    registry: Registry,
    user: CurrentUser,
    last_event_id: Annotated[
        str | None, Header(alias="Last-Event-ID", max_length=32)
    ] = None,
    since: Annotated[int | None, Query(ge=0)] = None,
) -> StreamingResponse:
    """Server-Sent Events for one job.

    Replaces polling: the browser opens one connection and receives status,
    stage, progress and the terminal result as they happen (directive §10).

    **Reconnection is lossless.** Every event carries a monotonic `id`. On a
    dropped connection `EventSource` reconnects automatically with
    `Last-Event-ID`, and the stream replays everything after that sequence from
    PostgreSQL before attaching to the live feed — so nothing that happened
    during the gap is missed. `?since=` is the manual equivalent, for clients
    that are not `EventSource`.

    Ownership is verified BEFORE the stream opens; once a `StreamingResponse`
    starts, the status code is already sent and a 404 can no longer be returned.
    """
    bind_context(job_id=str(job_id))
    service = GenerationService(session, redis, registry)
    await service.require_for_user(job_id, user.id)

    resume_from = _parse_cursor(last_event_id, since)

    async def generate():
        # Tells the browser how long to wait before reconnecting.
        yield "retry: 3000\n\n"
        try:
            async for event in event_bus.stream_job_events(job_id, last_event_id=resume_from):
                yield event_bus.format_sse(event)
        except Exception:
            # A stream cannot return an error status — the headers are long
            # gone — so it is logged and the connection closed. The client
            # reconnects and resumes from its last id.
            logger.exception("sse_stream_error", extra={"job_id": str(job_id)})
            return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, no-transform",
            "Connection": "keep-alive",
            # Nginx buffers proxied responses by default, which would hold every
            # event until the connection closed and make SSE behave like a slow
            # single response.
            "X-Accel-Buffering": "no",
        },
    )


def _parse_cursor(last_event_id: str | None, since: int | None) -> int:
    if last_event_id:
        try:
            return max(0, int(last_event_id))
        except ValueError:
            # A malformed header is not worth failing the connection over —
            # replaying from the beginning is correct, just chattier.
            logger.info("sse_bad_last_event_id")
    return since or 0
