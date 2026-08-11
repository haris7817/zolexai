"""Generation event fan-out — the durable half plus the live half.

**The problem this solves.** A job is executed by one worker and watched by a
browser connected to one API instance. With several of each behind a load
balancer, the worker reporting progress and the API streaming it are almost
never the same process. A naive implementation — the API polling its own memory,
or the worker pushing straight to a connection — silently streams nothing as
soon as there is more than one instance.

**The design.**

    worker ──HTTP──▶ any API instance
                          │
                          ├─▶ PostgreSQL  generation_events   (durable, replayable)
                          └─▶ Redis PUBLISH zx:job:{id}:events (live, fire-and-forget)

    browser ──SSE──▶ any API instance
                          ├─ 1. SUBSCRIBE to the channel
                          ├─ 2. SELECT events WHERE seq > Last-Event-ID
                          └─ 3. stream the replay, then the live feed

Redis pub/sub delivers to whoever is listening *at that instant* and remembers
nothing, so on its own it drops every event that arrives while a client is
reconnecting. PostgreSQL is the record of truth; Redis only removes the latency
of polling for it.

**Subscribe-then-read, in that order.** Reading the database first would leave a
window between the last row read and the subscription taking effect, and any
event published in that window would be lost forever. Subscribing first can only
produce a duplicate, which `seq` lets us discard.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select

from app.core.enums import EventType, JobStatus, is_terminal
from app.core.logging import get_logger
from app.db.redis import new_redis_connection
from app.db.session import get_session_factory
from app.models.generation import GenerationEvent

logger = get_logger(__name__)

#: How long a stream sits idle before emitting an SSE comment. Proxies and load
#: balancers commonly close a connection after 60s of silence; a generating job
#: can easily be quiet for longer than that.
KEEPALIVE_SECONDS = 15.0

#: Ceiling on a single stream's lifetime. A browser tab left open for days would
#: otherwise pin a connection and a Redis subscriber indefinitely. EventSource
#: reconnects automatically, and `Last-Event-ID` makes that free.
MAX_STREAM_SECONDS = 3600.0


def channel_for(job_id: uuid.UUID) -> str:
    return f"zx:job:{job_id}:events"


@dataclass(frozen=True)
class JobEvent:
    """One lifecycle event, in the exact shape the browser receives."""

    seq: int
    event_type: str
    status: str
    stage_label: str
    progress: int
    message: str
    payload: dict[str, Any]
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_row(cls, row: GenerationEvent) -> JobEvent:
        from app.core.enums import STATUS_LABELS

        return cls(
            seq=row.seq,
            event_type=str(row.event_type),
            status=str(row.status),
            stage_label=STATUS_LABELS[JobStatus(row.status)],
            progress=row.progress,
            message=row.message,
            payload=row.payload or {},
            created_at=row.created_at.isoformat(),
        )


async def publish_event(redis: Redis, job_id: uuid.UUID, event: JobEvent) -> None:
    """Best-effort live delivery.

    Deliberately swallows Redis failures. The event is already committed to
    PostgreSQL by the time this runs, so a Redis outage costs latency (clients
    fall back to reconnect-and-replay) but never correctness. Letting it raise
    would fail a worker's progress report over a cache problem.
    """
    try:
        await redis.publish(channel_for(job_id), event.to_json())
    except Exception:
        logger.warning("event_publish_failed", extra={"job_id": str(job_id), "seq": event.seq})


async def _replay_from_db(job_id: uuid.UUID, after_seq: int) -> list[JobEvent]:
    """Events this client has not seen, straight from the durable log.

    Uses its own short-lived session rather than the request session: an SSE
    stream can live for an hour, and holding a pooled connection open that long
    would exhaust the pool at a handful of concurrent viewers.
    """
    async with get_session_factory()() as session:
        rows = (
            await session.execute(
                select(GenerationEvent)
                .where(GenerationEvent.job_id == job_id, GenerationEvent.seq > after_seq)
                .order_by(GenerationEvent.seq)
            )
        ).scalars()
        return [JobEvent.from_row(row) for row in rows]


async def stream_job_events(
    job_id: uuid.UUID, *, last_event_id: int = 0
) -> AsyncIterator[JobEvent | None]:
    """Yields events for one job; `None` means "emit a keepalive".

    Terminates once a terminal event has been delivered, so a completed job
    closes its stream instead of holding a connection forever.
    """
    connection = new_redis_connection()
    pubsub = connection.pubsub(ignore_subscribe_messages=True)
    started = asyncio.get_running_loop().time()
    highest_seq = last_event_id

    try:
        # 1. Subscribe BEFORE reading. See the module docstring — the reverse
        #    order has an unrecoverable gap.
        await pubsub.subscribe(channel_for(job_id))

        # 2. Catch up on everything already recorded.
        for event in await _replay_from_db(job_id, last_event_id):
            highest_seq = max(highest_seq, event.seq)
            yield event
            if is_terminal(JobStatus(event.status)):
                return

        # 3. Follow the live feed.
        while True:
            if asyncio.get_running_loop().time() - started > MAX_STREAM_SECONDS:
                logger.info("event_stream_max_age", extra={"job_id": str(job_id)})
                return

            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=KEEPALIVE_SECONDS
            )
            if message is None:
                yield None  # keepalive comment
                continue

            try:
                event = JobEvent(**json.loads(message["data"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("event_stream_bad_payload", extra={"job_id": str(job_id)})
                continue

            # Duplicates are expected: an event may arrive live having already
            # been replayed from the database. `seq` makes them free to discard.
            if event.seq <= highest_seq:
                continue

            highest_seq = event.seq
            yield event

            if is_terminal(JobStatus(event.status)):
                return
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(channel_for(job_id))
            await pubsub.aclose()
        with contextlib.suppress(Exception):
            await connection.aclose()


def format_sse(event: JobEvent | None) -> str:
    """Wire format.

    `id:` is what the browser echoes back as `Last-Event-ID` after a dropped
    connection — it is the entire reconnection mechanism, not decoration.
    """
    if event is None:
        return ": keepalive\n\n"
    return f"id: {event.seq}\nevent: {event.event_type}\ndata: {event.to_json()}\n\n"


def build_event(
    *,
    seq: int,
    event_type: EventType,
    status: JobStatus,
    progress: int,
    message: str,
    payload: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> JobEvent:
    from app.core.enums import STATUS_LABELS

    return JobEvent(
        seq=seq,
        event_type=str(event_type),
        status=str(status),
        stage_label=STATUS_LABELS[status],
        progress=progress,
        message=message,
        payload=payload or {},
        created_at=(created_at or datetime.now().astimezone()).isoformat(),
    )
