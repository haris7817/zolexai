"""Data access for worker nodes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import WorkerStatus
from app.models.worker import WorkerNode


def _now() -> datetime:
    return datetime.now(UTC)


class WorkerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(
        self,
        *,
        name: str,
        runtime: str,
        version: str,
        capabilities: dict[str, Any],
        max_concurrency: int,
    ) -> WorkerNode:
        """Upsert by name.

        Re-registering under the same name rather than inserting a new row is
        what keeps the table from growing by one row per container restart —
        and it means a restarted worker keeps the identity its past jobs
        reference.
        """
        worker = (
            await self.session.execute(select(WorkerNode).where(WorkerNode.name == name))
        ).scalar_one_or_none()

        if worker is None:
            worker = WorkerNode(name=name)
            self.session.add(worker)

        worker.runtime = runtime
        worker.version = version
        worker.capabilities = capabilities
        worker.max_concurrency = max_concurrency
        worker.status = WorkerStatus.ONLINE
        worker.last_heartbeat_at = _now()

        await self.session.flush()
        return worker

    async def heartbeat(
        self, worker_id: uuid.UUID, *, status: WorkerStatus | None = None
    ) -> WorkerNode | None:
        worker = await self.session.get(WorkerNode, worker_id)
        if worker is None:
            return None
        worker.last_heartbeat_at = _now()
        if status is not None:
            worker.status = status
        await self.session.flush()
        return worker

    async def get(self, worker_id: uuid.UUID) -> WorkerNode | None:
        return await self.session.get(WorkerNode, worker_id)

    async def mark_stale_offline(self, *, stale_after_seconds: int) -> int:
        """Flags workers that stopped heartbeating.

        Cosmetic only — job recovery does NOT depend on this. Leases expire on
        their own schedule, so a job is requeued whether or not anyone has
        noticed its worker is gone. Keeping the two independent means a bug here
        cannot strand work.
        """
        cutoff = _now() - timedelta(seconds=stale_after_seconds)
        result = await self.session.execute(
            sa.update(WorkerNode)
            .where(
                WorkerNode.status != WorkerStatus.OFFLINE,
                sa.or_(
                    WorkerNode.last_heartbeat_at.is_(None),
                    WorkerNode.last_heartbeat_at < cutoff,
                ),
            )
            .values(status=WorkerStatus.OFFLINE)
            .returning(WorkerNode.id)
        )
        return len(result.scalars().all())

    async def count_online(self, *, stale_after_seconds: int) -> int:
        cutoff = _now() - timedelta(seconds=stale_after_seconds)
        return (
            await self.session.execute(
                select(sa.func.count())
                .select_from(WorkerNode)
                .where(
                    WorkerNode.status != WorkerStatus.OFFLINE,
                    WorkerNode.last_heartbeat_at.is_not(None),
                    WorkerNode.last_heartbeat_at >= cutoff,
                )
            )
        ).scalar_one()
