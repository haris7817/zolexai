"""Worker node registry.

Workers register on boot and heartbeat while alive. The row exists so the
platform can answer "is anything able to run this workflow right now?" and so a
job can name the worker that produced it — not so the API can command a worker.
Control flows the other way: workers pull work (directive §8, §16).

`capabilities` lists the workflow ids a node can execute. In M1 the mock worker
claims all six; in M2 a GPU node may advertise a subset, and the claim query
filters on it — which is how heterogeneous hardware becomes possible without an
API change.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import WorkerStatus
from app.db.base import Base, created_at_col, enum_column, updated_at_col, uuid_pk


class WorkerNode(Base):
    __tablename__ = "worker_nodes"

    id: Mapped[uuid.UUID] = uuid_pk()

    name: Mapped[str] = mapped_column(sa.String(120), nullable=False, unique=True)
    """
    Stable per deployment slot (e.g. "mock-worker-1"), not per process. A worker
    that restarts re-registers under the same name and keeps its identity, so
    the table does not accumulate a row per crash.
    """

    status: Mapped[WorkerStatus] = mapped_column(
        enum_column(WorkerStatus), nullable=False, default=WorkerStatus.ONLINE
    )

    # Adapter kind the node runs — "mock" in M1. Never a provider or model name
    # in any client-visible surface; this column is internal only (rule §12).
    runtime: Mapped[str] = mapped_column(sa.String(40), nullable=False, default="mock")
    version: Mapped[str] = mapped_column(sa.String(40), nullable=False, default="")

    capabilities: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sa.text("'{}'::jsonb")
    )
    """{"workflows": ["text-to-video", ...], "max_concurrency": 2}"""

    max_concurrency: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False, default=1)

    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()

    __table_args__ = (
        # "Which workers are alive?" — a readiness and capacity question asked
        # on a schedule, so it gets its own index.
        sa.Index("ix_worker_nodes_status_heartbeat", "status", "last_heartbeat_at"),
    )

    def __repr__(self) -> str:
        return f"<WorkerNode {self.name} {self.status}>"
