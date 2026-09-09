"""generation_jobs.result — the worker's structured report (Music Lyrics Workflow v2.0)

Revision ID: 3c9d2e7a1b40
Revises: b066c7ac256b
Created: 2026-09-09 12:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3c9d2e7a1b40"
down_revision: str | None = "b066c7ac256b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable, no default: every existing row reads as "no report", which
    # is exactly what every existing job has. Adding a nullable column is a
    # metadata-only change on PostgreSQL — no table rewrite, no lock worth
    # scheduling around.
    op.add_column(
        "generation_jobs",
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("generation_jobs", "result")
