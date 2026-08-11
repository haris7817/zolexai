"""SQLAlchemy models.

Importing this package registers every table on `Base.metadata`. Alembic's
`env.py` imports it for exactly that reason — a model that is not reachable from
here is invisible to autogenerate and will silently never be migrated.
"""

from app.db.base import Base
from app.models.asset import Asset
from app.models.generation import (
    GenerationEvent,
    GenerationJob,
    GenerationJobInput,
    GenerationJobOutput,
)
from app.models.user import User
from app.models.worker import WorkerNode

__all__ = [
    "Asset",
    "Base",
    "GenerationEvent",
    "GenerationJob",
    "GenerationJobInput",
    "GenerationJobOutput",
    "User",
    "WorkerNode",
]
