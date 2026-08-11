"""Worker configuration.

The worker knows three things about the outside world: where the API is, the
service token to reach it, and (optionally) where Redis is for wake-ups. It has
no database URL and no storage credentials for reading — everything it fetches
or writes uses presigned URLs the API hands it per job.

That narrowness is the security property: a compromised GPU node cannot read the
database, cannot enumerate other users' media, and holds no long-lived storage
key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root(start: Path | None = None) -> Path:
    """Locates the repository root, in a way that survives the container.

    In the repo this file is `apps/worker/worker/core/config.py`, so the root
    is four parents up. Inside the image it is `/app/worker/core/config.py` —
    only three parents exist. Indexing `parents[4]` unconditionally raised
    `IndexError` at import time, before the worker could even start.

    The only thing derived from this is the OPTIONAL `.env`, which exists in
    development and never in the image (where every value is supplied by the
    environment), so falling back to the filesystem root is harmless: pydantic
    ignores an env_file that is not there.

    `start` exists so a test can pass a simulated location; production always
    uses this module's own path.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / ".env").is_file() or (parent / "workflow-definitions").is_dir():
            return parent
    return here.parents[-1]


REPO_ROOT = _repo_root()


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    worker_name: str = "mock-worker-1"
    """Stable per deployment slot. Re-registering under the same name keeps the
    node's identity across restarts."""

    runtime: str = "mock"
    """Which adapter to run. M2 adds a real one; nothing else changes."""

    worker_version: str = "0.1.0"

    api_base_url: str = "http://localhost:8000"
    worker_api_token: str = ""

    redis_url: str = "redis://localhost:6379/0"
    use_redis_wakeup: bool = True
    """
    Redis only removes polling latency. With it off the worker polls on
    `idle_poll_seconds` and behaves identically, just less promptly — the queue
    lives in PostgreSQL, so nothing is lost either way.
    """

    max_concurrency: int = 2
    idle_poll_seconds: int = 3
    wake_timeout_seconds: int = 10

    heartbeat_interval_seconds: int = 20
    """Also renews leases on in-flight jobs, so a long silent stage cannot have
    its job reaped out from under it."""

    request_timeout_seconds: float = 20.0
    upload_timeout_seconds: float = 120.0

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    #: Mock-only pacing, matching the timings the client approved during PRE-M1.
    mock_speed: float = 1.0

    @property
    def api_v1(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/api/v1"


@lru_cache(maxsize=1)
def get_settings() -> WorkerSettings:
    return WorkerSettings()


settings = get_settings()
