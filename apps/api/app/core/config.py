"""Environment-driven configuration.

Every value the application needs comes from the environment, never from a
literal in source (scalability rule #14, security baseline §17). `.env.example`
at the repo root documents the full set and contains no real secrets.

Settings are constructed ONCE at import and shared; nothing mutates them at
runtime, so multiple API instances behave identically.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _repo_root(start: Path | None = None) -> Path:
    """Locates the repository root, in a way that survives the container.

    In the repo this file is `apps/api/app/core/config.py`, so the root is four
    parents up. Inside the image it is `/app/app/core/config.py` — only three
    parents exist, because the Dockerfile copies just what the service needs
    and there is no repository. Indexing `parents[4]` unconditionally raised
    `IndexError` at import time and killed the process before any handler ran.

    So: walk upwards looking for the workflow definitions, which are the one
    directory both layouts genuinely have (the image places them at
    `/workflow-definitions`). Fall back to the filesystem root, where the only
    two things derived from this — the optional `.env` and the definitions
    directory — are either absent-and-ignored or supplied by the environment.

    `start` exists so a test can pass a simulated location; production always
    uses this module's own path.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "workflow-definitions").is_dir():
            return parent
    return here.parents[-1]


REPO_ROOT = _repo_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "ZolexAI API"
    api_v1_prefix: str = "/api/v1"

    # ── Datastores ───────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://zolexai:change-me@localhost:5432/zolexai"
    redis_url: str = "redis://localhost:6379/0"

    # ── Object storage (S3-compatible; MinIO locally) ────────────────────
    storage_provider: Literal["minio", "s3", "r2"] = "minio"
    storage_endpoint: str = "http://localhost:9000"
    """Endpoint the API itself talks to. Inside Docker this is the service name."""
    storage_public_endpoint: str = ""
    """
    Endpoint that goes into presigned URLs handed to a BROWSER.

    These differ whenever the API runs in Docker and the browser does not:
    the API reaches MinIO at http://minio:9000, but a presigned URL containing
    that host is unreachable from the user's machine. Empty means "same as
    storage_endpoint".
    """
    storage_bucket: str = "zolexai-dev"
    storage_region: str = "us-east-1"
    storage_access_key: str = ""
    storage_secret_key: str = ""
    storage_presign_expiry_seconds: int = 900

    # ── Worker coordination ──────────────────────────────────────────────
    worker_api_token: str = ""
    """Shared service credential for /api/v1/internal/*. NEVER a user login."""
    job_lease_seconds: int = 120
    """How long a claim is valid before another worker may take the job."""
    job_max_attempts: int = 3
    job_default_timeout_seconds: int = 1800
    """
    Render ceiling assumed for a workflow that declares no
    `execution.timeout_seconds`. Mirrors the worker's own default
    (`worker.core.config.job_timeout_seconds`); the API needs it only to sign the
    output upload URL for long enough, so the duplication is deliberate and
    harmless — signing generously costs nothing, signing short loses renders.
    """
    worker_upload_grace_seconds: int = 900
    """
    How much longer than the render ceiling the worker's output upload URL stays
    valid. Covers staging inputs, validating the result and streaming it up — the
    worker allows `upload_timeout_seconds` (900s) for that last part alone.
    """

    # ── Limits (foundation only — full enforcement is M3 billing) ────────
    default_user_concurrency_limit: int = 3
    """Max simultaneously-running jobs per user, so one user cannot occupy every worker."""
    idempotency_ttl_seconds: int = 86_400

    # ── HTTP ─────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000"
    """Comma-separated. Wildcards are rejected outside development."""

    # ── Observability ────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ── Auth seam (real sessions arrive at M3.01) ────────────────────────
    dev_user_email: str = "demo@zolexai.local"
    """
    M1 has no authentication. Every request resolves to this single seeded user
    so `generation_jobs.user_id` is a real FK from the first migration and M3
    only has to swap how the identity is derived. See app/core/security.py.
    """

    workflow_definitions_dir: Path = Field(default=REPO_ROOT / "workflow-definitions")

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard_origins(cls, value: str) -> str:
        # Caught here rather than in the middleware so a misconfigured deploy
        # fails at startup instead of silently serving a permissive API.
        if "*" in value:
            raise ValueError("cors_origins must list explicit origins, never '*'")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def public_storage_endpoint(self) -> str:
        return self.storage_public_endpoint or self.storage_endpoint

    def assert_production_ready(self) -> None:
        """Fails fast on a production boot that would be insecure.

        Development is allowed to run on defaults; production is not. Doing this
        at startup means a bad deploy dies in the health check rather than
        leaking an open internal API.
        """
        if not self.is_production:
            return
        missing = [
            name
            for name, value in (
                ("WORKER_API_TOKEN", self.worker_api_token),
                ("STORAGE_ACCESS_KEY", self.storage_access_key),
                ("STORAGE_SECRET_KEY", self.storage_secret_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Refusing to start in production without: {', '.join(missing)}")
        if len(self.worker_api_token) < 32:
            raise RuntimeError("WORKER_API_TOKEN must be at least 32 characters in production")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
