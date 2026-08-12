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

import tempfile
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
    """This node's primary runtime, reported at registration."""

    runtimes: str = ""
    """
    Comma-separated runtimes this node can actually execute; empty means "just
    `runtime`".

    This is what stops a node claiming work it cannot do. Routing is per
    *workflow* — the YAML's `execution.runtime` decides who should run a job —
    but until M2 nothing checked that the claiming worker agreed. A mock node
    would happily claim a GPU-routed job, find no adapter, and fail it with
    `retriable=False`, which is a permanently dead job from the user's side.

    The API intersects this list with each workflow's runtime at claim time, so
    a mixed fleet is safe: mock nodes see only mock-routed workflows.
    """

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
    """
    Jobs at once. Two is right for the mock runtime; a node with one GPU should
    set this to 1, because concurrency here is asyncio tasks sharing a single
    event loop, not isolated processes.
    """

    idle_poll_seconds: int = 3
    wake_timeout_seconds: int = 10

    heartbeat_interval_seconds: int = 20
    """
    Proof of life for the worker NODE. It does not touch job leases — those are
    renewed by progress reports, and by the keepalive below when an adapter is
    working silently. (An earlier version of this docstring claimed otherwise;
    it was wrong, and a long silent stage really did get its job reaped.)
    """

    lease_keepalive_seconds: int = 45
    """
    How often to re-report the last known progress while an adapter is running.

    A lease is `JOB_LEASE_SECONDS` (120s by default on the API) and only a
    progress report renews it. A real render is silent for far longer than that
    between stages, and a lapsed lease means the reaper hands the job to another
    worker while this one is still burning compute on it — two workers, one
    output, and a user watching a bar that restarts.

    Comfortably under a third of the lease so two consecutive failures are
    survivable.
    """

    job_timeout_seconds: int = 1800
    """
    Wall-clock ceiling for one adapter run. A workflow may override it with
    `execution.timeout_seconds`.

    Without this a hung provider call holds a concurrency slot forever: the job
    never fails, never completes, and the slot never returns to the pool.
    """

    shutdown_drain_seconds: int = 300
    """
    How long to let in-flight jobs finish on SIGTERM before cancelling them.

    Was 30s, which is fine for a 7-second mock and pointless for a real render —
    the job got cancelled anyway and waited out its lease. Long enough to matter
    now, and cancelled jobs still clean up.
    """

    request_timeout_seconds: float = 20.0
    """API calls only. Small on purpose — these are control-plane round trips."""

    download_timeout_seconds: float = 300.0
    """Media in. Separate from the API timeout: a 500 MB source video is not a
    control-plane call."""

    upload_timeout_seconds: float = 900.0
    """Media out. Generous — the result is the whole job's value, and losing it
    to a timeout wastes everything spent producing it."""

    # ── Workspace ────────────────────────────────────────────────────────

    workspace_dir: Path | None = None
    """Scratch root. Defaults to the system temp directory."""

    min_free_disk_mb: int = 2048
    """
    Refuse a job when the workspace has less room than this.

    Checked before work starts, because failing early is cheap and running out
    of disk halfway through a render is not — and a full disk tends to take the
    next job down too.
    """

    keep_workspace_on_failure: bool = False
    """Debugging aid: leave a failed job's scratch directory behind."""

    # ── Media tooling ────────────────────────────────────────────────────

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    max_segment_seconds: int = 10
    """
    Default ceiling for one generation pass before the long-form layer splits
    the work. A workflow overrides it with `execution.max_segment_seconds`; the
    real value comes from benchmarking the selected model.
    """

    # ── LTX runtime (M2, GPU nodes only) ─────────────────────────────────

    ltx_repo_dir: Path = Path("/workspace/ltx2-benchmark")
    """
    Where the LTX repository and its `uv` environment live on a GPU node. The
    adapter shells out to `uv run` with this as the working directory, so the
    model's Python environment stays completely separate from the worker's —
    the worker itself never imports torch.
    """

    ltx_model_dir: Path | None = None
    """Model weights root. Defaults to `<ltx_repo_dir>/models/ltx-2.5`."""

    ltx_quantization: str = "nvfp4-prequant"
    """
    NVFP4 is the only mode that fits the client's RTX 5090: the BF16
    transformer alone is ~40 GB against 32 GB of VRAM. Verified by benchmark
    on 2026-08-12 (docs/internal/ltx-2.5-licensing-review.md is the licensing
    side; the VRAM ceiling is an engineering fact independent of it).
    """

    ltx_max_seconds: int = 30
    """
    Longest single-pass generation the GPU survived in benchmarking. 30s
    completed (with VRAM pressure during audio decode); 60s hard-OOMed at
    29.6/31.4 GiB. Long-form beyond this is the segmentation layer's job.
    """

    ltx_frame_rate: int = 24
    """LTX-2.5's native rate; num_frames = seconds x this."""

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    #: Mock-only pacing, matching the timings the client approved during PRE-M1.
    mock_speed: float = 1.0

    @property
    def api_v1(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/api/v1"

    @property
    def runtime_list(self) -> list[str]:
        """Runtimes this node serves, always including its primary one."""
        declared = [item.strip() for item in self.runtimes.split(",") if item.strip()]
        if self.runtime and self.runtime not in declared:
            declared.insert(0, self.runtime)
        return declared

    @property
    def workspace_root(self) -> Path:
        return self.workspace_dir or Path(tempfile.gettempdir()) / "zolexai-worker"

    @property
    def ltx_models_root(self) -> Path:
        return self.ltx_model_dir or self.ltx_repo_dir / "models" / "ltx-2.5"


@lru_cache(maxsize=1)
def get_settings() -> WorkerSettings:
    return WorkerSettings()


settings = get_settings()
