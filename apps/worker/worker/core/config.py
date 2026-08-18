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

import shlex
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

    person_matte_command: str = ""
    """
    How to invoke person matting, which produces the mask behind
    `execution.v2v_person_lock`. Empty means "the script shipped in this
    checkout, run in the LTX environment" — see `person_matte_argv`.

    A command rather than an import for the same reason the pipelines are: the
    matting model needs torch and CUDA, and this worker deliberately has
    neither. It runs in the LTX environment (`ltx_repo_dir` is the working
    directory) and speaks a small, stable CLI, so the segmentation model behind
    it can be replaced without touching a line of worker code.
    """

    ltx_quantization: str = "nvfp4-prequant"
    """
    NVFP4 is the only mode that fits the client's RTX 5090: the BF16
    transformer alone is ~40 GB against 32 GB of VRAM. Verified by benchmark
    on 2026-08-12 (docs/internal/ltx-2.5-licensing-review.md is the licensing
    side; the VRAM ceiling is an engineering fact independent of it).
    """

    ltx_max_seconds: int = 60
    """
    Operational brake on single-pass length, not the real ceiling.

    The real ceiling is per-grid and lives in `adapters/ltx._GRID_CEILINGS`,
    because the VAE fails on particular shapes rather than above a size: on the
    RTX PRO 6000, 1024x576 sustains 60s while 896x512 — fewer pixels — does not.
    A single global number can only ever encode the worst shape, which is how
    every 60s render came to be six passes with five seams.

    This value is the emergency lever: lowering it via the environment pulls
    every shape down immediately, with no deploy. That is exactly what
    contained the 14 Aug incident, so it stays in the clamp chain.

    60 because that is the longest length the product offers and every current
    grid was measured at it (16 Aug 2026, after NATTEN replaced the failing
    Triton fallback kernel). Raising it above 60 does nothing on its own — a
    grid still cannot exceed its measured entry.
    """

    ltx_frame_rate: int = 24
    """LTX-2.5's native rate; num_frames = seconds x this."""

    ltx_max_source_seconds: float = 330.0
    """
    Longest upload the source-duration workflows will accept.

    Music video and restyle take their length from the file the customer
    uploads, and nothing bounded it. The upload cap is 64 MB, which at ordinary
    MP3 bitrates is over an hour of audio — and an hour of audio is roughly 120
    render passes at a minute or two each. Such a job cannot finish inside its
    own timeout, so it ran for hours and then failed having produced nothing,
    holding the card the whole time and blocking every other customer.

    330 seconds is five and a half minutes: the product's own music range tops
    out at five, and the margin covers a track that probes slightly long. It is
    a refusal the customer sees IMMEDIATELY, before any compute is spent, with
    the actual length named — which is the difference between "try a shorter
    track" and a job that appears to hang.
    """

    ltx_max_extend_source_seconds: float = 1800.0
    """
    Longest source Extend Video will continue from — deliberately far looser
    than `ltx_max_source_seconds`, because the cost model is different in kind.

    For music video and restyle the source's length IS the render bill: every
    second of upload is a second the model must generate. An extension renders
    only the requested continuation; the source contributes its final frame and
    is then re-encoded once on the CPU for the stitch. Holding extensions to
    the render ceiling anyway is what made "extend it again" stop working at
    five and a half minutes total — the second extension's SOURCE was the first
    extension's output (client ask #1, 17 Aug 2026).

    30 minutes bounds what the ceiling actually protects here — ffmpeg time,
    disk, and the upload itself — while being far past anything a chain of
    generated-then-extended videos reaches in practice.
    """

    # ── Music runtime (M2) ───────────────────────────────────────────────
    #
    # No music model is selected yet (docs/milestones.md tracks it as a pending
    # decision), so unlike LTX there are no weight paths here — only the seam a
    # selected model plugs into. The worker owns the CLI contract and whatever
    # model is chosen gets a thin wrapper that satisfies it:
    #
    #     <music_launcher> --prompt TEXT --duration-seconds N --seed N
    #                      --output-path PATH [--lyrics-path PATH]
    #                      [--structure TEXT] [--continue-from PATH]
    #
    # Owning the contract rather than adapting to whichever CLI wins is what
    # keeps the model choice from reaching any of the code above the adapter.

    music_provider: str = "acestep"
    """Which provider implementation serves music jobs. See worker/music/."""

    acestep_base_url: str = "http://127.0.0.1:8001"
    """
    Where the music service listens.

    Unlike LTX, the model is NOT launched per job — it is a long-lived service
    holding ~24 GB of weights that answers requests in seconds. The worker
    treats it like a database: it connects, it never manages its lifecycle.
    """

    acestep_api_key: str = ""
    """Sent as `Authorization` when the service is started with one. Empty
    means the service is unauthenticated, which is correct on loopback."""

    acestep_max_seconds: int = 600
    """
    Longest single generation the service will accept.

    Measured, not assumed: the service reports a 600s ceiling and produced a
    240s song in 5.5s at flat VRAM on the RTX 5090. Since the product's
    longest song is 5 minutes, this covers the whole range in one pass and the
    adapter's sectioning path never triggers.
    """

    acestep_request_timeout: float = 30.0
    """Per-HTTP-call budget. Small: these are control-plane round trips, and
    the generation itself is awaited by polling rather than by one long call."""

    acestep_generation_timeout: float = 900.0
    """
    Whole-generation budget, from submit to audio.

    Generous relative to the ~6s a four-minute song actually takes, because the
    cost of being wrong is asymmetric: a job killed early wastes GPU time
    already spent, while a slow one merely finishes late.
    """

    acestep_poll_seconds: float = 1.0
    """How often to ask whether the task is done."""

    music_seconds_per_line: float = 13.0
    """
    How much song one line of lyrics needs.

    Measured on the GPU (2026-08-13 and 2026-08-16) and load-bearing in BOTH
    directions: eight lines inside a 60-second song sang only the chorus and
    silently dropped both verses, while five lines across 120 seconds produced
    an 82-second instrumental intro. Nine lines at 120s sang everything with
    vocals from 30s — so 13s/line is the densest point proven safe. See
    `worker/music/lyrics.py:line_budget`.
    """

    music_lyrics_writer: str = "template"
    """
    Which lyrics writer fills the song plan with words when the customer's
    prompt is the only input. "template" is the built-in, dependency-free
    writer (worker/music/writer.py); empty disables writing entirely.

    Load-bearing: the music model treats an empty lyric sheet as "make an
    instrumental" (verified on the GPU, 2026-08-16), so a music platform with
    no writer configured produces NO sung words on any track — which was the
    client's "lyrics not present" complaint, in its entirety.
    """

    music_crossfade_seconds: float = 1.5
    """
    Overlap between generated sections of one song.

    A butt-join between two independently generated sections is audible. The
    planner adds this back into what it asks for, so a five-minute song is
    still five minutes after the fades have eaten into it.
    """

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

    @property
    def person_matte_argv(self) -> list[str]:
        """The matting command, as argv.

        Defaults to the script shipped beside this package, invoked through the
        LTX environment's interpreter. Resolving the path from the worker's own
        location rather than naming an installed module is deliberate: the
        script travels with this checkout, so a node that has pulled the worker
        has the matter, and enabling person lock needs no separate file to be
        copied anywhere. Overridable for a node that keeps it elsewhere.
        """
        if self.person_matte_command:
            return shlex.split(self.person_matte_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "person_matte.py"
        return ["uv", "run", "python", str(script)]


@lru_cache(maxsize=1)
def get_settings() -> WorkerSettings:
    return WorkerSettings()


settings = get_settings()
