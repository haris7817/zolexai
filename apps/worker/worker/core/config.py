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

from pydantic import AliasChoices, Field
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

    `person_anchor_command` below is its sibling for the composited identity
    anchor (`scripts/person_anchor.py`), on the same seam for the same
    reason.

    A command rather than an import for the same reason the pipelines are: the
    matting model needs torch and CUDA, and this worker deliberately has
    neither. It runs in the LTX environment (`ltx_repo_dir` is the working
    directory) and speaks a small, stable CLI, so the segmentation model behind
    it can be replaced without touching a line of worker code.
    """

    person_anchor_command: str = ""
    """How to build the composited identity anchor for
    `execution.v2v_reference_identity` — see `person_anchor_argv`."""

    director_planner_command: str = ""
    """
    How to invoke the Director-mode scene planner, which turns a one-line idea
    into a structured dialogue plan. Empty means "the script shipped in this
    checkout, run in the LTX environment" — see `director_planner_argv`. Same
    seam and same reasoning as `person_matte_command`: it is model work, the
    worker has no torch, and a subprocess CLI keeps the planning model
    swappable without worker changes.
    """

    director_gemma_dir: Path | None = None
    """
    HF directory of the generative Gemma instruct checkpoint the planner runs.
    Defaults to `<ltx_repo_dir>/models/gemma-4-e2b-it` — deliberately the same
    checkpoint the LTX 2.5 runtime documents as its official prompt enhancer
    (Apache 2.0), so one ~10 GB download serves both roles.
    """

    director_planner_timeout_seconds: float = 900.0
    """
    Wall-clock ceiling for one planning subprocess, model load included. A cold
    load plus a long plan is minutes, not seconds; a planner that has hung is
    better killed and retried than left holding the job's budget.
    """

    director_vision_enabled: bool = False
    """
    Whether Image-to-Video Director mode may LOOK at the uploaded image before
    planning — a subprocess that asks the local checkpoint to state what the
    photograph visibly shows, so the plan's continuity facts come from the
    image rather than only from the idea.

    OFF by default for the same reason the guided tier is: whether the
    on-box checkpoint accepts image input is a measurement nobody has made,
    and this codebase does not ship unmeasured model paths as defaults.
    Planning works without it — the planner is then forbidden to invent
    visual details, and identity rides on the conditioned frames alone.
    A failure while enabled degrades to exactly that posture; it never fails
    the job.
    """

    director_vision_command: str = ""
    """
    How to invoke the image-facts describer. Empty means "the script shipped
    in this checkout, run in the LTX environment" — see
    `director_vision_argv`. Same seam as `director_planner_command`.
    """

    director_vision_timeout_seconds: float = 300.0
    """
    Wall-clock ceiling for one image-description subprocess, model load
    included. Shorter than the planner's: this step is optional garnish, and
    holding a job for minutes over it would cost more than the facts are
    worth.
    """

    ltx_quantization: str = "nvfp4-prequant"
    """
    NVFP4 is the only mode that fits the client's RTX 5090: the BF16
    transformer alone is ~40 GB against 32 GB of VRAM. Verified by benchmark
    on 2026-08-12 (docs/internal/ltx-2.5-licensing-review.md is the licensing
    side; the VRAM ceiling is an engineering fact independent of it).
    """

    vocal_separator_python: Path | None = None
    """Python of the stem-separation venv (demucs), e.g.
    /workspace/vocal-sep/.venv/bin/python. None disables vocal-aware
    performance direction entirely — the music-video prompt then behaves as
    it always did. A dedicated venv because the worker's own environment
    stays light and the pinned pipeline venvs stay untouched."""

    vocal_separator_timeout: float = 300.0
    """Stem separation budget. htdemucs on the production GPU does a 3-minute
    track in well under a minute; five covers a cold model download."""

    ltx_unquantized_offload: str = "cpu"
    """How the UNQUANTIZED tiers (audio-conditioned music video, IC-LoRA,
    reference anchor) fit the card: "cpu" streams the 22B transformer's
    weights from host RAM each pass — safe anywhere, and measured 23-30%
    slower than "none", which keeps the weights resident. "none" needs the
    headroom that lazy ComfyUI eviction created on the 96 GB production node
    (27 Aug 2026); a node where it OOMs sets this back to "cpu". Per-node
    hardware property, hence a setting and not a workflow key."""

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

    music_seconds_per_line: float = 6.0
    """
    How much song one line of lyrics needs — the CEILING on a sheet's size.

    Re-measured 2026-08-21 against the ACE-Step build actually in production,
    twelve cells across three durations and four densities, vocal presence read
    from a separated stem rather than from a transcriber. The previous 13.0 came
    from an RTX 5090 and an older checkpoint, and against this one it is where
    coverage falls apart: a three-minute song at that density sang for 52.8% of
    its length, with a 43-second hole in the middle and nothing at all for the
    first thirty seconds.

    This is the ceiling; `_TARGET_SECONDS_PER_LINE` in `worker/music/lyrics.py`
    is the number a writer is actually told to hit, and that file carries the
    full matrix.
    """

    music_lyrics_writer: str = "cerebras,template"
    """
    Which lyrics writer fills the song plan with words when the customer's
    prompt is the only input — and, as a comma-separated list, in what order.

    Known names: "cerebras" (a hosted language model, writes any offered
    language) and "template" (the built-in, dependency-free English bank in
    worker/music/writer.py). Empty disables writing entirely.

    The default puts the model first and the bank behind it. With no
    CEREBRAS_API_KEY set, the first entry reports itself unavailable and the
    chain behaves exactly as it did when "template" was the only value — so
    this default is safe on a deployment that has not configured anything.

    A fallback is only tried for languages it can actually write; see
    worker/music/fallback.py. Falling back from Spanish to an English-only
    writer is not a degradation, it is the wrong song, and the chain refuses it.

    Load-bearing: the music model treats an empty lyric sheet as "make an
    instrumental" (verified on the GPU, 2026-08-16), so a music platform with
    no writer configured produces NO sung words on any track — which was the
    client's "lyrics not present" complaint, in its entirety.
    """

    # ── Cerebras (lyrics text only — never audio) ────────────────────────
    #
    # Cerebras writes words and nothing else. ACE-Step remains solely
    # responsible for composition, vocals and audio; the two never meet except
    # through a lyric sheet. See worker/music/cerebras.py.

    cerebras_api_key: str = ""
    """
    Read from the worker's environment as CEREBRAS_API_KEY, and used in exactly
    one place: the Authorization header in worker/music/cerebras.py.

    It never reaches the browser (the web app has no notion of a lyrics
    provider), never enters job parameters or job metadata, and is never
    logged — the log records latency, token counts and the model name, none of
    which identify the credential. Empty means the Cerebras writer reports
    itself unavailable and the chain moves to the next writer.
    """

    cerebras_base_url: str = "https://api.cerebras.ai"
    """The API root. Overridable so a test or a proxy can stand in front."""

    cerebras_lyrics_model: str = Field(
        default="gemma-4-31b",
        validation_alias=AliasChoices("CEREBRAS_LYRICS_MODEL", "CEREBRAS_AI_MODEL"),
    )
    """
    Which model writes the lyrics.

    Two accepted names, because a deployment already had this set as
    `CEREBRAS_AI_MODEL` and a configured value that is silently ignored is
    worse than one that is rejected — it looks connected and changes nothing,
    which is the same class of bug as a language selector that does not select.
    `CEREBRAS_LYRICS_MODEL` is the canonical name; the other is an alias kept
    for that existing config.

    `gemma-4-31b` because the product offers fourteen lyric languages and Gemma
    is the multilingual one of the two models on the Cerebras public endpoint
    (checked 2026-08-19); the other, `gpt-oss-120b`, is reasoning-first,
    English-centred, and emits a reasoning channel that fights a "return only
    the lyric sheet" contract.

    Configurable because that lineup changes — this default is a reasoned
    starting point, not a permanent fact.
    """

    cerebras_lyrics_enabled: bool = True
    """
    The feature switch for automatic lyrics via Cerebras.

    False makes the writer report itself unavailable, which is the same path a
    missing key takes: the chain moves on. It exists so the hosted writer can
    be turned off on a running deployment without editing the writer list and
    without a restart that changes anything else.
    """

    cerebras_lyrics_timeout_seconds: float = 45.0
    """
    Whole-request budget for one lyric generation.

    A few hundred tokens on this hardware is a couple of seconds, so this is
    already generous. It is bounded at all because the alternative is a stalled
    text call holding a music job — and therefore a GPU slot — open for as long
    as the job timeout allows.
    """

    cerebras_lyrics_max_retries: int = 1
    """
    Extra attempts after the first, within one call to the writer.

    One, deliberately. This budget covers BOTH kinds of second chance — a
    transient transport failure (timeout, 429, 5xx) and a sheet that came back
    in the wrong language, which is retried with a reinforced instruction. A
    permanent failure (bad key, unknown model) does not consume it at all and
    goes straight to the fallback.

    Kept small because it sits inside a job a customer is watching, and because
    the review loop above may call the writer a second time anyway.
    """

    # ── Director scene planning (Text to Video, Idea mode) ───────────────
    #
    # The same account and key as lyrics; a separate model, switch and budget
    # because the two tasks fail differently and are tuned separately. See
    # worker/director/cerebras.py.

    cerebras_director_model: str = Field(
        default="gemma-4-31b",
        validation_alias=AliasChoices("CEREBRAS_DIRECTOR_MODEL", "CEREBRAS_AI_MODEL"),
    )
    """
    Which model plans the scene.

    Defaults to the same Gemma the lyric writer uses, for the same measured
    reasons: it is the multilingual one of the two models on the Cerebras
    public endpoint, and it is the plainer instruction-follower for a "return
    only this JSON" contract — which matters more here than for lyrics,
    because a plan is parsed rather than read.
    """

    cerebras_director_enabled: bool = True
    """
    Whether Director mode may plan on the hosted model.

    False falls the chain through to the local Gemma checkpoint, which is
    slower but needs nothing external. This exists so the hosted planner can be
    turned off on a running deployment without the feature going with it.
    """

    cerebras_director_timeout_seconds: float = 60.0
    """
    Whole-request budget for one planning call.

    Longer than the lyrics equivalent because a plan is a bigger answer, and
    bounded at all because this call sits in front of a render: a stalled
    request would hold a GPU slot open producing nothing.
    """

    cerebras_director_temperature: float = 0.7
    """
    Sampling temperature for scene planning.

    Below the lyric writer's 0.8. A plan is a structure that gets parsed and
    validated, not a creative artefact read by a human — the dialogue inside it
    still needs life, but a planner that wanders off the JSON shape costs an
    attempt.
    """

    cerebras_lyrics_temperature: float = 0.8
    """
    Sampling temperature for lyric writing.

    Above the middle on purpose: a near-deterministic writer hands the same
    song to every customer whose prompt rhymes with another's. Low enough that
    the structure rules in the prompt are still followed.
    """

    music_crossfade_seconds: float = 1.5
    """
    Overlap between generated sections of one song.

    A butt-join between two independently generated sections is audible. The
    planner adds this back into what it asks for, so a five-minute song is
    still five minutes after the fades have eaten into it.
    """

    # ── H3 through the pinned ComfyUI INT8 stack (client-test runtime) ───
    #
    # The service is the client pack proven on 25 Aug 2026: ComfyUI v0.33.3,
    # Extender 6a3583d, Easy-Use 4de1ab3, official Comfy-Org INT8 weights.
    # Like ACE-Step, it is a long-lived local process the worker connects to
    # and never manages. `docs/internal/h3-client-runtime-freeze.md` is the
    # source of truth for every pin.

    h3_comfy_base_url: str = "http://127.0.0.1:8188"
    """Where the pinned ComfyUI listens. Loopback on the GPU node."""

    h3_comfy_workflows_dir: Path = REPO_ROOT / "benchmarks" / "client-pack"
    """The frozen client workflow graphs. These files are the contract; the
    adapter edits only what the pack itself sanctions."""

    h3_comfy_input_dir: Path | None = None
    """ComfyUI's own `input/` directory. Required to run: LoadImage validates
    files there at submit time, so the adapter stages job inputs into it.
    None means this node does not carry the H3 ComfyUI runtime."""

    h3_comfy_models_dir: Path | None = None
    """Root of the official Comfy-Org weights, for health verification
    (existence + exact published size every check; full SHA at provisioning)."""

    h3_comfy_request_timeout: float = 30.0
    """Per-HTTP-call budget — control-plane round trips only."""

    h3_comfy_poll_seconds: float = 3.0
    """How often to ask whether the prompt finished."""

    h3_comfy_generation_timeout: float = 3600.0
    """Whole-generation ceiling. The measured worst case is the 60 s quality
    run at ~13 minutes; an hour covers a cold model load plus a slow run
    without letting a wedged service hold a lease forever."""

    h3_comfy_draft_canvas: tuple[int, int] = (544, 320)
    """R2V draft tier — the pack's shipped canvas, measured at ~11-12x real
    time. Multiples of 32, as the guide requires."""

    h3_comfy_free_after_job: bool = False
    """Unload ComfyUI's models after each H3 job.

    Default OFF since 25 Aug 2026 — the lazy policy: H3 keeps its ~52 GB warm
    between H3 jobs (saving the measured 40-60 s reload every job paid), and
    the LTX and music adapters evict it just before they need the card
    (`evict_comfy_vram`). Same OOM safety, paid only on an actual engine
    switch instead of on every job. Set True to restore eager freeing on a
    node whose job mix makes back-to-back H3 rare."""

    h3_comfy_quality_canvas: tuple[int, int] = (960, 544)
    """R2V delivery tier — measured at ~33x real time with the best identity
    adherence of the whole H3 evaluation. Cost scales linearly in pixels."""

    h3_comfy_video_to_video: bool = False
    """Whether H3 may serve Video to Video at all. Default OFF since
    28 Aug 2026.

    The R2V graph consumes IMAGES. Mapped the way the proven D1 run mapped it
    — reference photo to Picture 1, the source video's FIRST FRAME to
    Picture 2 — it generates a new performance in a place that resembles the
    source's opening shot, and the customer's footage is otherwise unused.
    That is a legitimate thing to sell, but it is not what Video to Video
    promises, and the client said so in the plainest possible terms on 28 Aug
    2026: "I put a video and press better and give me a whole different
    video."

    Following the footage is LTX transform's job — an edge map of the source
    drives every frame, and `v2v_reference_identity` replaces the person
    inside it (GPU-verified 19 Aug 2026). So both quality levels route there
    and this switch is what a benchmark flips to reach the R2V path again.
    With it off, `supports()` declines the workflow and the resolver's
    safety net serves the job on the base runtime rather than failing it."""

    # ── H3 availability (client decision, 5 Sep 2026: hidden, not used) ──

    enable_h3: bool = False
    """Whether this node may serve the H3 engine at all (env `ENABLE_H3`).

    Off (the default): the `h3_comfy` adapter declines every workflow, refuses
    to run, is not advertised in the node's runtime list at registration, and
    the benchmark router refuses `provider=h3`. The API side refuses to boot
    on YAML that routes to it. Nothing is deleted — `ENABLE_H3=true` restores
    the 28 Aug 2026 behaviour exactly, which is the rollback."""

    # ── LTX 2.5 through the client's ComfyUI graphs (Sep 2026) ───────────
    #
    # A SECOND ComfyUI instance. The client's graphs use core nodes stamped
    # 0.34.0 and KJNodes/LTXVideo commits newer than the H3 freeze (v0.33.3),
    # so they cannot share that instance without an H3 compatibility pass
    # nobody has run. Own venv, own port, own supervisord program — see
    # docs/internal/ltx-comfy-runtime.md.

    ltx_comfy_base_url: str = "http://127.0.0.1:8189"
    """Where the LTX ComfyUI listens. Loopback on the GPU node."""

    ltx_comfy_workflows_dir: Path = REPO_ROOT / "benchmarks" / "client-pack" / "ltx25"
    """The frozen client graphs. The files are the contract; the compiler
    edits only what a job must supply."""

    ltx_comfy_models_dir: Path | None = None
    """Root of the ComfyUI `models/` tree, for the deep health check (file
    presence and size). None skips the filesystem half; the combo-option
    check against `/object_info` still runs."""

    ltx_comfy_request_timeout: float = 60.0
    """Per control-plane call. Uploads and downloads use their own budget."""

    ltx_comfy_transfer_timeout: float = 900.0
    """Ceiling for one input upload or output download over HTTP — a 512 MB
    source clip on loopback is seconds; the number is a guard, not a pace."""

    ltx_comfy_poll_seconds: float = 3.0

    ltx_comfy_generation_timeout: float = 3600.0
    """Whole-submission ceiling. UNMEASURED on this pack — the ZIP shipped no
    timings. Revisit after the GPU benchmark (`scripts/ltx_comfy_bench.py`)."""

    ltx_comfy_expected_wall_per_output_second: float = 7.5
    """Progress pacing only — never a completion claim. Measured 5 Sep 2026
    on the RTX PRO 6000 (client T2V graph, 1280x704): 48.8 s for 5 s, 76 s
    for 10 s, 106 s for 15 s, 215 s for 30 s — about 20 s fixed plus 6.5 s
    per output second. 7.5 keeps the bar honest across the ladder."""

    ltx_comfy_frame_rate: int = 24
    """What the graphs' FPS constants say. Every product duration lands on
    the model's 8k+1 lattice at this rate (121/241/361/721 frames)."""

    ltx_comfy_max_segment_seconds: float = 30.0
    """One graph submission's ceiling — the pack's own slider maximum and the
    client's sample length. Longer results are chained continuations through
    the extension engine, never one pass."""

    ltx_comfy_free_after_job: bool = False
    """Unload the LTX ComfyUI's models after every job. Off: the lazy policy —
    models stay warm between LTX jobs and are evicted only when another
    engine needs the card (`evict_comfy_vram`)."""

    ltx_comfy_input_dir: Path | None = None
    """ComfyUI's `input/` directory when the worker shares a filesystem with
    it. Optional: inputs travel over HTTP either way; this only enables
    cleanup of a job's uploads afterwards."""

    character_replacement_max_seconds: int = 20
    """The longest source window one character-replacement pass renders. The
    pack's own note lists 5/10/20 s as the lengths to try; the client's
    sample was 8 s. Longer sources are cut to this window (the delivered
    length is stated to the customer) until the chained variant is measured."""

    character_replacement_canvas: tuple[int, int] = (736, 1280)
    """The pack's pinned canvas (portrait). Oriented to match the source
    clip; the pixel budget is never changed."""

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    #: Mock-only pacing, matching the timings the client approved during PRE-M1.
    mock_speed: float = 1.0

    @property
    def api_v1(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/api/v1"

    @property
    def runtime_list(self) -> list[str]:
        """Runtimes this node serves, always including its primary one.

        `h3_comfy` is dropped unless `ENABLE_H3` is on, whatever the env
        declares: the API intersects this list with each workflow's runtime
        at claim time, so a node that does not advertise the engine can never
        be handed one of its jobs.
        """
        declared = [item.strip() for item in self.runtimes.split(",") if item.strip()]
        if self.runtime and self.runtime not in declared:
            declared.insert(0, self.runtime)
        if not self.enable_h3:
            declared = [item for item in declared if item != "h3_comfy"]
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

    @property
    def person_anchor_argv(self) -> list[str]:
        """The composited-identity-anchor command, as argv — same pattern."""
        if self.person_anchor_command:
            return shlex.split(self.person_anchor_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "person_anchor.py"
        return ["uv", "run", "python", str(script)]

    @property
    def director_gemma_root(self) -> Path:
        return self.director_gemma_dir or self.ltx_repo_dir / "models" / "gemma-4-e2b-it"

    @property
    def director_planner_argv(self) -> list[str]:
        """The planning command, as argv — the `person_matte_argv` pattern:
        the script travels with this checkout and runs in the LTX environment,
        so a node that has pulled the worker already has the planner."""
        if self.director_planner_command:
            return shlex.split(self.director_planner_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "director_plan.py"
        return ["uv", "run", "python", str(script)]

    @property
    def director_vision_argv(self) -> list[str]:
        """The image-facts command, as argv — same pattern as the planner's."""
        if self.director_vision_command:
            return shlex.split(self.director_vision_command)
        script = Path(__file__).resolve().parents[2] / "scripts" / "director_image_facts.py"
        return ["uv", "run", "python", str(script)]


@lru_cache(maxsize=1)
def get_settings() -> WorkerSettings:
    return WorkerSettings()


settings = get_settings()
