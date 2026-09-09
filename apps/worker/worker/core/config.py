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

    # ── Music Lyrics Workflow v2.0 (client specification, 9 Sep 2026) ──
    #
    # Lyrics first, validated, then sung; then the song is transcribed and
    # measured. worker/music/workflow.py is the orchestrator and
    # docs/internal/music-lyrics-workflow-v2.md the account of it.
    music_lyrics_workflow: Literal["v2", "v1"] = "v2"
    """Which lyric pipeline the music adapter runs. "v1" is the pre-9-Sep
    path unchanged (plan → write → sing, no gates, no verification);
    `execution.music_lyrics_workflow` overrides per job."""

    music_vocal_coverage_target: float = 0.90
    """The client's floor: at least this fraction of the song must be sung.
    The workflow never plans for less than 0.90 whatever this says."""

    music_max_filler_ratio: float = 0.10
    """Instrumental time, pauses, ad-libs and humming together, at most."""

    music_v2_seconds_per_line: float = 4.0
    """Seconds of song per sung line the v2 blueprint targets. Denser than
    the v1 target (8.0) on purpose: the August matrix in lyrics.py shows
    ninety-percent sung coverage arriving only at roughly 3.5-4 s/line."""

    music_rhyme_mode: Literal["strict", "relaxed"] = "strict"
    """Default rhyme validation. Strict requires an exact key match from the
    stressed vowel; relaxed accepts a shared vowel (assonance). A request's
    `rhyme_mode` overrides it."""

    music_verify_transcription: Literal["auto", "disabled"] = "auto"
    """Post-generation lyric recall via faster-whisper (the music-video
    extra's model and settings are reused). "disabled" leaves recall
    unmeasured; the stem-based coverage measure is unaffected."""

    music_verify_policy: Literal["fail", "deliver_best"] = "fail"
    """What happens when the song still fails verification after every
    retry: "fail" returns the workflow's error code (the specification);
    "deliver_best" ships the best take with the numbers in its report."""

    music_verify_max_retries: int = 2
    """Whole-track regenerations after a failed verification. Each one is a
    new seed and a firmer production brief; the sheet is kept. Cheap on the
    current model (a four-minute song renders in seconds)."""

    music_lyric_recall_threshold: float = 0.5
    """Fraction of lines the transcript must contain — heard as written or
    with some words changed. Sung vocals transcribe imperfectly in every
    language (measured 9 Sep 2026: 40% exact on Spanish takes where every
    line was audibly sung), so this is a floor for "the lines were sung",
    not a fidelity score; the exact-match share is reported beside it."""

    music_reference_fetch_python: Path | None = None
    """Interpreter with yt-dlp for reference links. None uses the LTX venv,
    where yt-dlp already lives for Music Video."""

    music_reference_fetch_timeout: float = 300.0
    music_reference_max_bytes: int = 200 * 1024 * 1024
    music_reference_max_seconds: int = 600

    cerebras_lyrics_reasoning_effort: str = "low"
    """`reasoning_effort` sent to reasoning models (gpt-oss, qwen) for lyric
    writing. "low" measured 9 Sep 2026: the default effort burned the whole
    output budget thinking and returned nothing. "none" sends no field."""

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

    # ── LTX 2.5 guideline pack (client pack, 9 Sep 2026) ─────────────────

    ltx25_guidelines_enabled: bool = False
    """
    Rewrite the customer's prompt to the client's LTX 2.5 guideline pack
    before rendering. See `worker/prompt/ltx25/`.

    Off by default, and this default is load-bearing rather than cautious.
    The prompt text IS the product on these workflows, this rewrites all of
    it through a language model, and the structuring it replaces carries GPU
    measurements the pack has no equivalent of. It wants an A/B on a real
    node before it becomes the default, not a deployment note.

    `execution.ltx25_guidelines` overrides per job, which is what makes that
    A/B one request rather than one redeploy.
    """

    # ── Automatic dialogue (client pack, 7 Sep 2026) ─────────────────────

    auto_dialogue_enabled: bool = False
    """
    Write spoken lines into a prompt that has none, on the single-pass video
    workflows.

    Off by default, and that default is the honest one: this changes what a
    customer's video SAYS, on a surface a client is mid-test on, and a feature
    that starts talking without being asked is a surprise rather than an
    improvement. A job's own `auto_dialogue` parameter overrides it either way.

    What it does is small — a prompt with no quoted words gets some — but what
    that unlocks is not, because `worker/longform/language.py` already hands
    the soundtrack to the scene when nobody speaks and to the people on screen
    when somebody does. See `worker/dialogue/__init__.py`.
    """

    auto_dialogue_layout: str = "native"
    """
    How written lines are laid into the prompt: "native" (the client's
    second-revision format, `worker/dialogue/native.py` — the writer returns
    a screenplay with one stable voice per visible speaker and a
    per-duration word range, composed as `says`/`replies` turns with no cues
    and no timing labels), "paragraph" (all lines in one block) or "beats"
    (each line led by a prose cue).

    "native" became the default on 8 Sep 2026 after the client reviewed a
    "beats" render: "After a short pause" was spoken aloud, gaps of 1.3–3 s
    sat between lines, a manner on a speaker's first line read as the voice
    changing, a speaker who never appeared spoke, and 40 words was too few
    for 30 seconds. Their validator rejects every one of those; this is it.

    The first real render (7 Sep 2026) took four lines in one paragraph and
    delivered the gist of them: one verbatim, three paraphrased, one phrase
    twice. Director mode, measured delivering lines verbatim on this same
    runtime, spreads them across events with a beat between. This is that
    lever, isolated, so a single render can say whether separation is what
    was missing. Default unchanged until it does.
    """

    auto_dialogue_local_fallback: bool = True
    """
    Fall back to the local Gemma checkpoint when the hosted writer is
    unavailable.

    Kept on: the fallback costs GPU seconds before the render starts, which is
    why it is second, but a video that quietly stops speaking whenever a
    hosted service is rate-limited is the kind of intermittency nobody can
    reproduce.
    """

    auto_dialogue_timeout_seconds: float = 30.0
    """
    How long either writer may take.

    Deliberately far under the Director planner's 900 s. This runs before a
    render that itself takes minutes, and it is optional — a writer that has
    not answered in half a minute has already cost more than the feature is
    worth, and failing open renders the customer's own prompt.
    """

    auto_dialogue_max_tokens: int = 1200
    """
    Output budget for the hosted writer.

    Room for a handful of short lines and their speaker locks, plus the
    reasoning headroom the lyrics writer's measurement demands: a reasoning
    model that overspends `max_completion_tokens` returns an empty string with
    `finish_reason: stop` and no error at all.
    """

    auto_dialogue_temperature: float = 0.7
    """
    Sampling temperature for dialogue writing.

    The Director planner's value, because this is the same job on a smaller
    canvas: the words need life, and the JSON around them still has to parse.
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

    ltx_comfy_disabled_loras: str = ""
    """Comma-separated file-name fragments to switch OFF in the pack's
    `Power Lora Loader (rgthree)` (7 Sep 2026, the client's ComfyUI operator:
    the Text to Video and First/Last Frame graphs mix LTX 2.3 adapters —
    `LTX-2.3-OmniNFT-RL-Lora_bf16` @0.4 and `ltx2.3-transition` @0.8 — with
    LTX 2.5 core files, and they suspect colour, adherence and speed).
    Empty runs the pack exactly as delivered. rgthree's loader applies an
    entry only when its `on` flag is true, so this is the same switch a
    ComfyUI operator would flick; the node stays, the adapter is never
    loaded. `execution.disabled_loras` overrides per workflow."""

    ltx_comfy_bypass_detailer: bool = False
    """Bypasses `ltx-2-19b-ic-lora-detailer` in the generation graphs, so the
    model reaches both `LTXVDualCFGGuider` nodes directly (the same client
    request). It is an LTX-2 19B adapter loaded on the 2.5 22B transformer;
    no source claims the two are compatible. Off until the A/B says
    otherwise. `execution.bypass_detailer` overrides."""

    ltx_comfy_megapixels: float | None = None
    """
    Text to Video's DELIVERED size on the pack graph, as the megapixel budget
    of its own `ResolutionSelector` (0.9 is 1280x736 at 16:9; 2.0 is
    1920x1088). None keeps the pack's 0.9. `execution.megapixels` overrides.

    The pack's first pass runs at half this (see `ltx_comfy_base_scale`) and
    its latent upsampler + 3-step refine bring it back up. Measured clean on
    the RTX PRO 6000, 15 s at 16:9 (8 Sep 2026): 0.9 MP delivers in 92 s;
    the FAST 1080 graph's native single pass takes ~310 s; a 0.9 MP delivery
    with the base at 0.75 takes 240 s at 0.46x native detail. Every faster
    path measured costs detail — see docs/internal/text-to-video-speed.md.
    """

    ltx_comfy_base_scale: float | None = None
    """
    The pack graph's `ImageScaleBy` that sizes the FIRST pass, as a fraction
    of the delivered size. The pack's own is 0.5. None keeps it.
    `execution.base_scale` overrides. Named `final_scale_by` for a few hours
    on 8 Sep 2026 under a misreading of the graph; it never touched the
    delivery.
    """

    ltx_comfy_transformer: str = ""
    """Overrides the diffusion transformer the GENERATION graphs load. Empty
    runs the pack's own `LTX-2.5-Distilled-Q8_0.gguf`. The node also carries
    Lightricks' `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
    — the file the character graph already runs — and an nvfp4 build. A GGUF
    is dequantized on every forward pass, so this is the first thing to
    measure against "the videos are taking longer" (client, 7 Sep 2026).
    `execution.transformer` overrides."""

    ltx_hd_upscaler: str = "lanczos"
    """
    How a 720p-canvas render reaches 1080p: "lanczos" (the client graph's
    own closing `ImageScale` — a resize, and what the client's own package
    specifies: "upscale afterward with FFmpeg ... lanczos") or "seedvr2" (a
    second ComfyUI prompt running SeedVR2, the temporal AI upscaler ComfyUI
    supports natively; `worker/comfy/seedvr2.py`). Their review also asked
    for "a temporal AI upscaler", so both exist; the default is the one in
    their package until the other is measured. `execution.upscaler`
    overrides per job. Needs the two SeedVR2 weight files on the node.
    """

    ltx_hd_delivery: str = "1080p"
    """
    The frame the FAST 1080 path delivers: "1080p" (1920x1080 / 1080x1920 /
    1080x1080, via the graph's own closing node) or "4k" (3840x2160 /
    2160x3840 / 2160x2160, via one lanczos resize in ffmpeg with NVENC, the
    soundtrack copied through — the client's own package method). Client
    request, 8 Sep 2026: 4K "in the same way we do 1920x1080". Measured
    2.3 s for a 10 s clip; the file is ~3x the 1080p size.
    `execution.delivery` overrides per job.
    """

    ltx_hd_canvas: str = "native"
    """
    Generation canvas for the FAST 1080 graph: "native" (the graph's own
    1920x1088), **"720p"** (the client's speed plan — 1280x704 landscape,
    704x1280 vertical, 960x960 square, each upscaled to 1080p by the graph's
    own closing node; see `DRAFT_CANVAS` in the adapter), or "WxH" such as
    "1280x736".

    The speed lever the user asked for (7 Sep 2026): generate at a
    720p-class size and let the graph's own final `ImageScale` (lanczos,
    crop=center) bring it to 1920x1080. Native 15 s measured 306 s; this
    exists so the alternative can be measured against it on the same seed
    rather than assumed. Both sides must be multiples of 32.
    """

    ltx_hd_max_seconds: float = 15.0
    """The longest Text to Video HD renders. Benchmarked 7 Sep 2026 on the
    RTX PRO 6000: 8 s = 121 s, 15 s = 306 s, 30 s = 1051 s. Thirty seconds
    works and is not hardware limited (58 % of the card, 84 % of the
    container's memory ceiling), but it holds a node that serves one job at a
    time for as long as eight 8 s renders, so the ladder stops at 15 until
    there is capacity. Raise it here and add the length to the definition's
    `supported_durations` together."""

    ltx_hd_expected_wall_per_output_second: float = 21.0
    """Progress pacing only. Measured 15.1 s of compute per second of video at
    8 s and 20.4 at 15 s; the bar is a time estimate, not a promise."""

    # ── Music Video on the client's music-video worker (v1.8.0, 8 Sep 2026) ──
    #
    # The `music_video` runtime runs the client's own orchestrator package
    # (vendored as `zolex_music_worker`) and points every backend it leaves
    # to the deployer at this node's services. See worker/musicvideo/.

    music_video_render_backend: Literal["command", "ltx", "mock"] = "command"
    """How each shot is rendered. `command` (default) is this repository's
    `scripts/mv_render.py`: it frees ComfyUI's VRAM — the anchor stage left
    Qwen-Image-Edit warm there — and then launches the official
    `ltx_pipelines.a2vid_two_stage` CLI with this node's model files and
    the platform's own audio-tier flags. `ltx` is the package's built-in
    direct adapter, the same CLI without the eviction. `mock` draws a moving
    slate from the anchor and is for the pipeline smoke only."""

    music_video_render_command: str = ""
    """JSON argv replacing `scripts/mv_render.py` for the `command` backend
    (a warm render service, the package's recommended path at scale), with
    the package's placeholders (`{request_json}`, `{output}`, …)."""

    music_video_render_engine: Literal["cli", "comfy"] = "cli"
    """Which engine `scripts/mv_render.py` uses for a shot.

    `cli` launches the official `ltx_pipelines.a2vid_two_stage` process: the
    development transformer plus the distilled LoRA, unquantized, 24 steps.
    Measured 8 Sep 2026: 96 s per 121-frame shot, of which about 35 s is
    process start and weight loading, every shot.

    `comfy` submits Lightricks' own audio-to-video graph to the warm ComfyUI
    instead (`worker/comfy/ltx_a2v.py`): the distilled transformer, 8 steps
    on the same schedule the client's FAST 1080 graph uses, then the 2x
    latent upscale and a 3-step refine. Measured the same day on the same
    anchor and the same second of the song: **24.2 s** per 121-frame shot,
    38.4 s at 193 and 49 s at 241, repeatable to two seconds, with the song
    arriving on the clip at 0.998 correlation to the source.

    The conditioning is identical either way — the audio is encoded, frozen
    with a zero noise mask and denoised against — so this is a speed and a
    LOOK decision, not a change to whether the model hears the song. It
    stays `cli` until the client has seen both."""

    music_video_tight_frames: bool = True
    """Render only the frames a shot delivers, snapped up to the model's
    8k+1 lattice, rather than the package's 121-frame floor.

    That floor is a property of the CLI decoder — 121 is the smallest count
    measured safe on this card for `a2vid_two_stage` — and the package
    applies it to every shot, so a 4.4-second shot renders 121 frames to
    deliver 105. On the ComfyUI path the tiled decoder has no such floor.
    The package trims to the delivered count either way, which is what its
    own render contract promises, so this changes nothing about the result
    and drops about an eighth of the work. `comfy` engine only."""

    music_video_image_strength: float = 0.7
    """How hard the anchor still is written into the shot's first frame on
    the `comfy` engine. Lightricks' own audio-to-video graph ships 0.7 for
    the first stage and 1.0 for the refine; the CLI path pins 1.0."""

    music_video_negative_prompt: str = (
        "identity drift, face change, deformed hands, extra fingers, extra limbs, "
        "duplicate person, cropped head, cropped feet, flicker, exposure pulsing, "
        "sudden darkness, captions, text, logo, watermark"
    )
    """The package's own negative prompt, sent by the render command."""

    music_video_a2v_guidance_scale: float | None = None
    """`--a2v-guidance-scale` for the render command; None keeps the
    pipeline's default (3.0). LTX's help: higher may improve lip-sync."""

    music_video_anchor_backend: Literal["command", "reference", "mock"] = "command"
    """Where each shot's starting still comes from. `command` renders it on
    this node's ComfyUI with Qwen-Image-Edit (`scripts/mv_anchor.py`,
    worker/comfy/qwen_edit.py) — the performer's real face composed into the
    shot. `reference` is the package's development fallback: the customer's
    photo fitted to the frame, no composition. `mock` is a slate."""

    music_video_ltx_extra_args: str = ""
    """Extra official CLI flags for the direct render backend, as a JSON
    array. The node's offload setting (`ltx_unquantized_offload`) is added
    automatically unless an `--offload` is given here."""

    music_video_inference_steps: int = 24
    """Stage-1 denoising steps per shot — the package's default, and the one
    lever that moves wall time. `execution.inference_steps` overrides per
    deployment. The platform's own audio tier runs 15 (a user-approved
    quality trade, 27 Aug 2026)."""

    music_video_shot_min_seconds: float = 5.0
    music_video_shot_target_seconds: float = 8.0
    music_video_shot_max_seconds: float = 10.0
    """The shot ladder, at the client's instruction (8 Sep 2026).

    It does NOT make the model cheaper: rendering costs about 0.21 s per
    frame with no meaningful per-shot fixed cost, measured at 121, 193 and
    241 frames, so the same song costs the same however it is cut. What it
    saves is everything a shot carries with it — one anchor still, one
    normalise and one QA pass each. A 3-minute song goes from 41 shots to
    23, which is about 100 s off the job.

    The cost is a slower cut rhythm, which is a creative choice and theirs
    to make. `execution.shot_*_seconds` override per deployment."""

    music_video_max_source_seconds: int = 300
    """The package's ceiling: five minutes of song."""

    music_video_max_attempts: int = 2
    """Render attempts per shot before the job fails, at the client's
    instruction (8 Sep 2026). The package retries only the shot that failed
    its technical QA, with a new seed, and never restarts the job."""

    music_video_command_timeout: int = 7200
    """Wall-clock ceiling for any one external command the package runs."""

    music_video_transcription_backend: Literal["faster_whisper", "disabled"] = "faster_whisper"
    """Lyric transcription for a prompt that asks to follow the lyrics.
    faster-whisper runs inside the worker process on this node's GPU;
    `disabled` makes such a prompt fail early with the package's own
    message instead of rendering an unrelated mood video."""

    music_video_whisper_model: str = "large-v3"
    """faster-whisper model name, or an absolute directory."""
    music_video_whisper_device: str = "cuda"
    music_video_whisper_compute_type: str = "float16"
    music_video_whisper_download_root: Path | None = None
    """Where faster-whisper keeps its weights (3.1 GB for large-v3). Set on
    the node so a worker restart never downloads."""

    music_video_upscale_backend: Literal["command", "cuda", "cpu"] = "command"
    """The 4K finishing pass.

    `command` runs `scripts/mv_upscale.py`, which states the frame count and
    passes timestamps through. The package's built-in `cuda` path uses
    `-shortest` instead and loses the tail of a long job: measured 8 Sep
    2026, a 3-minute master of 4,319 frames delivered 4,315 and the
    package's own QA refused it, while a 40-second job was exact. `cuda`
    and `cpu` are the package's own paths, kept for comparison."""
    music_video_upscale_cq: int = 18
    music_video_upscale_timeout: int = 1800

    music_video_enforce_prompt: bool = True
    """
    Hold the client's music-video planner to the customer's prompt
    (`worker/musicvideo/enforce.py`).

    ON by default, and this is the one place in the music-video adapter where
    the default is not "leave their package alone" -- because leaving it alone
    is the fault. Their `director.expand_direction` reads the prompt only to
    pick a genre profile; every location, palette and camera move comes from a
    table, and not one noun the customer wrote reaches a shot. A customer
    asked for a Beverly Hills penthouse full of hay bales and got a shoreline
    (client report, 9 Sep 2026). `execution.music_video_enforce: false` turns
    it off per job, which is also the A/B.
    """

    music_video_brief_writer: bool = True
    """
    Read the prompt into a creative brief with the hosted writer (the same
    Cerebras/Gemma chain Auto Dialogue uses) before falling back to the
    regex extractor. The writer splits beats more cleanly; the extractor is
    the guarantee -- it keeps the customer's sentences verbatim -- and the
    writer's brief is merged OVER it, never instead of it. Off means regex
    only, which costs nothing and still reaches every word.
    """

    music_video_reference_vision: bool = False
    """
    Analyse a pasted reference video with Qwen2.5-VL-3B as well as the
    package's built-in visual metrics. The model is not installed on the
    node (9 Sep 2026); the metrics alone give shot rhythm, palette and
    motion, which is what the treatment actually consumes. Turn on once the
    weights are there.
    """

    music_video_lyric_mode: Literal["automatic", "always", "off"] = "automatic"
    """When lyrics are transcribed: `automatic` only for a prompt that says
    so ("according to the lyrics"), `always` for every song, `off` never.
    `execution.lyric_mode` overrides."""

    music_video_anchor_steps: int = 4
    """Sampling steps per anchor still — 4 with the Lightning LoRA."""

    music_video_anchor_scale: float = 0.75
    """Fraction of the shot's size an anchor is GENERATED at, before being
    resized to the size the package asked for.

    An anchor is a conditioning frame, not delivered pixels: the graph
    VAE-encodes it to 640x352 for the first stage and to the working size
    for the refine, so generating it at the full 1280x704 buys detail that
    is immediately thrown away in stage one. Measured 8 Sep 2026, the anchor
    stage is 218 s of a 1300 s three-minute job — the largest block that is
    not the model. Three two-performer anchors took 10.1 s each at full size
    and 6.6 s at 0.75, and both hold the faces and the wardrobe; 0.6 costs
    the same 6.6 s, so below 0.75 there is nothing left to win. 1.0 keeps
    the full size."""
    music_video_anchor_lightning: bool = True
    """Use the Lightning 4-step LoRA (seconds per still) rather than the
    base 20-step schedule (a minute per still)."""

    ltx_comfy_input_dir: Path | None = None
    """ComfyUI's `input/` directory when the worker shares a filesystem with
    it. Optional: inputs travel over HTTP either way; this only enables
    cleanup of a job's uploads afterwards."""

    character_replacement_max_seconds: int = 10
    """The longest source window one character-replacement pass renders.

    Measured on the RTX PRO 6000 (6 Sep 2026): a 10 s window peaks at 85 GB
    of VRAM, 85 GB of process RAM and 110 GiB of container memory; a 20 s
    window grew past the container's 241 GiB RAM limit and the kernel killed
    ComfyUI mid-render. The pack's note lists 5/10/20 s; on this card 10 is
    the ceiling. Longer sources are cut to this window and the delivered
    length is logged."""

    character_replacement_canvas: tuple[int, int] = (736, 1280)
    """The pack's pinned canvas (portrait). Oriented to match the source
    clip; the pixel budget is never changed."""

    character_replacement_max_total_seconds: int = 120
    """The longest source the tool follows in TOTAL, as a chain of windows
    (client request, 6 Sep 2026: "the length of the source video, like
    Video to Video"). Each window is one run of the unchanged client graph
    at most `character_replacement_max_seconds` long; the next window's
    reference picture is the last frame the previous window produced, so
    the character carries across the seam. Every 10 s of source costs about
    5.5 minutes on the RTX PRO 6000 (measured 6 Sep: 323 s per 10 s window),
    and the node serves one job at a time — that is why this is 2 minutes
    and not Video to Video's 5. Raise it here or per deployment
    (`execution.max_total_seconds`) once the wait is acceptable."""

    character_replacement_chain_reference: Literal["previous_frame", "photo"] = "previous_frame"
    """What the second and later windows use as their reference picture.
    `previous_frame` (default) is the last frame the previous window
    produced — continuity across the seam, the character's pose carries on.
    `photo` is the customer's picture again for every window — no identity
    drift over a long chain, but every window starts on the photo's pose.
    Both are the graph's own one-image input; nothing else differs."""

    character_replacement_anchor_reference: bool = True
    """Colour-anchor every chained seed frame before it becomes the next
    window's reference picture (6 Sep 2026 fix for "he went darker through
    the video"). Each window is seeded from a RENDERED frame, and the graph
    renders that frame's look a little darker and flatter than it was given
    (measured on the client's clip: 90th-percentile luminance 178 → 164 →
    157 → 146 → 145 across four windows, chroma unchanged). Left alone the
    loss compounds. Anchoring matches the seed's luminance level and spread,
    and its chroma means, to the first window's own rendering just after
    its handoff — the look the customer approved — with bounded gains, so
    each window starts where window 1 started. Sources within one window
    are untouched; `photo` mode needs no anchor."""

    character_replacement_chain_skin_clause: bool = True
    """Adds the hands clause (`CHARACTER_REPLACEMENT_SKIN`) to the prompt of
    every window of a CHAINED character replacement, between the pack's lead
    sentence and the customer's text. Never touches a source within one
    window. On since the A/B of 7 Sep 2026 on the client's clip: with the skin
    anchor it took the hands at 16-33 s from 83/91 to 107/108 on the
    source-hand-masked metric and the hand/face ratio from 0.94/0.97 to
    1.23/1.21, face unchanged (`execution.chain_skin_clause` overrides per
    deployment)."""

    character_replacement_skin_anchor: bool = True
    """Skin-region re-anchoring of every chained seed (`worker.media.skin`,
    7 Sep 2026): inside the source performer's skin silhouette at the seam,
    skin that is dark AS A WHOLE against the first window's own skin level
    is lifted back to it (bounded additive offsets) before the seed becomes
    the next reference. Containment for "the hands get darker": stops the
    compounding at each seam; the within-window slide is the clause's job.
    Never touches a source within one window. On since the A/B of 7 Sep 2026
    (see `character_replacement_chain_skin_clause`); `execution.skin_anchor`
    overrides per deployment."""

    character_replacement_exposure_clause: bool = True
    """
    The client's lighting lock in the character-replacement prompt.

    Their MULTI4 graph (7 Sep 2026) answers the darkening report with two
    things this platform did not have: a positive sentence pinning the
    scene's light and the character's skin brightness to what the first frame
    established, and a negative that names the DIRECTION of the fault
    ("darker face", "underexposure", "crushed blacks") where ours only ever
    named a change. Both are carried; this switches the positive half.

    On by default because the client asked for it. It is not the whole of
    their fix — see `character_replacement_chain_reference`, whose default
    does the very thing their backend contract forbids.
    """

    character_replacement_skin_hold: bool = True
    """Per-frame skin hold on every window of a CHAINED character
    replacement (`worker.media.skin_hold`, 7 Sep 2026). The seed pass puts
    each seed back at the first window's skin level, but the skin slides
    again inside the next window (measured on the client's clip: Y ≈ 136 in
    the first second to ≈ 100 by the end of an 8 s window, whatever the
    prompt), so the seams read as brightness steps. The hold measures every
    delivered frame under the same silhouette gate and ramp and lifts dark
    skin back to the first window's own level with a smoothly varying,
    bounded offset, following the source performer's own skin level so real
    shadows stay. The GPU renders and the seeds are unchanged; a source
    within one window is untouched. `execution.skin_hold` overrides."""

    character_replacement_delivery: str = "native"
    """The frame the finished Character Replacement video is delivered at:
    "native" (the generation canvas, unchanged) or "4k".

    Client request, 8 Sep 2026: "let's use the same approach upscale to 4k
    after video is done". It is the same lanczos-and-NVENC finish Text to
    Video HD ships (`worker/media/upscale.py`), and it runs ONCE, after the
    windows are joined and the source audio is laid over — one encode, one
    audio stream, no seam. It buys nothing in detail: 4K here is the
    generated frame enlarged, which is what the client asked for and what
    their own package does at 1080p. Default "native" so no existing
    deployment starts writing files four times the size without being told.
    `execution.delivery` overrides."""

    character_replacement_free_after_chain: bool = True
    """Ask ComfyUI to release its memory after a CHAINED job (7 Sep 2026).
    A chain submits one prompt per window and ComfyUI's resident set grows
    with them: measured at 91.7 GB after an afternoon of jobs, dropping to
    21.1 GB on `POST /free`, and the container's cgroup killed the server
    four times in two days at its 241 GiB limit — once mid-render of the
    client's own job, which cost 10 minutes of GPU and re-ran from scratch.
    NOTE: ComfyUI unloads its models on any `/free`, whatever `unload_models`
    says (its `main.py` reads `flags.get("unload_models", free_memory)`), so
    the next job pays a model load. That is why this is scoped to chained
    jobs — the ones that accumulate, and that run for tens of minutes anyway;
    a single-window job leaves the models warm exactly as before."""

    character_replacement_ripple_strength: float | None = None
    """Overrides the Ripple LoRA's `strength_model` in the client's graph
    (1.35 as shipped; the client's advisor suggests 1.45-1.50 when the edit
    does not carry strongly enough). None runs the graph as delivered. A
    per-deployment lever for the A/B, never a silent default change;
    `execution.ripple_strength` overrides."""

    character_replacement_chain_window_seconds: int | None = None
    """Window length for CHAINED sources only (None = `max_seconds`, as
    today). Read only when a source already exceeds `max_seconds`, so a
    source that ran as one window keeps running as one window whatever this
    says; capped by `max_seconds` (the memory ceiling). Shorter windows bound
    how far the hands can slide inside one window before the next seed is
    re-anchored; the price is more seams. `execution.chain_window_seconds`
    overrides."""

    character_replacement_expected_wall_per_output_second: float = 30.0
    """Progress pacing only. Measured 6 Sep 2026 on the RTX PRO 6000: an 8 s
    window took 164 s (20.5 s per second), a 10 s window 323 s (32)."""

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
