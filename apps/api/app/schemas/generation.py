"""Public request/response contracts for generation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import JobStatus

RoleName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=48)]


class PerformerSpec(BaseModel):
    """One member of the band a Music Video is about (client package
    v1.8.0, 8 Sep 2026: "bands/groups of up to five people").

    `slot` pairs this entry with the picture in the `performer_{slot}` input
    role, so a member's face, role and description travel together without a
    new column: the picture is an input like any other, and the words ride
    in the parameters. A member may be described without a picture (the
    worker then creates a fictional identity from the description) and a
    picture may arrive without an entry (a plain performer)."""

    model_config = ConfigDict(extra="forbid")

    slot: int = Field(ge=1, le=5)
    role: str = Field(default="performer", max_length=48)
    """`lead_vocalist`, `guitarist`, `drummer`, `dj`… — free text; the
    worker's prompts turn recognised instrument roles into instrument
    actions instead of singing."""
    description: str = Field(default="", max_length=300)
    """What must stay the same on this person — wardrobe, jewellery."""


class GenerationParameters(BaseModel):
    """The creative settings. Which of these apply is decided by the workflow.

    Every field is optional at the schema level because whether it applies is
    decided by the workflow, not by the shape: an automatic-duration workflow
    takes its length from the uploaded file and REJECTS a supplied duration,
    while a fixed-duration workflow requires one. Sending a parameter a
    workflow does not use is rejected rather than ignored, so a client bug
    surfaces instead of silently producing something the user did not ask for.
    """

    model_config = ConfigDict(extra="forbid")

    duration: str | None = Field(default=None, max_length=16)
    aspect_ratio: str | None = Field(default=None, max_length=16)
    quality: str | None = Field(default=None, max_length=32)

    motion_strength: int = Field(default=60, ge=0, le=100)
    prompt_adherence: int = Field(default=75, ge=0, le=100)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    lyrics: str | None = Field(default=None, max_length=10_000)
    """The customer's own words, passed to the music runtime untouched — they
    are the one thing the platform must never rewrite. The music model sings
    whatever language the sheet is written in (50+ supported natively), so
    this field IS multilingual lyrics support."""
    lyrics_language: str | None = Field(default=None, max_length=32)
    """Requested language for GENERATED lyrics, when the customer supplies
    none. Recorded and passed to the worker's lyric writer; writers that
    cannot honour it say so in the log rather than silently singing English."""

    # ── Music Lyrics Workflow v2.0 (client specification, 9 Sep 2026) ──
    # Only workflows whose definition sets `settings.lyrics_workflow` accept
    # these; absent means the workflow's defaults, so every existing client
    # keeps its exact behaviour.
    rhyme_scheme: Literal["auto", "AABB", "ABAB", "AAAA"] | None = None
    """The rhyme scheme every section's lines follow. `auto` picks by genre."""
    rhyme_mode: Literal["strict", "relaxed"] | None = None
    """Strict requires an exact rhyme from the stressed vowel; relaxed
    accepts a shared vowel sound. The deployment default is strict."""
    point_of_view: Literal["auto", "first_person", "second_person", "third_person"] | None = None
    clean_mode: bool | None = None
    """No profanity or explicit content. Absent means clean."""
    reference_audio_url: str | None = Field(default=None, max_length=2048)
    """A public song whose tempo, energy, structure and vocal cadence the
    new song should be LIKE. HTTPS, on a host the worker can fetch (YouTube,
    Vimeo, SoundCloud). Its lyrics, melody and voice are never copied; the
    worker's originality gate refuses a sheet that shares a run of words
    with it. An uploaded `reference_audio` input wins over a link."""
    dry_run: bool | None = None
    """Plan and validate the lyrics without generating audio. The job ends
    with the plan's numbers in its message and no output; the worker keeps
    the full report. For checking a brief before spending GPU time."""

    sound: bool | None = Field(default=None)
    """Whether the finished video carries its soundtrack. Absent means yes —
    every existing client keeps its exact behaviour, and the worker's own
    default agrees. Only workflows whose definition sets `settings.sound`
    accept it — on every quality level (client round two, 27 Aug): both
    engines deliver an audio track."""

    prompt_mode: str | None = Field(default=None, max_length=32)
    """How the prompt should be read: `standard` (the default — the text IS
    the generation prompt) or `director` (the text is an IDEA, and the worker
    plans characters, dialogue and timing from it before generating). Only
    workflows whose definition sets `settings.prompt_modes` accept it; absent
    means standard, so every existing client keeps its exact behaviour."""
    dialogue_language: str | None = Field(default=None, max_length=32)
    """Language the video's speech is in. `auto` (or absent) leaves it to the
    deployment's `spoken_language`. Accepted by workflows that offer prompt
    modes and by workflows that offer Auto Dialogue — both write speech, and
    the language question is the same question either way."""

    auto_dialogue: bool | None = Field(default=None)
    """Whether the worker should write spoken lines into a prompt that has
    none. Absent means the deployment's own default (off unless
    `AUTO_DIALOGUE_ENABLED`), so every existing client keeps its exact
    behaviour. Set explicitly, it is a promise: a job that asks for dialogue
    and cannot be given any FAILS rather than delivering a silent video that
    looks like a success. Only workflows whose definition sets
    `settings.auto_dialogue` accept it."""
    maximum_speakers: int | None = Field(default=None, ge=1, le=4)
    """How many visible people Auto Dialogue may give words to. Four is the
    client's validator's ceiling and the default; one pass holds one voice per
    speaker and no more than this many convincingly."""

    reference_video_url: str | None = Field(default=None, max_length=2048)
    """A public video the customer wants the CAMERA of -- angles, composition,
    pacing -- on workflows whose definition sets `settings.reference_video`
    (Music Video, 9 Sep 2026). HTTPS, on a host the worker can fetch
    (YouTube, Vimeo). Its people, places and shot order are never copied:
    the worker's originality policy takes only the filmmaking grammar."""

    performers: list[PerformerSpec] | None = Field(default=None, max_length=5)
    """The band, for workflows whose definition sets `settings.performers`
    (Music Video since the client's music-video worker, 8 Sep 2026). Absent
    means no named performers — the worker invents a fictional adult lead —
    so every existing client keeps its exact behaviour."""


class GenerationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(max_length=64)
    prompt: str = Field(default="", max_length=20_000)
    parameters: GenerationParameters

    inputs: dict[RoleName, uuid.UUID] = Field(default_factory=dict)
    """
    Role -> asset id. Roles come from the workflow definition, so Video to Video
    can accept `{"source_video": ..., "reference_image": ...}` with the second
    optional, and no other endpoint or column changes (directive §14).
    """


class GenerationOutput(BaseModel):
    asset_id: uuid.UUID
    kind: str
    is_primary: bool
    url: str | None = None
    """Short-lived presigned URL. Responses carrying one are marked no-store."""

    # ── What the file actually is ────────────────────────────────────────
    #
    # Measured by the worker from the finished artifact, not echoed back from
    # the request. The client reported two symptoms that were both this data
    # being absent: a 9:16 video displayed inside a hard-coded 16:9 frame, and
    # an extension showing "5s" (the requested extension) beside a 14-second
    # result (the actual file).
    #
    # Nothing here is provider detail — a duration and a pixel size are
    # properties of the media a customer downloaded, which is why they are safe
    # to project while `execution` never is.
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


class GenerationInput(BaseModel):
    role: str
    asset_id: uuid.UUID
    kind: str


class GenerationError(BaseModel):
    code: str
    message: str
    """Customer-safe copy only. Never a stack trace or worker detail (§23)."""


class GenerationJobPublic(BaseModel):
    id: uuid.UUID
    workflow_id: str
    workflow_name: str

    status: JobStatus
    stage_label: str
    """Human label for `status`, resolved server-side so every client agrees."""
    progress: int
    hint: str

    prompt: str
    parameters: dict[str, Any]

    inputs: list[GenerationInput] = Field(default_factory=list)
    outputs: list[GenerationOutput] = Field(default_factory=list)

    error: GenerationError | None = None

    result: dict[str, Any] | None = None
    """Structured, customer-safe facts the worker reported with the output
    beyond the file itself — for Music, the lyrics (plain, LRC, SRT and
    timed JSON), the planned and measured vocal coverage, the rhyme result
    and any warnings. Bounded at 64 KB; never model names or paths."""

    attempt_count: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    last_event_seq: int
    """Where to resume an SSE stream from — pass as `Last-Event-ID`."""

    is_terminal: bool


class GenerationAccepted(BaseModel):
    """202 body — the job exists and is queued; nothing has been generated yet.

    Returned immediately so the HTTP request never waits on generation
    (directive §6, scalability rule #3).
    """

    job_id: uuid.UUID
    status: JobStatus
    stage_label: str
    events_url: str
    """Where to subscribe for progress, so the client does not construct URLs."""
