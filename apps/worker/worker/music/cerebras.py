"""The Cerebras lyrics writer — a hosted language model behind the writer seam.

## What this is

One implementation of `LyricsWriter` (see `worker/music/lyrics.py`). It receives
a `LyricBrief` and a `SongPlan` — the platform's own vocabulary — and returns a
lyric sheet. It is the *primary* writer: `TemplateLyricsWriter` remains as the
floor beneath it, and `FallbackLyricsWriter` is what puts them in that order.

It knows nothing about music generation, and the music model knows nothing
about it. Cerebras writes words; ACE-Step sings them. That split is the reason
this file can be deleted and replaced without touching the adapter.

## Why Cerebras, and why this model

The API is OpenAI-shaped (`POST /v1/chat/completions`, bearer auth), so no SDK
is needed — the worker already depends on `httpx` and nothing else is added.

The default model is `gemma-4-31b`, chosen from the two models Cerebras serves
on its public endpoint (checked 2026-08-19; the other is `gpt-oss-120b`).

BOTH were then run against all fourteen offered languages on the live API
(`scripts/lyrics_smoke.py`, 2026-08-19), because the documentation does not
answer the only question that matters here — will it write Urdu:

  * `gemma-4-31b`  — 14/14 languages, ~1.5s median.
  * `gpt-oss-120b` — 14/14 languages, ~1.7s median, but only after
    `_REASONING_HEADROOM` below existed. Without it: 5/14.

Gemma is the default because it reaches the same result without needing the
reserve, is the plainer instruction-follower for a "return only this" contract,
and is the family this codebase already runs for Director planning. `gpt-oss`
is a supported choice, not a broken one — the headroom made it work, and that
fix belongs in the writer regardless of which model is configured.

It is configurable (`CEREBRAS_LYRICS_MODEL`, or `CEREBRAS_AI_MODEL`) precisely
because that lineup changes, and the default above is a measured starting point
rather than a permanent fact.

## The failure posture

Every way this can fail sorts into exactly two buckets, and the bucket decides
the behaviour:

  * **Transient** — timeout, network error, 429, 5xx, a malformed or empty
    body, a sheet in the wrong language. Worth one more try.
  * **Permanent** — no API key, disabled by configuration, 401/403 (bad or
    revoked key), 404 (unknown model), 400/422 (bad request), 402 (billing).
    Retrying is guaranteed to fail again, so it goes straight to the fallback.

Both end at the same place — `LyricsWriteFailed`, which the chain treats as
"try the next writer". The distinction only decides how much time is spent
before getting there.

## Language

The check that the returned sheet is *actually* in the requested language is
not optional and is not advisory. A model that answers a Spanish request in
English produces a sheet indistinguishable from a good one at every layer
below this file, and the resulting song is English words sung with Spanish
phonetics. `worker/music/detect.py` measures it; a mismatch spends the
remaining attempt on a reinforced instruction and then fails rather than
returning the sheet.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.music.detect import written_in
from worker.music.language import resolve_language
from worker.music.lyrics import (
    LyricBrief,
    LyricsWriteFailed,
    SongPlan,
    target_lines,
)
from worker.music.writer import detect_mood, extract_subject, singable_details

logger = get_logger(__name__)

#: HTTP statuses that mean "this exact request will fail the same way forever".
#: Everything else that is not a success is treated as worth one more attempt.
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 405, 413, 422})

#: Statuses that are explicitly worth retrying. 429 is rate limiting; 5xx is
#: the service. Listed for readability — the default for an unknown status is
#: already "transient".
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Roughly how many tokens one sung line costs, across scripts. Generous:
#: running out of output tokens mid-chorus produces a truncated sheet, which
#: costs a whole retry, while over-allocating costs nothing at all — a model
#: stops when the song is done, and unused allowance is not billed.
_TOKENS_PER_LINE = 60

#: Tokens reserved for a model that thinks before it answers.
#:
#: MEASURED, and load-bearing. `max_completion_tokens` is a budget for
#: *everything the model emits*, and a reasoning model spends an unpredictable
#: share of it on a hidden channel before writing a word of the answer. On
#: `gpt-oss-120b` this was 386-696 reasoning tokens for one lyric request
#: (GPU box, 2026-08-19), against a budget of 540 — so the sheet came back
#: EMPTY, with `finish_reason: stop` and no error of any kind.
#:
#: The symptom is the worst kind: not a failure, but an empty string that looks
#: like a model declining to answer. It cost nine of fourteen languages before
#: this reserve existed; with it, the same model passes all fourteen. Sized
#: above the highest reasoning cost observed, because the failure is silent and
#: the over-allocation is free.
_REASONING_HEADROOM = 900

#: Floor and ceiling on the output allowance, so a 1-minute song still has room
#: to finish a thought and a 5-minute one cannot run away.
_MIN_TOKENS = 1400
_MAX_TOKENS = 3000


# ── The brief, as the model reads it ─────────────────────────────────────
#
# Written against the shape the rest of the pipeline already enforces:
# `parse_sections` reads `[tag]` lines, `review_lyrics` blocks a sheet longer
# than the plan's line budget, and `polish_lyrics` collapses repeated choruses.
# So the prompt asks for exactly what those three will accept — anything else
# is a round trip spent producing something that will be measured and rejected.

_SYSTEM_PROMPT = """You are a professional songwriter. You write lyrics that a \
singer will actually sing.

LANGUAGE — this is the most important rule:
Write every single sung line in {language_name}. Not English. Do not translate \
anything into English. Do not add an English translation, transliteration, \
gloss, or explanation of any kind. If you cannot write in {language_name}, \
write nothing at all.

STRUCTURE:
Use only the section tags you are given, each exactly once, in the order given. \
Put each tag alone on its own line in square brackets, like [verse]. Write the \
tags in English exactly as given even though the lyrics are in \
{language_name} — they are markup for the music software, not sung words.

CRAFT:
Write natural, singable lines that a person would say out loud, not poetry on \
a page. Give the song one memorable chorus built on a single repeated idea. \
Keep the verses concrete and specific to the theme. Do not pad with filler, do \
not repeat a line just to fill space, and do not describe the music, the \
production, the instruments or the singer — those are directions about the \
recording, not words anyone sings.

OUTPUT:
Return the lyric sheet and nothing else. No title, no commentary, no notes, no \
markdown code fences, no section numbering beyond the tags you were given."""

#: Appended when a first attempt came back in the wrong language. Deliberately
#: blunt and repetitive: the failure being corrected is a model that has
#: already read one polite instruction and ignored it.
_LANGUAGE_REINFORCEMENT = """
CRITICAL CORRECTION — YOUR PREVIOUS ATTEMPT WAS REJECTED.
You wrote the lyrics in the wrong language. Every sung line must be in \
{language_name} and only {language_name}. Write in {language_name} now. Do not \
write in English."""


def _user_prompt(brief: LyricBrief, plan: SongPlan, notes: list[str] | None) -> str:
    """The per-song half of the request.

    Numbers are computed here rather than described, for the same reason the
    Director planner computes its own: an instruct model reliably obeys a
    number it is given and unreliably derives one it is asked to work out.
    """
    language = resolve_language(brief.language)
    language_name = language.name if language else brief.language

    # Each word-carrying kind once, in plan order. Not the full outline: the
    # music model repeats a chorus by itself (observed at 240s on the GPU), so
    # writing it twice would spend the line budget twice for the same words.
    tags = list(dict.fromkeys(s.kind for s in plan.sections if s.carries_words))
    target = target_lines(plan)

    lines = [
        f"LANGUAGE: {language_name}",
        # Both halves of the prompt, labelled. The full text carries the genre,
        # mood and vocal direction the model should write *towards*; the
        # subject is what it should write *about*. Handing over only the raw
        # prompt is how a chorus ends up opening "Summer nights, catchy and
        # energetic" — the writer singing the brief back at the customer.
        f"STYLE AND PRODUCTION DIRECTION (shapes the writing; never sing these "
        f"words): {brief.topic}",
        f"WRITE ABOUT: {extract_subject(brief.topic) or brief.topic}",
        f"GENRE: {plan.genre}",
        f"MOOD: {brief.mood or detect_mood(brief.topic)}",
        f"SONG LENGTH: {plan.total_seconds:.0f} seconds",
        f"SECTION TAGS, in this order: {' '.join(f'[{tag}]' for tag in tags)}",
        # A TARGET and a ceiling, not a ceiling alone. Given only "at most 9"
        # and "about 2 per section", a model multiplies the two smallest
        # numbers it was given and stops — three sections, six lines, and a
        # two-minute song that is mostly instrumental. Observed in production
        # on the first real Spanish track (job 08f7c37d, 2026-08-19): budget 9,
        # sheet 6 lines, which is one line per 20s and sits in the band the GPU
        # measured as "82-second instrumental intro plus wordless padding".
        #
        # The ceiling still matters in the other direction — `review_lyrics`
        # blocks a sheet that exceeds it, because the music model drops the
        # excess silently rather than compressing it.
        f"TOTAL SUNG LINES: write {target} lines across the whole song. "
        f"Not fewer than {max(2, target - 1)}, and never more than "
        f"{plan.line_budget}.",
        f"LINES PER SECTION: at least 2. Spread the {target} lines across the "
        f"{len(tags)} sections above, giving the chorus the most.",
    ]
    if brief.perspective:
        lines.append(f"PERSPECTIVE: {brief.perspective}")
    keep = singable_details(brief.must_keep)
    if keep:
        lines.append(
            "THESE EXACT DETAILS FROM THE REQUEST MUST APPEAR IN THE LYRICS: "
            + ", ".join(keep)
        )
    if notes:
        lines.append(
            "\nYOUR PREVIOUS DRAFT HAD THESE PROBLEMS. Fix every one of them "
            "and keep the language the same:\n"
            + "\n".join(f"- {note}" for note in notes)
        )
    return "\n".join(lines)


class CerebrasLyricsWriter:
    """Writes lyrics with a hosted language model. One `LyricsWriter`."""

    name = "cerebras"

    supported_languages: frozenset[str] = frozenset()
    """
    Empty means "any", which is the protocol's documented way for a language
    model to say it has no list to give.

    That is the honest answer here and it is why the *check* moved downstream:
    this writer will attempt any language, and whether it succeeded is settled
    by reading what came back rather than by consulting a table beforehand.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
        enabled: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else settings.cerebras_api_key
        ).strip()
        self._model = (model or settings.cerebras_lyrics_model).strip()
        self._base_url = (base_url or settings.cerebras_base_url).rstrip("/")
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.cerebras_lyrics_timeout_seconds
        )
        self._max_retries = int(
            max_retries if max_retries is not None else settings.cerebras_lyrics_max_retries
        )
        self._temperature = float(
            temperature if temperature is not None else settings.cerebras_lyrics_temperature
        )
        self._enabled = (
            enabled if enabled is not None else settings.cerebras_lyrics_enabled
        )
        self._transport = transport
        """Injected HTTP transport, for tests only — the real one is built per
        call so no connection is held open between jobs."""

    # ── Availability ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether this writer can be attempted at all.

        A missing key is not an error and not a warning every job repeats — it
        is a deployment that has not turned this on. The chain reads this and
        skips straight to the next writer, which is what "skip Cerebras cleanly
        and attempt fallback" means in practice.
        """
        return bool(self._enabled and self._api_key and self._model)

    def unavailable_reason(self) -> str:
        if not self._enabled:
            return "CEREBRAS_LYRICS_ENABLED is false"
        if not self._api_key:
            return "CEREBRAS_API_KEY is not set"
        if not self._model:
            return "CEREBRAS_LYRICS_MODEL is empty"
        return ""

    # ── The write ────────────────────────────────────────────────────────

    async def write(
        self, brief: LyricBrief, plan: SongPlan, notes: list[str] | None = None
    ) -> str:
        if not self.available:
            raise LyricsWriteFailed(
                f"cerebras writer unavailable: {self.unavailable_reason()}",
                retriable=False,
            )

        language = brief.language.strip().lower() or "en"
        attempts = max(1, 1 + self._max_retries)
        reinforce = False
        problems: list[str] = []

        for attempt in range(1, attempts + 1):
            started = asyncio.get_running_loop().time()
            try:
                text, usage = await self._complete(brief, plan, notes, reinforce)
            except LyricsWriteFailed as failure:
                problems.append(f"attempt {attempt}: {failure}")
                if not failure.retriable or attempt == attempts:
                    raise LyricsWriteFailed(
                        " || ".join(problems), retriable=failure.retriable
                    ) from failure
                # Short, fixed backoff. This sits inside a job a user is
                # watching, so the point is to ride out a moment of rate
                # limiting, not to wait out an outage.
                await asyncio.sleep(min(2.0, 0.5 * attempt))
                continue

            elapsed = asyncio.get_running_loop().time() - started
            verdict = written_in(text, language)
            logger.info(
                "cerebras_lyrics_attempt",
                extra={
                    "attempt": attempt,
                    "model": self._model,
                    "language": language,
                    # `None` is "not enough text to tell", which is a different
                    # state from "checked and correct" and worth seeing.
                    "language_ok": verdict,
                    "latency_ms": round(elapsed * 1000),
                    "characters": len(text),
                    # Usage as the API reports it, so cost per song is
                    # measurable later without building billing now.
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            )

            if verdict is False:
                problems.append(
                    f"attempt {attempt}: returned text that is not {language!r}"
                )
                if attempt == attempts:
                    # Deliberately a failure rather than a return. Handing this
                    # sheet back would put English words into a Spanish request
                    # with nothing downstream able to notice.
                    raise LyricsWriteFailed(" || ".join(problems), retriable=True)
                reinforce = True
                continue

            return text

        raise LyricsWriteFailed(" || ".join(problems) or "no attempt succeeded")

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _messages(
        self,
        brief: LyricBrief,
        plan: SongPlan,
        notes: list[str] | None,
        reinforce: bool,
    ) -> list[dict[str, str]]:
        language = resolve_language(brief.language)
        language_name = language.name if language else brief.language
        system = _SYSTEM_PROMPT.format(language_name=language_name)
        if reinforce:
            system += _LANGUAGE_REINFORCEMENT.format(language_name=language_name)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _user_prompt(brief, plan, notes)},
        ]

    def _max_completion_tokens(self, plan: SongPlan) -> int:
        """The output allowance: room for the song, plus room to think.

        The headroom is added rather than assumed to fit inside the per-line
        figure, because it is not proportional to the song — a reasoning model
        deliberates about as much for a one-minute song as a five-minute one.
        """
        wanted = plan.line_budget * _TOKENS_PER_LINE + _REASONING_HEADROOM
        return max(_MIN_TOKENS, min(_MAX_TOKENS, wanted))

    async def _complete(
        self,
        brief: LyricBrief,
        plan: SongPlan,
        notes: list[str] | None,
        reinforce: bool,
    ) -> tuple[str, dict[str, Any]]:
        """One request. Raises `LyricsWriteFailed` with the bucket set."""
        payload = {
            "model": self._model,
            "messages": self._messages(brief, plan, notes, reinforce),
            "max_completion_tokens": self._max_completion_tokens(plan),
            # Lyrics are a creative task and a deterministic writer produces
            # the same song for every customer with a similar prompt. High
            # enough to vary, low enough to keep following the structure rules.
            "temperature": self._temperature,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                # The one place the key is used. It is set as a header on a
                # per-call client and never stored on the request, logged, or
                # written to the workspace.
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                transport=self._transport,
            ) as client:
                response = await client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LyricsWriteFailed(
                f"timed out after {self._timeout:.0f}s", retriable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise LyricsWriteFailed(
                f"could not reach the lyrics service: {type(exc).__name__}",
                retriable=True,
            ) from exc

        if response.status_code >= 400:
            retriable = response.status_code not in _PERMANENT_STATUS
            raise LyricsWriteFailed(
                # Bounded, and the body is the service's own error text — it
                # never contains the key, which travelled in a header.
                f"lyrics service returned {response.status_code}: "
                f"{response.text[:200]}",
                retriable=retriable,
            )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LyricsWriteFailed(
                "lyrics service returned a body that is not JSON", retriable=True
            ) from exc

        text = _first_message(body)
        if not text.strip():
            raise LyricsWriteFailed(
                "lyrics service returned an empty message", retriable=True
            )

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return _strip_fences(text), usage


def _first_message(body: Any) -> str:
    """The assistant's text, or "" if the response is not the shape we expect.

    Tolerant by design: an unexpected shape becomes an empty string and then a
    clean retriable failure, rather than a `KeyError` that reaches the job
    runner as an unhandled exception.
    """
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _strip_fences(text: str) -> str:
    """Removes a markdown code fence the model was told not to add.

    It is told not to, and mostly does not. When it does, the fence lines would
    survive into the sheet and reach the music model as lyrics — so this is
    repaired rather than treated as a reason to retry, because the words inside
    are fine.
    """
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines).strip()
