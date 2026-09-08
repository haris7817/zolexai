"""Where generated dialogue comes from, and what makes it acceptable.

Two providers behind one method, the same seam and the same order as
`worker/director/provider.py`: the hosted model first because it answers in
about two seconds against a much larger model on someone else's hardware, the
local Gemma checkpoint behind it because a hosted dependency can be down,
rate-limited or revoked and none of those should take a feature with it.

The local provider reuses `scripts/director_plan.py` unchanged. That script
takes a system prompt, a user prompt and a pair of markers and returns what
the model wrote between them; nothing in it is specific to a Director plan,
so nothing new had to be written or deployed for this.

## Failure posture: open

This is the one place this module differs sharply from Director mode. Director
fails the job when planning fails, because a customer who asked for a planned
dialogue scene and got a silent one was answered with a different request.
Automatic dialogue is the opposite: the customer asked for a video. If no line
can be written, the right outcome is the video they asked for, rendered from
the prompt they wrote, exactly as it would have been last week. Every failure
here returns nothing and is logged.

## Validation

A line the model wrote is not accepted on trust. It must come from a speaker
the model itself declared, contain no nested quotes (a quote inside a quoted
line breaks the sentence the prompt is built from), stay inside the measured
word budget, and not repeat itself — a repeated line is the exact artefact
this feature exists to prevent, and one arriving from the planner rather than
the renderer is no better.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol

import httpx

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.dialogue.decide import (
    Dialogue,
    Line,
    Speaker,
    line_ceiling,
    line_target,
    speaker_ceiling,
    word_budget,
)

logger = get_logger(__name__)

_BEGIN = "===DIALOGUE_BEGIN==="
_END = "===DIALOGUE_END==="

#: Statuses no retry can fix: no key, wrong key, wrong model, bad request.
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 422})

#: Grace over the measured word budget before a plan is refused outright. The
#: same 1.15 the Director plan uses, and for the same reason: a line or two
#: over is fine, double is not.
_BUDGET_SLACK = 1.15


class DialogueUnavailable(Exception):
    """This provider cannot be asked. Try the next one."""


class DialogueRejected(Exception):
    """This provider answered with something unusable."""


class DialogueProvider(Protocol):
    name: str

    async def write(self, request: DialogueRequest) -> dict[str, Any]: ...


class DialogueRequest:
    """One ask: this prompt, this long, in this language."""

    def __init__(
        self,
        *,
        prompt: str,
        seconds: float,
        language: str = "",
        system: str = "",
        user: str = "",
    ) -> None:
        self.prompt = prompt.strip()
        self.seconds = float(seconds)
        self.language = language.strip()
        # A caller may hand over the exact instruction to send — the client's
        # native-dialogue format (`worker/dialogue/native.py`) writes its own
        # — and the providers then carry it unchanged. Empty means this
        # module's own `system_prompt` / `user_prompt`.
        self.system = system
        self.user = user

    @property
    def system_text(self) -> str:
        return self.system or system_prompt(self)

    @property
    def user_text(self) -> str:
        return self.user or user_prompt(self)

    @property
    def words(self) -> int:
        return word_budget(self.seconds)

    @property
    def lines(self) -> int:
        return line_target(self.seconds)

    @property
    def max_lines(self) -> int:
        return line_ceiling(self.seconds)


# ── The ask ────────────────────────────────────────────────────────────────


def system_prompt(request: DialogueRequest) -> str:
    return (
        "You write the spoken dialogue for a short AI-generated video, from a "
        "description of what the video shows. Return one bare JSON object and "
        "nothing else, between "
        f"{_BEGIN} and {_END}.\n\n"
        "Keys: has_speaker (boolean), speakers (array), lines (array).\n"
        "Each speaker: id (a short lowercase word), description (the noun "
        'phrase the video would call them, beginning with "the": "the taxi '
        'driver"), voice (two or three words of audible manner: "low and '
        'weary").\n'
        "Each line: speaker (a declared id), text (the words said aloud), and "
        "optionally delivery (audible manner for this line only).\n\n"
        "Set has_speaker to false, with empty arrays, when the scene has no "
        "visible human or human-like character who could plausibly speak — a "
        "landscape, a product shot, an animal, a vehicle, an abstract or "
        "time-lapse scene. Never introduce a character the description does "
        "not already contain, and never change what the video shows.\n\n"
        f"Write about {request.lines} lines and at most {request.max_lines}, "
        f"totalling no more than {request.words} spoken words, from at most "
        f"{speaker_ceiling()} speakers. Spread them across the whole clip: a "
        "stretch with nothing to say is filled by the video model repeating a "
        "line or reading the description aloud, so too few lines is a worse "
        "failure than too many.\n\n"
        "Every line must be natural speech a person would actually say, "
        "specific to what is happening, and different from every other line. "
        "No line may repeat, paraphrase or echo another. Write only the words "
        "spoken: no speaker labels, no stage directions, no quotation marks, "
        "no captions, no lyrics, and nothing that would be read as narration "
        "about the scene."
    )


def user_prompt(request: DialogueRequest) -> str:
    language = (
        f"Write the dialogue in {request.language}."
        if request.language
        else "Write the dialogue in the language the description implies."
    )
    return (
        f"Video length: {request.seconds:g} seconds\n"
        f"{language}\n\n"
        f"What the video shows:\n{request.prompt}"
    )


# ── The hosted provider ────────────────────────────────────────────────────


class CerebrasDialogueProvider:
    """Writes dialogue with the hosted model the platform already uses."""

    name = "cerebras"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        enabled: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else settings.cerebras_api_key).strip()
        self._model = (model or settings.cerebras_director_model).strip()
        self._base_url = (base_url or settings.cerebras_base_url).rstrip("/")
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.auto_dialogue_timeout_seconds
        )
        self._enabled = (
            enabled if enabled is not None else settings.cerebras_director_enabled
        )
        self._transport = transport

    @property
    def available(self) -> bool:
        return bool(self._enabled and self._api_key and self._model)

    def unavailable_reason(self) -> str:
        if not self._enabled:
            return "CEREBRAS_DIRECTOR_ENABLED is false"
        if not self._api_key:
            return "CEREBRAS_API_KEY is not set"
        return "CEREBRAS_DIRECTOR_MODEL is empty"

    async def write(self, request: DialogueRequest) -> dict[str, Any]:
        if not self.available:
            raise DialogueUnavailable(self.unavailable_reason())

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_text},
                {"role": "user", "content": request.user_text},
            ],
            "max_completion_tokens": settings.auto_dialogue_max_tokens,
            "temperature": settings.auto_dialogue_temperature,
            "stream": False,
            # No `response_format`, for the reason measured in
            # worker/director/cerebras.py: on this model family the
            # constrained decoder runs away and truncates. The prompt asks for
            # a bare object and `_extract_json` tolerates prose around it.
        }
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                transport=self._transport,
            ) as client:
                response = await client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise DialogueRejected(
                f"the dialogue service timed out after {self._timeout:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise DialogueRejected(
                f"could not reach the dialogue service: {type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            detail = f"dialogue service returned {response.status_code}"
            if response.status_code in _PERMANENT_STATUS:
                raise DialogueUnavailable(detail)
            raise DialogueRejected(detail)

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise DialogueRejected("the dialogue service returned a body that is not JSON") from exc

        text = _first_message(body)
        if not text.strip():
            raise DialogueRejected("the dialogue service returned an empty message")
        if _finish_reason(body) == "length":
            raise DialogueRejected(
                f"the dialogue service ran past its output limit ({len(text)} characters)"
            )
        return _extract_json(text)


# ── The local provider ─────────────────────────────────────────────────────


class GemmaDialogueProvider:
    """Writes dialogue with the local checkpoint, through the LTX environment."""

    name = "gemma"

    async def write(self, request: DialogueRequest) -> dict[str, Any]:
        payload = json.dumps(
            {
                "gemma_root": str(settings.director_gemma_root),
                "system_prompt": request.system_text,
                "user_prompt": request.user_text,
                "sample": False,
                "seed": 0,
                "max_new_tokens": 700,
                "begin_marker": _BEGIN,
                "end_marker": _END,
            }
        ).encode()

        try:
            process = await asyncio.create_subprocess_exec(
                *settings.director_planner_argv,
                cwd=str(settings.ltx_repo_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise DialogueUnavailable(f"local planner could not start: {exc}") from exc

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(payload), timeout=settings.auto_dialogue_timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise DialogueRejected(
                f"local dialogue writer timed out after "
                f"{settings.auto_dialogue_timeout_seconds:.0f}s"
            ) from None

        text = (stdout or b"").decode("utf-8", "replace")
        if process.returncode != 0:
            tail = " | ".join(text.strip().splitlines()[-4:])
            raise DialogueRejected(f"local dialogue writer exited {process.returncode}: {tail}")
        return _extract_json(text)


def default_providers() -> list[DialogueProvider]:
    """Hosted first, local behind it — the order Director settled on."""
    providers: list[DialogueProvider] = []
    hosted = CerebrasDialogueProvider()
    if hosted.available:
        providers.append(hosted)
    else:
        logger.info(
            "auto_dialogue_hosted_unavailable", extra={"reason": hosted.unavailable_reason()}
        )
    if settings.auto_dialogue_local_fallback:
        providers.append(GemmaDialogueProvider())
    return providers


# ── Reading the answer ─────────────────────────────────────────────────────


def _first_message(body: Any) -> str:
    try:
        return str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _finish_reason(body: Any) -> str:
    try:
        return str(body["choices"][0].get("finish_reason") or "")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(re.escape(_BEGIN) + r"(.*?)" + re.escape(_END), text, re.DOTALL)
    body = match.group(1) if match else text
    brace = re.search(r"\{.*\}", body, re.DOTALL)
    if not brace:
        raise DialogueRejected("the dialogue writer returned no JSON at all")
    try:
        parsed = json.loads(brace.group(0))
    except json.JSONDecodeError as exc:
        raise DialogueRejected(f"the dialogue writer returned malformed JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise DialogueRejected("the dialogue writer returned something that is not an object")
    return parsed


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def parse(raw: dict[str, Any], request: DialogueRequest) -> Dialogue | None:
    """The model's answer as a `Dialogue`, or None when it declined.

    Raises `DialogueRejected` for an answer that claims a speaker and then
    fails a contract. The caller treats that as this provider having failed,
    not as a reason to render silently — the next provider gets a turn.
    """
    if not raw.get("has_speaker"):
        return None

    speakers: list[Speaker] = []
    for item in raw.get("speakers") or []:
        if not isinstance(item, dict):
            raise DialogueRejected("a speaker is not an object")
        speaker = Speaker(
            id=str(item.get("id", "")).strip().lower(),
            description=_clean(item.get("description", "")),
            voice=_clean(item.get("voice", "")),
        )
        if not speaker.id or not speaker.description:
            raise DialogueRejected("a speaker is missing its id or description")
        speakers.append(speaker)
    if not speakers:
        raise DialogueRejected("has_speaker is true but no speaker was described")
    if len(speakers) > speaker_ceiling():
        raise DialogueRejected(f"{len(speakers)} speakers, over the {speaker_ceiling()} ceiling")
    ids = {speaker.id for speaker in speakers}
    if len(ids) != len(speakers):
        raise DialogueRejected("two speakers share an id")

    lines: list[Line] = []
    seen: set[str] = set()
    for item in raw.get("lines") or []:
        if not isinstance(item, dict):
            raise DialogueRejected("a line is not an object")
        text = _clean(item.get("text", ""))
        speaker_id = str(item.get("speaker", "")).strip().lower()
        if not text:
            raise DialogueRejected("a line has no words")
        if speaker_id not in ids:
            raise DialogueRejected(f"a line is spoken by an undeclared speaker '{speaker_id}'")
        key = _normalise(text)
        if key in seen:
            # The artefact the whole feature exists to prevent. Arriving from
            # the writer rather than the renderer makes it no less a repeat.
            raise DialogueRejected("two lines say the same thing")
        seen.add(key)
        lines.append(Line(speaker=speaker_id, text=text, delivery=_clean(item.get("delivery", ""))))

    if not lines:
        raise DialogueRejected("has_speaker is true but no line was written")
    if len(lines) > request.max_lines:
        raise DialogueRejected(f"{len(lines)} lines, over the {request.max_lines} ceiling")
    words = sum(len(line.text.split()) for line in lines)
    if words > request.words * _BUDGET_SLACK:
        raise DialogueRejected(f"{words} spoken words, over the {request.words}-word budget")

    spoken_ids = {line.speaker for line in lines}
    return Dialogue(
        speakers=tuple(s for s in speakers if s.id in spoken_ids),
        lines=tuple(lines),
        language=request.language,
    )


def _clean(value: Any) -> str:
    """One line of plain text: no newlines, no wrapping or embedded quotes.

    Quotes are stripped rather than escaped because the composed prompt puts
    the line inside quotation marks, and a nested pair ends the quoted span
    early — the model then speaks half a line and reads the rest as prose.
    """
    text = " ".join(str(value or "").replace("\n", " ").split())
    return text.replace('"', "").replace("“", "").replace("”", "").strip()


__all__ = [
    "CerebrasDialogueProvider",
    "DialogueProvider",
    "DialogueRejected",
    "DialogueRequest",
    "DialogueUnavailable",
    "GemmaDialogueProvider",
    "default_providers",
    "parse",
    "system_prompt",
    "user_prompt",
]
