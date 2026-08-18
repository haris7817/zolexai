"""Director planning on a hosted model — one `DirectorProvider`.

## Why this exists beside the local one

`GemmaDirectorProvider` loads a 10 GB checkpoint on the GPU node for every
job. It works, and it was the right first implementation: no external
dependency, no key, nothing to buy. But it costs **18-26 seconds of wall clock
before a single frame renders** (measured in production, 19 Aug 2026), and it
spends that time on the same card the render needs.

The platform already talks to Cerebras for lyrics, over an OpenAI-shaped API
with no SDK. The same call plans a scene in roughly two seconds against a much
larger model, on someone else's hardware. For a feature whose whole job is
writing words, that is the better tool.

## The seam is the point

Both providers satisfy one method and return the same raw plan object. Every
contract the product depends on — user dialogue preserved verbatim, speech
budget, speaker ownership, pacing — is enforced afterwards in
`worker/director/plan.py`, identically for both. So this file can be switched
off with an environment variable and the feature keeps working, slower.

That ordering is deliberate: a hosted model is a dependency that can be down,
rate-limited or revoked, and none of those should take a video feature with
it.

## Failure posture

Identical in shape to the lyrics writer, for the same reasons:

  * **Transient** — timeout, network, 429, 5xx, an unparseable or empty body.
    Worth another attempt.
  * **Permanent** — no key, disabled, 401/403, 404, 400/422. Retrying is
    guaranteed to fail, so the caller falls through to the local provider.

`_REASONING_HEADROOM` is carried across verbatim and for the measured reason
recorded there: `max_completion_tokens` budgets everything a model emits
INCLUDING a hidden reasoning channel, and a reasoning model that overspends it
returns an empty string with `finish_reason: stop` and no error at all. A plan
is a larger answer than a lyric sheet, so the allowance here is larger.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.director.plan import DirectorPlanError

logger = get_logger(__name__)

#: Statuses that mean "this exact request fails the same way forever".
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 405, 413, 422})

#: Reserve for a model that thinks before answering. See the module docstring
#: and `worker/music/cerebras.py`, where this was measured and cost nine of
#: fourteen languages before it existed.
_REASONING_HEADROOM = 900

#: A DirectorPlan is a bigger answer than a lyric sheet — characters with
#: appearance and voice, plus one object per timeline event — so the floor sits
#: well above the lyrics writer's. Over-allocating is free; running out
#: truncates the JSON mid-object and costs the whole attempt.
_MIN_TOKENS = 2500
_MAX_TOKENS = 6000
_TOKENS_PER_EVENT = 120


class DirectorProviderUnavailable(Exception):
    """This provider cannot be attempted at all — no key, or switched off."""


class CerebrasDirectorProvider:
    """Plans a scene with a hosted language model."""

    name = "cerebras"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        enabled: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else settings.cerebras_api_key).strip()
        self._model = (model or settings.cerebras_director_model).strip()
        self._base_url = (base_url or settings.cerebras_base_url).rstrip("/")
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.cerebras_director_timeout_seconds
        )
        self._temperature = float(
            temperature if temperature is not None else settings.cerebras_director_temperature
        )
        self._enabled = enabled if enabled is not None else settings.cerebras_director_enabled
        self._transport = transport

    # ── Availability ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self._enabled and self._api_key and self._model)

    def unavailable_reason(self) -> str:
        if not self._enabled:
            return "CEREBRAS_DIRECTOR_ENABLED is false"
        if not self._api_key:
            return "CEREBRAS_API_KEY is not set"
        if not self._model:
            return "CEREBRAS_DIRECTOR_MODEL is empty"
        return ""

    # ── The plan ─────────────────────────────────────────────────────────

    async def generate_plan(self, request: Any) -> dict[str, Any]:
        """One planning attempt, as `DirectorProvider` requires.

        Raises `DirectorProviderUnavailable` when it cannot be attempted (the
        caller falls through to the local provider) and `DirectorPlanError`
        when the service answered with something unusable (the caller counts
        it as a failed attempt, exactly like a bad plan from the local model).
        """
        # Imported here rather than at module scope: provider.py imports this
        # module for the chain, so a top-level import would be a cycle.
        from worker.director.provider import system_prompt, user_prompt

        if not self.available:
            raise DirectorProviderUnavailable(self.unavailable_reason())

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(request)},
            ],
            "max_completion_tokens": self._max_completion_tokens(request),
            "temperature": self._temperature,
            "stream": False,
            # NO `response_format: json_object`, and that is a measured
            # decision rather than an omission.
            #
            # It looks like exactly the right belt to add — ask for JSON at the
            # protocol level as well as in the prompt — and on `gemma-4-31b` it
            # is what breaks the call. Measured on the box, 19 Aug 2026, same
            # prompt and seed, 3 runs each: WITH the constraint 1 of 3 returned
            # usable JSON, the failures running away to 8-49 KB of output and
            # truncating at `finish_reason: length`; WITHOUT it, 3 of 3 came
            # back clean at ~2.3 KB. (`gpt-oss-120b` is unaffected either way,
            # which is why this reads as a constrained-decoding interaction
            # with that model rather than a service fault.)
            #
            # The prompt already asks for a bare JSON object and `_extract_json`
            # tolerates prose or fences around it, so nothing is lost.
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
            raise DirectorPlanError(
                [f"the planning service timed out after {self._timeout:.0f}s"]
            ) from exc
        except httpx.HTTPError as exc:
            raise DirectorPlanError(
                [f"could not reach the planning service: {type(exc).__name__}"]
            ) from exc

        if response.status_code >= 400:
            detail = f"planning service returned {response.status_code}: {response.text[:200]}"
            if response.status_code in _PERMANENT_STATUS:
                # Permanent for THIS provider — the chain should stop asking it
                # and use the local one rather than burning the retry here.
                raise DirectorProviderUnavailable(detail)
            raise DirectorPlanError([detail])

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise DirectorPlanError(
                ["the planning service returned a body that is not JSON"]
            ) from exc

        text = _first_message(body)
        if not text.strip():
            raise DirectorPlanError(["the planning service returned an empty message"])

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        finish = _finish_reason(body)
        logger.info(
            "cerebras_director_attempt",
            extra={
                "model": self._model,
                "finish_reason": finish,
                "characters": len(text),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        )
        if finish == "length":
            # Named rather than left to the JSON parser, which would report a
            # truncated plan as "expecting ',' delimiter" and send whoever
            # reads the log hunting for a syntax bug that is not there. A run
            # that ends this way produced too much, not malformed output.
            raise DirectorPlanError(
                [f"the planning service ran past its output limit ({len(text)} characters)"]
            )
        return _extract_json(text)

    def _max_completion_tokens(self, request: Any) -> int:
        """Room for the plan, plus room to think.

        Scaled by duration because a longer video is a longer timeline — more
        events, each its own JSON object — while the reasoning reserve is flat,
        since a model deliberates about as much either way.
        """
        events = max(4, int(getattr(request, "duration_seconds", 20.0) // 3))
        wanted = events * _TOKENS_PER_EVENT + _REASONING_HEADROOM
        return max(_MIN_TOKENS, min(_MAX_TOKENS, wanted))


def _finish_reason(body: Any) -> str:
    """Why the model stopped, or "" when the reply does not say."""
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else ""


def _first_message(body: Any) -> str:
    """The assistant's text, or "" if the response is not the expected shape."""
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


def _extract_json(text: str) -> dict[str, Any]:
    """The plan object, from a reply that may carry fences or commentary."""
    import re

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace:
        raise DirectorPlanError(["the planning service returned no JSON at all"])
    try:
        parsed = json.loads(brace.group(0))
    except json.JSONDecodeError as error:
        raise DirectorPlanError([f"the planning service returned invalid JSON: {error}"]) from None
    if not isinstance(parsed, dict):
        raise DirectorPlanError(["the planning service returned JSON that is not an object"])
    return parsed
