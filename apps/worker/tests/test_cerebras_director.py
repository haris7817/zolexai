"""The hosted scene planner — `worker/director/cerebras.py`.

The seam this suite protects: a planning service that is slow, down, rate
limited, misconfigured or answering nonsense must degrade Director mode to the
local checkpoint, never take the feature out. Everything the product actually
depends on — user dialogue preserved, speaker ownership, pacing — is enforced
after this file returns, so what is tested here is narrower on purpose: does it
ask the right question, and does it sort every failure into the right bucket.
"""

from __future__ import annotations

import json

import httpx
import pytest

from worker.director.cerebras import (
    CerebrasDirectorProvider,
    DirectorProviderUnavailable,
)
from worker.director.plan import DirectorPlanError
from worker.director.provider import DirectorRequest

PLAN = {
    "scene": "A dim office at night.",
    "tone": "tense",
    "ambience": "a low lamp hum",
    "characters": [
        {
            "id": "detective",
            "role": "detective",
            "appearance": "a weathered man in a gray suit",
            "voice": "low",
        }
    ],
    "timeline": [
        {
            "start": 0,
            "end": 4,
            "action": "He leans on the desk",
            "camera": "medium shot, static",
            "speaker": "detective",
            "dialogue": "You knew.",
            "delivery": "quiet",
        }
    ],
}


def request(**overrides) -> DirectorRequest:
    defaults = dict(
        idea="A detective confronts his boss.",
        duration_seconds=20.0,
        language="english",
        seed=1,
        sample=False,
    )
    return DirectorRequest(**{**defaults, **overrides})


def endpoint(
    *,
    status: int = 200,
    content: str | None = None,
    body: object | None = None,
    fail: type[Exception] | None = None,
    seen: dict | None = None,
) -> httpx.MockTransport:
    recorded = seen if seen is not None else {}

    def handle(http_request: httpx.Request) -> httpx.Response:
        recorded.setdefault("requests", []).append(json.loads(http_request.content))
        recorded.setdefault("headers", []).append(dict(http_request.headers))
        if fail is not None:
            raise fail("injected")
        if status != 200:
            return httpx.Response(status, text="service said no")
        if body is not None:
            return httpx.Response(200, json=body)
        text = content if content is not None else json.dumps(PLAN)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 260},
            },
        )

    return httpx.MockTransport(handle)


def provider(transport: httpx.MockTransport, **overrides) -> CerebrasDirectorProvider:
    defaults = dict(
        api_key="sk-test-not-a-real-key",
        model="gemma-4-31b",
        base_url="https://api.cerebras.test",
        timeout_seconds=5.0,
        enabled=True,
    )
    return CerebrasDirectorProvider(transport=transport, **{**defaults, **overrides})


# ── The happy path, and what it asks for ─────────────────────────────────


async def test_a_plan_comes_back_parsed() -> None:
    raw = await provider(endpoint()).generate_plan(request())
    assert raw["scene"] == "A dim office at night."
    assert raw["timeline"][0]["dialogue"] == "You knew."


async def test_the_request_carries_the_shared_brief_and_the_job_numbers() -> None:
    """Both providers must send the SAME instructions. If they drifted, a
    fallback would quietly produce a differently shaped plan than the primary
    and it would only ever show up in a customer's video."""
    seen: dict = {}
    await provider(endpoint(seen=seen)).generate_plan(request(duration_seconds=30.0))

    sent = seen["requests"][0]
    system = sent["messages"][0]["content"]
    user = sent["messages"][1]["content"]

    assert sent["model"] == "gemma-4-31b"
    assert "video director" in system
    assert "TOTAL_LINES: write 8" in user  # the target for 30s, computed here
    assert "DURATION: 30 seconds" in user


async def test_json_mode_is_never_requested() -> None:
    """Constrained decoding is what BREAKS this model. Measured 19 Aug: with
    `response_format: json_object`, 1 of 3 runs returned usable JSON and the
    rest ran away to tens of kilobytes before truncating; without it, 3 of 3
    were clean. This assertion is the guard against someone re-adding the
    obvious-looking safety measure."""
    seen: dict = {}
    await provider(endpoint(seen=seen)).generate_plan(request())
    assert "response_format" not in seen["requests"][0]


async def test_a_truncated_reply_is_named_as_truncation() -> None:
    """A run that ends at the output limit produced too much, not malformed
    output — reporting it as a JSON syntax error sends the next reader hunting
    for a bug that is not there."""
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": '{"scene": "a room'},
             "finish_reason": "length"}
        ]
    }
    with pytest.raises(DirectorPlanError, match="output limit"):
        await provider(endpoint(body=body)).generate_plan(request())


async def test_the_output_allowance_reserves_room_to_think() -> None:
    """`max_completion_tokens` budgets the hidden reasoning channel too — the
    trap that returned empty lyric sheets with no error at all."""
    seen: dict = {}
    await provider(endpoint(seen=seen)).generate_plan(request(duration_seconds=60.0))
    assert seen["requests"][0]["max_completion_tokens"] >= 2500


async def test_the_key_travels_in_a_header_and_never_in_the_body() -> None:
    seen: dict = {}
    await provider(endpoint(seen=seen)).generate_plan(request())
    assert seen["headers"][0]["authorization"] == "Bearer sk-test-not-a-real-key"
    assert "sk-test" not in json.dumps(seen["requests"][0])


async def test_a_reply_wrapped_in_prose_or_fences_still_yields_the_plan() -> None:
    wrapped = f"Here is the plan:\n```json\n{json.dumps(PLAN)}\n```\nHope it helps."
    raw = await provider(endpoint(content=wrapped)).generate_plan(request())
    assert raw["scene"] == "A dim office at night."


# ── Failure sorting ──────────────────────────────────────────────────────


@pytest.mark.parametrize("reason", [(True, "", "enabled"), (False, "", "key")])
async def test_an_unconfigured_provider_reports_itself_unusable(reason) -> None:
    """Never an error the customer sees — the chain reads this and uses the
    local planner instead."""
    enabled, key, _ = reason
    entry = provider(endpoint(), enabled=enabled, api_key=key)
    assert entry.available is False
    with pytest.raises(DirectorProviderUnavailable):
        await entry.generate_plan(request())


@pytest.mark.parametrize("status", [401, 403, 404, 400, 422])
async def test_a_permanent_status_abandons_this_provider(status: int) -> None:
    """A bad key or an unknown model fails identically forever, so it must not
    consume the retry that a transient blip deserves."""
    with pytest.raises(DirectorProviderUnavailable):
        await provider(endpoint(status=status)).generate_plan(request())


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_transient_status_is_a_failed_attempt_not_a_dead_provider(
    status: int,
) -> None:
    with pytest.raises(DirectorPlanError):
        await provider(endpoint(status=status)).generate_plan(request())


async def test_a_timeout_is_a_failed_attempt() -> None:
    with pytest.raises(DirectorPlanError, match="timed out"):
        await provider(endpoint(fail=httpx.ReadTimeout)).generate_plan(request())


async def test_an_empty_message_is_refused_rather_than_returned() -> None:
    """The measured reasoning-token failure: `finish_reason: stop`, no error,
    and nothing in the content. It must not reach the parser as a valid plan."""
    body = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
    with pytest.raises(DirectorPlanError, match="empty"):
        await provider(endpoint(body=body)).generate_plan(request())


async def test_a_reply_with_no_json_is_refused() -> None:
    with pytest.raises(DirectorPlanError, match="no JSON"):
        await provider(endpoint(content="I'd rather not.")).generate_plan(request())


async def test_an_unexpected_body_shape_is_refused_cleanly() -> None:
    """A shape change at the service must arrive as a handled failure, not a
    KeyError surfacing in the job runner."""
    with pytest.raises(DirectorPlanError):
        await provider(endpoint(body={"unexpected": True})).generate_plan(request())
