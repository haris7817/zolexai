"""The hosted dialogue writer must reserve room to think.

`max_completion_tokens` budgets *everything the model emits*, and
`gpt-oss-120b` spends an unpredictable share of it on a hidden reasoning
channel before writing a word of the answer. Music learned this on 19 Aug
2026 (empty lyric sheets, `finish_reason: stop`, no error) and the Director
learned it too; both reserve 2500.

**Auto Dialogue never got the same fix, and it took the feature down.**
Measured on the client-test node 10 Sep 2026: every Text to Video job that
explicitly asked for dialogue failed all three attempts, for hours, with
"the dialogue service ran past its output limit" at 412, 875, 1715 and 2158
characters. The budget was a flat 1200 for the answer AND the reasoning
together — below the reserve its two siblings had already measured.

The failure mode is worth naming, because it is not the obvious one: the
customer did not get a silent video, they got no video. `requested_explicitly`
fails closed (the client's own 8 Sep request), so a writer that cannot answer
stops the job.
"""

from __future__ import annotations

import pytest

from worker.core.config import settings
from worker.dialogue.provider import (
    _REASONING_HEADROOM,
    CerebrasDialogueProvider,
    DialogueRejected,
    DialogueRequest,
)


class _Endpoint:
    """Records the request bodies the provider sends."""

    def __init__(self, body: dict | None = None) -> None:
        self.requests: list[dict] = []
        self._body = body or {
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"turns": []}'},
                    "finish_reason": "stop",
                }
            ]
        }

    async def handler(self, request):  # httpx.MockTransport signature
        import json

        import httpx

        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self._body)


def _provider(endpoint: _Endpoint) -> CerebrasDialogueProvider:
    import httpx

    return CerebrasDialogueProvider(
        api_key="sk-test-not-a-real-key",
        model="gpt-oss-120b",
        base_url="https://api.cerebras.test",
        timeout_seconds=5.0,
        enabled=True,
        transport=httpx.MockTransport(endpoint.handler),
    )


def _request() -> DialogueRequest:
    return DialogueRequest(prompt="two people on a dock", seconds=15.0, language="English")


def test_the_reserve_matches_the_two_modules_that_measured_it() -> None:
    """Music and the Director both reserve 2500 on this model family. A
    smaller number here would be the same bug in a third place."""
    assert _REASONING_HEADROOM >= 2500


def test_the_configured_budget_is_room_for_the_ANSWER_not_the_thinking() -> None:
    """The setting keeps meaning what its name says: a deployment that wants
    longer dialogue raises it without needing to know a hidden channel exists.
    The reserve is added on top rather than carved out of it — which is what
    the old flat 1200 got wrong."""
    assert settings.auto_dialogue_max_tokens > 0
    total = settings.auto_dialogue_max_tokens + _REASONING_HEADROOM
    assert total > _REASONING_HEADROOM
    # Whatever the deployment sets, the answer never has to compete with the
    # thinking for the same tokens.
    assert total - _REASONING_HEADROOM == settings.auto_dialogue_max_tokens


async def test_the_sent_budget_carries_the_reserve() -> None:
    """The regression guard, asserted on the wire rather than on a constant.

    A flat 1200 went out for months and read as deliberate — the setting had a
    docstring saying it already included the reasoning headroom. Only the
    request body proves it does."""
    endpoint = _Endpoint()
    await _provider(endpoint).write(_request())
    sent = endpoint.requests[0]["max_completion_tokens"]
    assert sent >= 2500, "below the reserve is the bug that took dialogue down"
    assert sent == settings.auto_dialogue_max_tokens + _REASONING_HEADROOM


async def test_a_truncated_reply_is_still_named_as_truncation() -> None:
    """The reserve makes truncation unlikely, not impossible, and the message
    that named it is what let this be diagnosed from a log rather than
    reproduced. It stays."""
    endpoint = _Endpoint(
        body={
            "choices": [
                {"message": {"role": "assistant", "content": '{"turns": ['},
                 "finish_reason": "length"}
            ]
        }
    )
    with pytest.raises(DialogueRejected, match="output limit"):
        await _provider(endpoint).write(_request())
