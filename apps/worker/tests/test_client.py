"""The worker's only channel to the platform, and what it must never break.

`test_runner.py` fakes this client to observe what the runner decides. That
leaves the client's own contract with the API untested — which is where a
failure report grew past a length limit, was rejected with 422, and turned a
clean failure into a silent retry at full GPU cost.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from worker.core.client import (
    MAX_INTERNAL_DETAIL,
    MAX_USER_MESSAGE,
    WorkerApiClient,
    fit_detail,
)

WORKER_ID = "11111111-1111-1111-1111-111111111111"
LEASE = "33333333-3333-3333-3333-333333333333"


def capturing(captured: dict[str, Any], status_code: int = 200) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = __import__("json").loads(request.content)
        return httpx.Response(status_code, json={"accepted": True})

    return httpx.MockTransport(handle)


def client(transport: httpx.MockTransport) -> WorkerApiClient:
    return WorkerApiClient(
        httpx.AsyncClient(base_url="http://api.test", transport=transport)
    )


# ── fit_detail ───────────────────────────────────────────────────────────


def test_a_short_detail_is_untouched() -> None:
    assert fit_detail("LTX pipeline exited 1") == "LTX pipeline exited 1"


def test_an_oversized_detail_keeps_both_ends() -> None:
    """The head names the stage and exit code; the tail carries the actual
    exception. A truncation that keeps only one of them loses the diagnosis,
    which is the entire reason the field is sent."""
    detail = (
        "LTX pipeline exited 1; output tail: "
        + ("filler line that means nothing | " * 400)
        + "RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR"
    )
    assert len(detail) > MAX_INTERNAL_DETAIL

    fitted = fit_detail(detail)
    assert len(fitted) <= MAX_INTERNAL_DETAIL
    assert "LTX pipeline exited 1" in fitted
    assert "CUBLAS_STATUS_INTERNAL_ERROR" in fitted
    assert "trimmed" in fitted


def test_the_limit_is_respected_at_the_boundary() -> None:
    for length in (MAX_INTERNAL_DETAIL - 1, MAX_INTERNAL_DETAIL, MAX_INTERNAL_DETAIL + 1):
        assert len(fit_detail("x" * length)) <= MAX_INTERNAL_DETAIL


# ── The failure report itself ────────────────────────────────────────────


async def test_a_cuda_traceback_still_produces_a_valid_report() -> None:
    """The bug, end to end. A real CUDA traceback runs to several thousand
    characters; the API caps internal_detail at 2000 and rejects the whole
    request with 422. The job is then never marked failed — it waits out its
    lease and is retried, spending the same minutes of GPU to reach the same
    deterministic crash. Observed twice on 2026-08-16.
    """
    captured: dict[str, Any] = {}
    huge = "Traceback (most recent call last):\n" + ("  File x, line 1\n" * 500)

    await client(capturing(captured)).report_failure(
        "job-1",
        worker_id=WORKER_ID,
        lease_token=LEASE,
        user_message="This generation could not be completed. Please try again.",
        internal_detail=huge,
        retriable=False,
    )

    body = captured["json"]
    assert len(body["internal_detail"]) <= MAX_INTERNAL_DETAIL
    assert len(body["user_message"]) <= MAX_USER_MESSAGE
    assert body["retriable"] is False


async def test_an_overlong_user_message_is_cut_rather_than_rejected() -> None:
    captured: dict[str, Any] = {}

    await client(capturing(captured)).report_failure(
        "job-1",
        worker_id=WORKER_ID,
        lease_token=LEASE,
        user_message="way too chatty " * 100,
        internal_detail="short",
        retriable=True,
    )

    assert len(captured["json"]["user_message"]) <= MAX_USER_MESSAGE


async def test_a_rejected_report_is_surfaced_as_data_not_an_exception() -> None:
    """A 4xx is the API saying the worker is wrong — a lost lease, an unknown
    worker. The runner has to be able to drop the job cleanly rather than
    treat it as an outage."""
    captured: dict[str, Any] = {}
    result = await client(capturing(captured, status_code=422)).report_failure(
        "job-1",
        worker_id=WORKER_ID,
        lease_token=LEASE,
        user_message="x",
        internal_detail="y",
        retriable=True,
    )
    assert result["_rejected"] is True
    assert result["status_code"] == 422


async def test_a_server_error_is_an_outage() -> None:
    from worker.core.client import ApiUnavailable

    captured: dict[str, Any] = {}
    with pytest.raises(ApiUnavailable):
        await client(capturing(captured, status_code=503)).report_failure(
            "job-1",
            worker_id=WORKER_ID,
            lease_token=LEASE,
            user_message="x",
            internal_detail="y",
            retriable=True,
        )


# ── Transport failures are retried; API failures are not ─────────────────


def flaky(fail_times: int, error: Exception) -> tuple[httpx.MockTransport, list[int]]:
    """A transport that raises `error` the first `fail_times` calls.

    Returns the transport and a one-element list counting every attempt, so a
    test can assert not only that the call succeeded but how many sockets it
    took.
    """
    calls = [0]

    def handle(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] <= fail_times:
            raise error
        return httpx.Response(200, json={"accepted": True})

    return httpx.MockTransport(handle), calls


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries are real; their waits are not worth a second of test time."""
    import worker.core.client as module

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(module.asyncio, "sleep", instant)


async def test_a_dropped_connection_is_retried_rather_than_losing_the_job() -> None:
    """THE BUG, 2026-08-17. The worker reaches the API down a long-lived tunnel;
    a pooled socket the far end has already closed raises RemoteProtocolError on
    the next write. Six jobs died of this in one day — including a
    video-to-video that had rendered seven of its eight sections — because one
    routine progress update could not be delivered. Asking again on a fresh
    socket is the whole fix."""
    transport, calls = flaky(1, httpx.RemoteProtocolError("server disconnected"))

    result = await client(transport).report_progress(
        "job-1",
        worker_id=WORKER_ID,
        lease_token=LEASE,
        status="generating",
        progress=40,
        message="Generating your video…",
    )

    assert result["accepted"] is True
    assert calls[0] == 2, "the first socket failed; the second carried the report"


@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("server disconnected"),
        httpx.ReadError("connection reset"),
        httpx.ConnectError("tunnel down"),
        httpx.WriteError("broken pipe"),
    ],
)
async def test_every_transport_failure_class_seen_in_production_is_retried(
    error: Exception,
) -> None:
    """RemoteProtocolError and ReadError are the two the logs actually show;
    the others are the same fault arriving through a different socket call, and
    treating any of them as an outage would keep the bug alive under a new
    name."""
    transport, calls = flaky(1, error)

    result = await client(transport).heartbeat(WORKER_ID)

    assert result["accepted"] is True
    assert calls[0] == 2


async def test_retries_are_bounded_and_then_reported_as_an_outage() -> None:
    """A genuinely unreachable API must still surface as an outage — quickly.
    Retrying forever would hold a GPU against a platform that is not coming
    back."""
    from worker.core.client import _TRANSPORT_ATTEMPTS, ApiUnavailable

    transport, calls = flaky(99, httpx.ConnectError("tunnel down"))

    with pytest.raises(ApiUnavailable) as raised:
        await client(transport).heartbeat(WORKER_ID)

    assert calls[0] == _TRANSPORT_ATTEMPTS
    assert "ConnectError" in str(raised.value)


async def test_a_read_timeout_is_not_repeated() -> None:
    """The request reached the API and may still be running. Sending it again
    would be a second instruction, not a retry — the one transport-shaped
    failure where asking twice is worse than failing once."""
    from worker.core.client import ApiUnavailable

    transport, calls = flaky(99, httpx.ReadTimeout("no answer"))

    with pytest.raises(ApiUnavailable):
        await client(transport).heartbeat(WORKER_ID)

    assert calls[0] == 1


async def test_claiming_a_job_is_never_retried() -> None:
    """A claim whose response was lost has already taken a job on the server;
    asking again takes a SECOND one this worker will never run. The poll loop
    is the retry, and it comes round in a second."""
    from worker.core.client import ApiUnavailable

    transport, calls = flaky(99, httpx.RemoteProtocolError("server disconnected"))

    with pytest.raises(ApiUnavailable):
        await client(transport).claim(WORKER_ID, ["text-to-video"])

    assert calls[0] == 1


def test_pooled_connections_expire_before_the_server_closes_them() -> None:
    """The ROOT CAUSE, not the symptom. The API runs uvicorn with no
    `--timeout-keep-alive`, so it drops an idle connection at 5s — and httpx's
    own default is also 5.0s. Both ends expiring the same socket at the same
    instant is a race the worker loses roughly 28 times a day, each one a
    RemoteProtocolError on a connection it still believed in.

    Expiring earlier removes the race instead of surviving it. If this value
    ever creeps back up to the server's, the retries above will hide the
    problem while it happens on every call again."""
    from worker.core.client import _KEEPALIVE_EXPIRY_SECONDS, WorkerApiClient

    UVICORN_DEFAULT_KEEPALIVE = 5.0
    assert _KEEPALIVE_EXPIRY_SECONDS < UVICORN_DEFAULT_KEEPALIVE

    pool = WorkerApiClient()._client._transport._pool
    assert pool._keepalive_expiry == _KEEPALIVE_EXPIRY_SECONDS


async def test_a_server_error_is_not_retried_either() -> None:
    """A 5xx is the API answering. It is an outage to report, not a socket to
    re-open, and hammering a struggling API is what this module has always
    warned against."""
    from worker.core.client import ApiUnavailable

    calls = [0]

    def handle(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(503, json={})

    with pytest.raises(ApiUnavailable):
        await client(httpx.MockTransport(handle)).heartbeat(WORKER_ID)

    assert calls[0] == 1
