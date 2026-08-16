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
