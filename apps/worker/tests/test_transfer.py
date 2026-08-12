"""Media transfer: streamed both ways, and correctly framed on the wire.

M1 held the whole of an input and the whole of an output in memory. That was
invisible with a 40 KB placeholder PNG and is a node-killer with a minute of
1080p video, so both directions now go through a file.

The `Content-Length` assertion below looks fussy and is not: httpx falls back to
`Transfer-Encoding: chunked` for any streamed body whose length it cannot infer,
and S3 answers a chunked PUT with `411 Length Required`. That failure would only
appear against real object storage, on large outputs, after a full generation —
the most expensive possible moment to discover it.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from worker.adapters.base import AdapterError
from worker.storage import transfer
from worker.storage.transfer import download_input_to, upload_output_file


def _capturing_transport(captured: dict, status_code: int = 200) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = await request.aread()
        return httpx.Response(status_code)

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch):
    """Routes worker HTTP through a caller-supplied transport."""

    def install(transport: httpx.MockTransport) -> None:
        original = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(transfer.httpx, "AsyncClient", factory)

    return install


# ── Upload ───────────────────────────────────────────────────────────────


async def test_upload_sends_an_explicit_content_length_and_not_chunked(
    tmp_path: Path, mock_http
) -> None:
    """S3 rejects chunked PUTs. The header must be a real byte count."""
    payload = b"z" * (3 * 1024 * 1024)
    source = tmp_path / "output.mp4"
    source.write_bytes(payload)

    captured: dict = {}
    mock_http(_capturing_transport(captured))

    size = await upload_output_file("https://storage.test/signed", source, "video/mp4")

    assert size == len(payload)
    assert captured["headers"]["content-length"] == str(len(payload))
    assert "transfer-encoding" not in captured["headers"]
    assert captured["headers"]["content-type"] == "video/mp4"
    assert captured["body"] == payload


async def test_upload_preserves_the_signed_content_type(tmp_path: Path, mock_http) -> None:
    """The type is part of the presigned signature, so a worker cannot upload
    something other than what the workflow declared."""
    source = tmp_path / "track.mp3"
    source.write_bytes(b"audio")

    captured: dict = {}
    mock_http(_capturing_transport(captured))

    await upload_output_file("https://storage.test/signed", source, "audio/mpeg")
    assert captured["headers"]["content-type"] == "audio/mpeg"


async def test_a_rejected_upload_is_a_customer_safe_failure(tmp_path: Path, mock_http) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"data")
    mock_http(_capturing_transport({}, status_code=403))

    with pytest.raises(AdapterError) as raised:
        await upload_output_file("https://storage.test/signed", source, "video/mp4")

    assert "403" in raised.value.internal_detail
    assert "403" not in raised.value.user_message
    assert "storage.test" not in raised.value.user_message


# ── Download ─────────────────────────────────────────────────────────────


async def test_download_writes_the_input_to_disk(tmp_path: Path, mock_http) -> None:
    payload = b"source video bytes" * 1000

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    mock_http(httpx.MockTransport(handler))
    destination = tmp_path / "inputs" / "source_video.mp4"

    result = await download_input_to(
        "https://storage.test/signed", destination, role="source_video"
    )

    assert result == destination
    assert destination.read_bytes() == payload


async def test_a_failed_download_leaves_no_partial_file(tmp_path: Path, mock_http) -> None:
    """A half-written file is worse than none: an adapter would read it as
    valid input and produce a plausible, wrong result."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    mock_http(httpx.MockTransport(handler))
    destination = tmp_path / "source_video.mp4"

    with pytest.raises(AdapterError):
        await download_input_to("https://storage.test/x", destination, role="source_video")

    assert not destination.exists()


async def test_an_oversized_input_is_refused_without_retrying(
    tmp_path: Path, mock_http, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying cannot make a file smaller, so the attempt budget should not be
    spent on it."""
    monkeypatch.setattr(transfer, "MAX_INPUT_BYTES", 512)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    mock_http(httpx.MockTransport(handler))
    destination = tmp_path / "huge.mp4"

    with pytest.raises(AdapterError) as raised:
        await download_input_to("https://storage.test/x", destination, role="source_video")

    assert raised.value.retriable is False
    assert not destination.exists()


async def test_download_urls_never_reach_the_customer_message(tmp_path: Path, mock_http) -> None:
    """A presigned URL carries a signature — it is a credential, and must not
    appear in anything a user can see."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    mock_http(httpx.MockTransport(handler))

    with pytest.raises(AdapterError) as raised:
        await download_input_to(
            "https://storage.test/bucket/key?X-Amz-Signature=secret",
            tmp_path / "in.mp4",
            role="source_video",
        )

    assert "X-Amz-Signature" not in raised.value.user_message
    assert "X-Amz-Signature" not in raised.value.internal_detail
