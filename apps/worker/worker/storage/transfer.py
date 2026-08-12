"""Media transfer via presigned URLs.

The worker holds no storage credentials. Every read and write uses a
short-lived signed URL issued by the API for one specific object, so a
compromised node cannot enumerate the bucket or reach another user's media
(directive §13, §17).

Bytes go straight between the worker and object storage — never through the API,
and never fully into memory. Both directions stream through a file on disk:
inputs can be half a gigabyte and outputs can be larger, and buffering either
one is how a single oversized job takes a whole node down.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from worker.adapters.base import AdapterError
from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)

#: Refuse to accept more than this from a presigned download. The API enforces
#: upload limits at their source; this is the second line, and it protects the
#: worker's disk rather than its memory.
MAX_INPUT_BYTES = 1024 * 1024 * 1024

#: Streaming chunk size. Large enough that syscall overhead is irrelevant, small
#: enough that a chunk is never a memory concern.
_CHUNK_BYTES = 1024 * 1024


async def download_input_to(url: str, dest: Path, *, role: str) -> Path:
    """Streams a presigned GET to `dest`, returning it.

    Written incrementally: at no point is more than one chunk resident, so the
    ceiling below is about disk, not RAM.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        async with httpx.AsyncClient(timeout=settings.download_timeout_seconds) as client:
            async with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise AdapterError(
                        "One of the selected files could not be read.",
                        internal_detail=f"GET {role} returned HTTP {response.status_code}",
                    )

                with dest.open("wb") as handle:
                    async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                        written += len(chunk)
                        if written > MAX_INPUT_BYTES:
                            raise AdapterError(
                                "That input file is too large to process.",
                                internal_detail=f"{role} exceeded {MAX_INPUT_BYTES} bytes",
                                retriable=False,
                            )
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise AdapterError(
            "One of the selected files could not be read.",
            internal_detail=f"{type(exc).__name__} downloading {role}",
        ) from exc
    except BaseException:
        # A partial file is worse than none — an adapter would read it as valid.
        dest.unlink(missing_ok=True)
        raise

    logger.info("input_staged", extra={"role": role, "size_bytes": written})
    return dest


async def _file_chunks(path: Path) -> AsyncIterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            yield chunk


async def upload_output_file(url: str, path: Path, content_type: str) -> int:
    """PUTs a file to the presigned URL, streaming from disk. Returns its size.

    Two headers matter and both are deliberate:

    `Content-Type` must match what the API signed exactly — it is part of the
    signature. A mismatch is rejected by storage, which is precisely the
    guarantee: a worker cannot upload something other than the type the workflow
    declared.

    `Content-Length` is set explicitly because httpx would otherwise fall back
    to `Transfer-Encoding: chunked` for a streamed body, and S3 rejects chunked
    PUTs with 411. Supplying the length also suppresses that fallback — httpx
    skips its own `Transfer-Encoding` header when the caller has set a length.
    """
    size = path.stat().st_size

    try:
        async with httpx.AsyncClient(timeout=settings.upload_timeout_seconds) as client:
            response = await client.put(
                url,
                content=_file_chunks(path),
                headers={"Content-Type": content_type, "Content-Length": str(size)},
            )
    except httpx.HTTPError as exc:
        raise AdapterError(
            "The result could not be saved. Please try again.",
            internal_detail=f"{type(exc).__name__} uploading output",
        ) from exc

    if response.status_code >= 400:
        raise AdapterError(
            "The result could not be saved. Please try again.",
            internal_detail=f"PUT output returned HTTP {response.status_code}",
        )

    logger.info("output_uploaded", extra={"size_bytes": size})
    return size
