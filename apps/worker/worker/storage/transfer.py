"""Media transfer via presigned URLs.

The worker holds no storage credentials. Every read and write uses a
short-lived signed URL issued by the API for one specific object, so a
compromised node cannot enumerate the bucket or reach another user's media
(directive §13, §17).

Bytes go straight between the worker and object storage — never through the API.
"""

from __future__ import annotations

import httpx

from worker.adapters.base import AdapterError
from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)

#: Refuse to buffer more than this from a presigned download. A worker holding a
#: multi-gigabyte input in memory is how one oversized job takes a whole node
#: down; the API enforces upload limits, and this is the second line.
MAX_INPUT_BYTES = 1024 * 1024 * 1024


async def download_input(url: str, *, role: str) -> bytes:
    async with httpx.AsyncClient(timeout=settings.upload_timeout_seconds) as client:
        try:
            async with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise AdapterError(
                        "One of the selected files could not be read.",
                        internal_detail=f"GET {role} returned HTTP {response.status_code}",
                    )

                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_INPUT_BYTES:
                        raise AdapterError(
                            "That input file is too large to process.",
                            internal_detail=f"{role} exceeded {MAX_INPUT_BYTES} bytes",
                            retriable=False,
                        )
                return bytes(chunks)
        except httpx.HTTPError as exc:
            raise AdapterError(
                "One of the selected files could not be read.",
                internal_detail=f"{type(exc).__name__} downloading {role}",
            ) from exc


async def upload_output(url: str, content: bytes, content_type: str) -> None:
    """PUTs the result to the presigned URL.

    `Content-Type` must match what the API signed exactly — the header is part
    of the signature. A mismatch is rejected by storage, which is precisely the
    guarantee: a worker cannot upload something other than the type the workflow
    declared.
    """
    async with httpx.AsyncClient(timeout=settings.upload_timeout_seconds) as client:
        try:
            response = await client.put(
                url, content=content, headers={"Content-Type": content_type}
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

    logger.info("output_uploaded", extra={"size_bytes": len(content)})
