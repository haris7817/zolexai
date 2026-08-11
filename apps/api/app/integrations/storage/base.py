"""Object storage abstraction.

Generated and uploaded media never lives on an API or worker disk
(scalability rule #2). Everything goes to S3-compatible object storage: MinIO
locally, Cloudflare R2 or S3 in production. The provider is a configuration
value, not a code path.

The upload shape this enables is the one that matters:

    Browser ──presigned PUT──▶ Object storage

NOT:

    Browser ──▶ Next.js ──▶ FastAPI ──▶ Object storage

A 500 MB video streamed through the API would occupy a worker process for
minutes and cap throughput at whatever one instance can buffer. Presigning moves
the bytes out of the request path entirely, so the API only ever handles small
JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PresignedUpload:
    url: str
    method: str
    headers: dict[str, str]
    """Headers the browser MUST send. Signing binds Content-Type, so a mismatch
    is rejected by storage — this is what stops a client declaring `image/png`
    and uploading something else."""
    expires_in: int


@dataclass(frozen=True)
class ObjectStat:
    size_bytes: int
    content_type: str


@runtime_checkable
class ObjectStorage(Protocol):
    """The seam. Swapping MinIO for R2 means a different implementation here and
    nothing else — no service, route or model imports a storage SDK directly."""

    def presign_upload(
        self, key: str, *, content_type: str, max_size_bytes: int
    ) -> PresignedUpload: ...

    def presign_download(self, key: str, *, filename: str | None = None) -> str: ...

    def stat(self, key: str) -> ObjectStat | None:
        """Object metadata, or None when it does not exist.

        Used to confirm an upload actually landed before an asset is marked
        ready — a client claiming success is not evidence."""
        ...

    def delete(self, key: str) -> None: ...

    def ensure_bucket(self) -> None:
        """Creates the bucket if absent. Local convenience; a no-op in production
        where buckets are provisioned deliberately."""
        ...

    def health(self) -> bool: ...
