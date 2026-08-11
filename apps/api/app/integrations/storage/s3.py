"""S3-compatible storage, used for MinIO, AWS S3 and Cloudflare R2 alike.

boto3 is synchronous, which would normally be disqualifying inside async
handlers. It is safe here because of *what* is called:

  `generate_presigned_url` performs no network I/O at all — it is local HMAC
  signing. The blocking calls (`head_object`, `delete_object`, bucket setup) are
  either startup-time or run in a thread via `asyncio.to_thread` at the service
  layer.

That keeps a single well-supported SDK rather than adding an async S3 client for
operations that are mostly not I/O.
"""

from __future__ import annotations

import threading

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings
from app.core.logging import get_logger
from app.integrations.storage.base import ObjectStat, PresignedUpload

logger = get_logger(__name__)


class S3ObjectStorage:
    def __init__(
        self,
        *,
        endpoint: str,
        public_endpoint: str,
        bucket: str,
        region: str,
        access_key: str,
        secret_key: str,
        presign_expiry: int,
    ) -> None:
        self._bucket = bucket
        self._presign_expiry = presign_expiry
        self._endpoint = endpoint.rstrip("/")
        self._public_endpoint = public_endpoint.rstrip("/")

        common = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
            # SigV4 with path-style addressing: MinIO does not do virtual-host
            # buckets on a bare hostname, and R2 requires SigV4.
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }

        self._client = boto3.client("s3", endpoint_url=self._endpoint, **common)

        # A SECOND client, signing against the endpoint a browser can reach.
        #
        # Inside Docker the API talks to http://minio:9000, but a presigned URL
        # containing that host is unreachable from the user's machine — and the
        # host is part of the signature, so it cannot be rewritten afterwards
        # without invalidating it. Two clients is the only correct fix.
        self._public_client = (
            self._client
            if self._public_endpoint == self._endpoint
            else boto3.client("s3", endpoint_url=self._public_endpoint, **common)
        )

        self._bucket_ready = False
        self._bucket_lock = threading.Lock()

    # ── Presigning (local computation, no I/O) ───────────────────────────

    def presign_upload(
        self, key: str, *, content_type: str, max_size_bytes: int
    ) -> PresignedUpload:
        url = self._public_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": self._bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=self._presign_expiry,
            HttpMethod="PUT",
        )
        # Content-Type is signed, so the browser must send exactly this value or
        # storage rejects the PUT. That is the enforcement point for the
        # declared MIME type — a client cannot claim one type and upload another.
        return PresignedUpload(
            url=url,
            method="PUT",
            headers={"Content-Type": content_type},
            expires_in=self._presign_expiry,
        )

    def presign_download(self, key: str, *, filename: str | None = None) -> str:
        params: dict[str, str] = {"Bucket": self._bucket, "Key": key}
        if filename:
            safe = filename.replace('"', "").replace("\\", "").replace("\r", "")
            params["ResponseContentDisposition"] = f'attachment; filename="{safe}"'
        return self._public_client.generate_presigned_url(
            ClientMethod="get_object", Params=params, ExpiresIn=self._presign_expiry
        )

    # ── Blocking operations — callers wrap in asyncio.to_thread ──────────

    def stat(self, key: str) -> ObjectStat | None:
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return ObjectStat(
            size_bytes=int(head.get("ContentLength", 0)),
            content_type=head.get("ContentType", "application/octet-stream"),
        )

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def ensure_bucket(self) -> None:
        with self._bucket_lock:
            if self._bucket_ready:
                return
            try:
                self._client.head_bucket(Bucket=self._bucket)
            except ClientError:
                try:
                    self._client.create_bucket(Bucket=self._bucket)
                    logger.info("storage_bucket_created", extra={"bucket": self._bucket})
                except ClientError as exc:
                    # A concurrent instance winning the race is success, not failure.
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                        raise
            self._bucket_ready = True

    def health(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return True
        except (ClientError, BotoCoreError) as exc:
            logger.warning("storage_unhealthy", extra={"reason": type(exc).__name__})
            return False


_storage: S3ObjectStorage | None = None


def get_storage() -> S3ObjectStorage:
    global _storage
    if _storage is None:
        _storage = S3ObjectStorage(
            endpoint=settings.storage_endpoint,
            public_endpoint=settings.public_storage_endpoint,
            bucket=settings.storage_bucket,
            region=settings.storage_region,
            access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key,
            presign_expiry=settings.storage_presign_expiry_seconds,
        )
    return _storage
