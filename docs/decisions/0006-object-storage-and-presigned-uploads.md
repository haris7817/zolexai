# ADR 0006 — Media lives in object storage and never passes through an app server

**Status:** Accepted (M1)
**Related:** [0003](./0003-stateless-api.md)

## Context

ZolexAI moves large files: source videos up to 1 GB, generated results, audio.
The default architecture most frameworks encourage is:

```
Browser → Next.js → FastAPI → disk or storage
```

Every one of those arrows is a problem. A 500 MB upload occupies a server
process for the whole transfer, so concurrent uploads are capped by however many
a single instance can buffer. Writing to local disk breaks the moment there is a
second API instance, because the file exists on exactly one of them. And a
worker on a different host cannot read it at all.

## Decision

**Bytes go directly between the browser and object storage. The API only ever
handles small JSON.**

```
Browser ──presigned PUT──▶ Object storage
Worker  ──presigned GET──▶ Object storage
```

Provider is configuration: MinIO locally, S3 or Cloudflare R2 in production,
behind one `ObjectStorage` protocol (`app/integrations/storage/base.py`). No
service, route or model imports a storage SDK.

### Uploads are two-phase, and the second phase is not a formality

1. `POST /assets/upload-url` validates the *declared* type and size, writes a
   `pending` asset row, returns a presigned PUT.
2. The browser uploads directly.
3. `POST /assets/{id}/confirm` re-reads the object's **real** size and content
   type from storage and flips the row to `ready`.

Everything in step 1 is a claim. A client can declare `image/png` at 2 MB and
upload a 4 GB file. Two independent things stop that: `Content-Type` is part of
the signature, so storage itself rejects a mismatched PUT; and the confirm step
checks what actually landed, deleting the object if it is oversized rather than
leaving an orphan on paid storage.

Only `ready` assets may be used as generation inputs, so nothing unverified ever
reaches a worker.

### The two-client detail

`S3ObjectStorage` builds **two** boto3 clients. Inside Docker the API reaches
MinIO at `http://minio:9000`, but a presigned URL containing that host is
unreachable from the user's machine — and the host is part of the signature, so
it cannot be rewritten afterwards without invalidating it. The second client
signs against `STORAGE_PUBLIC_ENDPOINT`.

### Why boto3, which is synchronous

Presigning performs **no network I/O** — it is local HMAC computation. The
genuinely blocking calls (`head_object`, `delete_object`, bucket setup) run via
`asyncio.to_thread`. That avoids adding an async S3 client for operations that
are mostly not I/O.

## Consequences

**Good.** Upload throughput is the storage provider's problem, not ours. The
API stays stateless and small. Workers need no standing storage credential —
every file they can reach arrives as a URL scoped to one object for one job, so
a compromised GPU node cannot enumerate the bucket.

**Costs.** Uploads are three round trips instead of one, and the browser must
send the signed headers verbatim. Local development needs a MinIO CORS rule, or
the direct PUT fails with an opaque browser error despite a valid signature —
`minio-setup` in the compose file handles it.

**Deferred.** Multipart upload for very large files, background cleanup of
`pending` assets that were never confirmed, and a storage lifecycle policy.
