# ADR 0003 — The API holds no application state

**Status:** Accepted (M1)
**Related:** [0002](./0002-monorepo-with-independent-services.md), [0005](./0005-sse-progress-delivery.md)

## Context

The fastest way to build a generation platform is to keep jobs in a dictionary:

```python
jobs: dict[str, Job] = {}          # never do this
active_users: dict[str, Session] = {}
```

It works perfectly with one API process, and fails the moment there are two —
not loudly, but by serving a different answer depending on which instance the
load balancer picked. That failure appears at exactly the point where scaling
is urgent, and the fix at that point is a rewrite.

## Decision

No mutable application state in API process memory. Ever.

| State | Home |
|---|---|
| Jobs, assets, users, events | PostgreSQL |
| Cache, coordination, idempotency, pub/sub | Redis |
| Media bytes | S3-compatible object storage |

An API instance holds a database connection pool, a Redis pool and an immutable
workflow registry loaded from disk at startup. Nothing else. Restarting one, or
adding a tenth, changes no observable behaviour.

### The three deliberate exceptions

All three are **caches of facts owned elsewhere**, not sources of truth:

1. **The workflow registry** — parsed from version-controlled YAML at startup
   and never mutated. Every instance loads the same files.
2. **The development user's id** (`app/core/security.py`) — a cache of one
   immutable database row, rebuildable by any instance with one query.
3. **The lease reaper loop** (`app/main.py`) — a timer, not state. Running it in
   every instance is safe because the recovery is a single conditional `UPDATE`;
   concurrent runs contend on rows rather than duplicating work.

## Consequences

**Good.** `docker compose up --scale api=3` behind any load balancer works with
no sticky sessions and no coordination. A rolling deploy drops no jobs. An
instance that crashes mid-request loses that request and nothing else.

**Costs.** Every read is a query — there is no free in-process cache. Mitigated
where it matters: history uses keyset pagination against composite indexes, and
the workflow catalogue carries a short `Cache-Control`.

**What this rules out.** WebSockets with per-connection server state, in-process
job queues, and sticky-session assumptions. SSE was chosen over WebSockets
partly for this reason — see [ADR 0005](./0005-sse-progress-delivery.md).

## How to tell if this regresses

A module-level mutable dict, list or set in `apps/api/app/` that is written to
during a request. `grep` for `= {}` and `= []` at module scope.
