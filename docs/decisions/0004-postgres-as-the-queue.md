# ADR 0004 — PostgreSQL is the queue; Redis is a doorbell

**Status:** Accepted (M1)
**Related:** [0003](./0003-stateless-api.md)

## Context

A worker needs to be handed exactly one job, no two workers may get the same
one, and a worker that dies must not take the job with it.

The obvious choice is a Redis list: `LPUSH` to enqueue, `BRPOP` to claim. It is
fast and it is what most tutorials show. It also **loses work**: `BRPOP`
removes the item, so a worker that crashes between popping and persisting has
deleted a paid job with no record it existed. `BRPOPLPUSH` into a processing
list helps, but now there are two sources of truth to reconcile, and the
reconciliation needs durable state — which is the thing Redis was chosen to
avoid needing.

A dedicated broker (RabbitMQ, Kafka, Celery) solves this properly and adds an
operational component, a failure mode and a deployment step for a system that
currently runs one worker.

## Decision

**The `generation_jobs` table is the queue.** A claim is one statement:

```sql
SELECT * FROM generation_jobs
WHERE status = 'queued' AND workflow_id = ANY(:workflows)
ORDER BY created_at          -- fair: oldest first
FOR UPDATE SKIP LOCKED       -- concurrent, non-blocking, exactly-once
LIMIT 1
```

`SKIP LOCKED` is what makes this correct *and* concurrent: workers each lock a
different row instead of queueing behind one another, and no two ever see the
same row. A worker that dies mid-transaction releases its lock on disconnect and
the row is untouched — still `queued`, still claimable.

**Ownership is a lease, not an assignment.** A claim sets `worker_id`, a rotated
`lease_token` and `lease_expires_at`. Every progress report must present the
token, which is what stops a stalled process from overwriting the state of
whichever worker took over. A reaper requeues expired leases up to
`max_attempts`, then fails the job with a customer-safe message.

**Redis is a doorbell.** `notify_job_available` pushes to a list workers block
on, purely to remove polling latency. If Redis is flushed, stale or down,
workers fall back to polling and nothing is lost.

## Consequences

**Good.** No broker to operate. Enqueue and job creation are the same
transaction, so a job cannot exist in the queue but not the database (or the
reverse). Crash recovery is the lease reaper, which handles a killed container,
a network partition and a hung process identically — because from PostgreSQL
they are identical. Scaling workers needs no coordination.

**Costs.** Claiming is a database round trip, so throughput is bounded by
PostgreSQL rather than by Redis. At the scale generation implies — jobs take
seconds to minutes, not microseconds — this is irrelevant by orders of
magnitude. The partial index `ix_generation_jobs_claimable` keeps the claim an
index scan over queued rows only, not the whole table.

**When to revisit.** If claim contention ever becomes measurable — thousands of
workers, or sub-second jobs. The migration path is a real broker behind the same
`claim_next` repository method; nothing above the repository layer knows how
claiming works.
