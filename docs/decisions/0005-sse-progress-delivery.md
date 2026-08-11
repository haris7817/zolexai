# ADR 0005 — Progress over SSE, with a durable event log behind it

**Status:** Accepted (M1)
**Related:** [0003](./0003-stateless-api.md), [0004](./0004-postgres-as-the-queue.md)

## Context

A generation takes seconds to minutes. The user must see it advance. Three
options:

1. **Polling** — `GET /generations/{id}` every second. Simple, and it costs one
   request per second per watching user for the entire duration, most of them
   returning nothing new.
2. **WebSockets** — bidirectional, and nothing here is bidirectional: the
   browser sends no data during a generation. They also need per-connection
   server state, which cuts against [ADR 0003](./0003-stateless-api.md), and
   need their own reconnection and resume logic.
3. **Server-Sent Events** — one-way, plain HTTP, and `EventSource` reconnects
   automatically, resending the last event id it saw.

## Decision

SSE at `GET /api/v1/generations/{id}/events`, with delivery split in two:

```
worker ──HTTP──▶ any API instance
                      ├─▶ PostgreSQL  generation_events   (durable, replayable)
                      └─▶ Redis PUBLISH zx:job:{id}:events (live, fire-and-forget)

browser ──SSE──▶ any API instance
                      ├─ 1. SUBSCRIBE to the channel
                      ├─ 2. SELECT events WHERE seq > Last-Event-ID
                      └─ 3. stream the replay, then the live feed
```

### Why both halves are necessary

**Redis pub/sub alone loses events.** It delivers to whoever is listening at
that instant and remembers nothing, so everything published during a reconnect
is gone permanently. A user whose train enters a tunnel comes back to a progress
bar frozen at 22%.

**PostgreSQL alone means polling.** It has the events, but nothing pushes them.

Together: PostgreSQL is the record of truth, Redis removes the latency of
asking for it. `publish_event` deliberately swallows Redis failures — the event
is already committed, so an outage costs latency, never correctness.

### Two orderings that matter

**Commit, then publish.** Every service method writes and commits before
announcing. Publishing first can deliver an event describing a state that then
rolls back, and no client retry recovers from that.

**Subscribe, then read.** The stream subscribes to Redis *before* querying
PostgreSQL. The reverse leaves a gap between the last row read and the
subscription taking effect, and anything published in that gap is lost forever.
Subscribing first can only produce a duplicate, which the per-job `seq` makes
free to discard.

## Consequences

**Good.** Reconnection is lossless and costs no client code — `EventSource`
sends `Last-Event-ID` on its own. Any API instance can stream any job regardless
of which worker is running it or which instance the worker reported to. A
completed job closes its stream instead of holding a connection.

**Costs.** One row per lifecycle event (~7 per job). Every event is a row and a
publish. `generation_events` will need a retention policy before it becomes
large — noted as deferred work, not implemented.

**Guards in place.** `MAX_STREAM_SECONDS` caps a stream at an hour so an
abandoned tab cannot pin a connection and a Redis subscriber indefinitely;
`X-Accel-Buffering: no` stops nginx buffering the stream into a single slow
response; a keepalive comment every 15s stops proxies closing an idle
connection.
