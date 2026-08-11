# ZolexAI documentation

## Project control documents

Authored outside this repository and copied in as-is. They are the planning and
commercial reference for the whole engagement.

| File | Source document | Purpose |
|---|---|---|
| `architecture.md` | ZolexAI Implementation Architecture & Engineering Blueprint | Technical blueprint |
| `milestones.md` | ZolexAI Milestones & Deliverables | Commercial structure — M1/M2/M3, acceptance criteria |
| `delivery-tracker.pdf` | ZolexAI Project Delivery, Milestone & Task Tracker | Task register, gates, dependencies, risks |
| `demo-feedback-guide.md` | ZolexAI Client UI Demo & Feedback Guide | Demo script, feedback questions, freeze criteria |

> **Not yet copied in.** They live outside the repo. Drop them here so the
> repository is self-contained — nothing in the build depends on it.

## Decision records

Written here, in [`decisions/`](./decisions):

| ADR | Subject | Milestone |
|---|---|---|
| [0001](./decisions/0001-unified-design-system.md) | Unifying three screens onto one design system | PRE-M1 |
| [0002](./decisions/0002-monorepo-with-independent-services.md) | One repository, three independently deployable services | M1 |
| [0003](./decisions/0003-stateless-api.md) | The API holds no application state | M1 |
| [0004](./decisions/0004-postgres-as-the-queue.md) | PostgreSQL is the queue; Redis is a doorbell | M1 |
| [0005](./decisions/0005-sse-progress-delivery.md) | Progress over SSE with a durable event log | M1 |
| [0006](./decisions/0006-object-storage-and-presigned-uploads.md) | Media never passes through an app server | M1 |
| [0007](./decisions/0007-provider-abstraction.md) | Workflow definitions as source of truth; providers behind adapters | M1 |

## Reports

| File | Covers |
|---|---|
| [`PRE-M1-HANDOFF.md`](./PRE-M1-HANDOFF.md) | Client UI/UX demo delivery (design approved) |
| [`M1-REPORT.md`](./M1-REPORT.md) | Platform foundation delivery |

## Phase status

| Phase | State |
|---|---|
| PRE-M1 — Client UI/UX approval | **Complete** — design approved |
| M1 — Platform & core setup | **Complete** — see [M1-REPORT.md](./M1-REPORT.md) |
| M2 — AI workflows | Not started (GPU dependency D-03) |
| M3 — Subscription, testing, deployment | Not started |

### What M1 did and did not deliver

M1 built the platform. It did **not** build AI generation, authentication or
billing, and no M2/M3 task is advanced by it.

| Area | M1 state |
|---|---|
| Generation job system | **Implemented** — real jobs, real queue, real lifecycle |
| Worker architecture | **Implemented** — real claiming, leasing, retries; **mock** execution |
| AI generation | **Not started** — M2 |
| Authentication | **Not started** — M3.01. One seeded dev user; `user_id` is a real FK |
| Subscription / billing | **Not started** — M3.11. `plan_code` and `concurrency_limit` are unenforced extension points |
| Media library | **Implemented** — real uploads, real storage, real listing |

The PRE-M1 distinction still applies to anything still on mock data: a screen
existing does not mark a milestone task complete.

## Authority order

When documents disagree, resolve in this order:

1. The latest **directive** from the project owner.
2. **Client UI Demo & Feedback Guide** — for demo, feedback and freeze process.
3. **Tracker / Milestones / Architecture** — for scope, sequencing and engineering.
