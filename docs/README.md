# ZolexAI documentation

## Project control documents

Authored outside this repository and copied in as-is. They are the planning and
commercial reference for the whole engagement.

| File | Source document | Purpose |
|---|---|---|
| [`milestones.md`](./milestones.md) | ZolexAI Milestones & Deliverables — rev. 12 Aug 2026 | Commercial structure — M1/M2/M3, acceptance criteria |
| [`delivery-tracker.md`](./delivery-tracker.md) | ZolexAI Project Delivery, Milestone & Task Tracker — rev. 12 Aug 2026 | Task register, gates, dependencies, risks |
| `architecture.md` | ZolexAI Implementation Architecture & Engineering Blueprint | Technical blueprint |
| `demo-feedback-guide.md` | ZolexAI Client UI Demo & Feedback Guide | Demo script, feedback questions, freeze criteria |

The first two are current as of **12 August 2026**. They are client-facing
control documents, not engineering records: they describe agreed scope and
commercial state, and are **replaced wholesale** when a new revision is issued
rather than edited to track implementation. What the code actually does belongs
in [`M1-REPORT.md`](./M1-REPORT.md) and the decision records.

> **`architecture.md` and `demo-feedback-guide.md` are not yet copied in.** They
> live outside the repo. Drop them here so the repository is self-contained —
> nothing in the build depends on it.

### Open client revisions carried by these documents

Recorded in the tracker's change log; **none are implemented yet**:

| CR | Revision | State |
|---|---|---|
| CR-001 | Three-step section heading → **IMAGINE IT. GENERATE IT. GO VIRAL.** | Approved, not applied |
| CR-002 | Latest first-fold/mobile hero reference (prominent logo, Start Creating CTA) | Approved, not applied |
| CR-003 | Final brand logo integration | Pending asset |

CR-004 … CR-010 are M2 workflow/quality requirements and are captured in
[`milestones.md`](./milestones.md) §§5–13.

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

## Operations

Not customer-facing — see [`internal/`](./internal):

| File | Covers |
|---|---|
| [`internal/production-runbook.md`](./internal/production-runbook.md) | Production server, deploy and rollback procedure, go-live checks |

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
