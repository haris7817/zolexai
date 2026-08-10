# ZolexAI documentation

## Project control documents

These four are authored outside this repository and should be **copied in as-is**. They are the
planning and commercial reference for the whole engagement.

| File | Source document | Purpose |
|---|---|---|
| `architecture.md` | ZolexAI Implementation Architecture & Engineering Blueprint | Technical blueprint — stack, layering, DB model, API contract, security |
| `milestones.md` | ZolexAI Milestones & Deliverables | Commercial structure — M1/M2/M3, $1,200, acceptance criteria |
| `delivery-tracker.pdf` | ZolexAI Project Delivery, Milestone & Task Tracker | Task register, gates, dependencies, risks, test matrix |
| `demo-feedback-guide.md` | ZolexAI Client UI Demo & Feedback Guide | Demo script, feedback questions, freeze criteria |

> **Not yet copied in.** They currently live outside the repo. Drop them here so the repository is
> self-contained — the plan calls for this, and nothing in the build depends on it.

## Decision records

Written here, in [`decisions/`](./decisions):

| ADR | Subject |
|---|---|
| [0001](./decisions/0001-unified-design-system.md) | Unifying three screens onto one design system |

## Authority order

When these documents disagree, resolve in this order:

1. The latest **PRE-M1 directive** from the project owner.
2. **Client UI Demo & Feedback Guide** — for demo, feedback and freeze process.
3. **Tracker / Milestones / Architecture** — for scope, sequencing and engineering.

One conflict is already known and resolved: the demo guide (§2) anticipates that non-core pages
"may still be preview/placeholder states", while the PRE-M1 directive requires full visual mockups
of All Tools, Generations, Media Library, Subscription and Settings. **The directive wins** — all
five are built.

## Phase status

| Phase | State |
|---|---|
| PRE-M1 — Client UI/UX approval | **In progress** |
| M1 — Platform & core setup | **BLOCKED** until PREUI-20 |
| M2 — AI workflows | Not started (GPU dependency D-03) |
| M3 — Subscription, testing, deployment | Not started |

### DESIGNED ≠ IMPLEMENTED

A screen existing as a demo mockup **never** marks an M1/M2/M3 task complete.

| State | Meaning | Set by |
|---|---|---|
| **DEMO READY** | Visual/UX approval-ready, mock data | PREUI-xx |
| **IMPLEMENTED** | Real data, persistence, backend | M1/M2/M3 |

Specifically: `/app/media` reaching DEMO READY does **not** advance M3.07; `/app/generations` does
not advance M3.08/M3.09; `/app/subscription` does not advance M3.15; `/app/settings` does not
advance M3.18; Login/Register do not advance M3.02.
