# ADR 0007 — Workflow definitions are the source of truth; providers hide behind adapters

**Status:** Accepted (M1)
**Related:** [0002](./0002-monorepo-with-independent-services.md)

## Context

Two things must be true when M2 introduces real generation:

1. Changing or replacing a model must not require a frontend change.
2. No provider, model or infrastructure name may ever reach a customer.

Before M1 the frontend had its own hard-coded workflow registry in TypeScript
*and* `workflow-definitions/*.yaml` existed separately. They had **already
drifted** — the YAML had no icon, no marketing copy and no thumbnail. Two
definitions of the same six objects is a guarantee of divergence.

## Decision

### One source of truth

`workflow-definitions/*.yaml` defines what ZolexAI can do. Python validates it
at API startup; TypeScript parses the same files at render time. Two independent
readers of one declarative file — loose coupling, not duplication — and
`qa-catalog-parity.mjs` fails the build if they disagree.

A malformed definition **aborts the API boot**. A container that will not start
is a far cheaper failure than one that starts and quietly offers a control the
backend rejects.

### Public and private halves

Each definition has an `execution:` block: runtime today, model and graph
reference in M2. It is stripped by an **explicit allowlist projection**
(`WorkflowDefinition.to_public()`), never `model_dump(exclude=...)` — an exclude
list is a denylist, and a denylist silently ships the next private field
somebody adds.

The block crosses exactly one boundary: to an authenticated worker, on the
private network, in the claim response. Three tests and one QA script assert it
appears nowhere else.

### The adapter seam

```
Frontend → ZolexAI API → Workflow Service → Workflow Adapter → Provider
```

`GenerationAdapter` (`apps/worker/worker/adapters/base.py`) has three members:
`name`, `supports()`, `run(job, on_progress)`. Which adapter runs is decided by
the workflow's `execution.runtime`, so **moving a workflow to a real provider in
M2 is a one-line YAML change plus a new adapter class.** No route, schema,
migration or component changes.

An unknown runtime is a hard, non-retriable error. A silent fallback to the mock
would ship placeholder images while every dashboard looked healthy.

### Honesty about what M1 produces

The mock runtime emits a placeholder PNG, not video or audio. Rather than let
the UI imply otherwise, each definition declares
`execution.output_content_type: image/png`. The API signs the worker's upload
for exactly that type, the asset is stored as an image, and the UI picks its
renderer from **the asset's kind, not the workflow's declared output type** — so
the same components keep working unchanged when M2 starts producing real media.

## Consequences

**Good.** Adding a workflow is one YAML file: it appears in navigation, All
Tools, the landing grid, the settings panel and request validation with no code
change. Provider swaps do not reach the frontend. Nothing customer-facing can
name a provider without a test failing.

**Costs.** UI metadata (icon, gradient, marketing copy) lives in YAML rather
than in the frontend. That is the price of one file describing a tool
completely, and the directive lists "UI settings" as publishable workflow
information.

**Watch for.** Any `if (workflow.id === ...)` in a component, or a second
hard-coded list of workflows. Both mean the abstraction has been bypassed.
