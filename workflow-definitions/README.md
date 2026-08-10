# Workflow definitions

Public workflow metadata, version controlled per architecture doc §12.

## The one rule

These files contain **PUBLIC metadata only** — what the customer-facing UI is
allowed to know: name, category, output type, inputs, supported durations,
aspect ratios, quality levels, settings and capabilities.

They must **never** contain an `execution:` block, or any provider, model,
runner, workflow-file, VRAM or timeout value. That is private execution
metadata; it stays server-side and is served to no one.

> Non-negotiable rule #1 — do not expose internal AI/provider names in the
> customer UI. Guarded by test T-API-01 and risk R-10.

## Current status (PRE-M1)

These mirror `apps/web/src/features/workflows/registry.ts`, which is what the
demo actually reads. At **M1.14** the API becomes the owner: it loads and
validates these at startup, and **M1.15** exposes them via
`GET /api/v1/workflows`, at which point the frontend registry is deleted and
replaced by the API response (M1.16).

Scope is frozen at six workflows (milestones §8.1). Adding a seventh is a
change request, not an edit to this folder.
