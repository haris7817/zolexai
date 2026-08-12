# Workflow definitions

The single source of truth for what ZolexAI offers. The API validates these at
startup (a bad file aborts boot), serves the public parts at
`GET /api/v1/workflows`, and hands the private parts to the worker with each
claimed job. The frontend builds every tool surface from the API response; the
landing page reads these files directly at build time.

## The one rule

Everything in a file is **public** — served verbatim to any browser — **except
the `execution:` block**, which is private and stripped from every API response
by an explicit allowlist projection (`WorkflowDefinition.to_public()`).

Provider, model, runner, hardware and tuning details belong under `execution:`
and nowhere else. No provider or model name may appear in any public field.

> Guarded by tests: `test_execution_block_never_reaches_a_client` and
> `test_no_provider_or_infrastructure_names_anywhere`, plus the
> `qa:parity` script, which diffs these files against the live API response
> and fails if anything private escapes.

## Duration modes

`duration_mode` decides how a workflow's output length is chosen:

| Mode | Meaning | `supported_durations` |
|---|---|---|
| `fixed` | User picks one | `["5s", "10s", …]` |
| `source` | Matches the uploaded source file automatically | must be `[]` |
| `minutes` | User picks a song length in minutes | `["1m", …, "5m"]` |

The API rejects a supplied duration on a `source` workflow and requires one on
the others. Ceilings (e.g. music's `5m`) are provisional until the M2 model
benchmark; raising one is an edit to the list, not a code change.

## Scope

Frozen at six workflows (milestones §8.1). Adding a seventh is a change
request, not an edit to this folder.
