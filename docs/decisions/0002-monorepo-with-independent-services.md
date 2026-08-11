# ADR 0002 — One repository, three independently deployable services

**Status:** Accepted (M1)
**Supersedes:** nothing
**Related:** [0003](./0003-stateless-api.md), [0004](./0004-postgres-as-the-queue.md)

## Context

ZolexAI has three runtime components — a Next.js frontend, a FastAPI backend and
a generation worker — plus shared contracts and infrastructure definitions.

Two failure modes were available:

1. **Three repositories.** Every contract change becomes a multi-repo pull
   request, a version bump and a coordination problem, at a stage where the
   contracts are still moving weekly.
2. **One repository that is also one deployment.** A single artifact containing
   all three, which makes "scale the workers" mean "run more copies of the
   frontend too", and puts database credentials on a GPU node.

## Decision

One repository, three deployable units, and no shared runtime.

```
apps/web       Next.js      → any Node host / CDN
apps/api       FastAPI      → N stateless instances behind a load balancer
apps/worker    Python       → N instances, anywhere with outbound HTTPS
```

**Same repository does not mean same server.** Each has its own dependency
manifest, its own Dockerfile and its own scaling story:

- `apps/api` and `apps/worker` share **no Python package**. The worker has no
  SQLAlchemy, no database driver and no storage SDK. It reaches the platform
  only through the internal HTTP API, using presigned URLs scoped to one job.
- Shared *contracts* live in `packages/workflow-contracts` (TypeScript) and
  `workflow-definitions/*.yaml` (language-neutral). Shared *code* does not.

## Consequences

**Good.** A contract change is one commit that updates the YAML, the Python
schema and the TypeScript schema together, and CI verifies all three agree. The
worker's dependency list is four packages, so a compromised GPU node cannot
read the database or enumerate another user's media — it holds no credential
that would let it.

**Costs.** Two Python virtual environments to maintain, and a small amount of
duplication between the API's and the worker's logging setup (~30 lines). That
duplication is deliberate: giving the worker an import from `apps/api` would
couple two independently deployable services over formatting code.

**What this rules out.** Importing `app.models` from the worker. If a future
change appears to need that, the correct move is to extend the internal API,
not to reach across the boundary.
