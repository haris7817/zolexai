# ZolexAI

Premium AI media-generation platform. Six creation workflows — text to video,
image to video, video to video, extend video, music and music video — behind one
workspace.

**Milestone status:** M1 (platform foundation) complete. No real AI generation
yet: a mock worker executes the full job lifecycle and produces a placeholder
image. See [M1 limitations](#m1-limitations).

---

## Architecture

```
                          ┌──────────────┐
   browser ──────────────▶│  apps/web    │  Next.js 15 · React 19 · TanStack Query
        │                 │  (frontend)  │  React Hook Form · Zod · Tailwind v4
        │                 └──────┬───────┘
        │                        │ server-renders from workflow-definitions/
        │  REST + SSE            │
        └───────────────▶┌───────▼──────┐         ┌──────────────┐
                         │  apps/api    │────────▶│  PostgreSQL  │  jobs, assets,
                         │  (FastAPI)   │         └──────────────┘  events, users
                         │              │────────▶┌──────────────┐
                         │  stateless   │         │    Redis     │  pub/sub, cache,
                         └───────▲──────┘         └──────────────┘  idempotency
                                 │ internal API           
                                 │ (service token)  ┌──────────────────┐
                         ┌───────┴──────┐           │  Object storage  │
                         │ apps/worker  │──────────▶│  MinIO / S3 / R2 │
                         │  pull-based  │ presigned └──────────────────┘
                         └──────────────┘    URLs           ▲
                                                            │ presigned PUT
                                              browser ──────┘  (direct upload)
```

Three services, one repository, **independently deployable**. Same repo does not
mean same server — see [ADR 0002](docs/decisions/0002-monorepo-with-independent-services.md).

| Path | What it is | Scaling |
|---|---|---|
| `apps/web` | Next.js frontend | N instances / CDN |
| `apps/api` | FastAPI public + internal API | N stateless instances behind a load balancer |
| `apps/worker` | Generation worker | N instances, anywhere with outbound HTTPS |
| `packages/workflow-contracts` | Shared TypeScript API contracts (Zod) | — |
| `workflow-definitions/*.yaml` | **The source of truth for what the platform can do** | — |
| `infrastructure/compose` | Local PostgreSQL, Redis, MinIO | — |

### The load-bearing decisions

Each has an ADR under [`docs/decisions/`](docs/decisions):

- **[0003](docs/decisions/0003-stateless-api.md)** — the API holds no application state, so instances are interchangeable.
- **[0004](docs/decisions/0004-postgres-as-the-queue.md)** — PostgreSQL is the queue (`FOR UPDATE SKIP LOCKED` + leases); Redis is only a doorbell.
- **[0005](docs/decisions/0005-sse-progress-delivery.md)** — progress over SSE, with a durable event log so reconnection loses nothing.
- **[0006](docs/decisions/0006-object-storage-and-presigned-uploads.md)** — media never passes through an app server.
- **[0007](docs/decisions/0007-provider-abstraction.md)** — workflows are YAML; providers hide behind adapters.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Node.js | ≥ 20.9 | tested on 20.18 |
| Python | 3.11 – 3.13 | Docker images use 3.12 |
| Docker + Compose | v2 | for PostgreSQL, Redis, MinIO |

---

## Local setup

### 1. Configuration

```bash
cp .env.example .env
```

`.env` is gitignored and is read by **all three services** — the API and worker
load it directly, and `apps/web/next.config.ts` loads it too so
`NEXT_PUBLIC_API_URL` reaches the browser bundle.

> **Windows note.** Docker may be unable to bind port 6379 — Windows reserves
> ranges for Hyper-V. Check with
> `netsh interface ipv4 show excludedportrange protocol=tcp` and set
> `REDIS_PORT` / `REDIS_URL` to a free port if 6379 falls inside one.

### 2. Backing services

```bash
npm run infra:up        # PostgreSQL + Redis + MinIO, and creates the bucket
```

| Service | Port | Credentials |
|---|---|---|
| PostgreSQL | 5432 | from `.env` |
| Redis | 6379 (see note) | none |
| MinIO API | 9000 | `STORAGE_ACCESS_KEY` / `STORAGE_SECRET_KEY` |
| MinIO console | 9001 | same |

### 3. Install

```bash
npm install                                       # workspaces: web + contracts

cd apps/api    && python -m venv .venv && ./.venv/Scripts/python -m pip install -e ".[dev]"
cd ../worker   && python -m venv .venv && ./.venv/Scripts/python -m pip install -e ".[dev]"
```

On macOS/Linux use `.venv/bin/python`.

### 4. Migrate

```bash
cd apps/api && ./.venv/Scripts/python -m alembic upgrade head
```

### 5. Run (three terminals)

```bash
# API — http://localhost:8100  (docs at /docs)
cd apps/api    && ./.venv/Scripts/python -m uvicorn app.main:app --reload --port 8100

# Worker — no port; pulls work from the API
cd apps/worker && ./.venv/Scripts/python -m worker.main

# Frontend — http://localhost:3000
npm run dev
```

Open <http://localhost:3000/app/create/text-to-video>, type a prompt, press
Generate. The job is created, a worker claims it, and progress streams back over
SSE.

### Everything in Docker instead

```bash
npm run stack:up        # + API, worker and web containers
npm run stack:down
```

Workers scale with no coordination — claiming is atomic:

```bash
docker compose --env-file .env -f infrastructure/compose/docker-compose.yml \
  --profile apps up -d --scale worker=3
```

---

## Environment variables

Full list with comments in [`.env.example`](.env.example). The ones that matter:

| Variable | Used by | Notes |
|---|---|---|
| `DATABASE_URL` | api | `postgresql+asyncpg://…` |
| `REDIS_URL` | api, worker | |
| `STORAGE_ENDPOINT` | api | where the API reaches storage |
| `STORAGE_PUBLIC_ENDPOINT` | api | where a **browser** reaches it — differs under Docker, and the host is part of the signature |
| `STORAGE_BUCKET` / `STORAGE_ACCESS_KEY` / `STORAGE_SECRET_KEY` | api | |
| `WORKER_API_TOKEN` | api, worker | service credential for `/api/v1/internal/*`. ≥ 32 chars in production, and the API refuses to start without it |
| `CORS_ORIGINS` | api | explicit origins; a wildcard is rejected |
| `NEXT_PUBLIC_API_URL` | web | **inlined at build time.** Empty = same origin |
| `JOB_LEASE_SECONDS` / `JOB_MAX_ATTEMPTS` | api | lease length and retry ceiling |
| `DEFAULT_USER_CONCURRENCY_LIMIT` | api | simultaneous jobs per user |

No secret is ever committed. `alembic.ini` deliberately has no `sqlalchemy.url`.

---

## API

Base: `/api/v1`. Versioned from the first endpoint.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` · `/health/live` · `/health/ready` | live = process only; ready = dependencies |
| GET | `/workflows` · `/workflows/{id}` | public metadata only |
| POST | `/generations` | **202** + job id. Accepts `Idempotency-Key` |
| GET | `/generations` | keyset-paginated; `status`, `workflow_id` filters |
| GET | `/generations/{id}` | |
| POST | `/generations/{id}/cancel` | |
| GET | `/generations/{id}/events` | **SSE**; honours `Last-Event-ID` |
| POST | `/assets/upload-url` | presigned direct upload |
| POST | `/assets/{id}/confirm` | verifies what actually landed |
| POST | `/assets/{id}/download-url` | short-lived signed GET |
| GET | `/media` · `/media/counts` | |

Internal, service-token only, **excluded from the public OpenAPI document and
blocked at the edge**: `/internal/workers/register|heartbeat`,
`/internal/jobs/claim`, `/internal/jobs/{id}/progress|complete|fail`,
`/internal/maintenance/reap-leases`.

Every error uses one envelope:

```json
{ "error": { "code": "unsupported_parameter", "message": "…",
             "details": { "fields": [...] }, "request_id": "…" } }
```

`code` is stable and safe to branch on. No stack trace ever reaches a client.

---

## Generation lifecycle

```
queued → assigned → preparing → generating → post_processing → uploading → completed
                                                                        ↘ failed
                                                                        ↘ cancelled
```

Defined once in `apps/api/app/core/enums.py` and imported everywhere. The public
API returns both `status` (machine contract) and `stage_label` (what the user
reads), so internal granularity can change without touching a component.

Terminal is terminal: a late report from a superseded worker is rejected rather
than resurrecting a job.

---

## Tests

Tests run against **real** PostgreSQL and Redis. `FOR UPDATE SKIP LOCKED`,
partial unique indexes and `SET NX` atomicity cannot be verified against SQLite
or a mock, and those are exactly the properties that matter.

```bash
npm run infra:up                                  # required

cd apps/api    && ./.venv/Scripts/python -m pytest -q      # 66 tests
cd apps/worker && ./.venv/Scripts/python -m pytest -q      # 14 tests

npm run typecheck && npm run lint --workspace=web
```

With the full stack running:

```bash
npm run qa:parity   # YAML ↔ API agree; nothing private leaked
npm run qa:e2e      # browser: submit → SSE → result → history → library
```

---

## M1 limitations

Deliberate, and each has a named home:

| Limitation | Arrives at |
|---|---|
| **No real AI generation.** The mock worker emits a placeholder PNG; no GPU, no model, no provider account | M2 |
| **No authentication.** Every request resolves to one seeded development user; `user_id` is already a real FK | M3.01 |
| **No billing or plans.** `plan_code` and `concurrency_limit` exist as extension points and are unenforced | M3.11 |
| **Video-to-video reference image is contract only.** Accepted, validated, stored, handed to the worker — but it performs no identity or character replacement | M2 |
| Prompt search filters loaded pages, not all history (labelled as such in the UI) | server-side search |
| No retention policy on `generation_events` | before it grows large |
| No multipart upload; no cleanup of unconfirmed `pending` assets | as needed |

The previous PRE-M1 client demo is unaffected and still live at its own URL — it
is served from a standalone copy of the old static build.

---

## Repository layout

```
apps/
  web/         Next.js frontend
  api/         FastAPI — app/{api,core,db,models,schemas,repositories,services,integrations}
  worker/      Worker  — worker/{adapters,jobs,storage,workflows,core}
packages/
  workflow-contracts/   shared TypeScript API contracts (Zod)
workflow-definitions/   the six workflows — SOURCE OF TRUTH
infrastructure/
  compose/     docker-compose.yml (PostgreSQL, Redis, MinIO, + apps profile)
  nginx/       edge config: blocks /api/v1/internal/, SSE buffering off
docs/
  decisions/   ADRs
```
