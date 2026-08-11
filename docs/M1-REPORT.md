# Milestone 1 — Final Report

**Scope:** the production-oriented platform foundation. Real API, real database,
real queue, real worker architecture, real storage — with generation itself
**mocked** (no GPU, no model, no provider account). M2 replaces exactly one
seam; nothing else moves.

**Status: complete.** All 14 acceptance criteria verified (§15 below).
M2 has **not** been started, per the stop conditions.

---

## 1. Files and folders added or significantly changed

```
apps/api/                        NEW — the entire FastAPI service
  app/core/                        config, enums (status contracts), errors,
                                   logging, middleware, security
  app/db/                          async engine/session, Redis client, base
  app/models/                      users, assets, generation_*, worker_nodes
  app/schemas/                     public + internal Pydantic contracts
  app/repositories/                all SQL lives here (generation, asset, worker)
  app/services/                    workflow_registry, generation, storage,
                                   events (SSE), idempotency, rate_limit, queue
  app/integrations/storage/        ObjectStorage protocol + S3/MinIO impl
  app/api/v1/                      health, workflows, generations, assets, internal
  migrations/                      Alembic (async, env-driven URL)
  tests/                           66 tests
  Dockerfile · pyproject.toml · alembic.ini

apps/worker/                     NEW — the generation worker
  worker/adapters/                 GenerationAdapter protocol, MockAdapter, registry
  worker/jobs/runner.py            claim → inputs → adapter → upload → report
  worker/core/                     config, structured logging, API client
  worker/storage/transfer.py       presigned download/upload
  worker/workflows/resolver.py     claim payload → adapter job
  tests/                           14 tests
  Dockerfile · pyproject.toml

apps/web/                        CONVERTED — demo → real application
  src/lib/api/client.ts            the ONLY fetch; Zod-validated responses
  src/lib/query.ts                 QueryClient + centralized query keys
  src/services/                    workflows, generations, assets
  src/features/generation/         queries (TanStack), form (RHF+Zod),
                                   useGenerationStream (SSE)
  src/features/workflows/          queries + catalog.server.ts (YAML reader)
  components rewired               workspace, panel, canvas, job strip, cards,
                                   history, media library, dropzone, nav
  DELETED                          mockPipeline, useGenerationJobs,
                                   useWorkflowParams, TS workflow registry,
                                   mocks/generations, mocks/media

packages/workflow-contracts/     NEW — shared TS contracts (Zod schemas)
workflow-definitions/*.yaml      REWRITTEN — now THE source of truth
infrastructure/compose/          docker-compose.yml (pg, redis, minio, apps profile)
infrastructure/nginx/            edge config (internal blocked, SSE unbuffered)
docs/decisions/0002–0007         six ADRs
.env.example                     full M1 variable set
README.md                        architecture, setup, API reference, limitations
```

## 2. Architecture implemented

Three independently deployable services in one repository (ADR 0002):

```
browser ──REST+SSE──▶ FastAPI (stateless, N instances) ──▶ PostgreSQL / Redis
browser ──presigned PUT──▶ object storage ◀──presigned GET/PUT── worker (pull-based, N instances)
```

The mandatory scalability rules and where each is enforced: §11 below.

## 3. Database

7 tables + `alembic_version`, one migration (`20260811_1239_initial_schema`),
generated with a deterministic naming convention and verified by a test that
upgrades a throwaway database and diffs it against the models.

| Table | Purpose |
|---|---|
| `users` | seeded dev user; `plan_code`, `concurrency_limit` are M3 extension points |
| `assets` | pointers into object storage; `pending → ready` upload lifecycle |
| `generation_jobs` | the queue and the record: status, progress, lease (worker_id, lease_token, lease_expires_at), attempt_count/max_attempts, idempotency_key, error_code/message, timestamps |
| `generation_job_inputs` | role-tagged inputs (`source_video`, optional `reference_image`) |
| `generation_job_outputs` | produced assets, `is_primary` |
| `generation_events` | append-only lifecycle log; per-job `seq` = SSE cursor |
| `worker_nodes` | registration, heartbeat, capabilities |

**Indexes** (each guarded by a test): `(user_id, created_at DESC)`,
`(user_id, status, created_at DESC)`, `(user_id, workflow_id, created_at DESC)`,
partial `claimable` on `status='queued'`, partial `lease_expiry`, partial-unique
`(user_id, idempotency_key)`. All history endpoints are keyset-paginated —
nothing ever loads a full table.

## 4. API endpoints

Public under `/api/v1` (health ×3, workflows ×2, generations ×5 incl. SSE,
assets ×3, media ×2) and six internal worker endpoints plus a maintenance
endpoint — token-guarded, hidden from OpenAPI, blocked at the nginx edge. Full
table in the README. One error envelope with stable machine codes; no stack
trace can reach a client.

## 5. Workflow registry

`workflow-definitions/*.yaml` is the single source of truth. Python validates at
startup (a bad file aborts the boot); TypeScript reads the same files at render
time for the landing page and navigation; `qa:parity` proves both readings are
identical and that nothing private leaks. The `execution:` block (runtime, and
M2's model/provider detail) is stripped by an explicit allowlist projection and
crosses exactly one boundary: the authenticated worker claim.

## 6. Generation lifecycle

`queued → assigned → preparing → generating → post_processing → uploading →
completed | failed | cancelled` — defined once in `app/core/enums.py`.
Transitions are validated (terminal is final; no backwards moves), progress is
monotonic, and the public API returns both `status` and a customer `stage_label`.

## 7. Worker architecture

Pull-based; workers need only outbound HTTPS (no port, no DB, no storage
credential — everything arrives as presigned URLs scoped to one job). Claiming
is `FOR UPDATE SKIP LOCKED` (ADR 0004): concurrent, exactly-once, fair
(oldest-first). Ownership is a **lease** — rotated token + expiry; every report
must present the token, so a zombie process cannot overwrite its successor. A
reaper (in-process timer + maintenance endpoint) requeues expired leases up to
`max_attempts`, then fails the job with a customer-safe message. The provider
seam is `GenerationAdapter`; M1 registers only `MockAdapter` (~6.8s staged
lifecycle, honest placeholder PNG output declared as such in the YAML).

## 8. SSE implementation

`GET /generations/{id}/events`. Durable half: every event is a row with a dense
per-job `seq` that becomes the SSE `id:`. Live half: Redis pub/sub fan-out, so
any API instance streams any job. Reconnection is lossless: `EventSource`
resends `Last-Event-ID`, the API replays newer rows from PostgreSQL, then
attaches live (subscribe-before-read closes the gap). Commit-then-publish
ordering throughout. Keepalives every 15s; 1h stream cap; nginx buffering off.

## 9. Storage implementation

S3-compatible behind an `ObjectStorage` protocol — MinIO locally, S3/R2 by
config. Browser uploads: presign (validated) → direct PUT (Content-Type bound
into the signature) → confirm (re-reads real size/type from storage; only then
`ready`, only `ready` assets can feed a generation). Downloads are short-lived
signed GETs; the bucket stays private. Two signing clients handle the
Docker-internal vs. browser-reachable endpoint split.

## 10. Tests and results

| Suite | Count | Result |
|---|---|---|
| `apps/api` pytest (real PostgreSQL + Redis) | 66 | **all passing** |
| `apps/worker` pytest | 14 | **all passing** |
| `tsc --noEmit` / `eslint --max-warnings=0` | — | clean |
| `next build` | 19 routes | clean |
| `qa:parity` (YAML ↔ API ↔ leak scan) | 6 workflows | **PASS** |
| `qa:e2e` (real browser, full stack) | 20 checks | **PASS** (see §15) |

Highlights: two workers never receive the same job (real `SKIP LOCKED`);
concurrent double-click creates one job (real `SET NX`); a stale lease token is
refused; retries cap at 3; a traceback sent by a worker never reaches the
customer; SSE resumes from `Last-Event-ID` without loss; migration ↔ model
parity.

## 11. Scalability safeguards (the 15 mandatory rules)

| # | Rule | Where enforced |
|---|---|---|
| 1 | API stateless | ADR 0003; no mutable module state (3 documented cache exceptions) |
| 2 | Media never on app-server disk | presigned flows only; nginx body cap 2 MB as a tripwire |
| 3 | Generation never blocks HTTP | 202 + job id; verified by test and e2e |
| 4 | Workers independent of API | separate app, zero shared packages, HTTP-only |
| 5 | Workers scale horizontally | `SKIP LOCKED` claim; `--scale worker=3` documented |
| 6 | API scales horizontally | statelessness + Redis fan-out for SSE |
| 7 | Pagination required | keyset cursors everywhere; `limit` capped at 100 |
| 8 | Indexes support common queries | §3; guarded by a named-index test |
| 9 | Provider details behind adapters | ADR 0007; leak tests + parity script |
| 10 | Frontend never touches workers | workers have no inbound surface at all |
| 11 | Redis/PostgreSQL never public | compose network + nginx; only 3000/8100 exposed |
| 12 | No single-process assumptions | lease reaper safe to run in every instance |
| 13 | No global in-memory truth | jobs/events/assets all in PostgreSQL |
| 14 | No hard-coded endpoints | all URLs from env; wildcard CORS rejected at boot |
| 15 | Replaceable components | storage protocol, adapter seam, repository layer |

Fair-use foundation: per-user concurrency limit (default 3, per-user override
column, plan seam for M3), request-rate counters (fail-open), edge rate limit.
Idempotency: `Idempotency-Key` → Redis `SET NX` + partial unique index (both
necessary; tested under concurrency).

## 12. Known limitations

- Generation output is a **placeholder PNG** — declared honestly end to end
  (the UI renders by actual asset kind, so M2's real media needs no UI change).
- One seeded user; no sessions. The UI badge says "Preview build — accounts and
  billing are not connected", which is now the *only* untrue-feeling surface
  left deliberately.
- Prompt search filters loaded pages only (labelled as such).
- No retention policy on `generation_events`; no multipart upload; no cleanup
  job for never-confirmed `pending` assets.
- Windows dev quirk: this machine runs the API on **8100** and MinIO on
  **9010** because another project holds 8000/9000 (see `.env` notes).

## 13. Deferred to M2 / M3

**M2:** real provider adapter (LTX or other) + GPU workers on Vast.ai; real
video/audio output; reference-image *behavior* for video-to-video (the contract
— optional `reference_image` input, validated, stored, delivered to the worker —
is done); per-workflow output thumbnails/previews.

**M3:** authentication/sessions (swap `get_current_user`; `user_id` FKs already
real), plans/subscriptions/billing (`plan_code`, `concurrency_limit`, and the
limit-resolution seam already exist), usage metering, media library management
(rename/delete), server-side prompt search.

## 14. Exact local run commands

```bash
cp .env.example .env                 # once; adjust ports if taken
npm install
npm run infra:up                     # PostgreSQL + Redis + MinIO + bucket

cd apps/api    && python -m venv .venv && ./.venv/Scripts/python -m pip install -e ".[dev]"
cd ../worker   && python -m venv .venv && ./.venv/Scripts/python -m pip install -e ".[dev]"

cd apps/api    && ./.venv/Scripts/python -m alembic upgrade head

# three terminals:
cd apps/api    && ./.venv/Scripts/python -m uvicorn app.main:app --reload --port 8100
cd apps/worker && ./.venv/Scripts/python -m worker.main
npm run dev                          # http://localhost:3000

# tests / verification:
cd apps/api    && ./.venv/Scripts/python -m pytest -q
cd apps/worker && ./.venv/Scripts/python -m pytest -q
npm run qa:parity && npm run qa:e2e  # with the stack running
```

Docker-only alternative: `npm run stack:up`.

## 15. M1 acceptance criteria — verification

| # | Criterion | Evidence |
|---|---|---|
| 1 | Run locally | §14; all services healthy (`/health`: db/redis/storage/workflows all true) |
| 2 | Approved frontend | design untouched; e2e renders workspace/history/library |
| 3 | Workflow metadata from the API | e2e: durations/aspects rendered match `GET /workflows` |
| 4 | Submit a generation | e2e: browser submit |
| 5 | Job id immediately | e2e: HTTP 202 with id |
| 6 | Job enters the queue | status `queued`; event seq 1 written |
| 7 | Mock worker claims it | worker log `job_claimed`/`job_started`; lease held |
| 8 | Realistic SSE progress | e2e observed Queued → Preparing → Generating → Finalizing live |
| 9 | Completes with a mock result | e2e: result rendered from `…/generated/{job}/output.png` on MinIO |
| 10 | In generation history | e2e: history lists it; detail page loads by id |
| 11 | Upload/register media | presign → direct PUT → confirm → listed (tested + e2e library check) |
| 12 | API restart keeps jobs | verified live: API restarted mid-session; jobs, history and worker all resumed (worker re-registered automatically) |
| 13 | Multiple instances safe | statelessness + `SKIP LOCKED` + `SET NX` tests; no in-memory truth |
| 14 | Tests pass | 66 + 14, plus parity and e2e harnesses |

## 15b. Post-delivery fix — container path resolution

**Found in production, 2026-08-11.** `alembic upgrade head` inside the API
container failed at import with `IndexError: 4`.

**Cause.** `REPO_ROOT` was `Path(__file__).resolve().parents[4]`. That is
correct in the repository — `apps/api/app/core/config.py` genuinely is four
levels below the root — and out of range in the image, where the same module
sits at `/app/app/core/config.py` with only three parents. The worker carried
the identical defect at `/app/worker/core/config.py`; it had simply not been
reached yet.

**Why the M1 verification missed it.** Every check ran against a host checkout,
where the assumption holds. The unit suites, the API suite and the browser
end-to-end run all imported the module from the repo. Building the image would
not have caught it either: the failure is at *runtime*, and `docker build`
never imports the module. The Dockerfiles were written and reviewed but never
built-and-run, and that is the specific gap.

**Fix.** Both resolvers now walk upwards for a marker that exists in either
layout and degrade to the filesystem root when there is none — safe, because
the only two values derived from it (`.env`, the definitions directory) are
optional or environment-supplied in a deployed service.

**Regression cover.** `apps/api/tests/test_deployment_layout.py` and
`apps/worker/tests/test_deployment_layout.py` exercise the resolver against a
simulated container layout at every depth down to the root, so a future fixed
`parents[N]` index fails in CI rather than in production.

## 16. What the next stage needs from the client

1. **GPU provider account** (Vast.ai per the architecture doc) with billing
   enabled — required before any M2 work can run.
2. **Model/provider decision** for video generation (LTX per the blueprint) —
   confirm licensing terms are acceptable for commercial output.
3. **Production hosting decisions**: a domain, a managed PostgreSQL, a Redis, an
   S3-compatible bucket (Cloudflare R2 recommended), and where the API/web
   containers run. The compose file and Dockerfiles are ready for any
   container host.
4. **A production `WORKER_API_TOKEN`** (≥ 32 random chars) and storage
   credentials — the API refuses to boot in production without them.
5. For M3: choice of billing provider (decision register D-05).

---

*M2 has not been started. No Vast.ai, LTX, ComfyUI, PyTorch or GPU work exists
in this repository, and no real subscription/payment integration was performed.*
