# ZolexAI — Milestones & Deliverables
## Updated Working Scope & Delivery Reference

> **Provenance.** Copied into the repository from the project-control document
> `ZolexAI_MILESTONES_DELIVERABLES_UPDATED_2026-08-12.md`, authored outside this
> repo. Content is preserved as written; only mis-decoded characters (em dashes,
> curly quotes) were restored. Do not edit to reflect implementation state — that
> belongs in [`M1-REPORT.md`](./M1-REPORT.md). Replace this file wholesale when a
> newer revision is issued.

**Project:** ZolexAI AI Media Generation Platform
**Original Commercial Value:** **$1,200**
**Commercial Structure:** **3 Milestones**
**Revision Date:** **12 August 2026**
**Current Delivery State:** **Milestone 1 foundation complete and deployed; Milestone 2 preparation / client requirement refinement underway**

> **Commercial note:** The client has stated that they intend to add **$200 more**. This document keeps the formal milestone total at **$1,200** until that additional amount is formally confirmed through Fiverr or otherwise added to the active commercial agreement.

---

# 0. Revision Summary — 12 August 2026

This revision updates the original milestone document to reflect the actual implementation status, production deployment already completed during Milestone 1, and the latest client requests.

## Newly recorded client requirements

- Evaluate **LTX 2.5** as the preferred current open-weight video model rather than remaining locked to LTX 2.3.
- Keep the model/provider layer modular so ZolexAI can change models without rebuilding the public application.
- Tune the video workflows to get **as close as realistically possible to the polished / Seedance-like visual quality target**, without promising identical proprietary-model output.
- **Video-to-Video duration should be automatic**:
  - detect the uploaded source-video duration;
  - aim for the final transformed output to match the source duration;
  - use backend segmentation/chaining/stitching where a model cannot process the full duration in a single pass.
- **Music Video duration should be automatic**:
  - detect the uploaded song/audio duration;
  - final music video should match the full audio duration;
  - for long songs (for example 4 minutes), generate visual sections and assemble/synchronize them automatically.
- **Video Extension** must offer:
  - 5 seconds
  - 10 seconds
  - 15 seconds
  - 30 seconds
  - 60 seconds
- **Music Generation** should allow the user to choose how many minutes they want the song to last rather than only exposing short fixed-duration presets.
- Longer music outputs may use segmentation/continuation/assembly behind the scenes when required by the final music model.
- Music/lyrics generation should strongly prioritize **natural, catchy and consistently rhyming lyrics**.
- Where practical, add a lyric-quality pass to improve/regenerate weak lines before final song generation.
- Do not promise perfect rhyme in every generation.
- The client will provide/approve the GPU when real model integration and benchmarking are ready to begin.
- The final ZolexAI brand logo still needs additional integration/polish in the live frontend.
- Replace the section heading currently reading **"From idea to output in three steps"** with:
  - **"IMAGINE IT. GENERATE IT. GO VIRAL."**
- The opening/first-fold mobile presentation should follow the latest client reference:
  - prominent ZolexAI logo at the top;
  - black/dark background;
  - neon lime/green accents;
  - strong **"CREATE THE IMPOSSIBLE."** hero treatment;
  - short supporting copy;
  - clear **Start Creating** CTA;
  - premium, mobile-first first impression.

---

# 1. Project Scope Summary

ZolexAI is a modular AI media-generation platform providing a premium creator experience for multiple AI workflows.

The currently agreed core scope includes:

- Text-to-Video
- Image-to-Video
- Video-to-Video
- Video-to-Video optional reference image / look guidance
- Video Extension
- Repeated / chained video extensions
- Music / Audio Generation
- Music Video / Audio-driven Video
- Workflow-aware duration handling
- Automatic long-duration orchestration where required
- Generation job management
- Generation history
- Media handling / media library
- User/account functionality
- Subscription functionality
- Responsive premium frontend
- Backend/API architecture
- GPU worker integration
- Testing and deployment

The platform must remain flexible so future AI tools and model providers can be added without rebuilding the core application.

## Scope boundaries still in force

- Seedance API itself is **not** part of the current implementation.
- ZolexAI may target a similar polished visual feel through open-weight/self-hosted workflow tuning, but **identical Seedance output is not guaranteed**.
- Voice cloning / AI voice remains outside the current three-milestone scope unless separately agreed.
- Unlimited customer-facing marketing does not mean unlimited physical GPU capacity; hidden fair-use, queue and concurrency controls remain required.

---

# 2. Milestone Overview

| Milestone | Name | Amount | Current State | Main Outcome |
|---|---|---:|---|---|
| 1 | ZolexAI Platform & Core Setup | $360 | **COMPLETE / DEPLOYED** | Real modular application foundation, backend, job system, mock worker and production foundation |
| 2 | AI Workflows & Generation Features | $480 | **PREPARATION / REQUIREMENTS REFINED** | Real GPU workflows, long-duration orchestration, model tuning and generation features |
| 3 | Subscription, Testing & Deployment | $360 | **PARTIALLY ADVANCED EARLY** | Auth/subscription, final QA/hardening; significant production infrastructure already deployed |

**Formal project total remains:** **$1,200**

---

# 3. Design / UI Approval & Current Brand Direction

## Completed / approved direction

- [x] Main ZolexAI creator workspace designed
- [x] Desktop creator layout designed
- [x] Responsive tablet/mobile direction designed
- [x] Workflow navigation designed
- [x] Generation settings experience designed
- [x] Generation progress states designed
- [x] Generation result actions designed
- [x] Workflow-driven UI concept established
- [x] Client live demo shared
- [x] Client approved overall direction
- [x] Original purple/violet direction replaced by the approved **black + neon lime/green** direction
- [x] Hero copy uses **CREATE THE IMPOSSIBLE.**

## Current UI revisions to implement

- [ ] Integrate the final ZolexAI logo properly across the live experience
- [ ] Improve first-fold/mobile logo prominence
- [ ] Match the client's latest mobile hero reference more closely
- [ ] Change **"From idea to output in three steps"** to **"IMAGINE IT. GENERATE IT. GO VIRAL."**
- [ ] Final visual polish after real-workflow screens are stable

---

# 4. MILESTONE 1 — ZolexAI Platform & Core Setup

## Commercial Value

**$360 — 30% of original project**

## Current State

# **COMPLETE — CORE FOUNDATION BUILT, TESTED AND DEPLOYED**

The original Milestone 1 objective has been met: ZolexAI exists as a real modular application with a production frontend/backend foundation and an end-to-end non-GPU generation pipeline.

---

## 4.1 Repository & Development Foundation

- [x] ZolexAI source repository created
- [x] Git initialized and private GitHub repository used
- [x] Monorepo structure created
- [x] `.gitignore` configured
- [x] `.env.example` created
- [x] README / architecture documentation created
- [x] Local-development instructions created
- [x] Linting/type checking established
- [x] Docker-ready project structure established

Current monorepo:

```text
zolexai/
├── apps/
│   ├── web/
│   ├── api/
│   └── worker/
├── packages/
├── workflow-definitions/
├── infrastructure/
├── docs/
├── .env.example
├── package.json
└── README.md
```

---

## 4.2 Frontend Foundation

- [x] Real Next.js frontend implemented
- [x] TypeScript/Tailwind frontend foundation implemented
- [x] Centralized design tokens implemented
- [x] Application shell implemented
- [x] Responsive mobile navigation implemented
- [x] Creator workspace implemented
- [x] Workflow-specific creator forms implemented
- [x] Prompt/media/settings controls implemented
- [x] Duration/aspect-ratio/quality metadata support implemented
- [x] Job progress UI connected to backend flow
- [x] Result/history/media foundation implemented
- [x] Responsive QA completed for Milestone 1
- [x] Browser E2E tests passed during M1 QA

### Remaining visual revision

- [ ] Final logo/brand integration and newest client hero/section copy revisions

---

## 4.3 Workflow-Driven Frontend Architecture

Initial workflow registry supports:

- [x] Text-to-Video
- [x] Image-to-Video
- [x] Video-to-Video
- [x] Extend Video
- [x] Music
- [x] Music Video

The frontend remains metadata-driven and does not publicly expose internal model/provider names.

---

## 4.4 Backend Foundation

Implemented:

- [x] FastAPI application
- [x] `/api/v1` API structure
- [x] Configuration / environment system
- [x] PostgreSQL integration
- [x] SQLAlchemy models
- [x] Alembic migrations
- [x] Redis integration
- [x] Health endpoints
- [x] Structured error/logging foundation
- [x] Workflow registry
- [x] Workflow list/detail API
- [x] Generation creation/detail/history APIs
- [x] Secure internal worker endpoints
- [x] SSE generation-event stream

Production health currently confirms:

```text
database: true
redis: true
storage: true
workflows: true
```

---

## 4.5 Generation Job Architecture

Implemented statuses:

```text
queued
assigned
preparing
generating
post_processing
uploading
completed
failed
cancelled
```

Implemented:

- [x] Persistent generation jobs
- [x] Inputs / outputs / events
- [x] Worker registration
- [x] Worker heartbeat
- [x] Claim / progress / complete / fail flow
- [x] Lease / retry-ready architecture
- [x] Backend-driven progress
- [x] Mock worker flow

---

## 4.6 Media / Object Storage Foundation

- [x] S3-compatible storage abstraction
- [x] Local/production MinIO integration
- [x] Direct presigned browser upload flow
- [x] Persistent media metadata
- [x] Public storage endpoint separated from internal Docker endpoint
- [x] Production storage domain configured

Current production storage:

```text
https://storage.zolexai.com
```

---

## 4.7 Milestone 1 QA Evidence

Completed during M1:

- [x] API tests passed
- [x] Worker tests passed
- [x] Frontend lint/typecheck/build passed
- [x] Browser E2E suite passed
- [x] Text-to-Video mock generation passed
- [x] Image-to-Video upload + generation flow passed
- [x] Video-to-Video source flow passed
- [x] Video-to-Video optional reference-image flow passed
- [x] Media library foundation passed
- [x] Persistence after restart checked
- [x] Manual M1 acceptance flow passed

The current worker remains a **mock runtime** until Milestone 2 real GPU/model integration.

---

## 4.8 Production Foundation Completed Early

Although final deployment was originally grouped into Milestone 3, the project foundation is already publicly deployed so M1 could be reviewed on the real domain.

Current routing:

```text
https://zolexai.com/
    -> Next.js frontend

https://zolexai.com/api/v1/*
    -> FastAPI

https://storage.zolexai.com/
    -> MinIO S3 API
```

Verified:

- [x] Production VPS selected and audited
- [x] Docker installed
- [x] Production Compose stack running
- [x] PostgreSQL private
- [x] Redis private
- [x] MinIO exposed only through controlled reverse proxy
- [x] Next.js host port loopback-only
- [x] FastAPI host port loopback-only
- [x] HTTPS / Let's Encrypt working
- [x] `www.zolexai.com` redirects to `zolexai.com`
- [x] Public worker-internal API path blocked
- [x] Main API health returns HTTP 200
- [x] Storage health returns HTTP 200
- [x] Deployment runbook created

---

## Milestone 1 Acceptance Criteria — Current

- [x] Real Next.js application exists
- [x] Responsive frontend exists
- [x] Workflow configuration is modular
- [x] FastAPI backend runs successfully
- [x] PostgreSQL connected
- [x] Redis connected
- [x] Workflow API works
- [x] Generation job can be created
- [x] Mock worker can claim/process jobs
- [x] Progress appears through backend events
- [x] Mock output appears in UI/history
- [x] Object storage works
- [x] Production foundation works over HTTPS
- [x] M1 can be demonstrated without real GPU

**M1 remaining note:** logo/latest marketing-copy polish is a visual revision, not a blocker to the completed core architecture.

---

# 5. MILESTONE 2 — AI Workflows & Generation Features

## Commercial Value

**$480 — 40% of original project**

## Current State

# **PREPARATION / REQUIREMENTS REFINED — GPU WORK NOT YET STARTED**

The client has confirmed that GPU access can be provided when requested.

## Primary Objective

Replace the mock runner with real GPU-backed workflows while preserving the stable provider-agnostic architecture.

Milestone 2 now also includes the client's latest duration, quality and music-generation behavior requirements.

---

## 5.1 GPU Environment

- [ ] Confirm final M2 workflow inventory
- [ ] Request client GPU / Vast.ai GPU when implementation is ready
- [ ] Validate GPU SSH/API access
- [ ] Configure CUDA/Python/PyTorch/FFmpeg
- [ ] Pin model/runtime dependencies
- [ ] Configure model cache
- [ ] Configure worker credentials
- [ ] Run worker diagnostics
- [ ] Benchmark VRAM/runtime by workflow

---

## 5.2 LTX 2.5 Evaluation

- [ ] Evaluate LTX 2.5 for ZolexAI video generation
- [ ] Verify official/open-weight license/commercial conditions before production use
- [ ] Compare LTX 2.5 against the previously planned LTX 2.3 setup
- [ ] Benchmark:
  - quality
  - motion
  - prompt adherence
  - V2V behavior
  - I2V behavior
  - extend/chaining behavior
  - VRAM usage
  - generation time
- [ ] Keep provider adapter generic so a future model can replace it without public API/UI changes

### Quality target

Tune the pipeline to get as close as realistically possible to the polished **Seedance-like** visual target requested by the client.

This is a quality direction, not a guarantee of identical proprietary-model output.

---

# 6. Text-to-Video

- [ ] Connect real Text-to-Video workflow
- [ ] Validate prompt/settings
- [ ] Validate model-supported duration
- [ ] Validate aspect ratio
- [ ] Validate quality
- [ ] Execute real generation
- [ ] Report progress
- [ ] Post-process output
- [ ] Upload to persistent storage
- [ ] Display result
- [ ] Preserve history
- [ ] Tune workflow/prompting for strongest practical visual quality

---

# 7. Image-to-Video

- [ ] Connect real Image-to-Video workflow
- [ ] Validate source image
- [ ] Preserve uploaded source
- [ ] Map user prompt/settings
- [ ] Generate and track progress
- [ ] Persist output
- [ ] Preserve history
- [ ] Test motion/identity consistency
- [ ] Tune for premium cinematic output

---

# 8. Video-to-Video

## Updated client behavior

**Duration should be automatic.**

The user should not need to manually select the final duration for V2V.

### Required behavior

- [ ] Detect uploaded source-video duration
- [ ] Set target output duration from source duration
- [ ] Keep optional reference-image input
- [ ] Use the reference image to guide appearance/look where supported
- [ ] For videos longer than the model's native processing window:
  - segment automatically;
  - process required sections;
  - preserve timing/order;
  - stitch output automatically;
  - minimize visual discontinuity;
  - keep total final duration aligned with the source.
- [ ] Hide chunking details from the customer UI
- [ ] Display useful progress for long V2V jobs
- [ ] Test short and long inputs
- [ ] Test audio handling strategy where relevant

### Acceptance

Examples:

```text
10 sec source  -> approx. 10 sec transformed result
45 sec source  -> approx. 45 sec transformed result
60 sec source  -> approx. 60 sec transformed result
```

Long outputs may be produced through orchestration rather than one native model call.

---

# 9. Video Extension

## Updated client duration requirement

The user-facing extension selector must include:

```text
5 seconds
10 seconds
15 seconds
30 seconds
60 seconds
```

### Tasks

- [ ] Implement all five requested selector options
- [ ] Validate each option against the final model/workflow
- [ ] Use native extension where supported
- [ ] Use repeated/chained extension where required
- [ ] Preserve parent/child lineage
- [ ] Preserve original output
- [ ] Allow extension of an extension
- [ ] Keep technical chaining hidden from the customer where practical
- [ ] Test continuity at joins

### Important rule

ZolexAI may offer a 60-second **user outcome** without falsely claiming that the underlying model produces a native 60-second extension in one inference.

---

# 10. Duration Orchestration

Duration behavior is now workflow-specific.

## Text-to-Video / Image-to-Video

- Show only technically supported direct options or clearly orchestrated options.
- Do not advertise unsupported native behavior.

## Video-to-Video

- Automatic from source-video duration.

## Video Extension

- Manual: 5 / 10 / 15 / 30 / 60 seconds.

## Music Video

- Automatic from uploaded/generated audio duration.

## Music Generation

- User chooses desired song length in minutes.

The API remains authoritative for validation.

---

# 11. Music / Audio Generation

## Updated client requirements

- [ ] Connect real music/audio model
- [ ] Support desired song duration in **minutes**
- [ ] Provide a simple duration control suitable for music rather than only video-style second presets
- [ ] Define safe technical maximum after benchmarking final model/GPU
- [ ] Support long-form orchestration/continuation where required
- [ ] Generate/store first-class audio assets
- [ ] Display audio player
- [ ] Download action
- [ ] History support

## Lyrics / rhyme quality

- [ ] Strongly prefer lyrics that rhyme naturally and consistently
- [ ] Avoid robotic forced-rhyme output
- [ ] Use a structured rhyme scheme where appropriate
- [ ] Add lyric-quality review before final music generation where practical
- [ ] Improve/regenerate weak lines where practical
- [ ] Keep semantics/story coherence
- [ ] Do not promise 100% perfect rhyming

### Acceptance direction

The generated song should generally feel:

- natural
- catchy
- coherent
- intentionally structured
- noticeably better at rhyme consistency than an unreviewed one-pass lyric generation

---

# 12. Music Video / Audio-Driven Video

## Updated client behavior

**Final video duration should automatically match the complete song/audio duration.**

Examples:

```text
30 sec audio  -> 30 sec final music video
2 min audio   -> 2 min final music video
4 min audio   -> 4 min final music video
```

### Tasks

- [ ] Detect source-audio duration
- [ ] Set target final video duration automatically
- [ ] Analyze / segment long audio
- [ ] Generate required visual sections
- [ ] Preserve scene/style direction across sections
- [ ] Create transitions where required
- [ ] Assemble sections
- [ ] Synchronize final video to the full audio
- [ ] Preserve full audio track
- [ ] Hide technical chunking from the customer UI
- [ ] Provide long-job progress
- [ ] Persist final assembled video
- [ ] Test multi-minute inputs

### Quality warning / engineering control

Long music videos will consume significantly more GPU time than short clips. Queue/fair-use controls must be designed before public unlimited usage is enabled.

---

# 13. Long-Form Generation / Stitching Layer

Because the client wants source-length V2V and full-song music videos, Milestone 2 now explicitly requires a long-form orchestration layer.

- [ ] Source duration detection
- [ ] Segment planning
- [ ] Segment overlap strategy where useful
- [ ] Prompt/style continuity strategy
- [ ] Seed/reference continuity strategy where supported
- [ ] Automatic segment generation
- [ ] Retry failed individual segments
- [ ] FFmpeg concatenation/composition
- [ ] Audio preservation/synchronization
- [ ] Transition/join QA
- [ ] Final-duration validation
- [ ] Persistent lineage/metadata
- [ ] Cleanup of temporary segment files

---

# 14. Media Storage Integration

Current production-compatible storage foundation is already running.

M2/M3 follow-up:

- [x] S3-compatible storage adapter foundation
- [x] Signed upload support foundation
- [x] Public production storage domain
- [ ] Final user ownership enforcement after authentication
- [ ] Signed private downloads
- [ ] Production lifecycle/retention policy
- [ ] Large long-form output testing
- [ ] Final backup strategy

---

# 15. Generation History / Media Library

Foundation exists from M1.

M2/M3 completion items:

- [ ] Real AI results in history
- [ ] Long-form parent/segment lineage hidden or summarized appropriately
- [ ] Extension lineage
- [ ] Reuse settings
- [ ] Variations
- [ ] Search/filter final pass
- [ ] Authenticated user ownership
- [ ] Private signed result access

---

## Milestone 2 Acceptance Criteria — Updated

Milestone 2 can be considered complete when:

- [ ] Real GPU worker connects successfully
- [ ] LTX 2.5 or another approved final model is benchmarked and selected
- [ ] Text-to-Video works
- [ ] Image-to-Video works
- [ ] Video-to-Video works with automatic source-duration behavior
- [ ] V2V optional reference-image behavior is tested
- [ ] Video Extension provides 5 / 10 / 15 / 30 / 60 second outcomes
- [ ] Repeated extension chain works
- [ ] Music generation works
- [ ] Music generation accepts desired duration in minutes
- [ ] Lyric/rhyme-quality improvement flow works as designed
- [ ] Music Video works
- [ ] Music Video automatically targets full source-audio duration
- [ ] Long-form segmentation/chaining/stitching works where needed
- [ ] Supported-duration handling is technically honest
- [ ] Outputs persist outside the GPU instance
- [ ] Real progress is reflected in the frontend
- [ ] Raw backend/model errors are not shown to customers
- [ ] Quality tuning has been performed against the client's premium/Seedance-like target
- [ ] GPU performance/VRAM/runtime benchmark is documented

---

# 16. MILESTONE 3 — Subscription, Testing & Deployment

## Commercial Value

**$360 — 30% of original project**

## Current State

# **PARTIALLY ADVANCED EARLY**

Significant production infrastructure was completed ahead of the original schedule during M1 deployment. Authentication, billing and final production hardening remain future M3 work.

---

# 17. Authentication & User Accounts

- [ ] User registration
- [ ] User login
- [ ] User logout
- [ ] Secure password storage
- [ ] Session management
- [ ] Protected routes
- [ ] Profile/settings
- [ ] Password recovery
- [ ] Ownership authorization
- [ ] User-specific job/media isolation

---

# 18. Subscription System

Initial business direction remains approximately **$70/month all-access**, subject to final client confirmation and billing implementation.

- [ ] Plan model
- [ ] Subscription model
- [ ] Final billing provider decision
- [ ] Checkout
- [ ] Webhook verification/idempotency
- [ ] Renewal/cancel/failure state
- [ ] Subscription management UI
- [ ] Paid-workflow entitlement
- [ ] Billing logs

---

# 19. Fair-Use / GPU Protection

Required before a public "unlimited" service is fully enabled:

- [ ] Maximum concurrent jobs per user
- [ ] Queue limits
- [ ] Rate limiting
- [ ] Long-form generation controls
- [ ] Abuse protection
- [ ] Retry limits
- [ ] Worker saturation controls
- [ ] Usage/GPU-time tracking
- [ ] Cost observability

---

# 20. Security Hardening

Already completed in the production foundation:

- [x] HTTPS on main domain
- [x] HTTPS on storage domain
- [x] Public DB not exposed
- [x] Public Redis not exposed
- [x] Web/API/MinIO host bindings restricted appropriately
- [x] Public internal-worker API route blocked at Nginx
- [x] Production secrets kept outside source control

Still required:

- [ ] Auth/session security
- [ ] Ownership enforcement
- [ ] Private signed downloads
- [ ] Rate limiting
- [ ] Billing webhook verification
- [ ] Full security regression pass
- [ ] Final production log sanitization

---

# 21. Frontend Finalization / Brand Polish

- [ ] Final ZolexAI logo integration
- [ ] Latest mobile first-fold reference implementation
- [ ] Keep strong black/neon-lime brand
- [ ] Prominent logo in opening view
- [ ] Keep **CREATE THE IMPOSSIBLE.** hero treatment
- [ ] Change the three-step section heading to **IMAGINE IT. GENERATE IT. GO VIRAL.**
- [ ] Remove preview/development copy before final launch
- [ ] Final login/register screens
- [ ] Subscription UI
- [ ] Settings/profile final pass
- [ ] Empty/error/loading states
- [ ] Responsive final pass
- [ ] Accessibility final pass
- [ ] Cross-browser final pass

---

# 22. Production Deployment — Current Actual State

Already completed:

- [x] Hostinger VPS audited
- [x] Docker installed
- [x] Production deployment user created
- [x] GitHub read-only deploy key configured
- [x] Repository cloned on server
- [x] Strong production environment secrets configured
- [x] PostgreSQL container deployed
- [x] Redis container deployed
- [x] MinIO deployed
- [x] API deployed
- [x] Worker deployed in M1 mock mode
- [x] Web frontend deployed
- [x] Database migration applied
- [x] DNS configured
- [x] CloudPanel reverse proxy configured
- [x] `zolexai.com` HTTPS configured
- [x] `www.zolexai.com` redirect configured
- [x] `/api/v1/` routed to FastAPI
- [x] `/api/v1/internal/` blocked publicly
- [x] `storage.zolexai.com` reverse proxy configured
- [x] Storage HTTPS configured
- [x] Large-upload proxy settings configured
- [x] Main API production health checked
- [x] Storage production health checked
- [x] Production deployment runbook created

Still required for final production:

- [ ] Real GPU worker
- [ ] Real generation workflows
- [ ] Authentication
- [ ] Billing
- [ ] Final long-form workload testing
- [ ] Final performance/load sanity
- [ ] Production backup/recovery drill
- [ ] Final handover

---

# 23. Testing — Updated

## M1 completed

- [x] API unit/integration testing
- [x] Worker tests
- [x] Frontend build/lint/typecheck
- [x] Mock browser E2E
- [x] Production web health
- [x] Production API health
- [x] Storage health
- [x] Internal worker-route public block
- [x] WWW redirect

## M2 required

- [ ] LTX 2.5 benchmark
- [ ] Real T2V
- [ ] Real I2V
- [ ] Real V2V short input
- [ ] Real V2V long input
- [ ] V2V source-duration validation
- [ ] V2V reference-image test
- [ ] Extension 5s
- [ ] Extension 10s
- [ ] Extension 15s
- [ ] Extension 30s
- [ ] Extension 60s
- [ ] Repeated extension continuity
- [ ] Music custom-duration generation
- [ ] Lyric-rhyme quality
- [ ] Music Video short audio
- [ ] Music Video multi-minute audio
- [ ] Audio/video synchronization
- [ ] Segment retry/recovery
- [ ] Final stitched-duration verification

---

# 24. Key Dependencies & Decisions

| Dependency | State | Owner / Action |
|---|---|---|
| Client visual direction | Approved with new revisions | Developer implements latest revision |
| Final logo integration | Pending | Developer |
| LTX 2.5 technical evaluation | Pending M2 | Developer |
| Development GPU | Client ready to provide when requested | Client / Developer |
| Final production GPU strategy | Pending | Client / Developer |
| Billing provider | Pending | Client / Developer |
| Long-form performance limits | Must be benchmarked | Developer |
| Music model final selection | Pending | Developer |
| Model/license compliance | Must be verified before production real-AI launch | Developer |

---

# 25. Active Risks

## R-01 — Long-form generation cost/time

**Risk:** Multi-minute V2V/music-video jobs can require many GPU inference segments.

**Mitigation:** segmentation, queue controls, per-user concurrency, benchmark-driven limits and progress reporting.

## R-02 — Continuity across generated video segments

**Risk:** visual jumps between sections.

**Mitigation:** overlap/reference frames, consistent prompting/seeds/references where supported, transition/post-processing and QA.

## R-03 — "60 seconds" may not be native

**Risk:** model may only generate shorter windows.

**Mitigation:** treat 60 seconds as a ZolexAI user outcome via chained extensions where needed; do not claim native single-pass support.

## R-04 — Seedance-like expectation

**Risk:** client expects premium proprietary-model quality.

**Mitigation:** benchmark/tune LTX 2.5, optimize workflow/prompting/post-processing and document practical limitations.

## R-05 — "Unlimited" commercial plan

**Risk:** unbounded multi-minute jobs can create excessive GPU cost.

**Mitigation:** hidden fair-use/concurrency/queue controls.

## R-06 — Music rhyme quality

**Risk:** forced or weak rhymes reduce perceived song quality.

**Mitigation:** lyric planning + quality/rewrite pass before final audio generation.

---

# 26. Change Request / Scope Clarification Log

| ID | Request | Type | Current Handling |
|---|---|---|---|
| CR-001 | Replace "From idea to output in three steps" with "IMAGINE IT. GENERATE IT. GO VIRAL." | UI revision | Add to frontend polish |
| CR-002 | Latest first-fold/mobile hero reference with prominent logo, black/neon-green style and Start Creating CTA | UI revision | Add to frontend polish |
| CR-003 | Final brand logo visibility/integration | UI revision | Pending |
| CR-004 | Evaluate LTX 2.5 | Technical scope clarification | M2 |
| CR-005 | Aim for Seedance-like premium quality | Quality target | M2 tuning |
| CR-006 | V2V duration automatic from source | Workflow behavior refinement | M2 |
| CR-007 | Music Video duration automatic from full song | Workflow behavior refinement | M2 |
| CR-008 | Extension 5/10/15/30/60 seconds | Workflow requirement | M2 |
| CR-009 | Music generation duration in minutes | Workflow requirement | M2 |
| CR-010 | Stronger natural rhyme behavior / lyric QA | Workflow-quality requirement | M2 |
| CR-011 | Client provides GPU when needed | Dependency confirmation | M2 |
| CR-012 | Client stated additional $200 | Commercial note | Do not change formal total until confirmed |

These entries are recorded so the team does not lose client decisions during implementation.

---

# 27. Progress Snapshot — 12 August 2026

## Completed

- Milestone 1 core application architecture
- Next.js frontend foundation
- FastAPI backend
- PostgreSQL / Redis / MinIO
- Workflow registry
- Generation-job architecture
- Mock worker
- SSE progress
- Media/history foundation
- M1 QA
- Production VPS setup
- Production Docker stack
- DNS
- SSL
- Main/storage reverse proxies
- Public internal-worker route protection
- Deployment documentation

## In progress / next

- Latest logo/marketing UI revisions
- Freeze updated M2 requirements
- Prepare LTX 2.5 evaluation plan
- Request GPU when model integration starts

## Blocked / dependency

- Real AI generation requires GPU/model environment
- Billing provider not yet selected
- Final real-AI scale/cost limits require benchmarking

---

# 28. Milestone Sign-Off

## Milestone 1 — $360

```text
[x] Core deliverables completed
[x] Internal testing completed
[x] Production foundation deployed
[x] Client live link provided
[x] Documentation created
[ ] Latest logo/copy visual revision applied
[ ] Commercial milestone approval/closure recorded as applicable
```

## Milestone 2 — $480

```text
[ ] GPU acquired
[ ] LTX 2.5 benchmark completed
[ ] Real workflows integrated
[ ] Long-duration orchestration validated
[ ] Music duration/rhyme behavior validated
[ ] Client workflow demo completed
[ ] Revisions completed
[ ] Milestone approved
```

## Milestone 3 — $360

```text
[ ] Authentication complete
[ ] Billing complete
[ ] Fair-use controls complete
[x] Core production web/API/storage deployment completed early
[ ] Real GPU production deployment complete
[ ] Final testing complete
[ ] Final documentation/handover complete
[ ] Milestone approved
```

---

# 29. Final Delivery Definition

The full project is complete when:

```text
Approved / polished ZolexAI UI
        +
Final logo / branding
        +
Real frontend
        +
Backend/API
        +
Database / Redis
        +
Generation queue
        +
Persistent object storage
        +
Real GPU worker
        +
Final approved video model/workflows
        +
Automatic V2V duration orchestration
        +
Video Extension 5/10/15/30/60
        +
Music custom-duration generation
        +
Improved lyric/rhyme quality
        +
Full-song Music Video generation
        +
Authentication
        +
Subscription / fair-use
        +
Testing
        +
Production deployment
        +
Documentation / handover
```

are functioning together as one platform.

---

# 30. Final Current Milestone Summary

## Milestone 1 — $360
**Platform & Core Setup — COMPLETE**

ZolexAI is a real modular application with frontend, backend, persistence, queue/job architecture, mock generation, object storage and a live HTTPS production foundation.

## Milestone 2 — $480
**AI Workflows & Generation Features — NEXT ACTIVE MILESTONE**

Main focus:

- LTX 2.5 evaluation
- premium quality tuning
- Text-to-Video
- Image-to-Video
- automatic-duration Video-to-Video
- 5/10/15/30/60 Video Extension
- custom-minute Music generation
- rhyme-quality handling
- automatic full-duration Music Video
- long-form segmentation/stitching
- GPU benchmarking

## Milestone 3 — $360
**Subscription, Final QA & Handover — PARTIALLY ADVANCED EARLY**

Core production hosting/routing/storage/SSL is already deployed. Authentication, billing, fair-use, real GPU production readiness, final testing and handover remain.

---

**End of updated working milestone document — 12 August 2026**
