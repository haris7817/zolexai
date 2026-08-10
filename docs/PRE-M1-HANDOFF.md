# PRE-M1 — Client UI/UX Demo · Handoff Report

**Phase:** PRE-M1 (gate G0) · **Status:** Ready for PREUI-17 deployment
**Milestone 1 is BLOCKED** until PREUI-20 (client approval / design freeze).

---

## 1. Completed demo screens

| # | Screen | Route | PREUI | State |
|---|---|---|---|---|
| 1 | Landing page | `/` | 03 | Demo ready |
| 2 | Sign in | `/login` | 11 | Visual direction |
| 3 | Create account | `/register` | 11 | Visual direction |
| 4 | Creator Dashboard | `/app` | 04 | Demo ready |
| 5 | Video Workspace | `/app/create/[workflowId]` | 05 | Demo ready |
| 6 | All Tools | `/app/tools` | 06 | Demo ready |
| 7 | Generations | `/app/generations` | 07 | Demo ready |
| 8 | Generation detail | `/app/generations/[id]` | 07 | Demo ready |
| 9 | Media Library | `/app/media` | 08 | Demo ready |
| 10 | Subscription | `/app/subscription` | 09 | Demo ready |
| 11 | Settings | `/app/settings` | 10 | Demo ready |

**31 routes** build statically, including all six workflow workspaces and twelve
generation-detail pages.

All six frozen-scope workflows are live and visibly different from one another:
Text to Video · Image to Video · Video to Video · Extend Video · Music · Music Video.

### Design-system work (PREUI-01 / 02)

- One token file (`apps/web/src/styles/tokens.css`) drives every screen.
  Landing and Dashboard were retro-fitted off their older `#0B0A14 / #110E1E`
  palette onto the approved Workspace tokens.
- All Unicode glyph icons (`✦ ◈ ⟲ ⇢ ♪ ▶ ⊞ ◫ ▤ ◇ ⚙ ⌂ ⏻`) replaced with Lucide
  SVG plus one custom `ExtendIcon`. **Verified zero glyph icons remain** in
  rendered code.
- Landing's seventh card, **"AI Editing Tools", was removed** — it is not in the
  frozen scope, and showing it invites the client to approve a tool nobody has
  committed to build.
- Rationale recorded in `docs/decisions/0001-unified-design-system.md`.

---

## 2. Responsive QA results

Automated across **12 routes × 6 widths = 72 page renders**
(`npm run qa:responsive`, Chromium).

### Horizontal overflow — the hard requirement

| Width | Result |
|---|---|
| 1440 | CLEAN |
| **1366** | CLEAN |
| 1024 | CLEAN |
| 768 | CLEAN |
| 430 | CLEAN |
| **390** | CLEAN |

Zero horizontal overflow on any route at any width.

### Shell mode — measured, not assumed

| Width | Sidebar | Settings panel | Panel role |
|---|---|---|---|
| 1440 | 224px | 320px inline | *(not a dialog)* |
| 1366 | 200px | 292px inline | *(not a dialog)* |
| 1024 | 200px | 292px inline | *(not a dialog)* |
| 768 | 64px icon rail | fixed right drawer | `dialog` |
| 430 | drawer | fixed bottom sheet | `dialog` |
| 390 | drawer | fixed bottom sheet | `dialog` |

Matches the approved design exactly. Zero console errors at every width.

### Demo-flow QA — **41/41 checks pass** (`npm run qa:flow`)

Covers the guide §7 walkthrough end to end:

- Workspace opens empty; Generate disabled until a prompt exists
- "Use example prompt" fills the scripted prompt; default duration stays 5s
- Pipeline runs Queued → Preparing → Generating → Finalizing → result
- Second concurrent job; job strip switching; active-jobs indicator
- **Settings preservation:** Text to Video @ 15s → Image to Video → falls back to 5s
- **Capability-driven actions:** Music shows Download / Reuse / Variation and
  **no Extend**; video results show all four
- Accessibility: one `<h1>`, `aria-pressed` on chips, `aria-current` on nav,
  labelled prompt, Escape closes the sheet, body scroll locks and releases

### Four real bugs found and fixed during QA

1. **Infinite render loop (React #185)** — a zustand selector returned a new
   array on every read, so `useSyncExternalStore` never saw a stable snapshot
   and the app crashed on every `/app` route.
2. **Every `<a>` rendered accent purple** — unlayered base CSS beats Tailwind's
   layered utilities regardless of specificity, so `a { color: accent }` was
   silently overriding every `text-zx-*` class. Base rules moved into `@layer base`.
3. **Primary CTAs had no fill** — `bg-[var(--gradient)]` emits `background-color`,
   which discards a gradient value. Switched to `bg-[image:var(--gradient)]`.
   The Generate button, "Start Creating" and the "Z" logo tile were all unfilled.
4. **Invalid ARIA** — `aria-pressed` on `role="listitem"` in the job strip
   (inherited from the source design); roles now split correctly.

---

## 3. Remaining visual issues

No blocking defects. Deliberate, worth knowing before you present:

| Item | Why | Action |
|---|---|---|
| All media is a **gradient placeholder** | No real generated content exists yet | Expected. If you want richer visuals for the client, supply sample stills/clips and I'll drop them into `src/mocks/` |
| **"Use example prompt"** is visible in the panel | Presenter convenience for the 2–5 min walkthrough | Behind `featureFlags.demoHelpers` — one boolean removes it |
| Landing footer links are **plain text, not links** | Those pages don't exist; dead links look worse than plain labels | Intentional |
| Audio results show a **decorative waveform** | No audio file to analyse | Expected |
| Light mode shown as **"coming later"** in Settings | Not in scope | Confirm with client if they expect it |
| `npm audit`: 3 high advisories | Both inside Next 15's bundled build-time deps (`postcss`, `sharp`); npm's only fix is a major bump to Next 16 | Not exploitable here — the demo takes no user input, uploads nothing and optimises no remote images. **Re-evaluate at M1**, not mid-demo |

---

## 4. Known mock-only functionality

**Nothing in this demo touches a backend.** Everything below is presentation only.

| Area | Reality | Arrives at |
|---|---|---|
| AI generation | **Simulated timer** (≈6.6s). No model, no GPU, no inference. | M2.08+ |
| Generation history | Static array in `src/mocks/generations.ts` | M3.08 |
| Media library | Static array; **dropzone accepts no files** | M3.05 / M3.07 |
| Search / filters / tabs | Client-side `useState`, **nothing persists** | M1/M3 |
| Download buttons | Inert no-ops | M3.06 |
| Login / Register | **No auth.** Submit routes to `/app` | M3.02 |
| Subscription | Plan, invoices, payment method all mock. **No payment integration** | M3.13 |
| Settings | Nothing saves | M3.18 |
| User "Maya Adler" | Fixture | M3.02 |

Refresh the page and every generation disappears — state is in memory only.

All mock data is quarantined in **`apps/web/src/mocks/`** — one directory to
delete when real data arrives. The simulated pipeline is isolated in
**`src/features/generation/mockPipeline.ts`**, so M1.22 swaps it for SSE in one file.

### Demo disclosure (PREUI-15)

Three muted placements — no modal, no banner, no alarm colours:

1. **"UI Preview"** pill in the sidebar footer and mobile drawer
2. *"Simulated preview — AI generation is not connected in this demo."* directly
   beneath the progress bar and the result — the load-bearing one, sitting
   exactly where a fake progress bar could be mistaken for real inference
3. *"Interactive product preview for design review."* in the landing footer

---

## 5. Running the demo locally

Requires **Node ≥ 20.9** (built on 20.18.0).

```bash
cd e:\zolexai
npm install
npm run dev
```

Open <http://localhost:3000>.

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server on :3000 |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint (zero warnings) |
| `npm run typecheck` | TypeScript (zero errors) |

QA harnesses — start `npm run start` first, then from `apps/web`:

```bash
npm run qa:responsive   # 12 routes x 6 widths, screenshots + metrics.json
npm run qa:flow         # 41-check demo walkthrough
```

> **Build gotcha:** don't pipe `next build` into `head` — the early pipe close
> sends SIGPIPE and leaves a half-written `.next` with no `BUILD_ID`, which then
> fails at `next start` or serves broken chunks. Use `> build.log` or `| tail`.
> This cost time during QA; noted so it doesn't cost it again.

---

## 6. Sharing the demo URL (PREUI-17)

Guide §5 requires: *click link → interact*. No ZIP, no install, no source, no commands.

Every route is Static/SSG, so the recommended approach is to **build locally and
deploy the finished folder**. That removes all build risk from the host.

### Why prebuilt rather than build-on-host

`apps/web/package.json` keeps `playwright` as a devDependency for the QA
harnesses, and its install step downloads browsers. A host that runs
`npm install && npm run build` will therefore be slow and may fail. Deploying the
prebuilt output sidesteps it entirely. (If you'd rather let the host build, set
`PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` in its environment.)

### Step 1 — build

```bash
cd e:\zolexai\apps\web
npm run build:static
```

Produces `apps/web/out` — 29 pages, ~2.6 MB. **Verified fully interactive** from
this folder: FAQ accordion, example-prompt helper, the complete simulated
pipeline through to a result, client-side navigation between workflows, and
Music correctly hiding its quality section.

### Step 2 — publish `out/`

**Vercel** (recommended — free, keeps the door open for `demo.zolexai.com`):

```bash
cd out
npx vercel            # first run: log in + link the project
npx vercel --prod     # returns the public URL
```

Vercel treats `out/` as a static site, so there is no install and no build.

**Netlify Drop** (fastest, no account needed to start): open
<https://app.netlify.com/drop> and drag the `out` folder in. Instant URL.

**Cloudflare Pages**: `npx wrangler pages deploy out`.

### Step 3 — subdomain (guide §5)

Point a dedicated subdomain at the deployment, kept separate from the eventual
production URL:

```
demo.zolexai.com
```

### Notes

- The demo is `robots: noindex, nofollow`, so it cannot be indexed while the
  real product doesn't exist.
- No environment variables are required — there is no backend.
- **Local check before serving `out/` yourself:** use `npx serve out`, not
  `python -m http.server`. Python's server is single-threaded and stalls on
  Next's parallel chunk requests, which silently breaks hydration and makes a
  perfectly good build look dead.
- Re-run `npm run build:static` after any revision; the folder is not live-linked.

---

## 7. Confirmation — no M1 work has started

**Confirmed. No Milestone 1, 2 or 3 implementation exists.**

Not started, not scaffolded, not stubbed with logic:

FastAPI · PostgreSQL · SQLAlchemy · Alembic · Redis · MinIO / object storage ·
Docker / Docker Compose · generation APIs · job queue · worker API · mock worker ·
SSE · authentication · sessions · billing · payments · Vast.ai · GPU · CUDA ·
PyTorch · ComfyUI · any model or provider integration.

`apps/api` and `apps/worker` contain only `.gitkeep` and a README saying when
they begin. `infrastructure/` has no compose file.

**Verified:** no provider, model or infrastructure name appears anywhere in the
frontend — no LTX, ComfyUI, Vast.ai, PyTorch, CUDA or model identifiers.
`workflow-definitions/*.yaml` carries public metadata only, with no `execution:`
block (architecture §11; guards T-API-01 and risk R-10).

### DESIGNED ≠ IMPLEMENTED

A screen reaching DEMO READY does **not** advance any milestone task:

| Demo screen | Does NOT complete |
|---|---|
| Media Library | M3.05, M3.07 |
| Generations / detail | M3.08, M3.09 |
| Subscription | M3.10, M3.13, M3.15 |
| Settings | M3.18 |
| Login / Register | M3.02, M3.03 |
| Workspace | M1.07 is *partly* satisfied visually, but M1.18–M1.22 (real jobs, worker, SSE) are untouched |

### Tracker updates to make

- **G0.1 / G0.2** → COMPLETE once the link is sent, with the URL as Evidence
- **CR-001** → log the Landing Page as *Scope Clarification*: it appears in no
  M1/M2/M3 task list nor in §21's screen list, but is confirmed in scope
- **§11 Progress Update Log** → record that the five app screens shipped as
  interactive mockups ahead of M1.09's "route skeletons", pulling some M3 UI
  forward without completing those M3 tasks

---

## 8. Client-demo checklist (guide §6)

Run through this yourself **before** sending the link.

### Desktop

- [ ] Loads without visible errors
- [ ] Sidebar displays correctly
- [ ] Text to Video opens
- [ ] Image to Video opens
- [ ] Video to Video opens
- [ ] Extend Video opens
- [ ] Music opens
- [ ] Music Video opens
- [ ] Duration options change per workflow *(Music → 30s/60s/120s)*
- [ ] Aspect ratios change per workflow *(Music → section disappears)*
- [ ] Quality renders correctly *(Music → section disappears)*
- [ ] Generate button works
- [ ] Simulated progress runs
- [ ] Result appears
- [ ] Result actions display *(Music → no Extend)*
- [ ] Multiple jobs appear in the strip
- [ ] Selected job can be changed

### Mobile

- [ ] No horizontal page overflow
- [ ] Hamburger menu opens
- [ ] Navigation drawer closes properly
- [ ] Settings bottom sheet opens
- [ ] Controls are usable
- [ ] Generate button is reachable
- [ ] Result area fits the screen
- [ ] Job strip remains usable

### Browser

- [ ] Chrome
- [ ] Edge
- [ ] One other if available

### Demo disclosure

- [ ] "UI Preview" visible in the sidebar
- [ ] Simulation note visible under progress **and** result
- [ ] You have said, in writing, that generation is simulated

---

## Appendix A — Client message (guide §11)

> Hey bro, I've completed the main ZolexAI UI/UX direction and I want to get your
> feedback before I move deeper into the actual implementation.
>
> Here is the interactive demo:
>
> **[DEMO LINK]**
>
> This stage is focused on the design and user experience. The generation process
> is currently simulated so we can first finalize how ZolexAI should look and work
> for the user before I connect the real GPU and AI workflows.
>
> Please mainly check the overall design, colors, creator layout, workflow
> navigation and generation experience.
>
> If you see anything you want changed in the UI, let me know now and I can
> include it before we freeze the design and continue with the backend/GPU
> implementation.

### Shorter version (guide §12)

> Hey bro, the main ZolexAI UI is ready for review. I've made an interactive demo
> so you can test the design, workflow navigation and generation experience
> yourself:
>
> **[DEMO LINK]**
>
> The AI generation is simulated at this stage — this demo is mainly to approve
> the UI/UX before I connect the real backend and GPU workflows.
>
> Have a look and let me know if you want any changes in the design, colors or
> overall user experience before I freeze this direction and continue
> implementation.

---

## Appendix B — Walkthrough script (guide §7), annotated

Target length **2–5 minutes**. Every step verified working.

| Step | Do this | Notes for this build |
|---|---|---|
| 1. First impression | Open `/app/create/text-to-video` | Opens **empty** — natural creator state, Generate correctly disabled |
| 2. Workflow switching | Click through all six in the panel | Music is the strongest demo: aspect ratio **and** quality vanish, durations become 30/60/120s, empty state says "track" |
| 3. Text-to-Video | Click **"Use example prompt"**, then select **10s** | The prompt is the scripted one. Duration defaults to 5s deliberately, so clicking 10s *shows the control working* |
| 4. Progress | Watch Queued → Preparing → Generating → Finalizing | ≈6.6s total. **Say aloud that this is simulated** — the on-screen note says so too |
| 5. Multiple generations | Type another prompt, hit **Generate Another** | Job strip appears, sidebar shows "N generations running", click between jobs |
| 6. Result actions | Point at Download / Extend / Reuse Settings / Variation | Then switch to a **Music** result: Extend is gone. That's capability-driven, not hardcoded |
| 7. Input workflows | Open Image to Video, Video to Video, Extend Video | Each shows its own dropzone label (INPUT IMAGE / INPUT VIDEO / SOURCE VIDEO) |
| 8. Music | Open Music and Music Video | Audio is a first-class output, not a video with no picture |
| 9. Responsive | Resize, or open on a phone | 1366 and 390 are the two worth showing |

**Do not discuss** (guide §9): Redis, PostgreSQL, SSE, Docker, queues, worker
leasing, API schemas, model adapters, CUDA.

---

## Appendix C — Feedback record (guide §16)

Ask the guide §8 questions A–H. Record here; keep required changes separate from
future ideas (§10) so new features don't silently enter the current scope.

| # | Area | Approved | Minor changes | Major revision | Notes |
|---|---|---|---|---|---|
| A | Overall look | ☐ | ☐ | ☐ | |
| B | Creator layout | ☐ | ☐ | ☐ | |
| C | Colors / branding | ☐ | ☐ | ☐ | |
| D | Workflow navigation | ☐ | ☐ | ☐ | |
| E | Generation settings | ☐ | ☐ | ☐ | |
| F | Generation experience | ☐ | ☐ | ☐ | |
| G | Result experience | ☐ | ☐ | ☐ | |
| H | Mobile experience | ☐ | ☐ | ☐ | |

**Required changes** (must be done before freeze)

1.
2.
3.

**Future ideas** (NOT current scope — log as CRs)

1.
2.
3.

### Freeze criteria (guide §17)

Design is frozen when branding/colors, creator layout, navigation, workflow
presentation, settings panel, progress UX, result experience and mobile
direction are all agreed.

Closing question (guide §19):

> If the overall design, colors and creator experience look good to you, can I
> treat this UI direction as approved and continue the full implementation based
> on it?

**On approval → PREUI-20 → design freeze → M1.01 unblocks.**
