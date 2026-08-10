# ZolexAI

Premium AI media-generation platform — video, music and visual content from one creator workspace.

> ## Current phase: **PRE-M1 — Client UI/UX Approval**
>
> This repository currently contains **only the frontend UI/UX demo** (`apps/web`), built entirely on
> mock data for client design approval.
>
> **AI generation is simulated.** There is no backend, no database, no queue, no object storage,
> no authentication, no payments and no GPU integration. `apps/api` and `apps/worker` are empty
> placeholders that begin at Milestone 1.
>
> **Milestone 1 is blocked until PREUI-20** (client approval / design freeze).

---

## Quick start

Requires **Node ≥ 20.9** (developed on 20.18.0) and npm.

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

| Command | What it does |
|---|---|
| `npm run dev` | Start the demo on :3000 |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint |
| `npm run typecheck` | TypeScript, no emit |

## Demo routes

| Route | Screen |
|---|---|
| `/` | Landing page |
| `/login`, `/register` | Auth visual direction (no real auth) |
| `/app` | Creator Dashboard |
| `/app/create/[workflowId]` | Video Workspace — the primary screen |
| `/app/tools` | All Tools |
| `/app/generations` | Generation history |
| `/app/media` | Media Library |
| `/app/subscription` | Subscription |
| `/app/settings` | Settings |

Valid `workflowId` values: `text-to-video`, `image-to-video`, `video-to-video`,
`extend-video`, `music`, `music-video`.

## Repository layout

```
zolexai/
├── apps/
│   ├── web/                  Next.js frontend  ← the only implemented app
│   ├── api/                  FastAPI backend   (M1 — placeholder)
│   └── worker/               GPU worker        (M2 — placeholder)
├── packages/                 shared libs       (M1+ — placeholders)
├── workflow-definitions/     public workflow metadata (YAML)
├── infrastructure/           docker / nginx / compose  (M1+ — placeholders)
└── docs/                     architecture, milestones, demo guide, decisions
```

## Architecture rules already enforced in the demo

Even with no backend, the frontend honours the non-negotiable controls so Milestone 1 needs no redesign:

- **Workflow-driven.** All six workflows come from one registry
  (`apps/web/src/features/workflows/registry.ts`). No per-workflow pages, no hard-coded
  duration or settings lists.
- **No provider leakage.** No model, provider or infrastructure name appears anywhere in the UI.
  `workflow-definitions/*.yaml` carries public metadata only — never an `execution:` block.
- **One design system.** Every screen reads `apps/web/src/styles/tokens.css`.
- **Swap-ready generation.** The simulated pipeline is isolated in
  `apps/web/src/features/generation/mockPipeline.ts` so M1 replaces it with SSE in one file.
- **Quarantined mock data.** Everything fake lives in `apps/web/src/mocks/` — one directory to
  delete when real data arrives.

## Documentation

See [`docs/`](./docs). Start with [`docs/README.md`](./docs/README.md).
