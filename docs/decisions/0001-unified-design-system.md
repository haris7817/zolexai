# ADR 0001 — Unify three screens onto one ZolexAI design system

**Status:** Accepted · **Phase:** PRE-M1 · **Date:** 2026-08-10

## Context

The approved design lives in Claude Design project `4d46333d-680e-47ff-8334-4ab992fd7360`
("Three target screens ready") and contains three screens plus superseded duplicates:

| File | Role |
|---|---|
| `Video Workspace.dc.html` | Newest. Full `:root` token block, SVG icons, 4-mode responsive system |
| `Creator Dashboard.dc.html` | Older. Hardcoded hex, Unicode-glyph icons, fixed 240px sidebar, no responsive |
| `Landing Page.dc.html` | Older. Hardcoded hex, Unicode-glyph icons, no responsive |
| `Video Workspace v1.dc.html`, `Video Workspace export.html`, `ZolexAI Video Workspace.html` | Superseded duplicates |

Landing and Dashboard predate the Workspace's design system. Shipped as-is, marketing and product
would not look like the same company.

## Decisions

### 1. The Video Workspace `:root` block is the single source of truth

| Token | Landing + Dashboard (old) | Canonical |
|---|---|---|
| Page background | `#0B0A14` | `--zx-bg-primary: #0D0C13` |
| Nav / section background | `#0E0C1A` | `--zx-bg-secondary: #111017` |
| Card surface | `#110E1E` | `--zx-surface: #17161F` |
| Text primary | `#F5F3FF` | `--zx-text-primary: #F2F0FA` |
| Text secondary | `#A29BC4` | `--zx-text-secondary: #B0ABC6` |
| Text muted | `#565073` / `#6F6893` | `--zx-text-muted: #847E9E` |
| Border | `rgba(167,139,250,0.1–0.25)` | `--zx-border: rgba(255,255,255,0.08)` |

The old palette is warmer and purple-tinted; the canonical one is cooler and neutral, letting the
purple brand accent carry the identity rather than the greys competing with it.

### 2. Architecture doc §8 is stale — the design wins

The blueprint's §8 token list disagrees with the approved design: it gives
`--zx-surface: #15121F` (design says `#17161F`) and omits `--zx-surface-hover`,
`--zx-primary-hover` and `--zx-shadow-cta` entirely.

Non-negotiable rule #13 is "do not rewrite the approved design during implementation."
**§8 should be corrected to match the design**, not the reverse.

### 3. One icon system — Lucide SVG everywhere

Landing and Dashboard used Unicode glyphs as product icons: `✦ ◈ ⟲ ⇢ ♪ ▶ ⊞ ◫ ▤ ◇ ⚙ ⌂ ⏻ ?`.

Rejected because glyph coverage varies by OS and font: `⏻`, `◫`, `▤` and `⇢` fall back
inconsistently on Windows and Android, so the same build looks broken on some client machines and
fine on others. They also cannot inherit stroke weight, so they never optically match the
Workspace's 1.8px-stroke SVGs.

Resolution: `lucide-react` throughout. One glyph has no Lucide equivalent — the "extend" mark
(`M2 12h17 / m15 8 4 4-4 4 / M22 5v14`) — kept as a local `ExtendIcon` at matching stroke weight.

### 4. Breakpoints follow the design, not the blueprint

Architecture doc §10 says desktop begins at 1200px; the Workspace implements 1440px, with
1024 and 768 beneath it. **The implemented values win.** Named `tablet` / `laptop` / `desktop`
so code reads in the same vocabulary as the design.

Sidebar 224 → 200 → 64px rail → drawer. Settings panel 320 → 292px inline → 340px right drawer →
78vh bottom sheet.

### 5. "AI Editing Tools" is removed from the demo

Landing's design shows seven tool cards; the frozen scope
(milestones §8.1) contains six. The seventh, "AI Editing Tools", is a design placeholder.

Showing it — even labelled "coming soon" — invites the client to approve a seventh tool as
committed. It is removed. Landing renders six cards from the same registry as the sidebar,
All Tools and Dashboard, so all four surfaces agree by construction.

### 6. Naming reconciliation

Dashboard's sidebar said **"Video Generator"** where the Workspace said **"Text to Video"**.
The Workspace name wins; the registry is the only place either name is defined.

## Consequences

- Landing and Dashboard shift slightly cooler and more neutral. Subtle, but visible side-by-side
  with the original design files — worth mentioning when presenting, so it reads as intentional.
- Both gain full responsive behaviour they never had.
- Every screen depends on `apps/web/src/styles/tokens.css`; no component defines a colour.
- Because all four tool surfaces read one registry, adding a workflow in M2 touches one file
  and updates every screen — the outcome architecture doc §2.5 asks for.
