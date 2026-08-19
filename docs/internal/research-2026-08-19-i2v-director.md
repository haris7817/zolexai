# Image-to-Video Director mode: source-anchored planning

**Date:** 2026-08-19 · **Status:** implemented, unit-tested; NOT committed, NOT
deployed, NOT GPU-measured
**Scope rule:** Standard Image to Video stays byte-identical. The mode is
opt-in per request, exactly as it is on Text to Video. No other workflow gains
a prompt mode.

## The product principle

**The uploaded image defines WHO and WHAT. Director mode defines WHAT
HAPPENS.** Every design decision below is that sentence applied somewhere.

## What already existed (audit, 19 Aug 2026)

Less was missing than the feature description implies:

- `LtxAdapter._run_generation` serves BOTH text-to-video and image-to-video,
  and the T2V Director integration lives inside it. The only worker-side gate
  was `wants_director()` hard-scoping to `text-to-video`.
- **The original upload is already the identity anchor across chained
  sections**, in standard I2V and now in Director I2V, by the same code: pass
  1 pins the still at frame 0 / strength 1.0; every later pass pins the
  predecessor's final frame at 0 / 1.0 AND the original upload mid-window at
  strength 0.2 (`execution.i2v_reference_strength`). Nothing new was built
  for it; the director path simply flows through it.
- Anti-replay is by construction and transfers whole: one global plan before
  any section renders, events bucketed into windows by midpoint, each caption
  carrying only its own events plus a continue-mid-scene preamble.
- The API (`validate_request`) and the ENTIRE web form layer key generically
  on `settings.prompt_modes` from the workflow YAML. Enabling the control on
  image-to-video is one YAML line; the toggle, idea field, language selector,
  payload mapping and catalog projection all follow with no code change.

## The genuinely new problem: a text planner and a photograph

The planner cannot see the image, and its T2V habit — inventing concrete
appearances, which is CORRECT there ("identity stated early and repeated" is
the measured anti-drift lever when text is the only identity channel) — is
exactly wrong here. A caption describing a red coat over a photograph of a
blue one is drift pressure written into the prompt.

So an I2V plan is **source-anchored** (`DirectorPlan.source_anchored`):

- The brief gains a SOURCE IMAGE MODE register (appended for anchored
  requests only; the T2V brief is byte-identical, pinned by test): cast
  exactly whom the idea names, NEVER invent visible detail, `appearance`
  carries only idea-stated facts and may be `""`, action moves FORWARD from
  the photographed moment, camera stays inside the space the photograph
  establishes.
- The validator relaxes exactly one rule for anchored plans: empty
  `appearance` is legal. Everything else — verbatim user dialogue, speech
  budget, speaker ownership, pacing/vocabulary corrections — applies
  unchanged.
- The compiler ties identity to the frame instead of to prose: the anchored
  cast sentence is "…are already present in the opening frame, and they keep
  exactly the appearance that frame shows", and the per-section continuity
  block becomes "…keep exactly the same faces, clothing, hair, colours and
  voices **they have in the first frame**". In sections ≥2 "the first frame"
  resolves to that pass's own conditioned frame 0 — the predecessor's final
  image — which carries the same identity, so the sentence stays true at
  every seam. Idea-stated appearance (a red dress the user typed) still rides
  along: the refusal is of INVENTED detail, not of the user's own facts.

## Seeing the image: wired, OFF by default

The right grounding is measured facts about the photograph, and the stack has
no vision capability anywhere (both planner providers are text-only; checked).
`worker/director/vision.py` + `scripts/director_image_facts.py` add the seam:
the same subprocess pattern as the planner, asking the local checkpoint (via
`AutoModelForImageTextToText`) to state PEOPLE / OBJECTS / SETTING /
COMPOSITION, which is injected into the planner's user prompt as a
PHOTOGRAPH FACTS block the anchored rules treat as true.

`DIRECTOR_VISION_ENABLED` defaults to **false** — whether `gemma-4-e2b-it`
accepts image input is a measurement nobody has made, and this codebase does
not ship unmeasured model paths as defaults (the guided tier's posture).
Every failure mode — disabled, unstaged image, text-only checkpoint, timeout,
garbage — degrades to "no facts" with a log line, never to a failed job: the
anchored brief is written to work blind (then the planner simply may not
describe what it cannot see). Turning it on is a GPU-box measurement session,
then one env var.

## Wiring (one line each)

- `image-to-video.yaml`: `settings.prompt_modes: true`.
- `wants_director`: workflow set {text-to-video, image-to-video}; a new
  `source_anchored(job)` keys the register on the workflow, mirroring the
  adapter's dispatch rule.
- `ltx.py`: **zero changes.** The existing director branch and the existing
  I2V conditioning compose.
- Web: two strings — the Director hint and idea placeholder say "your image
  defines who and what is in the scene" on image-fed workflows, because
  promising "AI will create the scene" over a photo the user chose would be
  promising the wrong thing.

## What is deliberately NOT built

- No vision by default (above). No hosted vision at all — Cerebras hosts no
  image models.
- No plan preview/edit UI; no guided-tier coupling; nothing on extend /
  video-to-video / music video (their prompts describe continuations and
  restyles, not scenes to invent).
- No I2V-specific pacing model: the speech budget, line targets and pause
  cues are the T2V measurements and carry over unchanged until an I2V render
  says otherwise.

## Honest limits / what a GPU session must answer

1. Does the anchored register hold on the real model — does the planner
   actually leave appearances empty rather than inventing, across many ideas?
   (The validator cannot detect a plausible invention; only refusal-by-brief
   prevents it.)
2. Does "keep exactly the appearance they have in the first frame" hold
   identity as well as the T2V appearance-restating captions did — especially
   at section seams, where the 0.2 reference still is the only tie back to
   the upload?
3. Is `AutoModelForImageTextToText` viable on the on-box checkpoint, what
   does it cost in wall clock, and are its facts accurate enough to inject as
   "treat these as true"?
4. Camera behaviour: does the anchored brief's stay-inside-the-frame rule
   prevent the impossible-reveal failure, or does 2.5's multi-shot habit need
   a harder rule?
