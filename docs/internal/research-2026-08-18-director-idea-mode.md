# Research: the "Director / Idea" workflow for Text to Video

**Date:** 2026-08-18 · **Status:** research complete; implemented, GPU-tested; NOT committed, NOT deployed
**Scope rule:** Text-to-Video only. Music Video keeps its audio-master
architecture untouched. The standard T2V path stays byte-identical unless the
user explicitly selects the new mode.

---

## 1. What "LTX Director / Idea mode" actually is — the claim, verified

The client described a workflow where a bare idea ("A detective argues with a
corrupt police chief") automatically becomes scene direction, characters,
dialogue, speaker turns, emotional delivery, timing and camera — then LTX
renders video *with* the spoken lines.

**Verdict: this is NOT a native LTX model or runtime feature.** Three
independent sources agree:

1. **The installed LTX 2.5 runtime** (`/workspace/ltx2-benchmark`, read on the
   box): no director/idea/planning module exists. The only "idea → richer
   output" mechanism is the prompt enhancer (`--enhance-prompt` +
   `--prompt-enhancer-gemma-root`): a separate generative Gemma instruct
   checkpoint rewrites the user's request into ONE chronological audio-visual
   caption (~150–220 words) in the model's training-caption style. Its system
   prompt (read verbatim on the box:
   `packages/ltx-core/.../prompts/gemma4_t2v_system_prompt.txt`) quotes
   dialogue inline "exactly, in the original language" with tone of voice —
   and forbids timestamps, labels and section headers. No timeline, no
   speaker management, no long-form awareness.
2. **The client's own reference stack** (`E:\Downloads\ltx-main`, LTX-2.3
   direct-Python — verified, zero ComfyUI): no Director. It has a
   `POST /prompt-tools/generate-dialogue` endpoint ("Generar diálogos") that
   turns an already-written visual prompt into `{speaker, text}` pairs — the
   implementation is absent from the drop. No timeline, no generated emotion
   cues (the per-line style field is user-typed), no camera planning. Their
   spoken dialogue is **not LTX speech at all**: lines are rendered by a
   VoxCPM TTS service, concatenated with 450 ms pauses, and fed to
   `a2vid` as conditioning — LTX lip-syncs to pre-rendered audio.
   Their `ltx23CameraDirector` compiles a user-chosen enum (dolly_in/jib/…),
   it does not plan.
3. **Official Lightricks sources** (docs.ltx.io, ltx.io blog guides, GitHub,
   HF cards): nothing named Director/Idea mode. The idea→script→scenes
   automation lives in **LTX Studio**, Lightricks' separate SaaS platform
   (which drives VEO 3, Nano Banana, Flux.2 *and* LTX-2) — not in the open
   model stack we run.

So the feature is a **planning layer ZolexAI builds**, sitting in front of the
existing generation pipeline. Native LTX contributes: joint audio-video
generation (speech included), officially endorsed dialogue prompting, and a
small Apache-2.0 instruct model that can do the planning locally.

## 2. What LTX 2.5 natively supports (evidence per claim)

- **Joint audio+video, speech included, on the DISTILLED tier.** Read in
  `ltx_pipelines/distilled.py`: video and audio latents are denoised jointly
  and both decoded — every plain T2V render already carries model-generated
  audio. (ZolexAI already requires it: `require_audio=True`, and a silent pass
  fails the job.) HF card: "synchronized video and audio within a single
  model"; known limitation: "when generating audio without speech, the audio
  may be of lower quality."
- **Dialogue prompt format (official):** spoken dialogue in quotation marks;
  "specify language and accent if needed"; screenplay style with character
  cues is explicitly endorsed for dialogue-heavy scenes; the 2.3 guide adds
  the beat pattern — break lines into short phrases with acting directions
  between them ("speaks in a sad, slow-paced voice, '…' He pauses …").
- **Emotional/vocal delivery cues (official):** voice descriptors (whisper /
  mutter / shout / scream, "robotic monotone", "resonant voice with
  gravitas"), delivery adverbs ("says softly", "whispering dramatically",
  "cracking voice"); facial acting via physical cues rather than abstract
  emotion labels.
- **Timestamps: NOT supported as a prompt convention.** No official source
  recommends "[0–5 sec]" syntax; the official structure is one chronological
  paragraph with "Initially… / A moment later…" transitions and named cuts in
  prose; the enhancer system prompts *forbid* timestamps. → Timestamps are
  planning data for the WORKER, never prompt text.
- **Multilingual speech:** "characters can talk and sing in various
  languages" (official); Dub-It validated languages: English, French,
  Spanish, German, Russian. Treat those five as the exposable set until
  measured otherwise.
- **Multi-shot (2.5):** 2–4 shots per generation, cuts named in prose, with
  character/scene/voice consistency across cuts.
- **Semantic speech quality is NOT guaranteed** — prior ZolexAI sessions saw
  clear voice, unclear voice, and non-semantic sustained vowels. The A–F
  criteria (audio exists → words correct → speaker matches → mouth responds)
  are measured on the box, not assumed. Results in §8.

## 3. The current ZolexAI T2V flow (trace summary)

Form (`CreatorWorkspace` + `GenerationSettingsPanel`, fully YAML-driven) →
`POST /generations` (`GenerationParameters`, `extra="forbid"`) →
`WorkflowRegistry.validate_request` → job stored with the prompt **verbatim**
→ Postgres queue (Redis doorbell) → worker claims → `LtxAdapter.run`:
`structure_prompt` (if `execution.prompt_structuring`) → `_run_generation` →
`plan_section_prompts` per section → `render_chain` →
`ltx_pipelines.distilled` (or `ti2vid_two_stages` under
`execution.generation_engine: guided`) → per-section
`normalize_clip(audio=True)` → `concat_segments` → `verify_output`
(video+audio, duration) → upload.

Current pass geometry on the PRO 6000: all four grids measured to 60 s per
pass, `LTX_MAX_SECONDS=60` on the box → every menu duration (5–60 s) is a
**single pass** on the distilled tier; 60 s becomes 2 passes only if the env
brake is lowered. The guided tier runs 5 s passes (121-frame landing) — the
section machinery is live there.

## 4. Root gap

ZolexAI requires explicit dialogue prompting today because nothing in the
stack writes dialogue: `structure_prompt` is deliberately a non-LLM rule
layer (counts/colours), `plan_section_prompts` only *distributes* what the
user wrote, and no LLM exists anywhere in the repo (by design — the deferred
API-key decision). An idea-in/dialogue-out feature needs a generative
planner. The reference stack solved dialogue *text* with a server module (not
shared) and speech *audio* with an external TTS — the 2.5 runtime lets us do
both natively: a local Gemma writes the plan, LTX speaks the lines.

## 5. Chosen architecture

```
STANDARD (unchanged, byte-identical):
  user prompt → structure_prompt → plan_section_prompts → render_chain → LTX

DIRECTOR (opt-in, T2V only):
  user idea + language
    → DirectorProvider.generate_plan()          (local gemma-4-e2b-it, GPU box)
    → DirectorPlan {scene, tone, language, characters[], timeline[]}
    → deterministic validation (schema, speech budget, verbatim-dialogue
      preservation, speaker existence, coverage)
    → compile_section_prompts(plan, windows)    (official caption style,
                                                 per-section events only,
                                                 persistent character block)
    → render_chain → LTX (distilled, audio on)  [unchanged from here on]
```

Decisions and their evidence:

- **Planner model: `google/gemma-4-e2b-it`, local.** It is the *official*
  LTX-2.5 prompt-enhancer checkpoint (docs name `gemma4_e2b_it_bf16` as "the
  prompt enhancer"), Apache-2.0 licensed (commercially clean, unlike the
  Gemma-3 Terms-of-Use flow-downs), un-gated, 10.2 GB bf16, and now on the
  box at `/workspace/ltx2-benchmark/models/gemma-4-e2b-it`. One download
  serves both a future `--prompt-enhancer-gemma-root` pilot and the Director
  planner. No paid API needed unless measurement proves it insufficient.
- **Planner runs in the worker, per job, before rendering.** The stored job
  keeps the user's idea verbatim (same directive as prompt structuring); the
  planner is a subprocess in the LTX venv (torch lives there, the worker venv
  has none) exactly like `person_matte.py` — `settings.director_planner_argv`
  with a `DIRECTOR_PLANNER_COMMAND` override. Sequential with the render, so
  its ~11 GB VRAM never stacks on a render peak.
- **Timeline ownership: the worker.** Timestamps never reach the prompt. The
  plan's events are bucketed into generation windows by midpoint (same rule
  as `plan_section_prompts._distribute_timed`); each section's caption
  carries ONLY its own events plus the persistent character/scene block —
  the global-plan-then-split design that prevents per-section dialogue
  restarts by construction.
- **Prompt compilation: official caption style.** Chronological prose,
  dialogue quoted exactly with delivery cues and physical acting beats,
  camera as prose, soundscape line, no labels/timestamps. Sections ≥2 get a
  short continuity preamble (the reference engine appends the same on its
  extension path).
- **Exact user dialogue is contractual.** Quoted lines in the idea must
  appear verbatim in the plan; the validator enforces it (retry once, then
  fail). The planner never rewrites user-provided lines — same posture as
  the enhancer rule "DO NOT modify user-provided character dialogue".
- **Speech budget:** ≤ 2 spoken words per second of video, spread with
  silent reaction events. (Conversational English ~2.5 wps; leaving air is
  the official recommendation and the anti-"suuu" hedge.)
- **Failure handling:** planner retried once (sampled decode); a second
  failure fails the job with a clear message. It never silently falls back
  to a dialogue-less standard render and never invents a plan
  deterministically — random dialogue is worse than an honest error.
- **Quality tier:** distilled (default). Guided is NOT forced — it is 4.3x
  cost, its passes are capped at 5 s (choppy for dialogue), and its
  adherence advantage is unproven at defaults. Director+guided can be
  benchmarked later via the existing `generation_engine` key.

### Wiring (the `quality`/`lyrics_language` templates)

- YAML: `text-to-video.yaml` gains top-level
  `supported_prompt_modes: ["standard", "director"]` (absent list = feature
  hidden; only T2V declares it, so I2V/extend/V2V/MV are untouched).
- API: `GenerationParameters.prompt_mode` + `dialogue_language`;
  `validate_request` rejects `director` on workflows that don't declare it
  and `dialogue_language` outside director mode. Projection via
  `to_public()` + `WorkflowPublic`.
- Contracts/web: `workflowSchema` + `catalog.server.ts` (top-level field →
  hand-projected; qa-parity `MIRRORED` updated), form schema, mode toggle +
  language select in `GenerationSettingsPanel`, `FIELD_MAP`.
- Worker: `worker/director/` (plan model, provider, compiler);
  `LtxAdapter.run` skips `structure_prompt` for director jobs (the compiler
  owns structure); `_run_generation` plans → compiles → hands the existing
  `render_chain` a director-aware `prompt_for_step`. Nothing below
  `prompt_for_step` changes.

## 6. Free/local vs paid API

Local wins on every axis that matters here: the checkpoint is the official
enhancer model (already required for any future enhancer work), Apache 2.0,
runs on hardware we already pay for, and keeps prompts on-box (no new data
egress). The open question was capability — whether a ~5 B instruct model can
emit a valid structured plan with sensible dialogue in the requested
language. That is measured, not assumed: §8 records the pilot results. The
`DirectorProvider` seam keeps a future `ExternalLLMDirectorProvider` a
drop-in if plan quality ever proves insufficient.

## 7. What is intentionally NOT built now

- No plan preview/edit UI (option B in the brief) — v1 generates invisibly;
  the preview is a later additive step.
- No guided-tier coupling, no new quality ladder.
- No Music Video integration of any kind.
- No speaker-voice cloning / TTS path — that is the reference stack's 2.3
  workaround; 2.5 speaks natively.
- No exposure of model internals (Gemma/CFG/pipeline names) in the UI.

## 8. GPU measurements (RTX PRO 6000, 18 Aug 2026)

**Planner pilot — gemma-4-e2b-it CAN write DirectorPlans.** Six ideas
(detective, robot interview, mother/daughter, Spanish, verbatim-manual,
no-dialogue): valid JSON on every case, ~5–14 s per plan (greedy, ~65 tok/s),
model load ~13 s warm, **peak 10.3 GB VRAM** — it fits BESIDE a running
render, though production runs it before the render starts. Spanish dialogue
was real Spanish ("Esto no es aceptable." / "Cálmate, Leo."); user-quoted
lines were carried verbatim; speech budgets were respected. Three observed
model quirks now handled in code: the literal string "null" as a speaker,
character ids leaking into camera/action text, and one gratuitous word of
dialogue on a no-dialogue idea (prompt rule added).

**Dialogue speech probe — distilled 2.5 SPEAKS.** 10 s, 1024x576, official
caption style, two-line detective scene, 36.8 s wall. faster-whisper (small)
transcription:

    [0.00-2.00]  "You knew the entire time."   ← requested line, EXACT
    [6.00-8.00]  "This job can kill."          ← requested "I did what I had
                                                  to do." — off-script

Criteria: audio exists YES · voice exists YES · words intelligible YES ·
requested words approximately correct PARTIAL (1 of 2 exact) · scene/identity
adherence excellent (dim office, lamp, gray suit, white uniform shirt, chief
rises on cue, identities rock-stable across all 10 sampled frames).

**Director end-to-end battery** (idea → local plan → compiled caption → real
render, through the actual adapter on the box, isolated test checkout):

- **TC1 detective, 20 s, English:** 78.3 s wall including planning. The
  planner wrote a 6-line exchange (28 words); whisper transcribed **all six
  lines verbatim, in order, with natural pauses**, ending on the planned
  resolution beat ("Fine."). Frames show two stable distinct men, the
  planned camera coverage (medium → alternating close-ups → over-shoulder),
  and the planned final defeat on the chief's face.
- **TC2 robot interview, 30 s:** all five planned lines verbatim and the
  interview progressed — and it exposed the two defects of the day:
  (a) the ~10 s of tail the plan left uncovered was filled by the model
  **reading the caption's camera/ambience sentences aloud as narration**;
  (b) a compiler bug produced "the the humanoid robot" (id replacement not
  absorbing an existing article). Both fixed: sections whose events end
  >2.5 s before the window now append a described-silence closing beat, and
  id/role replacement is marker-based with article absorption. Re-run
  (tc2b) verified below.
- **TC4 Spanish, 20 s:** perfect. Whisper language detection: **es, p=0.99**;
  all five planned Spanish lines spoken verbatim ("Esto es inaceptable." /
  "Cálmate, detective." / "No puedo ignorarlo." / "Tienes un problema." /
  "Lo sé."). No English substitution anywhere.
- **TC2 re-run (tc2b) found two more planner-shaped compiler bugs**, both
  fixed and re-verified: plans tag silent reaction beats with a speaker and
  EMPTY dialogue (compiled to a literal `says, ""` — an open invitation to
  narrate), and the model shortens roles to their head noun ("the robot"),
  which matched neither id nor role and duplicated the subject. Silent
  speaker-events now compile as action prose; unambiguous head-noun aliases
  join the replacement set.
- **TC2 final (tc2c, 30 s):** narration leak GONE. All five lines verbatim,
  naturally spaced across the full 30 s, tail held in described silence.
  (Whisper reports a phantom line starting exactly AT the file's end on
  quiet tail ambience — a known whisper hallucination, timestamps ≥ file
  duration; the tail is ambience, verified non-silent at −45 dB.)
- **TC5 manual dialogue, 15 s:** the user's exact quoted lines ("Please
  don't leave." / "I have to go.") survived planner, compiler and render
  UNTOUCHED — spoken verbatim at 1–3 s and 8–10 s, nothing invented.
- **Long-form, 60 s forced into 2×30 s sections:** 203 s wall. One global
  plan, section-only captions. The transcript progresses continuously
  across the 30 s boundary (4 lines in section 1, 3 in section 2), **no
  opening-line restart**, boundary audio continuous (RMS ≈ −38.9 dB both
  sides of the seam, no reset or double mux), identity held across the seam
  (same chief, same room at 29 s and 31 s). One model-level echo: "We'll
  see." (planned once) was spoken twice inside section 2.
- **Lip behaviour classification: B.** The camera cuts to the planned
  speaker for their line and that character's mouth articulates during it
  (TC1 frames at 2 fps). Phoneme-level sync (C) is neither claimed nor
  disproven — it needs a finer-grained measurement than frame sampling.
- **Costs (distilled tier, 1024x576):** planning adds ~35–45 s to a job
  (subprocess + model load + generation); renders: 20 s video ≈ 78 s wall
  total, 30 s ≈ 112 s, 60 s (2 passes) ≈ 203 s. No guided tier anywhere.

### Density, repetition, and the hosted planner (19 Aug, after launch)

Two customer reports came back after the feature went live: *"can you make it
have dialogue the whole video"* and *"in some videos repetitive things are
spoken"*. They looked like two asks and share one origin.

**The plans were too sparse, and the brief was why.** Across nine renders,
every plan denser than ~0.2 spoken lines/second was clean and every plan below
it was not — a 60s plan carrying 7 lines opened with a **12.8-second silence**
and spoke one line twice. The cause was the shape of the instruction, not the
model: the brief gave a ceiling ("at most N") plus a vague "about N", and an
instruct model handed two numbers multiplies the small ones. The lyrics writer
had hit this three days earlier and fixed it the same way (`510c616`).

Fixes: `target_spoken_lines` states a **target with a floor**, separate from
the ceiling, at one line per 4 seconds; `pacing_problems` measures opening
silence, internal gaps and the tail, and hands the specific complaint back to
the planner as a correction. Pacing is reported, never raised — a sparse plan
still makes a valid video, and failing a job over it helps nobody. Measured
after: a 60s plan went from 7 lines with a 12.8s hole to **14 lines spanning
0→60s**, all spoken verbatim.

**Repetition is a SEPARATE bug, and density does not fix it.** With the denser
plan, three of fourteen lines were spoken twice — each repeat landing seconds
after its original, not filling distant dead air. The mechanism looks like the
model filling a line's remaining screen time by saying it again. A positively
phrased caption directive ("each line of dialogue is spoken a single time, and
the exchange moves forward") took the same idea and duration from **3 repeats
to 1**. The phrasing is not stylistic: this runtime has no negation mechanism,
so "no line is repeated" would read as a request for repetition — the rule
`worker/longform/enhance.py` is built around.

Measured with the directive: **1 repeat across 22 lines in two runs** (60s:
14 lines, 1 repeat; 30s: 8 lines, 0 repeats and a line landing every four
seconds from the first to the last). Against 3 repeats in 14 lines without it.
**Not eliminated**, and both survivors were the OPENING line, which is where
the model has the least preceding context to hold it — that is the next thing
to attack if a customer reports it again.

**The planner moved to Cerebras.** The local checkpoint costs 18-26s of the
GPU the render is waiting for; the hosted call plans in **~1 second** against a
far larger model. It sits first in a chain with the local one beneath it, so a
missing key or an outage makes the feature slower rather than absent. Five
live cases (20s, 60s, 30s, Spanish, user-quoted dialogue) all planned clean
and well-paced.

**A trap worth recording: `response_format: json_object` BREAKS
`gemma-4-31b`.** It is the obvious safety measure — ask for JSON at the
protocol level as well as in the prompt — and adding it took the success rate
to 1 in 3, the failures running away to 8-49 KB of output before truncating at
`finish_reason: length`. Without it, 3 of 3 clean. `gpt-oss-120b` is
unaffected either way, so this reads as a constrained-decoding interaction
with that model rather than a service fault. A test now guards against
re-adding it.

### Continuity and vocabulary (19 Aug, third round of customer feedback)

Three more reports, all "small but noticeable": a person **flickers out** for a
moment mid-shot; **a prop changes** after being handled (Santa removes his hat,
puts it back, and the hat is subtly different); and **a distinctive word
repeats** across lines ("excellent" twice).

The third is ours outright — the planner writes those words. `repeated_vocabulary`
finds any distinctive word used in more than one line (structural words like
"the" and "you" exempt, 4+ characters, counted per line so "no, no, no" inside
one line is left alone) and feeds it back as a correction. Verified: zero
repeated words across every live case since.

The first two are model artefacts, so the only lever is the prompt — but it is
a lever with measured backing (16 Aug: explicit repeated constraints fixed
colour and identity drift on this unguided runtime). The plan gained a
`continuity` list — facts that must look identical in every frame — which the
planner fills and the compiler restates at the end of EVERY section, alongside
a standing "everyone stays fully visible in every single frame".

Every one of those sentences is phrased as what STAYS. That is not style: this
runtime has no negation mechanism, so "the hat does not change" reads as a
changing hat. A test asserts no negative phrasing reaches the continuity block.

Asked for the customer's own scene, the planner independently produced exactly
the right constraints — *"The Santa hat is red velvet with a fluffy white
pom-pom"*, *"Three people are present in the scene"*, plus each child's
wardrobe. Rendered on the box at 30s: the hat comes off at ~14s and returns at
~20s **looking like the same hat**, wardrobe held on all three, nobody missing
from any sampled frame.

**Honest limits.** One render is not proof that flicker is gone; a two-frame
dropout would not show in 1 fps sampling, and both symptoms are intermittent
by nature. This lowers the odds, it does not remove them.

### Short-clip pacing (fixed after the first deploy)

The deployed 15 s render fused its opening two lines into one utterance
("This ends now relax detective. We have time"). **Word density was not the
cause** — that clip carried 0.93 words/second where a clean 20 s clip carried
1.1 — so a flat words-per-second cut would have been the wrong fix. What
distinguished it was two short lines placed back to back with nothing between
them. Three changes, all measured:

1. **The word budget now excludes an establishing head** (`ESTABLISH_SECONDS
   = 2.5`), so it bites hardest exactly where over-packing hurts: a 15 s clip
   loses ~17% of a flat allowance, a 60 s clip ~4%.
2. **The planner is given computed `SPOKEN_LINES` and `TOTAL_WORDS` figures**
   rather than arithmetic to perform, plus explicit rules to open on a silent
   establishing beat and never place two lines back to back. A line-count
   *cap* is deliberately NOT enforced in code: a 20 s scene with six short
   lines rendered cleanly, so trimming to the guide would discard working
   output.
3. **Consecutive spoken lines get an explicit pause cue** in the compiled
   prose ("After a short pause", "A beat of silence passes, and then") —
   the official pacing lever, applied wherever the preceding event spoke.

Re-measured at the failing duration (15 s, same idea): the planner wrote two
lines instead of four, and both were delivered **separately and verbatim**
(0.0–0.8 s and 10.2–11.3 s). The trade-off is real and worth stating: short
clips now favour clarity over density, so a 15 s scene may carry a single
exchange with several seconds of ambience around it.
