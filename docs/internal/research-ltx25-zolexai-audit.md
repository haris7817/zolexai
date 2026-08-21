# Research — ZolexAI vs actual LTX-2.5, full pipeline audit

**Date:** 21 August 2026 · **Branch:** `ltx25-alignment-audit` (from stable `3bd8016`, main untouched)
**Scope:** every LTX code path — T2V, I2V, V2V, extend, music video, Director, lyrics, audio conditioning, presets, seams, camera. No GPU was available; nothing was deployed, no worker restarted, no model file touched.

**Evidence classes used throughout** (per the audit rules):
- **[OFFICIAL]** — read from the official `Lightricks/LTX-2` repository, pinned to upstream commit `400fd31` (2026-08-16), or the `Lightricks/LTX-2.5` HF card. A local clone was made for this audit; every claim in §2 carries a file citation into it.
- **[SOURCE-VERIFIED]** — read earlier from the installed pipeline source at `/workspace/ltx2-benchmark` and recorded in `docs/internal/research-2026-08-*.md`.
- **[GPU-MEASURED]** — measured on the RTX 5090 / RTX PRO 6000 and recorded in the same docs.
- **[REPO]** — read from this repository at `3bd8016` (file:line).
- **[INFERENCE]** — the team's conclusion from evidence, marked as such.
- **GPU VALIDATION REQUIRED** — cannot be settled without a render.

Companion documents: `audit/ZOLEXAI-CURRENT-STATE-AUDIT.md` (150-setting extraction),
`audit/DIAGNOSIS-flash-and-camera.md` (the two symptom clusters), and the dated
research notes in this directory.

---

## 1. Executive summary

**What is correct.** Far more than a fresh audit would guess. The four entry
modules we invoke exist verbatim upstream; every flag name `_command` emits
matches the upstream CLI at `400fd31` (closing audit unknown B.3); every
checkpoint filename matches the official model list; the 8k+1 frame rule, /64
resolution rule, 24 fps default, fixed distilled sigma schedules, the
LoRA-drops-quantization rule, the a2vid frozen-audio mechanism, our 0.04 s
audio-window pad (exactly one audio latent at 25 latents/s), and the guided
tier's unset-by-default guidance flags all align with upstream source. The
guided-family defaults the repo's comments claimed (CFG 3.0 / STG 1.0 /
modality 3.0 / 30 steps) are now **confirmed** against upstream
`constants.py` — with STG blocks resolved to `[28]` for a 2.5 checkpoint.
Where we deviate (per-job seed instead of the fixed seed 10, product grids
instead of 1536×1024, explicit `--num-frames` instead of auto-duration,
measured frame-count landing tables, sectioned long-form), each deviation is
deliberate, evidenced, and in three cases *more* correct than upstream's own
entry point (which does not snap frames although its sibling wrappers do).

**What is incorrect.** Four confirmed defects, all in prompt construction,
none in model invocation: (A5) a worker-generated continuity bullet was
classified as one section's dialogue; (B5) the worker appended "the camera
keeps moving" over a user-asserted static camera; the blanket presence
assertion ignored user-stated departures (the measured rendered-ghost
failure class); and section 1 of every chain was told to continue from a
predecessor frame that does not exist. All four are fixed on this branch
behind `execution.prompt_structuring_v2` (§16).

**What is obsolete.** Nothing was found that belongs to LTX-2.0/2.3 by
accident. The one 2.3 artifact (Union-Control IC-LoRA) is deliberate,
GPU-verified on 2.5, and matches the official 2.3 model list. There is no
preset system to be stale (§5). Two stale documentation values in
`.env.example` were corrected (§16).

**What needs GPU validation.** Whether the v2 prompt fixes improve renders
(G1–G4), the Director camera flags (G5), per-section V2V prompts (G8), seam
density vs flash-back (G6), the upstream checkout version on the node,
`--offload none` and `--max-batch-size` as cost levers, official 2.3
camera-control LoRAs on 2.5, and the full checklist in §18.

---

## 2. Actual LTX-2.5 architecture [OFFICIAL]

Upstream is `Lightricks/LTX-2` — `packages/ltx-core` (model), `packages/
ltx-kernels` (NVFP4 CUDA), `packages/ltx-pipelines` (the CLIs we shell into),
`packages/ltx-trainer`. Seventeen pipeline modules exist; the docs name
`DistilledPipeline` the **recommended default**. Facts that govern this audit:

- **Defaults are checkpoint-generation-detected.** `resolve_cli_params()`
  reads `model_version` from safetensors metadata; there is no 2.5 params row,
  so a 2.5 checkpoint resolves to the (2,4) row: seed 10, stage-2 default
  1536×1024, 121 frames, 24.0 fps, 30 inference steps, video guider
  CFG 3.0 / STG 1.0 / rescale 0.7 / modality 3.0 / **stg_blocks [28]**, audio
  guider CFG 7.0. (`utils/constants.py#L40-L133`)
- **Distilled is unguided by construction**: fixed sigma schedules — 8 steps
  stage 1, 3 steps stage 2 — no CFG, no steps flag, no negative prompt on its
  parser. 2.5 checkpoints switch stage 1 to **ancestral Euler (eta 1.0)**,
  seeded at `seed + 10000`; stage 2 is always deterministic.
  (`distilled.py#L60-L209`, `constants.py#L11-L25`)
- **The guided family** (`ti2vid_two_stages`, `a2vid_two_stage`) takes the dev
  transformer plus the distilled LoRA (stage 2 only) and the full guidance
  flag set; a long **default negative prompt applies automatically** —
  beginning "has_subtitles, has_blurbox, …" and banning "motion blur, camera
  shake, jittery movement, tilted camera". Stage 2 never uses CFG.
- **Frames:** `num_frames = 8k + 1` (VAE temporal factor 8); `snap_frames_to_
  grid` enforces it in the wrappers, **but the `distilled` entry point itself
  does not snap** — callers must. Resolution: divisible by **64** for every
  two-stage pipeline (32 for one-stage/raw model).
- **Auto-duration** (2.5 feature): with no `--num-frames`, the DurationHead
  predicts a length in **[1 s, 20 s]** from the caption; explicit
  `--num-frames` wins. No hard frame cap exists; official examples top out at
  121 frames (~5 s).
- **Long video: one diffusion pass per clip, full stop.** Nothing in
  `ltx_pipelines` windows time, autoregresses, or chains segments. **No
  extension/continuation inference pipeline exists** — extension is only a
  trainable LoRA mode in the trainer configs. The official long-video
  technique is **multi-shot prompting** (2–4 cuts named in prose). Adjacent
  official tools: `keyframe_interpolation` (guiding latents between `--image`
  keyframes), `retake` (regenerate a time window), `dfr_pipeline` (fps
  densification, not duration).
- **a2vid** takes `--audio-path --audio-start-time --audio-max-duration`
  (default = `num_frames/frame_rate`), VAE-encodes the audio, **freezes the
  audio modality in both stages**, and muxes the **original waveform** into
  the output. It does not segment a song; segmentation with pinned frames is
  an integrator construct (ours, and the client reference engine's). Audio
  latent geometry: 16 kHz / hop 160 / downsample 4 → **25 latents per
  second**; vocoder output 24 kHz (read from the checkpoint).
- **Camera is prompt text.** README: *"Include specific movements,
  appearances, camera angles, and environmental details — all in a single
  flowing paragraph… Think like a cinematographer describing a shot list.
  Keep within 200 words."* No structured camera input exists anywhere in the
  CLI or pipeline signatures. The only structured helpers are the **2.3-only**
  Camera-Control LoRAs (Dolly-In/Out/Left/Right, Jib-Up/Down, Static — no 2.5
  versions exist) and IC-LoRA video conditioning.
- **ComfyUI** official 2.5 distilled workflow matches the repo source exactly:
  euler_ancestral CFG 1 → euler CFG 1, 960×544×121 @ 24 fps, empty negative
  prompt. Nothing in it contradicts our invocation.

## 3. Our current architecture [REPO]

One worker (`apps/worker`), no torch import, shells into the LTX environment
per pass: `uv run python -m <module>` with `cwd=settings.ltx_repo_dir`. Five
video workflows dispatch on `workflow_id`; four pipeline tiers
(`LtxPipeline` at ltx.py:806-995) occupy the role a preset table would:
distilled (default), ic_lora (v2v transform), a2vid (music-video audio tier,
off), ti2vid_two_stages (guided tier, off). Durations beyond a measured
per-pass ceiling become chained passes (`render_chain`): even windows, zero
overlap, one seam PNG (`--image <last frame> 0 1.0`; v2v 0.85) plus a
per-section prompt plan computed once per job. The committed YAML routes all
workflows to `runtime: mock`; the GPU node's local YAML flips it. Full
extraction: `audit/ZOLEXAI-CURRENT-STATE-AUDIT.md`.

---

## 4. Side-by-side comparison (invocation-level)

| Parameter | Our implementation [REPO] | Actual LTX-2.5 [OFFICIAL] | Correct? | Required action |
|---|---|---|---|---|
| Entry modules | `distilled`, `ti2vid_two_stages`, `ic_lora`, `a2vid_two_stage` | all four exist verbatim; distilled is the recommended default | YES | none |
| Flag names (26 emitted) | `_command` ltx.py:2652-2799 | every one present on the upstream parsers at `400fd31` | YES | verify node checkout version (§18) |
| CFG / STG / steps (distilled) | not emitted | **do not exist** on the distilled parser; fixed 8+3-step sigmas | YES | none |
| CFG (guided/a2vid) | not emitted → pipeline default | 3.0 | YES (default applies) | optional `guidance_scale: 4.5` — measured to unfreeze camera [GPU-MEASURED] |
| STG (guided/a2vid) | not emitted → default | 1.0, blocks [28], rescale 0.7 | YES | none |
| a2v modality scale | not emitted → default | 3.0 ("higher may increase lipsync"); raising to 6.0 measured no change | YES | none |
| Steps (guided stage 1) | not emitted → default | 30 | YES | none |
| Negative prompt | not emitted | distilled: n/a; guided family: **default negative prompt auto-applies** (bans camera shake/tilted camera) | YES | note: it biases guided toward locked-off shots (§7) |
| FPS | `--frame-rate 24` always | default 24.0, configurable float | YES | none |
| Resolution | grid table: 1024×576 / 576×1024 / 768×768 / 512×640 (+ synthesized /64 grids) | default 1536×1024; rule = divisible by 64 (two-stage) | YES (all /64) | grids are VRAM-measured product choices, not upstream constraints |
| Frames | `round(s*24)` → 8k+1 snap → measured landing tables | `8k+1` required; **entry point does not snap** — caller must | YES — our snap is mandatory compensation | none |
| Seed | crc32(job_id), +index per pass; user seed `(seed+index)%2^31` | fixed default 10 (ancestral stage seeds +10000) | YES — deliberate; fixed seed would hand identical videos to different users | none |
| Duration | explicit `--num-frames` always | auto-duration [1–20 s] when omitted; explicit wins | YES | none (product sells fixed durations) |
| Image conditioning | `--image PATH IDX STRENGTH`, ascending order | same, optional CRF (defaults 18 on ≥2.4) | YES | none |
| Conditioning strength | first frame 1.0; seam 1.0 (v2v 0.85); i2v anchor 0.2; v2v stills 0.45 | no official values — free parameters | YES (measured internally) | GPU: seam-pin strength A/B (§18) |
| Audio conditioning | a2vid only: whole master + `--audio-start-time` + window `frames/24 + 0.04` | a2vid only; default window `num_frames/fps`; audio frozen; original waveform returned | YES — pad = exactly 1 audio latent (1/25 s) | none |
| Prompt enhancer | opt-in `enhance_prompt`, off; requires gemma root | `--enhance-prompt` off by default; same flags | YES | keep off (it paraphrases) |
| Offload | distilled: none; LoRA tiers: cpu | default none; cpu/disk documented | YES for 32 GB era | GPU: `--offload none` on 96 GB (23-30% faster, measured) |
| Quantization | nvfp4-prequant; dropped whenever a LoRA loads | valid enum; LoRA+FP8 clash confirmed [SOURCE-VERIFIED]/[GPU-MEASURED] | YES | none |
| `--max-batch-size` | never passed (default 1) | "Set to 4 to batch all guidance passes" | UNKNOWN — cost lever | GPU VALIDATION REQUIRED (guided/a2vid wall-clock) |
| Scheduler/sampler | never passed | none exists as a flag; 2.5 distilled auto-uses ancestral stage 1 | YES | none |
| Output encoding | pipeline writes; we trim/normalize/concat/mux with ffmpeg | H.264 crf 19 veryfast; AAC at audio rate | YES | none |

## 5. Preset comparison

**There is no preset system in this repository** — no named quality levels,
no preset table, `supported_quality_levels: []` everywhere; the audit's §5.1
"fast preset" row is `— not present`. The example preset in the task brief
(`stage_1_steps: 8`, `cfg: 1.0`, `seed: 42`…) **does not exist here in any
form**. What occupies the preset role is the four-tier `LtxPipeline` table
plus workflow `execution:` keys, and every default in them traces to one of
two provenance classes:

| De-facto preset value | Ours | Official 2.5 | Source / class | Correct? |
|---|---:|---:|---|---|
| FPS | 24 | 24.0 default | official default | YES |
| Stage-1 steps (distilled) | not sent | fixed 8 sigmas | official, immutable | YES |
| Stage-2 steps (distilled) | not sent | fixed 3 sigmas | official, immutable | YES |
| Steps (guided) | not sent → 30 | 30 | official default | YES |
| CFG (guided) | not sent → 3.0 | 3.0 | official default | YES |
| STG (guided) | not sent → 1.0 [28] | 1.0, blocks [28] | official default | YES |
| Width×Height | 1024×576 grid table | 1536×1024 default; /64 rule | internal, VRAM-measured | YES (deliberate) |
| Seed | crc32 per job | 10 fixed | internal, deliberate | YES |
| Prompt enhancer | off | off | matches | YES |
| Pass ceilings 30/8/20.04/5 s | measured | n/a (single pass ≤ ~5 s examples) | internal, dated GPU measurements | YES |
| Frame landing tables | measured per tier | n/a upstream | internal, dated GPU measurements | YES |
| `_TWO_IMAGE_SAFE_FRAMES` {120,240,360} | measured | n/a | internal, production crash 20 Aug | YES — do-not-touch |

Nothing here is an LTX-2.0/2.3 leftover mistaken for a 2.5 value. The
`ltx-2.3` Union-Control LoRA filename is the official 2.3 artifact,
deliberately loaded on 2.5 (verified working on the box; no 2.5 version
exists upstream). No preset needs replacing, renaming, or remapping; no API
contract changes.

## 6. Long video / seams — the real mechanism

**Actual LTX-2.5 has no long-video mechanism.** One diffusion pass per clip;
no extension pipeline; auto-duration tops out at 20 s; examples at 121
frames. The official answer to "60 seconds" is: nothing ships that does it.
The client's own reference engine (LTX-2.3 productisation) answers it the
same way we do: segments, each conditioned on the previous segment's final
frame pinned at strength 1.0, per-segment seed = seed + index [SOURCE-
VERIFIED via reference engine].

**Requested 60 s T2V @ 24 fps (1440 frames):**

```text
Our system:                                Actual LTX-2.5 (official CLI):
Pass 1: frames 0-719 (renders 736,         Pass 1: --num-frames 1441 in ONE pass.
        trimmed to 30.000s), fresh                 Upstream publishes no example
        subprocess, seed crc32(job):0             beyond 121 frames and no
Pass 2: frames 720-1439 (720 exact),               coherence claim at this length.
        conditioned on pass 1's last               Our single-pass 60s render
        frame @ strength 1.0, its own              measured: departed man returns
        section prompt, seed :1                    for the final 12s; dialogue
Concat → verify 60.0s ± tolerance                  repeats [GPU-MEASURED, 20 Aug]
```

Passes per duration (t2v/i2v, 30 s ceiling): 5 s → 1 · 15 s → 1 · 30 s → 1 ·
60 s → 2 (1 seam). Overlap is 0.0 everywhere (machinery exists, unused);
each seam carries exactly one PNG, the section prompt, and a derived seed —
no latents, no RNG state, no KV cache. Consistency across the seam =
pinned frame + repeated identity text (+ Director's `exits`/survivor
scoping in Director mode). The first frame is *not* re-used after pass 1
except as I2V's identity anchor, which currently never fires at shipped
durations (⛔ blocked constants, §17).

Judgement: our sectioning **is not a deviation from a real LTX-2.5 long-form
architecture — it fills a gap upstream leaves open**, matches the reference
engine's construction, and its 30 s regime is a measured story-coherence
choice, not a GPU limit. What differs from the reference engine: they send
bare per-segment prompts (continuity via pinned frame only); we add
structured continuity text — the measured anti-drift lever on distilled
[GPU-MEASURED, 16 Aug]. Both are defensible; ours carries the A5/B5 defects
now fixed under `prompt_structuring_v2`.

## 7. Camera control — actual vs ours

**Actually supported by LTX-2.5:** camera concepts are **prompt vocabulary
only**. The official guidance explicitly endorses naming camera angles and
movements in prose. Structured control exists only as 2.3 camera LoRAs
(dolly/jib/static — six moves + static, no 2.5 builds) and IC-LoRA video
conditioning. Additionally, the guided family's **default negative prompt
bans camera shake and tilted camera** — consistent with our measured finding
that guided-at-defaults froze an orbit that distilled partially executed;
CFG 4.5 unfroze it [GPU-MEASURED].

Which prompt concepts are reliable is **per-concept GPU work**; the official
guide endorses the vocabulary but promises nothing. Measured so far: orbit
partially works on distilled, freezes on guided at defaults; "starts directly
behind the subject" failed on both tiers. Everything else (wide/medium/
close-up/OTS/POV/angles/pans/tilts/dolly/handheld/crane…) is plausible prompt
text with no measurement — **GPU VALIDATION REQUIRED** before any of it is
promised to users.

**Ours before this branch:** exactly one structured field
(`DirectorEvent.camera`, free text, unvalidated); a closed planner vocabulary
with no angle; no rule binding the planner to a user's camera request plus an
active counter-instruction; camera state reset per section; standard mode has
no camera concept at all — worse, it appended "the camera keeps moving" over
explicit static requests (B5). V2V compiles no prompt at all.

**Changed on this branch (all off by default):** `prompt_structuring_v2`
removes the static-camera contradiction; `director_camera_continuity`
carries the last shot across seams and phrases same-shot openings as
continuations; `director_camera_from_idea` adds the brief register making a
user camera request binding and expressible (angles included — matching the
official guidance that angles belong in prompts). What we deliberately did
NOT do: invent a structured camera schema (the model has none), promise
specific moves (unmeasured), or validate `DirectorEvent.camera` against an
enum (rejecting plans would destabilize planning; the compiler already
renders any text safely).

**Do not promise users:** specific angles ("low angle"), "directly behind",
orbit on the guided tier, or any named-move fidelity — all unmeasured or
measured-shaky. Camera-Control LoRAs are a GPU-checklist option (2.3 LoRAs
load on 2.5 per Union-Control precedent; licence-check first).

## 8. T2V findings

Invocation correct (§4). The 60 s symptom cluster decomposes as: repeated
dialogue / action reset — model-level echo, mitigated by Director's
once-only phrasing and 30 s sectioning [GPU-MEASURED]; characters
disappearing/returning — presence-blind captions (fixed in Director mode via
`exits`; standard mode carried the blanket presence rule, now
departure-aware under v2); scene repetition at section 1 — the misclassified
continuity bullet performed as an action (A5, fixed under v2); camera reset —
no cross-seam camera state (fixed under `director_camera_continuity`).
Classification: prompt-construction problems (fixed, flag-gated), one
Director-mode gap already fixed pre-audit (`exits`), on top of a genuine
model limitation (echo, presence-blindness) that no static change removes —
the flags' *visual* effect is G1/G2/G5: GPU VALIDATION REQUIRED.

## 9. I2V findings

Source-image conditioning correct (`--image upload 0 1.0` pass 1).
First-frame preservation: structural (pinned at full strength). Director I2V
is source-anchored: the planner is forbidden to invent appearance,
`_ground_visual_claims` strips ungrounded visual claims, anchored constancy
sentences reference the conditioned frame — the planner cannot contradict
the photograph [REPO, test-pinned]. Language: `dialogue_language` validated
against the Dub-It-validated set. The one real defect: **the identity anchor
never fires on any offered duration** (60 s is the only chaining duration;
its 720-frame passes are outside `_TWO_IMAGE_SAFE_FRAMES`) — every route to
fixing it goes through the two ⛔ do-not-touch constants, so it is
documented, not changed (§17, G7).

## 10. V2V findings

Reference-image handling, identity transfer (composited anchor, per-region
attention, edge-map control), source preservation, and frame-exact seam
delivery were audited 19–21 Aug and are correct per their own research docs;
identity refresh is deliberately 0.0 (the reference photo flashed into
renders at `frames//3`, twice). The YAML comments claiming defaults
0.65/0.2 are stale vs code (1.0/0.0) — left unedited deliberately (VPS
stash-pop hazard, runbook §44.1); recorded here instead. Lip-sync: the
shipping path tracks the source mouth at 0 ms offset, r 0.947 — not a model
problem. The genuine gap: **every section received byte-identical prompt
text** (up to 44 sections on a 330 s source). Fixed behind
`v2v_section_prompts` (off; G8 decides). Long-video identity decay past 30 s
and multi-person sources remain unmeasured — do not promise them. Face
replacement from low-resolution sources: not promised; the anchor is pixels,
not an identity embedding (no identity input exists in the LTX family
[SOURCE-VERIFIED]).

## 11. Music video findings

The chain verified end-to-end: song → probe (duration) → onsets (cut
placement) → sectioned silent picture → one mux of the original track.
**Which pipeline receives audio:** by default, none — prompt-only
generation; the model never hears the song, and lip-sync there is
structurally impossible (distilled's parser has no `--audio-path` and its
audio ModalitySpec has no initial latent — impossible, not unset
[SOURCE-VERIFIED], reconfirmed upstream [OFFICIAL]). With
`audio_conditioning: true` (committed OFF), sections render on a2vid with
the master seeked per section — measured sync −125…−208 ms at r≈0.45
(energy-level, "goal B"); phoneme accuracy never claimed. The 21 Aug fixes
(sections delivered at planned length against cut boundaries; a2vid landing
table; no sliver passes) shipped pre-branch. Remaining known quality items:
performer identity drifts across seams (no reference-image input exists on
this workflow), and the seam pin at 1.0 may carry a wrong-moment mouth —
both GPU items (§18).

## 12. Lyrics findings

The user requirement — auto-generate appropriate lyrics in the selected
language when the box is empty — is **implemented and deployed** (Cerebras
`gemma-4-31b` writer, 14 languages verified, template fallback English-only,
silent-degradation trap documented: no `CEREBRAS_API_KEY` ⇒ English-only).
"Female vocals / sung chorus" misread as instrumental was real (`lo-fi`/
`ambient` genre words overruled the sentence) and was fixed in `4e37a02`:
`vocal_intent()` reads refusals first, then vocal words in what remains;
explicit instrumental requests remain instrumental (regression-tested).
Density re-measured on the production model (8 s/line): solid for 1–3 min;
at 4–5 min run-to-run variance exceeds the density effect — honest limit,
not a bug to fix statically. Song structure (verse/chorus/bridge), full-song
coverage, timestamps: handled by the writer/plan layer, tests passing.
Nothing further changed on this branch.

## 13. Audio-conditioning findings

Exact pipeline `ltx_pipelines.a2vid_two_stage` [OFFICIAL]: dev transformer +
distilled LoRA (+ CPU offload, unquantized — the LoRA/quantization rule),
inputs `--audio-path` (whole master) `--audio-start-time` (per section)
`--audio-max-duration` (our `frames/24 + 0.04`; official default
`frames/24`; the pad is exactly one 25 Hz audio latent — the same pad the
reference engine applies), full guidance set at official defaults, audio
frozen both stages, original waveform returned. Supported frame counts on
this decode path: PASS 65…1441 with holes (289–361, 409, 457, 841 CUBLAS;
601/1081 OOM) — landing table (121, 241, 385, 481); ceiling 481 frames
(20.0417 s). Memory: 77.4 GB isolated peak; 5/15 failures when sharing the
card with ACE-Step; cost 10.6× real time. **Safe activation plan:**
1) keep `audio_conditioning` commented until pricing accepts ~3× compute;
2) resolve co-tenancy first (stop ACE-Step during audio-tier passes, or
   second GPU) — the §7b measurement says failures are scheduling, not shape;
3) consider `--offload none` (needs 76–94 GB free) and `--max-batch-size`
   (GPU VALIDATION REQUIRED) before quoting wall-clock;
4) enable on one node via the YAML key; run the §46.2 gates (60 s end-to-end
   music video, frame-count log lines, sync probe);
5) do not promise phoneme-accurate lip-sync — measured claim is energy-level
   following. Production stays untouched by this branch.

## 14. Problems fixable without GPU — all fixed on this branch

1. A5 continuity-bullet misclassification (with the D1 redesign it was
   sequenced behind) — `prompt_structuring_v2`.
2. B5 static-camera contradiction — same flag.
3. Exit-blind presence assertions — same flag.
4. Section-1 "continue from the predecessor frame" noise — same flag.
5. Director camera state reset per section — `director_camera_continuity`.
6. Planner free to drop user camera requests — `director_camera_from_idea`.
7. V2V byte-identical section prompts — `v2v_section_prompts`.
8. Stale `.env.example` values (LTX_MAX_SECONDS 30→60 note,
   MUSIC_SECONDS_PER_LINE 18→6).
9. Audit correction: §C.7's stated cause was wrong (operative regex is
   `_DIALOGUE_LINE`, not `_PERSISTENT_LINE`) — recorded here and in the
   diagnosis.

## 15. Problems requiring GPU validation

Everything in §18; headline items: do the v2 prompts reduce flash-back
(G1/G2) and honour static camera (G3/G4); Director camera flags' visual
effect (G5); seam density (G6); I2V anchor restoration (G7 — touches ⛔
constants); V2V per-section prompts (G8); node checkout vs upstream
`400fd31`; `--offload none` / `--max-batch-size` cost levers; camera LoRAs
on 2.5; music-video seam-pin strength; audio-tier co-tenancy.

## 16. Changes implemented

All changes are additive and flag-gated; with every flag absent (the shipped
state), prompts, captions, and argv are byte-identical to `3bd8016` —
asserted by the new default-off tests.

**1. File:** `apps/worker/worker/longform/enhance.py`
Before: one combined bullet — "One continuous scene: the camera keeps moving through the same environment, and every subject present at the start is still present at the end." — appended unconditionally; header unmatchable by `_ALREADY_STRUCTURED`.
After: under `v2` — separate bullets; static-camera requests get a static rule; departure vocabulary suppresses presence/count assertions; idempotent header. v1 path byte-identical.
Why: B5 contradiction; rendered-ghost class; A5 sequencing (D1).
Evidence: DIAGNOSIS §B5/§A1; GPU 20 Aug ghost measurement; negation-mechanism rationale (enhance.py's own docs).
Risk: none while off; ON changes prompt text on 4 workflows → G1-G4.
Test: `test_prompt_structuring.py` — 8 new tests incl. byte-identical-off, idempotency, static, exits.

**2. File:** `apps/worker/worker/longform/prompts.py`
Before: `- One continuous scene: …` matched `_DIALOGUE_LINE` → one section's "NEW ACTION"; section 1 told to continue from a nonexistent predecessor.
After: under `v2` — bulleted lines classify persistent; section 1 gets an opening header/closing without predecessor references. v1 path byte-identical.
Why: A5 (the module's own contract: "ambiguous prose remains persistent"); section-1 preamble noise (MV21 §13).
Evidence: DIAGNOSIS §A5 regex probes; caption-narration measurement (DIR18 TC2).
Risk: none while off.
Test: `test_prompt_structuring.py` — 4 new tests (bullets reach every section and never the action block; section-1 shape; single-pass unchanged).

**3. File:** `apps/worker/worker/director/compiler.py`
Before: `previous_camera = ""` per section — every section re-announces its shot as a fresh framing event; no section knows the previous shot.
After: `compile_section_prompts(..., camera_continuity=False)` computes each section's inherited shot; same-shot openings say "The same X continues from the previous moment…"; shot changes stay fresh framings. Off → byte-identical.
Why: B3/B4; Phase 7's one-continuous-sequence goal.
Evidence: DIAGNOSIS §B3; compiler.py:194 [REPO].
Risk: none while off; ON changes Director captions → G5.
Test: `test_director.py` — 4 new tests (byte-identical off, continues, deliberate change, moving shot).

**4. File:** `apps/worker/worker/director/provider.py`
Before: no brief rule binds the planner to a user camera request; closed vocabulary; "No fast camera moves" counter-instruction.
After: `DirectorRequest.camera_from_idea` (default False) appends the CAMERA REQUESTS register (request outranks framing preference; angles expressible).
Why: B6 — user camera text droppable in Idea mode; official guidance says angles/movements belong in prompts.
Evidence: DIAGNOSIS §B6; README prompting section [OFFICIAL].
Risk: none while off; ON changes planner output → G5.
Test: `test_director.py` — 3 new tests (absent by default, register content/position, execution-key wiring via CannedProvider).

**5. File:** `apps/worker/worker/adapters/ltx.py`
Before: v2v restyle and transform never passed `prompt_for_step` → every section byte-identical text; no wiring for the three new flags.
After: `_v2v_prompt_for_step` (returns None unless `v2v_section_prompts`); flags plumbed to `structure_prompt`, `plan_section_prompts`, `compile_section_prompts` at all call sites.
Why: A6/B4-v2v; wiring.
Evidence: DIAGNOSIS §B4 call-site table.
Risk: none while off (None → `job.prompt` fallback unchanged); ON → G8.
Test: `test_video_to_video.py` — 3 new tests (off→None; per-section prompts differ and keep user text; single pass byte-identical).

**6. Files:** `workflow-definitions/*.yaml` (5 files) — commented-out
documentation of the new keys, following the repo's opt-in-tier pattern; zero
functional change (`music-video.yaml`/`video-to-video.yaml` edits are
comment-only, mindful of the VPS stash hazard — read the pop output on
deploy). **7. File:** `.env.example` — two stale commented values corrected.

**Test baseline (this machine, 21-22 Aug):** pre-change 788 passed / 15
failed / 1 skipped; post-change **810 passed / 15 failed / 1 skipped** — the
same 15 environmental failures both times (14 × `test_runner.py` disk-floor:
1.69 GB free vs `min_free_disk_mb` 2048; 1 × the known LAME-MP3 padding
failure in `test_music_video.py`, verified pre-existing 17 & 21 Aug). All 22
new tests pass; zero regressions.

## 17. Remaining risks — honest list

- **Every flag's visual effect is unmeasured.** The fixes are textually
  correct; whether the model renders better under them is exactly G1–G5/G8.
  A5's repair promotes the presence rule from one section to all N when no
  exit is stated — the diagnosis's warning; v2 pairs it with the D1 redesign,
  but the direction of the *rendered* effect is still a GPU question.
- **The ⛔ constants stand.** The I2V identity anchor still never fires at
  shipped durations; fixing it requires touching `_TWO_IMAGE_SAFE_FRAMES` or
  `max_segment_seconds`, both off-limits with dated crash provenance.
- **`_EXIT_WORDS`/`_STATIC_CAMERA` are heuristics.** Conservative regexes;
  a miss reproduces today's behaviour (never worse), but a false positive on
  e.g. "the road vanishes into fog" would withhold the presence rule.
- **Node checkout version unknown.** All upstream comparisons are pinned to
  `400fd31`; the node's `/workspace/ltx2-benchmark` may be older (a pre-2.5
  tree lacks the ancestral sampler and DurationHead handling). Verify first.
- **Committed YAML is `runtime: mock`** — nothing on this branch reaches a
  GPU until the node's YAML carries the flags; conversely the VPS stash
  dance applies to two of the edited YAMLs (comment-only, but read the pop).
- **STG blocks [29] vs [28]:** one older internal doc recorded [29]; upstream
  (2,4) row and the checkpoint read both say [28]. Trust [28]; noted in case
  the node checkout differs.
- The music `prompt_adherence` fake control and the two stale v2v YAML
  comments remain, deliberately (deployed-contract and stash-hazard
  respectively).

## 18. GPU validation checklist (in execution order)

Environment first:
1. `git -C /workspace/ltx2-benchmark log -1` — diff against upstream
   `400fd31`; confirm local commit `d434411` (green-video fix) present;
   confirm NATTEN + ltx-kernels intact (`uv sync` dry-run shows no `-` lines).
2. Read checkpoint safetensors metadata (`safe_open(...).metadata()`):
   `model_version`, audio VAE sample rate, vocoder output rate.

Prompt-fix A/Bs (same seed, same prompt, flag off vs on):
3. G3/G4 — "locked-off static camera" prompt, `prompt_structuring_v2`:
   does the camera hold?
4. G1/G2 — a 60 s two-section prompt with a mid-video departure, v2 on:
   fewer flash-backs? And with no departure: does the all-sections presence
   rule help or hurt?
5. G5 — Director 60 s with `director_camera_continuity` on/off; and an idea
   naming "low angle" / "slow orbit" with `director_camera_from_idea`.
6. G8 — a 60 s v2v transform with `v2v_section_prompts` on/off: identity
   drift across 8 sections.
7. G6 — same 60 s prompt at per-pass 30/20/10 s: does seam density correlate
   with flash-back? (cheapest structural experiment)

Cost/reliability levers:
8. `--offload none` on the 96 GB card for LoRA tiers (measured 23–30%
   faster; needs headroom — test with ACE-Step stopped, then co-resident).
9. `--max-batch-size 4` on guided/a2vid — upstream-documented batching lever
   we have never exercised.
10. Audio-tier co-tenancy: repeat the 6× 481-frame cell with ACE-Step
    stopped; decide scheduling before enabling `audio_conditioning`.
11. Music-video seam-pin strength sweep (1.0 vs 0.85/0.6) against the sync
    probe — the suspected wrong-moment-mouth carrier.
12. Official 2.3 Camera-Control LoRAs on 2.5 (licence review FIRST — same
    Attachment-A diligence as Union Control; do not fetch the BFS head-swap
    LoRA, licence unverified).
13. G7 — only with client sign-off on the ⛔ constants: re-probe two-image
    720-frame passes (`frame_probe2.py`), or accept a shorter I2V pass length
    to restore the anchor.
14. Watermark/provenance probe of LTX-2.5 output (licence §7 obligation,
    still open).

---

## AUDIT STATUS

```text
AUDIT STATUS
-------------
Actual LTX-2.5 match: NOT QUANTIFIABLE as a single percentage — measured
  instead: 26/26 emitted CLI flags match upstream 400fd31; 10/10 checkpoint
  files match the official list; all 4 entry modules exist upstream; every
  unset guidance default now source-confirmed; deliberate deviations (seed,
  grids, explicit frames, sectioning, landing tables) each carry dated
  evidence. Node checkout version: GPU VALIDATION REQUIRED.

Preset match: NOT QUANTIFIABLE (no preset system exists to match) — the
  de-facto defaults table (§5) shows official values where official values
  exist, and measured internal values everywhere upstream has no opinion.

T2V:                PASS (invocation) / NEEDS WORK (60s quality — v2 flags
                    implemented, GPU A/B pending)
I2V:                PASS, with one blocked defect (identity anchor never
                    fires at shipped durations — ⛔ do-not-touch constants)
V2V:                PASS (identity/seams fixed pre-branch) / per-section
                    prompts implemented behind flag, GPU A/B pending
Long video:         PASS structurally (upstream has no long-form mechanism;
                    ours matches the reference-engine construction) — seam
                    QUALITY items are GPU work
Camera:             NEEDS WORK (structural fixes implemented behind 3 flags;
                    per-concept reliability unmeasured — promise nothing yet)
Music video:        PASS (timing/vocals fixed 21 Aug; lip-sync only via the
                    off-by-default audio tier, correctly wired)
Audio conditioning: PASS (implementation verified against upstream source;
                    activation is a pricing + co-tenancy decision, plan §13)
Lyrics:             PASS (multilingual auto-lyrics deployed; vocal-intent
                    fixed; 4-5 min variance is a documented model limit)

Safe to buy GPU:    YES — no software blocker found; the invocation layer is
                    upstream-correct, every known defect is either fixed
                    behind a flag or explicitly blocked on client-owned
                    constants, and the validation checklist is ready to run
                    the day the card arrives.

Recommended next GPU tests:
1. Verify the node checkout against upstream 400fd31 + checkpoint metadata,
   then run the prompt-fix A/Bs (G1-G5, G8) — they gate every flag flip.
2. Seam-density experiment (G6) + music-video seam-pin sweep — the two
   cheapest structural quality levers for long video.
3. Cost/reliability levers before any pricing decision: --offload none and
   --max-batch-size on the LoRA tiers, and the audio-tier co-tenancy cell
   with ACE-Step stopped.
```
