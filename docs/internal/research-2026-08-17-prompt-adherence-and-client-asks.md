# Research: prompt adherence, V2V editing, and the client's seven asks

**Date:** 2026-08-17 · **Status:** research complete, nothing implemented
**Rule for everything below:** the current pipelines stay untouched until a
replacement is measured on the GPU. Every new capability lands behind a private
`execution` flag or a new workflow id, never by mutating a working path.

---

## 1. Why generations don't follow the prompt (the root fact)

Our production runtime is `ltx_pipelines.distilled` — the CFG-distilled
checkpoint. Distillation **bakes the guidance in**: there is no
`--video-cfg-guidance-scale`, no `--negative-prompt`, no step count. The model
cannot be *pushed* toward a prompt at inference; it either understood the text
or it didn't. Every adherence complaint (music video ignoring direction, V2V
ignoring the requested change, colour/identity drift) is downstream of this
one fact.

Three levers exist, in ascending cost:

| Lever | Cost | What it fixes |
|---|---|---|
| Structure the prompt (enhancer) | ~free | most "ignored my prompt" cases |
| Guided tier (`ti2vid_two_stages_hq`, dev ckpt, CFG + negative prompt) | ~2× compute | hard adherence, negatives |
| IC-LoRA control (structure from a signal, content 100% from prompt) | ~distilled speed + preprocessing | V2V editing specifically |

Measured evidence for the first lever (16 Aug): "two cars" drifted colours;
the same scene held perfectly when the prompt used explicit counts, concrete
colour words, and each constraint stated twice. That is exactly the transform
an enhancer automates.

New evidence for the second (16 Aug, lip-sync session): the two-stage guided
pipeline **runs on this card** — dev transformer + distilled LoRA passed in
31s at 512×320. It was never wired because it had never been run; that excuse
is gone. Shape ceilings at production sizes still need probing (the decoder
shape bug applies to every pipeline; 896×512 proven, 1024×576 fails on the
audio variant).

---

## 2. V2V: the client wants a different product than we built

Our video-to-video is a **restyle**: stills lifted from the source pull the
output back toward the source's own composition. It is architecturally a
"same scene, new look" tool. "Replace the person", "change the background",
"add things" are **edits** — keep the motion, regenerate the content — and no
strength tuning of the restyle will produce them. The two knobs we sweep
(anchor density, structure strength) trade drift against style; neither can
swap a subject.

The client's reference video ("How to Replace Characters or Backgrounds in
Videos with LTX 2.3") demonstrates **IC-LoRA Union Control**: extract
structure signals (canny edges / depth / pose) from the source, condition on
those, and let the prompt supply all content. Subject swapped, motion kept.

**We already ship the pipeline.** `ltx_pipelines.ic_lora`:

- runs on the **distilled model** — the fast tier, not the 2× guided one;
- `--video-conditioning REF.mp4 STRENGTH` — the reference video;
- `--conditioning-attention-mask MASK.mp4 STRENGTH` — a **spatial mask**:
  0.0 = free generation in that region, 1.0 = follow the source. Mask the
  person → regenerate just the person. Mask the background → replace just the
  background. This is region editing, natively;
- `--skip-stage-2` for fast iteration at half resolution.

**What we do not have yet:**

1. **The IC-LoRA weights.** HF has `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`
   (canny+depth+pose in one adapter) and per-signal 19b variants. The 2.3
   HDR IC-LoRA is already referenced by our repo's own `hdr_ic_lora.py`, which
   suggests 2.3 LoRAs load against 2.5 — **must be verified on the box**, it
   is the single biggest unknown.
2. **A control-signal extractor.** Union Control conditions on
   canny/depth/pose *videos*. Canny is trivial (ffmpeg/OpenCV). Depth/pose
   need a model (the ComfyUI workflows use DepthCrafter). Canny-first is the
   pragmatic pilot: cheap, no new model, good for structure-preserving swaps.
3. **Mask generation** for region edits ("replace the person"): needs a
   segmentation pass (e.g. a person-matting model) or, v1, whole-frame
   transformation without masks — which already covers the reference video's
   headline demo.
4. **Measured shapes.** Same decoder bug family; the matrix must run for this
   pipeline before anything ships.

**Product shape:** a new mode/workflow ("Transform"), not a change to the
existing Restyle. Restyle stays for "same scene, new look"; Transform covers
"keep the motion, change the content". Both honest about what they do.

---

## 3. The prompt enhancer (client ask #7, and the cheapest big win)

Two tiers, both preserving the user's words verbatim as required:

**Tier A — worker-side structural enhancer (no new model, deterministic).**
Encode the measured prompt rules: extract subjects/counts/colours from the
user's prompt, restate each as a persistence constraint, order as
scene → subjects (counted, coloured) → action → camera → style, append the
user's original text verbatim. This is `plan_section_prompts`'s philosophy
applied within a single pass, and it is testable without a GPU. Rules, not a
model → it cannot hallucinate away details. Estimated: a day, including tests.

**Tier B — LTX's own Gemma enhancer.** `--enhance-prompt` +
`--prompt-enhancer-gemma-root`: because our text encoder is the gemma4
single-file, the enhancer needs a separate *generative* Gemma instruct
checkpoint (HF directory — the flag's own help names "gemma4 E2B-it"). That is
a download (gated repo, token exists), VRAM at generation time, and latency
per job. Worth piloting after Tier A, A/B on the same prompts.

UI: a toggle, **default on**, per the client's ask ("automatically structured
… while preserving the original idea"). The API contract keeps the raw prompt;
enhancement happens in the worker so the stored job always shows what the user
typed (directive: never silently rewrite what we store).

---

## 4. The seven client asks — feasibility and the honest cost

1. **Unlimited video extension.** The chain already renders arbitrary length;
   the 5s–60s menu is a product choice, and Extend-of-extend already works.
   Two real limits to respect: the source ceiling (now 330s, raisable once
   chained-input stability is measured) and **drift** — every 12–60s pass is
   a continuation, and identity degrades over many hops. Recommendation:
   raise the menu (e.g. +2m, +5m), keep a documented total ceiling, and be
   honest that a 10-minute extension will drift. Backend work: small.
2. **Lyrics language option.** ACE-Step 1.5 natively sings 50+ languages and
   auto-detects the lyric language, so the *model* is not the constraint —
   our lyric writer is: the template bank is English. Three steps:
   (a) expose the existing pass-through `lyrics` parameter in the UI (user
   pastes lyrics in any language — works today, zero model risk);
   (b) add a `language` parameter reaching the writer;
   (c) the real fix for *generated* non-English lyrics is the LLM writer
   (the deferred API-key decision) — template banks per language do not scale.
3. **"Generate Music Video" button under completed music.** Pure wiring: the
   music result is an asset; music-video takes `source_audio` by asset id.
   Frontend button + prefilled job. No model risk. Small.
4. **Review generation restrictions.** Split every restriction into
   *measured* (GPU ceilings, source length, frame lattice — keep, they are
   why the product stopped crashing) vs *product* (duration menus, aspect
   lists, upload caps — negotiable). Deliverable: a table for the client of
   which is which, then loosen the product ones they care about.
5. **Flashing ZolexAI logo while generating.** Frontend only.
6. **Multi-shot prompt behaviour.** `plan_section_prompts` already separates
   persistent constraints from sequential actions. Gaps: (a) explicit
   timestamped segments ("0:00–0:10 …") in the client's examples are not
   parsed — add a timestamp splitter that maps segments onto sections/musical
   boundaries; (b) single-pass jobs get no structuring at all — Tier A
   enhancer covers that.
7. **Prompt enhancer.** §3.

---

## 5. Sequencing (stability first, then capability)

Gate for every phase: full suite green + `ltx_matrix.sh` on any touched
pipeline + one real job through zolexai.com before the client sees it.

- **P0 — stabilise what exists** (the pre-requisite for "remains healthy"):
  V2V strength verdict (clips rendered, awaiting eyes), the VPS API drops
  (28/day — this will read as "broken" at volume regardless of model quality),
  music-video retest.
- **P1 — cheap, high-visibility:** Tier A enhancer · timestamp multi-shot ·
  Generate-Music-Video button · logo · restrictions table · lyrics
  pass-through + language param. No new models anywhere in P1.
- **P2 — IC-LoRA Transform pilot:** verify 2.3-LoRA-on-2.5 loads → canny
  control extractor → shape matrix → A/B against the client's reference
  video → new workflow behind a flag.
- **P3 — guided quality tier:** shape-probe `ti2vid_two_stages_hq` at
  production sizes → cost table → expose as a Quality level on T2V/I2V/MV.
- **P4 — lip-sync productisation** (recipe proven 16 Aug, awaiting the
  quality verdict on the 2-minute render) and the LLM lyric writer.

## Sources

- [LTX blog — IC-LoRA in LTX-2.5 ComfyUI workflow](https://ltx.io/blog/how-to-use-ic-lora-in-ltx-2)
- [RunComfy — LTX 2.3 IC-LoRA V2V motion-track workflow](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-ic-lora-in-comfyui-v2v-motion-track-video-workflow)
- [HF — Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control)
- [HF — LTX-2-19b IC-LoRA Depth](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Depth-Control) / [Pose](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control)
- [LTX blog — using LoRA adapters with LTX-2.5](https://ltx.io/blog/using-lora-adapters)
- [ACE-Step 1.5 — 50+ language support](https://ace-step.github.io/ace-step-v1.5.github.io/) · [GitHub](https://github.com/ace-step/ACE-Step-1.5)
- Local: `packages/ltx-pipelines/docs/pipelines.md`, `conditioning.md`,
  `ic_lora.py` CLI, `utils/args.py` (enhancer flags), plus the 16 Aug
  measurement sessions recorded in `docs/internal/`.
