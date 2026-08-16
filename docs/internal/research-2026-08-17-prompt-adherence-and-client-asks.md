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

---

## Addendum (17 Aug): the client's own LTX 2.3 backend, read in full

The client shared their Node.js LTX 2.3 stack (controller/routes/schema/
service/validation). We run 2.5; the value is seeing what they treat as table
stakes. What it changes:

**1. Their text-to-video default is `two_stage_hq`** — the guided pipeline,
with `negative_prompt`, step counts 15–40, and `fp8-cast`/`fp8-scaled-mm`
quantization. Every adherence comparison the client makes is against a GUIDED
render. This promotes the guided tier (P3) from "quality option" to "the gap
the client is actually measuring". Our distilled tier stays the fast default;
the guided tier is what "follows the prompt" means to them.

**2. Their prompt pipeline is layered, and enhancement is never silent.**
Normalizer → semantic compiler producing a **fingerprinted, human-approved
contract** (generation is refused if the approved plan drifted) → camera
director → LTX-native `enhance_prompt` with a fallback prompt. Confirms our
Tier A design (structure, don't paraphrase; keep the user's text verbatim) and
adds two ideas worth stealing later: an approval step for enhanced prompts,
and a post-completion "was it faithful?" feedback endpoint.

**3. Their Video Animate maps 1:1 onto the IC-LoRA pilot.** Targets
person/background/other; person modes replace/add/edit; up to 4 person
reference images; an **inpainting mask video — white regenerates, black
keeps** — which is exactly the shape of `ic_lora.py`'s
`--conditioning-attention-mask`; pose transfer via DWPose conversion; and a
head-swap engine for single-identity replacement. Their identity strategies
(`ltx_pose_transfer`, `ltx_full_body`, `face_lock`…) are productised names
over the same control-signal machinery we verified yesterday.

**4. Identity replacement requires explicit consent, audited.** A consent
checkbox with versioned audit strings, enforced server-side, before any person
swap runs. When our Transform mode ships person replacement, this is a
REQUIREMENT, not polish — copy the pattern.

**5. `extend/:jobId` — extension without re-upload.** They stream the engine's
own prior output back in as the source. Same pattern our "Generate Music
Video" button needs (we already have asset-id inputs, so ours is simpler), and
their extend menu is the same 5–60s ours is — "unlimited" in practice means
extend-of-extend chaining plus honest drift expectations.

**6. Their hard limits look like ours.** 60s per generation, 700 MB video /
100 MB audio uploads, 540p/720p/1080p. Useful for the restrictions table: our
ceilings are not out of line with the reference the client trusts.

---

## Addendum 2 (17 Aug): the reference PYTHON engine, read in full

`ltx-main/python` is the engine behind the client's reference product. It is
LTX 2.3 and we are 2.5, but the mechanics transfer. Findings in order of how
much they change for us:

**1. Our 8k+1 discovery is their stated invariant.** `_snap_frames`: "LTX
latent video scale requires (frames - 1) % 8 == 0", snap to nearest, ties up.
Audio-driven paths use `_snap_frames_ceil` because "snapping down can truncate
the conditioning audio and create one-latent mismatches" — the exact failure
our chunk-10 hit. Two days of measurement, stated in their wrapper as a code
comment. For the upstream report to Lightricks: the `distilled` entry point
does not snap, their own wrappers all do.

**2. LoRA + FP8/quantization is a KNOWN-BAD combination they guard against.**
`_effective_quantization` forces quantization to NONE whenever a LoRA is
loaded ("LoRA+FP8 fusion can use unsupported Triton fp8e4nv kernels") and they
fit the unquantized model with `--offload cpu` (their default). EVERY one of
our remaining a2vid (lip-sync) shape crashes ran distilled-LoRA under
nvfp4-cast. Next GPU experiment: a2vid at 1024x576, quantization none,
offload cpu — the resolution ceiling may simply vanish.

**3. Their music video IS our lip-sync prototype, productised.** Audio-driven
segments (default 30s on 2.3), each conditioned on its slice of the track;
first segment takes the user image at strength <=1.0 (0.65 on retry);
EVERY later segment pins the previous segment's final frame at strength 1.0;
per-segment seed = seed + index; conditioning audio is normalised to STEREO
first (mono is a known trap) and cut with +1s headroom; container padding
(90.04s probes for a 90s MP3) is absorbed by extending the last frame at
concat, never by rendering an extra segment.

**4. They deliberately send BARE per-segment prompts.** Tried and removed
section metadata: "segment index and timing are orchestration metadata …
adding universal prose only dilutes the user's scene". Continuity comes from
the pinned frame. Their EXTENSION prompt does append continuity prose (nearly
identical wording to our continuation block). Direct tension with our
LONG-FORM CONTINUATION headers → A/B on the GPU: headers vs bare + pinned
frame, same seed, same track. Whichever wins, wins.

**5. Person replacement mechanics, fully mapped.**
   - *Pose transfer:* source video → DWPose ONNX (yolox_l + dw-ll_ucoco_384)
     → OpenPose-style skeleton video → Union Control IC-LoRA conditions on the
     skeleton; new person's photo is the first-frame anchor. "Union Control
     does not consume ordinary RGB as pose conditioning — it expects an
     aligned control video (DWPose/OpenPose, depth, or Canny)."
   - *Full-body replacement:* the **In-Outpainting IC-LoRA**
     (`Lightricks/…IC-LoRA-in-outpainting-0.9`, a repo we did not know) +
     CPU-side OpenCV prep: tracked person boxes → loose temporal mask → GREEN
     inpainting guide → composite that preserves every pixel outside the
     edited region.
   - *Head swap:* a rank-64 LoRA (`head_swap_v3_rank_64.safetensors`).
   All three are LoRAs on the distilled model plus deterministic CPU prep —
   no exotic runtime. Their enhancer Gemma is `gemma-3-12b-it` (ours would be
   the gemma4 E2B-it path per the 2.5 flag help).

**6. Assorted keepers.** Native-enhancer failures retry ONCE with a validated
fallback prompt (the enhancer is never allowed to be the reason a job fails);
`negative_prompt_supported = pipeline not in {distilled, ic_lora}` matches our
understanding; quality ladders exist as explicit fallbacks with audit flags
(`quality_identity_fallback_used`, `motion_retry_used`) rather than silent
degradation.
