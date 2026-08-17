# The client's ComfyUI reference audited, and the guided quality tier wired

**Date:** 2026-08-17 · **Status:** audit complete; guided tier implemented
behind `execution.generation_engine: guided` (off everywhere), GPU-verified
mechanically; quality advantage **not yet demonstrated** — see §6 and §13,
they are the honest heart of this document.

Companion documents, not repeated here:
- `research-2026-08-17-v2v-music-video-lipsync.md` — V2V transform engine,
  music-video audio conditioning, lip-sync goals A/B/C (all implemented
  17 Aug, GPU-verified).
- `research-2026-08-17-prompt-adherence-and-client-asks.md` — the adherence
  root cause, the client's reference engine read in full, the seven asks.

---

## 1. Executive summary

The client sent a "LTX 2.5 — ComfyUI Complete Reference" (WhatsApp compilation,
17 Aug). It was treated as a set of hypotheses and audited claim by claim
against three sources: the client's own reference stack (`E:\Downloads\ltx-main`
— which turns out to be **LTX-2.3, no ComfyUI anywhere in it**), the LTX-2.5
runtime actually installed on our GPU node (read at source), and the official
ComfyUI/Lightricks releases.

Verdict in one line: **the document's hard rules are real and already
implemented here; its tuning tables are ComfyUI-specific expressions of
pipelines we already drive natively; its two workflow JSONs must not be
imported; and its architectural suggestions (MusicGen, SAM2, KSampler-denoise
V2V) are each either licence-blocked, superseded by a native mechanism, or
not applicable outside ComfyUI.**

Nearly everything the compilation asks the product to *do* was already built
and GPU-verified in the two prior sessions: V2V auto-duration and structural
transform, music-video auto-duration and audio conditioning, single-mux master
audio, prompt structuring with immutable facts, 8k+1 frame conforming. The one
capability that remained unreachable was the **guided tier** — CFG, STG and
negative prompts for T2V/I2V. It is now wired behind
`execution.generation_engine: guided`, off by default, with its own measured
frame landings and pass ceiling, at a measured cost of **~4.3x** the distilled
tier (not the ~10x previously estimated).

The surprise finding: on the two A/B probes run so far, **the guided tier at
its official defaults showed no adherence advantage over distilled + prompt
structuring — and on the camera-choreography probe it was *more* static than
distilled.** Enabling it in any workflow remains a measurement-driven product
decision, which is exactly why it ships off.

## 2. Existing architecture (before this session)

Fully documented in the two companion docs and in
`architecture-audit-2026-08-16.md`. The relevant summary:

- Six workflows, all `runtime: mock` in git; GPU routing is a deliberate,
  uncommitted VPS/worker-side flip (runbook §12/§16/§38).
- One adapter (`apps/worker/worker/adapters/ltx.py`) drives four LTX entry
  points as `LtxPipeline` descriptors: `_DISTILLED` (default, nvfp4-prequant),
  `_IC_LORA` (V2V transform), `_A2VID` (audio-conditioned music video), and —
  new today — `_GUIDED`.
- V2V and Music Video read their duration from the source probe; the API
  rejects a supplied duration; the frontend shows "Same as source video" /
  "Matches your audio". The client's auto-duration P0s were already met.
- Master audio is muxed exactly once, at the end, on both source-audio paths.
- Deterministic prompt structuring (counts, colours, persistence — immutable
  facts, verbatim user text) is on for t2v/i2v/extend/music-video.

## 3. Exact LTX runtime

Read from the installed source at
`/workspace/ltx2-benchmark/packages/ltx-pipelines` (the authority every claim
below is checked against):

| Component | Value |
|---|---|
| Package | `ltx_pipelines` (LTX-2.5 split layout) |
| Production entry point | `ltx_pipelines.distilled` — explicit 8-sigma schedule, unguided by construction |
| Transformer (serving) | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (18.7 GB, `--quantization nvfp4-prequant`) |
| Dev transformer (on box) | `ltx-2.5-22b-dev-transformer-bf16.safetensors` (42 GB) — first run TODAY |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (single-file Gemma-4 12B + projection) |
| Video VAE / Audio VAE | `ltx-2.5-video-vae-bf16` / `ltx-2.5-audio-vae-bf16` |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` |
| Guided defaults (source) | `num_inference_steps` 40 base / 30 for `ti2vid_two_stages` / 15 for hq-variant params; video CFG **3.0**, audio CFG **7.0**, STG scale **1.0**, blocks `[29]`, rescale 0.7, modality 3.0; default negative prompt constant |
| Samplers (source) | explicit sigma schedules; euler-ancestral stage 1 for some checkpoint generations; `res2s` for the HQ pipeline. **No scheduler/sampler selector, no DPM++ anything** |
| Frame rule | `(frames − 1) % 8 == 0` family, stated in `retake`/`dfr_layout`/trainer |
| Resolution rule | divisible by **32** (one-stage) / **64** (two-stage), from `utils/helpers.py` |
| Frame rate | 24 fixed in production; pipeline default 24.0, default 121 frames |
| Attention | NATTEN installed (see `issue-triton-na-kernel.md`) |

## 4. Client claims audit

Statuses: **SUPPORTED** · **PARTIALLY** · **UNSUPPORTED** · **OUTDATED** ·
**CHECKPOINT-SPECIFIC** · **COMFYUI-SPECIFIC** · **INCORRECT** ·
**NOT RELEVANT** (to ZolexAI).

| # | Claim | Status | Evidence | Decision |
|---|---|---|---|---|
| 1 | Distilled: 4–8 steps, degrades above 8 | CHECKPOINT/COMFYUI-SPECIFIC | Native distilled takes **no step count** — an explicit 8-sigma schedule (`distilled.py`, `DISTILLED_SIGMAS`); the client's own 2.3 engine never passes steps to distilled | Nothing to change; we already run the 8-sigma schedule |
| 2 | Full/Dev: 20–40 steps | SUPPORTED (range) | Source defaults: 40 base, 30 `ti2vid_two_stages`, 15 hq params; client engine restricted {15..40}. HF discussion has **no official numbers** — the doc's range is community-consistent, not official | Guided tier uses the pipeline default (30); `--num-inference-steps` exists if a sweep is ever wanted |
| 3 | Distilled CFG 1.0–1.5 | PARTIALLY | Distilled is CFG-distilled: guidance baked, effectively 1.0, **no knob at all** natively | No fake sliders (already policy: `settings.quality: false`) |
| 4 | Full/Dev CFG 3.0–3.5 | **SUPPORTED** | Official source default `cfg_scale=3.0` (video), 7.0 (audio) | Guided tier ships on the official default |
| 5 | CFG >5 causes flicker | PLAUSIBLE, UNVERIFIED | No official statement; rescale 0.7 default suggests high CFG needs rescaling | Not exposed; irrelevant until a CFG sweep is run |
| 6 | STG "3–4" (earlier client message) | PARTIALLY | STG **exists** on guided tiers (`--video-stg-guidance-scale`); official default is **1.0**, blocks [29] | Default kept; a sweep is now one flag away |
| 7 | Schedulers: Simple / LTXVScheduler / DPM++ 2M | COMFYUI-SPECIFIC | Zero scheduler/sampler selection in `ltx_pipelines`; nothing named DPM++ anywhere (2.5 source or client tree) | Ignore; native pipelines own their schedules |
| 8 | 8n+1 frame rule | **SUPPORTED** | Our measured tables; their `_snap_frames`; 2.5 `dfr_layout.py:71` | Already canonical (`conforming_frames`, one utility) |
| 9 | Resolution ÷32, ÷64 for two-stage | **SUPPORTED** | `utils/args.py:789,894`, `helpers.py:541-548`; client engine enforces /64 fail-closed | Already respected; plus our measured stage-2 even-latent-dim workaround at 1024x576 |
| 10 | Two-stage: half-res stage 1 → x2 latent upscale | **SUPPORTED** | All native two-stage pipelines; official upscaler file; client engine "request 2x and stop there" for IC-LoRA | Already used (a2vid, guided); `stage_1_only` workaround for ic_lora |
| 11 | I2V CFG 2.5–3.0 | UNSUPPORTED | No official I2V-specific CFG guidance found; distilled I2V has no CFG at all | No action |
| 12 | 50 FPS for I2V smooths motion onset | UNSUPPORTED / NOT RELEVANT | No official recommendation; production is 24 fps end-to-end; 50 fps would double per-second cost and halve pass ceilings | Rejected |
| 13 | V2V = VAEEncode → KSampler denoise 0.40–0.65 | COMFYUI-SPECIFIC | **No denoise-strength parameter exists anywhere in `ltx_pipelines`** or in the client's engine (their only "denoise" is a VoxCPM TTS boolean). Generic latent-video img2img is a ComfyUI construction | Rejected in favour of the native mechanism (IC-LoRA control conditioning — already shipped as the transform engine). Also root-cause-wrong for restyling: RGB carries the source's look, which is what the restyle must discard |
| 14 | V2V steps 10 / CFG 2.5 / euler / 0.50 denoise (JSON B) | COMFYUI-SPECIFIC | As above | Not imported |
| 15 | Audio VAE + joint AV latents | **SUPPORTED** | `ltx-2.5-audio-vae-bf16` on box; a2vid/AV cross-attention is the native mechanism; ComfyUI nodes (`LTXVAudioVAEEncode`, `LTXVConcatAVLatent`) verified real in official workflows | Already shipped (`execution.audio_conditioning`) via the native pipeline |
| 16 | Vocal-stem-only conditioning beats full mix | **NOT SUPPORTED** (measured 18 Aug) | A/B on the box (Demucs stem vs full mix, same seed, window spanning a measured vocal gap): mouth opens on vocals and closes in the gap **identically on both**. The client's own engine doesn't implement this either | Rejected for now — no `StemSeparatorProvider`. Re-probe with a dense heavy-percussion mix if customer results ever show beat-twitch. Full detail in Addendum 2 |
| 17 | Stem separators: Mel-Band RoFormer / UVR5 | NOT NEEDED | Consequence of 16 | Not added |
| 18 | Full song muxed once as final audio | **SUPPORTED** | Their `_replace_audio_from_source`; our `mux_audio` once at the end | Already implemented on both source-audio paths, tested |
| 19 | LipDub for music video | UNSUPPORTED | Native `dubit` has **no `--audio-path`** — it generates new speech from the prompt; it cannot sync to a supplied song. Client tree has no dubbing at all | Rejected for music video; a2vid audio conditioning is the correct native mechanism |
| 20 | IMG strength 0.70–0.85 (AV lip-sync) | PARTIALLY | Their engine: 0.75 default first segment, **1.0 forced** on chained segments, 0.65 retry clamp. Ours: first image ≤1.0, pinned frames 1.0 | In-family with ours; a strength sweep is possible but nothing indicates it is the binding constraint |
| 21 | SAM2 / BiRefNet masking | COMFYUI-SPECIFIC | Client's own engine uses **OpenCV HOG + DWPose ONNX + In-Outpainting IC-LoRA**, not SAM2/BiRefNet. Native region editing is `--conditioning-attention-mask` (white=regenerate) | Not added. If/when region edits ship, the native mask + in-outpainting LoRA path is first in line |
| 22 | High-denoise (0.75–0.90) masked background regen | COMFYUI-SPECIFIC | Their engine drives the same intent with mask videos at strength **1.0** on the in-outpainting LoRA | Not imported |
| 23 | IP-Adapter for LTX | INCORRECT | SD-family concept; no LTX IP-Adapter exists in the official ecosystem | Rejected |
| 24 | "ControlNet (Tile/Depth)" for LTX | INCORRECT NAME | The LTX mechanism is IC-LoRA control conditioning (canny/depth/pose) — which we ship | Concept already served natively |
| 25 | Motion bucket / motion scale | INCORRECT | SVD concept. Zero hits in 2.5 source and client tree | Never expose (matches existing `motion_strength: false`) |
| 26 | Negative prompts | CHECKPOINT-SPECIFIC | Dropped for distilled/ic_lora by the client's own engine (`pipeline not in {distilled, ic_lora}`); native distilled has no flag. Guided tiers support it with an official default constant | Available on the guided tier as of today; never faked on distilled |
| 27 | Hard-cut attention masking | UNSUPPORTED | No mechanism in source; LTX-2.5's **native multi-shot** (one chronological paragraph, named transitions) is the supported way to cut | Use multi-shot prompting; do not touch attention |
| 28 | MusicGen / AudioCraft for music | REJECTED | MusicGen weights are **CC-BY-NC** — commercially unusable; evaluated and ruled out in the ACE-Step benchmark (13 Aug) | Keep ACE-Step 1.5 XL (MIT, live in production, 50+ languages, ~1.5–5.5s per song) |
| 29 | Stable Audio for music | REJECTED | Stable Audio Open is instrumental-only; ruled out same benchmark | Keep ACE-Step |
| 30 | Frame tables 97/121/161/257 + durations | SUPPORTED | 8k+1 arithmetic | Already canonical |
| 31 | FPS lock across nodes / drift after 3–4s | SUPPORTED (mechanism) | Generic A/V timebase truth | Already enforced: 24 fps end-to-end, normalized concat, duration verification |
| 32 | Model files (`…comfy-int8-convrot`, gemma4-12b, audio VAE, upscaler) | SUPPORTED (ComfyUI artifacts) | Verified against docs.comfy.org — the filenames are the real official ComfyUI release | We run the split bf16/nvfp4 layout; nothing to change |
| 33 | Workflow JSON A (lip-sync) | DO NOT IMPORT | Malformed keys, duplicate audio links, checkpoint-loader vs Gemma inconsistency — the doc's own Appendix C lists 12 defects | Used as intent-documentation only; our native path supersedes it |
| 34 | Workflow JSON B (V2V) | DO NOT IMPORT | Well-formed but ComfyUI-only; no audio-latent path; KSampler denoise approach (see 13) | Same |

## 5. Root causes (confirmed, cumulative across the three 17 Aug docs)

1. Distilled tier is unguided by construction → adherence cannot be pushed at
   inference on the default tier.
2. Still-frame conditioning was the wrong signal for V2V restyling → fixed by
   the transform engine (edge-map control conditioning).
3. The music-video model never heard the song → fixed by `audio_conditioning`.
4. LoRA + quantization is a known-bad combination → encoded as a structural
   rule in `LtxPipeline`.
5. Frame-count safety is **per-pipeline**, not global — reconfirmed today by
   the newest pipeline on the box: 241 frames at 1024x576 is measured-safe on
   distilled, renders happily on a2vid, and is a reproduced illegal-memory-
   access crash on `ti2vid_two_stages`.
6. Prompt structure (explicit counts/colours, restated constraints) measurably
   fixes drift on the unguided tier — and today's probes suggest it already
   closes much of the gap the guided tier was expected to close.

## 6. T2V improvements

**Implemented:** the guided tier, `execution.generation_engine: guided`
(worker-only; YAML documents it commented-out; nothing routes to it yet).
Configuration: `ltx_pipelines.ti2vid_two_stages`, dev transformer + distilled
LoRA (the pipeline requires both), unquantized, `--offload cpu`,
`conforming_only`, `measured_landings=(121,)`, pass ceiling 5s
(`execution.guided_pass_seconds`).

**Benchmark (RTX PRO 6000, 17 Aug, 1024x576, 121 frames, same prompt+seed):**

| tier | wall | peak VRAM (incl. 24 GB resident music) | output |
|---|---|---|---|
| distilled (nvfp4-prequant) | **34s** | 44.4 GB | video+audio, 5.04s |
| guided (dev+LoRA, unquantized, offload) | **146s** | 39.3 GB | video+audio, 5.04s |

**Honest quality verdict — advantage NOT demonstrated yet.** Two probes:

- *Adherence probe* ("two red cars… camera behind… LA"): **both** tiers
  honoured every constraint at t=1s and t=4s. The prompt was written in the
  structured style production applies automatically — consistent with prompt
  structure, not guidance, having been the binding fix for this class.
- *Choreography probe* ("starts behind the hiker, orbits to a frontal
  close-up"): **guided locked a static frontal close-up for the whole clip;
  distilled executed a partial orbit** (side → frontal → moving close-up).
  One seed, one prompt — weak evidence, but in the wrong direction for
  "guided fixes static videos".

What the guided tier genuinely adds today is **reachability**: CFG, STG and
negative prompts were structurally unreachable before; now they are one YAML
line away, with correct weights, shape safety and tests. Turning it on — or
sweeping CFG/steps/negative prompts to find settings that DO beat distilled —
is the follow-up measurement, not this change.

## 7. I2V improvements

The guided engine covers I2V through the same handler: conditioning stills
(first-frame anchor + the low-strength mid-pass identity reference) ride along
unchanged, pinned by `test_guided_image_to_video_keeps_its_conditioning_stills`.
No claim of improved long-form continuity is made — the state-reset/ghosting
work remains driven by the continuation-vs-pinned-frame A/B recorded in the
prior docs.

## 8. V2V improvements

None this session; the transform engine (edge-map control conditioning,
auto-duration, single audio mux) shipped in the prior session and its repo
YAML has `v2v_engine: transform` enabled. Production activation remains an API
image rebuild — deliberately not performed (no deploys).

## 9. Music Video improvements

None this session; `audio_conditioning` shipped previously, off by default.
The stem-separation hypothesis (claims 16/17) is the one open experiment, and
the audit found it is **not** implemented in the client's own engine either.

## 10. Lip-sync findings

Unchanged from the companion doc: goal A delivered, goal B GPU-verified on a
real ACE-Step vocal track (mouth stops when the track ends), goal C
(phoneme-level, drift-free) **not claimed** pending a sync-offset measurement.
`dubit` remains unsuitable (cannot take supplied audio).

## 11. Audio architecture

Unchanged: conditioning audio (master, seeked per section) is strictly
separate from the delivered soundtrack (master muxed once). Stem separation,
if ever validated, slots in as a provider that feeds the *conditioning* input
only — the final mux must always be the user's original file.

## 12. Prompt enhancer

Already shipped as deterministic `structure_prompt` (immutable counts/colours,
verbatim user text first) plus per-section `plan_section_prompts` with
timestamp pinning. LTX's own Gemma enhancer stays available behind
`execution.enhance_prompt` (off: it paraphrases). Today's probes strengthen
the case that this layer, not guidance, is the workhorse of adherence.

## 13. Dynamic camera

The camera probe is the session's most instructive negative result: the
guided tier at official defaults was **more** static than distilled on an
explicit orbit instruction, and neither honoured "starts directly behind".
Conclusions: (a) do not sell the guided tier as the choreography fix;
(b) the productive levers to test next are multi-shot prompting (named
transitions in one chronological paragraph — native to 2.5), negative prompts
against static framing ("static camera, locked-off shot" as negatives on the
guided tier), and seed variety; (c) any cinematography planner should be
built on measured prompt patterns, not assumed model behaviour.

## 14. Inpainting / compositing

Not implemented, by decision (mega-brief allows "only if validated"). The
audit settled *which* mechanism is native when it is wanted: mask videos via
`--conditioning-attention-mask` + the In-Outpainting IC-LoRA (the client
engine's own person-replacement path), with SAM2/BiRefNet as optional mask
*sources* rather than requirements. Identity replacement additionally requires
the consent-gate pattern recorded in the prior doc.

## 15. RTX PRO 6000 benchmarks (17 Aug, this session)

| run | grid | frames | wall | peak VRAM | result |
|---|---|---|---|---|---|
| guided T2V | 1024x576 | 121 | 146s | 39.3 GB | PASS (video+audio) |
| distilled T2V | 1024x576 | 121 | 34s | 44.4 GB | PASS |
| guided T2V | 1024x576 | 241 | 141s | 54.4 GB | **FAIL** — illegal memory access in decode |
| guided camera probe | 1024x576 | 121 | 138s | — | PASS |
| distilled camera probe | 1024x576 | 121 | 31s | — | PASS |
| shape matrix (241 isolated retry; 121 @ 576x1024 / 768x768 / 512x640) | | | | | see addendum below |

Card context: ~24 GB idles resident (ACE-Step); everything above ran alongside
it with >40 GB headroom. Model residency (avoiding the per-pass 22B reload)
remains the single largest available performance win and is still not
implemented.

## 16. Files changed (this session)

- `apps/worker/worker/adapters/ltx.py` — `_GUIDED` pipeline descriptor,
  `_GUIDED_PASS_SECONDS`, `_guided_pass_seconds`, engine selection in
  `_run_generation`. Default argv untouched (pinned by existing tests).
- `apps/worker/tests/test_guided.py` — 9 new tests.
- `workflow-definitions/text-to-video.yaml`, `image-to-video.yaml` —
  comment-only documentation of the new keys (parsed YAML unchanged).
- `docs/internal/research-2026-08-17-comfyui-claims-audit-and-guided-tier.md`
  — this document.

## 17. Tests

| Suite | Command | Result |
|---|---|---|
| Worker (baseline, pre-change) | `pytest tests -q` | 531 passed, 1 skipped, 1 failed (pre-existing LAME/MP3 duration environment failure, documented in the companion doc §14) |
| New guided suite | `pytest tests/test_guided.py -q` | 9 passed |
| Adjacent pinning suites | `pytest tests/test_ltx.py tests/test_transform.py tests/test_music_video_audio.py tests/test_video_to_video.py -q` | 138 passed, 1 skipped |
| Worker (full, post-change) | `pytest tests -q` | **540 passed**, 1 skipped, 1 failed — the SAME pre-existing failure as baseline, nothing else |
| API | `pytest -q` (infra up) | **108 passed** |
| Frontend | `tsc --noEmit`, eslint `--max-warnings=0`, `next build` | all clean |
| Worker lint | `ruff check` on changed files | clean |
| Workflow YAML | parse check, all six files | clean |

## 18. Regression results

T2V / I2V / V2V / Extend / Music / Music Video defaults are all byte-identical
— the guided tier is unreachable without the execution flag, asserted directly
by `test_a_plain_generation_still_runs_the_distilled_tier` and
`test_an_unrecognised_engine_value_stays_on_the_default_tier`, and pinned by
the whole pre-existing suite. No API, frontend, storage, progress, history,
cancellation or lease code was touched.

## 19. Client suggestions rejected (with the technical reason)

1. **KSampler-denoise V2V** — no such control exists natively; RGB latent
   img2img is the wrong signal for restyling (measured); IC-LoRA control
   conditioning is the native, already-shipped mechanism.
2. **MusicGen / Stable Audio** — CC-BY-NC weights / instrumental-only;
   ACE-Step is live, MIT, and better on every axis we measured.
3. **IP-Adapter / ControlNet / motion bucket** — SD/SVD concepts that do not
   exist for LTX; the roles are filled natively (IC-LoRA, prompt structure).
4. **SAM2/BiRefNet as requirements** — the reference engine itself ships
   without them; native mask conditioning + CPU prep is the first-line path.
5. **50 FPS I2V** — doubles cost, halves pass ceilings, no official basis.
6. **Importing the ComfyUI JSONs** — defective as shipped (the doc's own
   Appendix C) and ComfyUI-only in their assumptions.
7. **LipDub for song sync** — `dubit` cannot consume supplied audio.
8. **Hard-cut attention masking** — no mechanism; native multi-shot prompting
   is the supported route.

## 20. Remaining limitations

**MODEL:** distilled has no guidance (by design); stage 2 cannot run at
1024x576 (odd latent dim — exact workaround in place); decoder bad-shape sets
are per-pipeline and non-monotonic (241 guided FAIL vs 121 PASS); `dubit`
cannot sync to supplied audio.

**PIPELINE:** guided landings measured at one count (121); guided quality
advantage unproven at defaults — needs a CFG/steps/negative-prompt sweep on
real failing prompts; stem-separation A/B unrun; lip-sync goal C unmeasured;
per-pass 22B reload (residency unimplemented) dominates short-pass cost.

**IMPLEMENTATION:** nothing in git routes to a real runtime (deliberate);
enabling any new tier requires the documented API-image rebuild; transform
engine still unproven on multi-pass chains over footage with hard cuts.

**QUALITY/RESEARCH:** camera choreography adherence is weak on both tiers;
multi-shot prompting untested; long-song identity drift beyond two sections
unmeasured.

## 21. Recommended production configuration

Unchanged defaults everywhere. Recommended next activations, in order, each
after its own measurement gate:

1. `v2v_engine: transform` (already in repo YAML) — activate via API image
   rebuild once the multi-pass/hard-cut chain test passes on real footage.
2. `audio_conditioning: true` for music-video — after the 3-minute drift
   check; price at ~4x the default tier.
3. `generation_engine: guided` — **only after** a sweep finds settings that
   measurably beat distilled+structuring on the client's failing prompts;
   price at ~4.3x.

## 22. Deployment readiness

**READY WITH LIMITATIONS.** All changes are additive, flag-gated, fully
tested, and the default behaviour is byte-identical. Nothing was deployed,
committed or pushed; production routing is untouched. The limitation is
honest: the new tier's quality advantage is not yet demonstrated, so its flag
should stay off until the sweep in §21.3 is run.

---

## Addendum: guided-tier shape matrix (completed)

121 frames passes at **all four product grids**; 241 at 1024x576 is a
**confirmed** fail (the isolated retry reproduced the illegal memory access,
ruling out contamination from the concurrently running camera probe).

| grid | frames | result | wall |
|---|---|---|---|
| 1024x576 (16:9) | 121 | PASS | 146s |
| 1024x576 (16:9) | 241 | **FAIL ×2** (incl. isolated retry) | — |
| 576x1024 (9:16) | 121 | PASS | 138s |
| 768x768 (1:1) | 121 | PASS | 149s |
| 512x640 (4:5) | 121 | PASS | 145s |

`_GUIDED.measured_landings=(121,)` is therefore measured at every grid the
product can request, and the 5s pass ceiling holds everywhere.

---

## Addendum 2 (18 Aug): the tuning sweep, the chain test, and the stem A/B

### The guided tuning sweep — CFG 4.5 unfreezes the camera

Ten runs, same seeds as the earlier probes (all passed, walls 85–149s guided /
31s distilled):

| variant | camera result on the orbit prompt |
|---|---|
| guided CFG 3.0 (default) | frozen frontal close-up (the earlier finding) |
| guided **CFG 4.5** | **clear arc around the subject** — ¾ side view with the sun left → frontal with the sunset centred behind |
| guided CFG 4.5 + anti-static negative | same arc; the negative adds little on top of CFG |
| guided anti-static negative alone | still frontal; background micro-drift only |
| guided STG 2.0 | slight movement, mostly frontal |
| guided 15 steps | static — but **85s vs 135–149s** (~37% cheaper), a real cost lever where motion doesn't matter |
| distilled | partial orbit (still the most motion of all variants) |

The suspected mechanism held up: the pipeline's **default negative prompt bans
"motion blur, camera shake, jittery movement, tilted camera"**, biasing guided
output toward locked-off shots — but removing those terms alone did not
restore motion; raising CFG to 4.5 did. No variant honoured "starts directly
behind" (the subject's face is visible from frame one everywhere) — that
instruction appears to be beyond both tiers at this length.

On the multi-constraint dance prompt, all three tiers honoured every
constraint (two people, green jacket, red dress, rain, night street); the
guided variants added letterbox bars — an artifact class the default negative
supposedly suppresses. The negative-prompt-capability probe (deserted square)
was a **null result**: distilled honoured "nobody anywhere" from the positive
prompt alone on this seed.

**Recommended guided configuration if/when the tier is enabled:**
`--video-cfg-guidance-scale 4.5`, default negative, default 30 steps. This
would land in the adapter as an `execution.guided_cfg` value — not yet added;
one more knob deserves one more measurement round on customer prompts first.

### Transform-engine multi-pass chain test — PASSED

A 14.03s source spliced from two distinct scenes with hard cuts at 5s and
10s, positioned so each of the two 7s transform passes crosses one mid-window.
Run through the real adapter (`ltx_smoke.py MODE=restyle`,
`v2v_engine: transform`), worker checkout at HEAD (`1fe0fa2`):

- rc=0, wall 122s, output **14.032s** (matches source), video+audio streams,
  source audio muxed once.
- **Both cuts survived at the correct timestamps** — the control clip carries
  scene cuts through the edge map, and a pass crossing a cut does not blend
  the scenes.
- The restyle ("charcoal sketch world") expressed as strong desaturation and
  tonal shift rather than full sketch texture, with minor edge artifacts near
  the first cut. Geometry-led restyles move lighting/colour/mood readily;
  material texture transforms remain partial. Product copy should promise
  "restyle", not "redraw".

This closes the "transform engine unproven on multi-pass chains over footage
with hard cuts" limitation from §20.

### Stem-separation A/B — the client's hypothesis is NOT SUPPORTED on this evidence

Design: a 60s ACE-Step ballad (solo female vocal) with a measured **2.7s vocal
gap at 38.4–41.1s where the instruments continue**; vocals isolated with
Demucs (MIT, htdemucs, 20s on CPU); then the SAME `a2vid_two_stage` pass
(seed 555, 1024x576, 481 frames, window 30–50s) rendered twice — once
conditioned on the full mix, once on the vocal stem only. The gap is the
discriminating window: a mouth driven by the vocal must close there; a mouth
confused by drums/bass keeps moving. Both renders passed (214s / 209s).

Frames at five timestamps (two inside the gap, three during vocals):

| song time | vocal state | full-mix mouth | stem-only mouth |
|---|---|---|---|
| 35.0s | singing (soft) | parted, mid-phrase | parted, mid-phrase |
| 39.5s | **GAP** (instruments only) | **closed, calm** | **closed, calm** |
| 40.2s | **GAP** | **closed** | **closed** |
| 43.0s | singing (full) | wide open, mid-phrase | wide open, mid-phrase |

**No discriminating difference at any timestamp.** The model's audio-video
cross-attention evidently separates the vocal from the arrangement on its
own; full-mix conditioning did not produce beat-induced mouth twitch on this
track. Identity also held near-identically across the two runs (same seed).

**Decision:** no `StemSeparatorProvider` — one fewer pipeline stage, licence
review and failure mode. Caveats recorded honestly: one track, one genre
(sparse ballad); the client's claim was specifically about mixes with heavy
drums and bass, so a dense EDM/hip-hop track is the follow-up probe if
customer results ever show beat-twitch. Claim 16 in §4 moves from UNVERIFIED
to **NOT SUPPORTED (single-track evidence, ballad-class)**.
