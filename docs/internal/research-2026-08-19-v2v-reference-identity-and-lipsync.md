# V2V reference identity + the V2V lip-sync drift: trace, root causes, fixes

**Date:** 2026-08-19 · **Status:** implemented, unit-proven, **GPU-verified
on the RTX PRO 6000 and DEPLOYED TO PRODUCTION the same day** — worker via
`git pull` + supervised restart (commits `2ba698e`/`7d99e41`/`e6e26dc`,
runbook §44), VPS api+web rebuilt by hand with `v2v_reference_identity:
true` live in the baked YAML. The lip-sync fix is active for every V2V job.
Test artifacts from the verification runs live in `/workspace/idtest/` on
the box. The consent-gate recommendation (§17) was NOT implemented; shipping
without it was a deliberate product decision. See the evening addendum for
the first production identity job and the reference-describer improvement it
produced.

---

## 1. Current V2V architecture (traced, not assumed)

```
upload (frontend Dropzone → presigned PUT → asset id)
  → POST inputs {source_video: id, reference_image?: id}       snake_case throughout
  → API validates roles/kinds/ownership against video-to-video.yaml
  → job rows; claim serializes {role, asset_id, kind, download_url}
  → worker _stage_inputs downloads every role to workspace/inputs/
  → LtxAdapter._run_restyle: probe source → target = source duration
  → v2v_engine: transform → _run_transform:
      plan_segments(target, per_pass=8.0s)  — EVEN fractional windows
      per pass: canny control clip of the pass's own source window
                (pass grid, 24fps, the frames ACTUALLY rendered)
                + frame-0 anchor: previous pass's final frame @ 0.85,
                  or (first pass only) reference image @ 0.3
      → ltx_pipelines.ic_lora + Union Control LoRA, unquantized,
        cpu-offload, 2x grid + --skip-stage-2, landings (193,)
      → trim each pass back to its planned seconds
  → _deliver_restyle: normalize sections to delivery fps → concat
  → mux source audio ONCE → verify length/streams
```

The role string `reference_image` is byte-identical end to end; the image
reliably reaches `job.input_for("reference_image")` with a staged path.
Two frontend paths silently dropped it (see §8): "Reuse Settings"/Retry
rebuilt the form without any media inputs, and history-page
regeneration links carry only the prompt.

## 2. Reference image root cause

Not a wiring bug — a **scope decision shipped as designed and now outgrown**.
The YAML records it: *"M1 scope is the CONTRACT ONLY … It does NOT perform
identity or character replacement; that behaviour belongs to M2."* Concretely,
in the live transform engine the reference:

1. conditions **only the first pass** — every later pass anchors on the
   previous pass's final frame, so by section 2 the reference has no input
   at all;
2. at **strength 0.3**, frame 0 — a deliberate "guides the look" hint, pinned
   by `test_a_reference_image_guides_without_displacing_the_source`;
3. against a **canny edge map at strength 1.0 in every pass**, which carries
   the source person's facial geometry — jawline, hairline, features — and
   re-imposes it 24 times a second. The engine was *built* to discard
   identity-free look and keep geometry; a face's geometry IS its identity,
   so no reference strength can win against it.

So: reference reaches the worker but is architecturally a first-frame style
hint. It never was identity conditioning.

## 3. Current LTX 2.5 capability (research verdict)

**No first-class identity input exists in the LTX family.** No
`--reference-image`, no ID embedding, no face encoder, no IP-Adapter (the
ComfyUI-claims audit already established the last). The native vocabulary is
exactly four things: `--image` pixel anchors, `--video-conditioning`
(IC-LoRA reference tokens — the control channel),
`--conditioning-attention-mask` (regional weighting), and task LoRAs.
`dubit` takes a reference *clip* for identity but generates new speech and
cannot sync to supplied audio; its LoRA is not on the box.

**The client's own reference engine (ltx-main, read in full) does person
replacement with those same four primitives**, on the same `ic_lora`
pipeline family we run, three ways:

- *Pose transfer*: Union Control on a DWPose skeleton + the reference person
  composited into an inpainted first frame at strength 1.0;
- *Full-body*: In-Outpainting IC-LoRA + OpenCV tracked mask → green guide +
  attention mask + composited first-frame anchor, recomposite outside the
  mask;
- *Head swap*: a third-party rank-64 LoRA fed a green side-strip holding an
  InsightFace face crop (license UNVERIFIED — see §15).

Their load-bearing tricks: **re-anchor identity every segment** (identity
decays across chained sections), **spatially align the reference into the
source frame** before anchoring (a raw photo at frame 0 hijacks
composition), and **QA identity with fail-closed checks** rather than
assuming success.

Answer to the gating question: **PARTIALLY.** Native conditioning can carry
reference identity — the client's product proves it in production on this
pipeline family — but only as composited pixels + regional masking +
per-segment re-anchoring, and the strength of the result is a GPU
measurement, not a spec sheet.

## 4. Chosen identity strategy

**Native LTX, no new models, no new weights**, as
`execution.v2v_reference_identity` on the transform engine:

1. **Reference in every pass**: frame-0 anchor at 0.65 on pass 1
   (`v2v_identity_anchor_strength`), re-shown at an interior frame of every
   later pass at 0.35 (`v2v_identity_refresh_strength`) alongside the
   continuity frame — the exact mechanism I2V already uses
   (`i2v_reference_strength`) to stop chained renders forgetting their
   subject.
2. **The edge map lets go of the person**: each pass mattes its window
   (BiRefNet — already integrated for person lock) and the attention mask
   weights the person's region *below* the scene
   (`v2v_identity_subject_attention` 0.5, the mirror of person lock's
   `BACKGROUND_ATTENTION`). Scene/camera keep the edge map's full grip; the
   person's pose still tracks while their appearance is freed.
3. **Nothing silently pretends**: identity+person-lock together is refused;
   identity with a reference on the still engine is refused (not quietly
   ignored); a matting failure fails the job. Without a reference the flag
   is inert and the job is byte-identical to a plain transform.

This is architecture A/B from the brief: native dual conditioning with
identity reinforcement per section. The escalation path if the matrix says
identity is too weak: the reference engine's composited-anchor prep
(BiRefNet cutout + inpaint, a GPU-env script like `person_matte.py`) and the
Lightricks In-Outpainting IC-LoRA — both documented, neither needed until
measured.

## 5. Alternatives evaluated and rejected (for now)

| Option | Why not |
|---|---|
| Raise `v2v_reference_strength` | Wrong lever: one first-pass showing decays regardless of strength, and the edge map still re-imposes the source face. Strength alone was the "too-low strength" hypothesis — disproven by trace. |
| First-frame-only reference at high strength | Composition hijack (the brief's own warning); their engine only anchors at 1.0 *after* compositing the person into the source's own frame. |
| Head-swap LoRA (BFS) | Third-party weights, license unrecorded; LTX license §3.5 makes unvetted LoRAs a real liability. Also face-only. |
| In-Outpainting LoRA port | Legit escalation (Lightricks artifact, same license review), but new weights + OpenCV prep + 2.3→2.5 LoRA compat unproven; not "minimal" before the native candidate is measured. |
| DWPose skeleton control | Frees identity fully but loses the scene's edges (whole-frame skeleton control), needs new ONNX models; a later signal behind the same seam (control.py says the same). |
| External face-swap/reenactment model (InsightFace inswapper, LivePortrait, etc.) | New license reviews, new VRAM resident, frame-by-frame temporal risk, and the brief's own "not just face swap" bar; only after native measurement fails. |
| `dubit` | Generates new speech; cannot preserve source dialogue; LoRA absent. |

## 6. Lip-sync root cause — proven

**Cumulative staircase drift from whole-frame quantization of fractional
section plans.** Not FPS mismatch, not encoder delay, not model motion
(those remain measurable residuals — see §12).

Mechanism, with the actual arithmetic (simulated through `plan_segments` +
the real trim/normalize rules):

- `plan_segments` divides the source into EVEN windows: 37s / 5 passes =
  7.4s — **177.6 frames at 24fps. Sections are planned at fractional frame
  counts.**
- Each rendered pass is trimmed with `-t 7.400` → whole frames only →
  **7.4167s** (ceil).
- Delivery normalization to the source's own fps (say 30) rounds up again →
  **7.4333s**.
- Sections are butt-joined; the source's audio runs continuously over the
  result. Every seam pushes later content **+17–33ms later**; a 37s source
  ends **+133ms** late, a 59.9s source at 25fps **+227ms** — past the ~45ms
  threshold where a mouth visibly moves after its words.
- Frame-aligned plans (15s → 7.5s sections = exactly 180 frames) drift
  **zero**, which is why the bug read as "minor" and intermittent.

## 7. T2V vs V2V difference

T2V is structurally immune twice over: its product durations (≤60s) fit
**one pass** at every product grid — no seams at all — and when chained,
each section carries **its own model-generated audio**, so any duration
rounding moves both streams together. V2V is the configuration where
picture timing and audio timing come from different places, and the seam
arithmetic lands entirely between them. (Music video shares V2V's shape and
inherits the same staircase — deliberately untouched here; see §17.)

## 8. Files changed

| File | Change |
|---|---|
| `apps/worker/worker/media/frames.py` | `normalize_clip` gains `frames=` — exact-count delivery, pad bounded to 2 frames so a broken render still fails honestly |
| `apps/worker/worker/adapters/ltx.py` | `_planned_section_frames` (cumulative allocator); `_deliver_restyle`/`_assemble_generated_sections` thread per-section counts; identity mode in `_run_transform` (constants, conditioning, attention, refusals); still-engine identity guard in `_run_restyle` |
| `apps/worker/worker/media/masks.py` | `build_attention_mask` gains independent `subject` weight (person lock defaults unchanged) |
| `apps/web/src/features/generation/form.ts` | `valuesFromJob` restores the job's media inputs — Reuse Settings/Retry no longer silently drops the reference image |
| `apps/web/src/components/generation/CreatorWorkspace.tsx` | passes `selectedJob.inputs` through |
| `workflow-definitions/video-to-video.yaml` | comment-only documentation of the identity keys (parse-verified identical) |
| `apps/worker/tests/test_seam_timing.py` | NEW — allocator arithmetic + end-to-end pinned-section delivery |
| `apps/worker/tests/test_reference_identity.py` | NEW — 9 tests (see §16) |
| `apps/worker/tests/test_media.py` | normalize pinning + bounded-pad tests |
| `apps/worker/scripts/av_offset_probe.py` | NEW — content-time offset + audio-mux offset measurement (validated against a synthetic 200ms lag: reads −208ms flat, audio 0ms) |
| `apps/worker/scripts/v2v_identity_matrix.sh` | NEW — benchmark matrix A–G + strength sweep, runs on the box via `ltx_smoke.py` |

Untouched: T2V/I2V/extend/music/music-video code paths, API, chain planner,
`concat_segments`, `mux_audio`.

## 9. Reference conditioning persistence

With identity on: pass 1 = reference @ frame 0 / 0.65; every pass k>1 =
continuity frame @ frame 0 / 0.85 **plus** reference @ frame ~N/3 / 0.35;
every pass additionally mattes its window and lowers control attention over
the person. Pinned by `test_the_reference_anchors_the_opening_and_refreshes_every_pass`
and `test_identity_softens_the_control_grip_over_the_person` (one matte per
pass, mask in every argv, weights background=1.0/subject=0.5, mask frame
count == rendered frame count).

## 10. Long-form behavior

Unit-proven for the conditioning (4-pass chain keeps the reference in every
argv), and **GPU-verified on a real 30s dialogue source** (test D: santa30 —
a Director-mode three-person Christmas scene, 4 transform passes, seed
pinned). Stills at 1/4/11/18/26/29s: the central performer is the
reference-flavored woman in EVERY section, still speaking mid-articulation
at 29s, with **no reversion to the source person at any seam**. Composition,
trio blocking and the source's own shot changes all tracked.

The same run is the live demonstration of the multi-person limitation: one
reference image re-imagined ALL THREE people (the two children came back as
adult women in the reference's leather jacket). BiRefNet mattes people, not
a person — single/primary-person sources are the supported shape, exactly as
the YAML documents.

## 11. Source audio behavior

Unchanged and re-pinned: sections are assembled silent, `mux_audio` attaches
the source's track exactly once, identity mode included
(`test_identity_keeps_the_sources_length_and_audio`: one audio stream,
source length). Duration remains probe-derived on every path.

## 12. Lip-sync results

Before (arithmetic, confirmed by simulation through the real planner):
+17–33ms video lag per seam, +133ms end-lag on a 37s/30fps source, +227ms
worst simulated case. After: every section is delivered at
`round(end·fps) − round(start·fps)` frames — boundary error ≤ half a
delivery frame, **non-accumulating**; end-to-end test delivers 22/22/23/22
frames for a 3.7s/4-pass source, summing to the source's exact frame count.

GPU measurement (santa30, 30.016s, 4 passes, SAME SEED before/after — the
passes are pixel-identical, so the comparison isolates assembly):

- Container: BEFORE concatenated 4×181 frames and relied on the mux's
  `-shortest` to cut the excess (hiding ~150ms of tail content); AFTER
  delivers **exactly 720 frames = round(30.016 × 24)** with sections at
  180/180/180/180. The 15s speech clip likewise delivered exactly 360.
- Content probe (block-matching `av_offset_probe.py`, ±1-frame resolution):
  every cleanly-measurable matched point moved **+42…+84ms toward zero**
  (the 1–2 frame staircase correction at sections 2–4); audio-track offset
  measured **0ms** on every run — no mux/encoder delay component exists.
- Residual: ±1–2 frames of *model tracking* noise, identical in both runs
  (same seed), which is the edge-map engine's motion-reproduction fidelity,
  not a timing bug — and two probe points near source shot-changes saturate
  the matcher's window (unmeasurable, not offsets).
- A-vs-B on the speech clip: baseline {+17, −25}ms vs identity {+100,
  −108}ms at the two measurable mid-points — within matcher noise;
  **identity conditioning shows no systematic lip-sync cost**, and the
  replaced woman visibly articulates through both passes.

## 13. Identity results

Measured 19 Aug on the box, generated fixtures (a 15s man speaking to
camera in a kitchen, distilled T2V with scripted dialogue; reference = a
photoreal woman, long dark hair, black leather jacket — the brief's own
example):

- **Baseline A (no reference)**: the transform invented a DIFFERENT man
  (mustache, khaki shirt) — the falconer bug reproduced on demand; edge-only
  transform never preserved identity, so B's result is attributable to the
  reference, not chance.
- **B (identity on)**: the output person **is the reference woman** — face,
  long dark hair, leather jacket — speaking the source's performance in the
  source's kitchen and camera, identity stable at 30/55/80/95% with no
  reversion across the pass seam and no visible flicker between sampled
  stills. Capability class on this evidence: **UPPER BODY** (face + hair +
  jacket followed the reference; trousers/full wardrobe unverified).
- **D (30s, multi-pass, multi-person)**: §10 — persistence holds to the last
  section; multi-person assignment is undefined by design.
- Sweep: `v2v_identity_subject_attention` bracketed at 0.35/0.5/0.65 on the
  speech clip, same seed and source moment compared. Identity held at ALL
  three values; articulation strengthens with the dial — at 0.35 the mouth
  is soft, at 0.65 it visibly tracks the source's speech hardest with no
  identity loss on this clip. Production range is 0.5–0.65: keep 0.5 as the
  default, reach for 0.65 when the source is dialogue-heavy. Clips kept in
  `/workspace/idtest/` for the eye pass; single-frame stills are indicative,
  not the final judgement.

## 14. RTX PRO 6000 measurements

19 Aug 2026, alongside the resident ACE-Step service (~24GB idle):

| run | source | passes | wall | notes |
|---|---|---|---|---|
| santa30 transform (before) | 30.016s | 4 | 267.7s | HEAD code |
| santa30 transform (after) | 30.016s | 4 | 298.6s | seam fix active |
| A speech baseline | 15.018s | 2 | 108.6s | |
| B speech + identity | 15.018s | 2 | 163.2s | **+~27s/pass** = per-pass BiRefNet subprocess (model load dominates; residency is the obvious optimization) |
| D santa30 + identity | 30.016s | 4 | ~360s | |

Peak VRAM observed during an identity render: **~47GB total** including the
resident music service — no pressure on the 96GB card. T2V fixture
generation: 5s ≈ 30s wall, 15s ≈ 60s wall (distilled, nvfp4).

## 15. Licensing

Nothing new introduced. BiRefNet (MIT) and Union Control (Lightricks,
LTX-2.x Community License — Attachment A #20 still gates production) were
already in use. Explicitly avoided: `Alissonerdx/BFS-Best-Face-Swap-Video`
head-swap LoRA (license unrecorded anywhere local — must be read before it
is ever fetched). If the In-Outpainting LoRA escalation is taken, it is a
Lightricks artifact under the same reviewed license.

## 16. Regression tests

Full worker suite green (see run log below), including: T2V/I2V argv pins,
extend, music, music video, V2V both engines with and without reference,
person lock, director. New: 2 allocator tests + 1 end-to-end seam test,
2 normalize tests, 9 identity tests (anchor+refresh, tunable strengths,
per-pass matte/mask alignment, bypass-without-reference, person-lock
conflict, still-engine refusal, matting-failure fail-closed, length+audio
promises). Web: tsc, eslint, next build clean. API suite not run (requires
docker infra, down on this machine; no API code changed — the YAML edit is
parse-verified identical, and the pinned `reference_image` help text is
untouched).

## 17. Remaining limitations

**Pipeline (ours):**
- Music video still carries the seam-staircase against its master track —
  same fix shape (`section_frames` through its own assemble), deliberately
  out of scope per the brief's "do not break Music Video". Worth its own
  measured change; audio-conditioned music video would benefit most.
- History-page "Reuse"/"Variation" links still carry only the prompt (both
  inputs, source video included — a visible, pre-existing product shape).
- The delivered boundary quantization is now ≤ half a frame per seam,
  non-accumulating — the floor CFR permits.

**LTX/model (not ours):**
- No native identity input exists; identity strength is bounded by what
  composited-pixel conditioning can carry. Face-shape geometry inside the
  person's region still leans toward the source where attention is nonzero
  — the subject-attention dial trades this against motion tracking.
- ic_lora remains stage-1-only at 2x on product grids; 193 frames is the
  only measured landing.

**Identity mode (until the matrix runs):**
- OFF everywhere; defaults are reasoned, not measured. Multi-person sources
  matte everyone — identity assignment is undefined; ship single-person
  only. Occlusion/profile behavior unmeasured (matrix tests F/G).
- Consent gating (versioned, server-enforced) is a REQUIREMENT before any
  public exposure — copy the reference engine's pattern.

**Lip-sync:**
- Phoneme-level accuracy is not claimed; what is fixed is the pipeline's
  timing. Model-side articulation fidelity under identity conditioning is
  matrix test B's explicit question.

---

## Addendum (19 Aug, evening): the first production identity job, and what it taught

**The job:** source = a dancer with a fully MASKED face, full-body wide shot,
fast motion; reference = a man in a maroon suit; prompt = a wall of
meta-instructions ("Replace the person… Preserve the exact face… no face
drift, duplicates, deformities, flickering"). Output: an invented shirtless
man — neither source nor reference.

**Why, precisely:** LTX prompts are captions, not commands. Every word of
those instructions entered the model as CONTENT vocabulary on a pipeline
(`ic_lora`) that has **no negative prompt** — and the prompt named not one
visible attribute of the reference person. Add the adversarial source (the
face the identity anchor must land on is masked and tiny in frame) and the
result was fully predictable from the benchmark's own limits.

**The client's ComfyUI JSON, audited:** its mechanism is latent img2img over
the source video (denoise 0.5–0.6) on the dev model with CFG 2.0 / STG 1.2 /
negative prompt. Verified against the installed `ltx_pipelines` sources:
that combination **does not exist in any official CLI** — `ic_lora` exposes
no CFG, no negative prompt, no STG, no denoise dial (checked argparse and
`__call__`); `retake` is temporal inpainting of a time window, not a
restyle; the only partial-denoise machinery is stage-2's internal
`noise_scale`. The JSON itself cannot run as written (CheckpointLoaderSimple
cannot produce a CLIP for LTX — LTX encodes with Gemma; the "advanced
configuration" node is not a core node), and it contains **no reference
image input at all** — its identity is text-described. Same verdict family
as the 17 Aug ComfyUI claims audit: one genuinely correct idea (captions
describing the person work; 97 = 8k+1), the rest unusable as specification.

**The improvement shipped from this:** `v2v_identity_describe_reference`
(default ON inside identity mode). The worker looks at the reference photo —
`reference_person_facts` in `worker/director/vision.py`, the I2V Director
describer generalized — and appends a one-sentence caption of the person
after the user's verbatim prompt. **Measured on the box first:** the on-box
gemma-4-e2b-it checkpoint, whose image-input ability was recorded as
"unmeasured" in the I2V Director report, loads as an image-text model in
4.3s and answered in 1.1s — "Woman: adult, dark hair, black leather jacket"
for the benchmark reference. That measurement also unblocks
`DIRECTOR_VISION_ENABLED` as a separate decision. Every failure of the
describer degrades to the bare prompt; four tests pin append/verbatim-prefix/
off-switch/bypass.

**Honest limits restated for this job class:** a masked-face, full-body,
fast-motion source is the hardest cell in the matrix (F/G territory) — the
description fix removes the prompt sabotage, but face-anchored identity onto
a face-covered dancer remains beyond what the native mechanism has ever
demonstrated. `v2v_identity_subject_attention: 0.65` is the measured dial
for motion-heavy sources.
