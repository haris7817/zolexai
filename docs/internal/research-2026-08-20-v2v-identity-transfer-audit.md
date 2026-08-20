# V2V reference-image identity transfer: audit, research, benchmark, decision

**Date:** 20 August 2026
**Scope:** why an uploaded reference image does not reliably replace the person
in a video-to-video job, whether a purpose-built model does it better, and what
(if anything) ZolexAI should adopt.
**Status:** research only. Nothing deployed, nothing committed.

Hardware for every measurement: the production GPU box, RTX PRO 6000 Blackwell
Workstation Edition, 96 GB. The production worker stayed up throughout and held
~24 GB resident; VRAM figures below are whole-GPU readings and say so.

---

## 1. The existing V2V reference flow, exactly as implemented

Traced from the workflow definition through to delivery. Line references are to
the tree at `1e7f757`.

| Stage | What happens to `reference_image` |
| --- | --- |
| `workflow-definitions/video-to-video.yaml` | Declared as an optional input role, `kind: image`, ≤25 MB. `execution.v2v_reference_identity: true` is set. |
| API | Accepts and stores it as a declared role; hands the worker a signed URL. Nothing in the API interprets it — it is an opaque input. |
| Worker staging | Downloaded to the job workspace like any input. |
| `LtxAdapter._run_transform` ([ltx.py:1315](../../apps/worker/worker/adapters/ltx.py#L1315)) | `reference = await self._conditioning_image(job, "reference_image")` — decoded once to prove it is real media, then kept as a `Path`. |
| Identity gate ([ltx.py:1457](../../apps/worker/worker/adapters/ltx.py#L1457)) | `identity = bool(execution.v2v_reference_identity) and reference is not None`. Mutually exclusive with `v2v_person_lock`; refused on any engine other than `transform`. |
| Describer ([ltx.py:1469](../../apps/worker/worker/adapters/ltx.py#L1469)) | A vision model looks at the photo and its caption is appended **after** the user's verbatim prompt. Text, not pixels. |
| Composited anchor ([ltx.py:1512](../../apps/worker/worker/adapters/ltx.py#L1512)) | `build_identity_anchor()` mattes both images, inpaints the source person out of frame 0, and pastes the reference person into their bounding box. This composite becomes the anchor. |
| Conditioning ([ltx.py:1547](../../apps/worker/worker/adapters/ltx.py#L1547)) | **Pass 1:** `[ConditioningFrame(anchor, 0, 1.0)]`. **Every later pass:** `[ConditioningFrame(previous_frame, 0, 0.85)]` — the reference itself is *not* present. |
| Control | A canny edge map of the source drives the IC-LoRA Union Control adapter at strength **1.0**, for **every frame of every pass**. |
| Attention mask ([ltx.py:1614](../../apps/worker/worker/adapters/ltx.py#L1614)) | A BiRefNet person matte weights the control signal to **0.5** over the person, 1.0 elsewhere. |
| `_deliver_restyle` ([ltx.py:1688](../../apps/worker/worker/adapters/ltx.py#L1688)) | Assembles sections at planned frame counts, lays the source's audio down **once**, normalises to the source's duration. |

Answering the audit's questions directly:

- **Does it reach the API?** Yes. **Survive serialization?** Yes. **Does the
  worker receive it?** Yes, as a staged file, decode-verified.
- **Does LTX receive it?** Yes — as **one pixel keyframe at index 0 of pass 1**,
  and (since the composited anchor) not even as the photograph itself but as a
  composite built from it.
- **In which passes?** Pass 1 only. `_V2V_IDENTITY_REFRESH_STRENGTH` is `0.0`:
  re-showing the reference inside later passes was implemented, measured, and
  **retired**, because at 0.35 *and* at 0.2 it flashed the reference photograph
  into the middle of a customer's video at exactly `frames // 3`.
- **At what strength?** 1.0 for the composite; capped to 0.65 if the composite
  cannot be built and the raw photo is used instead.
- **Is it present in every long-form section?** **No.** Later sections see a
  continuity frame — a rendering of a rendering — plus the caption text.
- **How is it treated?** As **generic first-frame image conditioning**. Not as
  an identity reference, because no such input exists (§3).
- **Does source conditioning overwhelm it?** Yes, and the arithmetic is not
  close. See §2.

---

## 2. Root cause

Three independent facts, each proven rather than inferred.

**(a) There is no identity-conditioning mechanism in the installed runtime.**
A search of the whole installed LTX stack (`ltx-core`, `ltx-pipelines`) for
identity vocabulary — `identity`, `ip.?adapter`, `face.?embed`, `arcface`,
`clip_image`, `reference_image` — returns **zero** functional hits. Every
`identity` match is `torch.nn.Identity`, a colour-primary identity transform, or
a cache key. The single `reference` hit is the dubbing pipeline's
`--reference-video`, where "audio identity" means *which audio track*. There is
no face encoder, no image-embedding cross-attention, no adapter that could carry
"who this person is" as a distinct signal.

The consequence is structural: the reference image **can only enter as pixels in
a frame**. Not because the integration chose that, but because the model exposes
no other door.

**(b) The evidence asymmetry is roughly three orders of magnitude.**

| Signal | Frames it occupies | Passes | Strength |
| --- | --- | --- | --- |
| Reference (as anchor) | 1 | first only | 1.0 |
| Source edge map | every frame | every pass | 1.0, attenuated to 0.5 over the person |

On a 15-second job that is 1 conditioned frame against ~360 control frames. The
edge map is not a style hint — it is a per-frame statement of where the eyes,
nose, jaw and hairline are, i.e. precisely the geometry that makes a face *that*
person's. The attention mask at 0.5 halves its grip; it does not remove it.

**(c) Identity after frame 0 is carried only by propagation and by text.**
Later passes receive a continuity frame, so each section reproduces the previous
section's rendering. Errors compound in the direction the control signal keeps
suggesting — the source person. The describer caption was added precisely
because text was the only channel still pushing the other way.

### (d) A canny edge map leaks appearance; a skeleton cannot

This one deserves separating out, because it explains a customer report that
looked like a tuning problem and is not.

The control signal under V2V is a **canny edge map**. Edges are drawn wherever
contrast changes — which includes the source person's hair silhouette, the fall
of their clothing, the seams of their jacket, the outline of dreadlocks. The
model is told, at strength 1.0, "put contrast boundaries *here*". When the
prompt then says the person has different hair, the render satisfies both
instructions at once and produces the compromise: **the reference person wearing
the source person's silhouette**. That is the reported "braided hair curling on
both ears" and "it amended the hair on the sides" — not a strength that needed
lowering, but the signal itself carrying appearance it was never meant to carry.

A **pose skeleton** carries joint coordinates and nothing else. It cannot leak
hair, clothing or silhouette, because that information is not in it. Inspecting
Wan-Animate's own driving materials for the full-body fixture shows exactly
this: `src_pose.mp4` is a stick figure of limb positions, and the appearance
channel is a separate reference image.

This is a structural difference between the two approaches, not a quality
difference that tuning can close.

### What this predicts, and what was observed

The failure should be **framing-dependent**: when the face is large in frame,
few edges describe it and the anchor's pixels dominate; when the face is small,
the edge map wins. That is exactly the reported pattern — the same build that
produces a clean replacement on a head-and-shoulders source produced "it is not
even close" on a full-body dance video.

Re-verified for this audit on the 19 August fixtures (a man speaking in a
kitchen, 15 s; reference: a woman in a leather jacket):

- **Current build, head-and-shoulders source:** the woman appears in the
  kitchen, identity holds across the pass seam, mouth moves with the speech.
  This case is genuinely good.
- **The earlier pre-composite build on the same inputs:** frame 0 is the
  reference photograph's grey studio backdrop — the "reference photo appears in
  my video" report, visible in the fixture itself.

So the current path is not broken; it is **operating at the ceiling of what
first-frame conditioning can do**, and that ceiling is a function of how large
the face is in the source.

---

## 3. Native LTX capability

Classified against the audit's own scale, from the installed runtime's pipeline
list and argument surface (`ic_lora`, `ti2vid_*`, `a2vid_two_stage`, `retake`,
`keyframe_interpolation`, `dubit`, …):

| Capability | Rating | Evidence |
| --- | --- | --- |
| Image conditioning (first frame) | **STRONG** | `--image PATH FRAME STRENGTH`, proven in production. |
| Multiple image conditioning / keyframes | **PARTIAL** | Supported, but only measured frame counts decode with two conditioning images (`_TWO_IMAGE_SAFE_FRAMES` = {120, 240, 360}); 720 fails deterministically. |
| Video conditioning / structure control | **STRONG** | IC-LoRA Union Control with canny; this is what V2V rides on. |
| Per-region conditioning weight | **STRONG** | `--conditioning-attention-mask`. |
| Subject consistency over time | **PARTIAL** | Via propagation and captions; decays across sections. |
| **Reference-image identity conditioning** | **UNSUPPORTED** | No mechanism exists (§2a). |

`ref0.5` in `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` was checked as
a possible reference modality: it is part of the published checkpoint filename
and the sibling Motion-Track checkpoint carries the identical suffix. It is a
release tag, not a reference-conditioning feature.

**Verdict: LTX cannot natively do "source motion + reference identity" as a
first-class operation.** Everything currently shipped is a well-built
approximation of a capability the architecture does not have.

---

## 4. Candidate research

Licensing was treated as a gate, not a footnote, because ZolexAI is a
commercial SaaS.

### The decisive capability split

Candidates fall into two classes, and the class matters more than the quality:

- **Driving-video motion transfer** — consumes a source video's pose/expression
  and re-performs it with a new identity. Preserves timing, camera and cuts by
  construction. *Wan2.2-Animate, MimicMotion, LivePortrait (face only).*
- **Reference-to-video generation** — composes reference subjects into a
  *newly generated* video from a prompt. *SkyReels-A2 / V3-R2V, and VACE's R2V
  mode.*

The brief's requirement — "SOURCE VIDEO = WHAT HAPPENS, REFERENCE IMAGE = WHO
PERFORMS IT", with source timing, camera and audio preserved — disqualifies the
second class regardless of how good its output looks, because it does not take a
timeline from the source at all.

### Per candidate

| Model | Class | Code licence | Weights licence | Commercial | Fit |
| --- | --- | --- | --- | --- | --- |
| **Wan2.2-Animate-14B** | driving-video | Apache 2.0 | Apache 2.0 | **Yes** (see caveat) | **Exact** — animation *and* replacement modes |
| Wan-Animate-2 (7 Aug 2026) | driving-video | Apache 2.0 | Apache 2.0 | Yes | Newer; animation-focused, viewpoint control; replacement mode not documented |
| Wan2.1-VACE-14B | multi-task | Apache 2.0 | Apache 2.0 (Wan2.1 variants) | Yes | Reference + control, but no dedicated face-expression branch |
| SkyReels-A2 / V3-R2V | reference-to-video | — | — | — | **Wrong class** — no source timeline |
| MimicMotion | driving-video (pose) | Apache 2.0 | Apache 2.0 *except third-party* | Restricted | Built on Stable Video Diffusion; SVD's own terms govern. Pose only, no face branch, 2024-era quality |
| LivePortrait | face-only refinement | MIT | MIT | Yes *if* InsightFace removed | Bundled InsightFace `buffalo_l` is **non-commercial research only** |

**Why VACE was not benchmarked, stated as a reason rather than an omission.**
VACE's applicable mode for this task is masked video-to-video (MV2V): source
video + a mask of the region to edit + reference images + prompt. Its control
signals are the structural family — pose, depth, edges, masks. It has **no
dedicated face-motion or expression encoder**; a face inside the mask is
regenerated from structural control and text, which is *architecturally the
same bet the current LTX path already makes*. The benchmark then measured that
this bet already scores 0 ms / r = 0.947 on lip-sync (§5) — so VACE's plausible
upside is on identity alone, at 14B-parameter cost, on the axis where a
purpose-built alternative with a real face branch was already available and
tested. One further licence note: VACE's weights **inherit the base model's**
terms — `VACE-Wan2.1` variants are Apache 2.0, but `VACE-LTX-Video-0.9` is
**RAIL-M**, a use-restricted licence. Only the Wan2.1 variants are candidates.

**Wan2.2-Animate licence caveat, stated precisely.** `LICENSE.txt` in the
official repository is unmodified Apache 2.0 with no use-restriction appendix,
and the README says: *"We claim no rights over the your generated contents,
granting you the freedom to use them"*. The **project page** separately says the
project is *"intended solely for academic research and effect demonstration"*.
The licence is the operative instrument and it permits commercial use; the
project-page line is a demo disclaimer. Given the stakes, this is worth a
one-line sign-off from counsel rather than an engineer's reading.

### Sub-model licences — where the real trap is

Wan-Animate's preprocessing needs auxiliary models, and they do **not** all
share the parent licence:

| Component | Purpose | Licence | Verdict |
| --- | --- | --- | --- |
| `sam2/sam2_hiera_large.pt` | person segmentation / mask tracking | Apache 2.0 | Clean |
| `pose2d/vitpose_h_wholebody.onnx` | whole-body pose | Apache 2.0 (ViTPose) | Clean |
| `det/yolov10m.onnx` | person detection | **AGPL-3.0** | **Blocker as bundled** |
| `FLUX.1-Kontext-dev` | *optional* enhanced pose retargeting | **Non-commercial** | Avoidable — animation-mode only, and deliberately not downloaded |

The YOLOv10 detector is AGPL-3.0 upstream. Redistribution inside an Apache-2.0
repository does not relicense it, and AGPL's network-service clause is exactly
the clause a SaaS cannot ignore. It is also the *least* load-bearing component
in the stack — a person bounding box. ZolexAI already ships a permissive person
matter (BiRefNet, MIT) that produces the same information. **Any adoption must
replace this detector**; it is a small, contained piece of work, not a reason to
reject the model.

---

## 5. Benchmark

### Method

Same source, same reference, every arm. Two source classes, because §2 predicts
the failure is framing-dependent and a single fixture would have hidden that:

- **T1 — speaking, head-and-shoulders.** `speech-15s.mp4`, 15.0 s, 1024×576,
  24 fps, with audio: a man speaking to camera in a kitchen.
- **T2 — full-body motion.** `dance-10s.mp4`, 10 s, 1024×576, 24 fps: a man in
  a maroon suit dancing on a neon street, whole body in frame. This is the
  shape of the source that produced "it is not even close" in production.
- **Reference for both:** `reference.png` — a woman with long dark hair in a
  black leather jacket. A different sex, hair colour, hair length and wardrobe
  from either source, so partial success is easy to see.

Three things were measured rather than eyeballed.

**Lip-sync — mouth-motion cross-correlation.** Speech makes the mouth region
change frame to frame. That per-frame change is extracted as a 1-D signal from
the source and from each output, resampled to a common 24 fps, and
cross-correlated over ±0.5 s. The peak's *height* says whether the mouth moves
when the source's mouth moves; the peak's *lag* says whether it moves late.

The probe was validated against controls before any result was believed:

| Control | Expected | Measured |
| --- | --- | --- |
| An unrelated clip (the dance fixture) | no correlation | lag 500 ms, **r = 0.132** |
| The source deliberately delayed 250 ms | −250 ms, near-perfect | **lag −250 ms, r = 1.000** |

It detects a quarter-second offset exactly and reports near-zero for unrelated
motion, so its verdicts below are load-bearing.

**Background preservation — PSNR on a fixed background patch.** A 220×220 patch
of kitchen cabinets that no person enters, source against output, both at 24 fps.

**Runtime and VRAM.** Wall-clock per cell, with whole-GPU memory sampled every
5 s. **The production worker stayed resident throughout and holds ~24 GB**, so
every VRAM figure below is a whole-GPU reading that includes it.

### T1 — speaking source: measured results

| Arm | Lip-sync lag | Lip-sync r | Background PSNR | Wall time |
| --- | --- | --- | --- | --- |
| **A** — LTX transform, no reference | 0 ms | **0.970** | 10.7 dB | — |
| **B** — LTX identity, raw-photo anchor (superseded) | 0 ms | 0.921 | 6.1 dB | — |
| **B′** — LTX identity, composited anchor (**shipping today**) | **0 ms** | **0.947** | **31.2 dB** | **218 s** |
| **E** — Wan2.2-Animate, replacement mode | **0 ms** | 0.930 | **32.6 dB** | **1642 s** |

Three findings here matter more than the ranking.

1. **Lip-sync is not the problem, and it is not a reason to adopt anything.**
   The shipping LTX path tracks the source's mouth at **0 ms** with r = 0.947 —
   *above* Wan-Animate's 0.930 on the identical fixture. The "minor V2V
   lip-sync issue" in the brief was a pipeline-timing defect (fractional
   section lengths against one continuous soundtrack), already fixed by
   frame-pinned section delivery; the model's facial motion was never the
   fault. Both arms lose a little mouth fidelity versus the no-reference
   baseline (0.970), which is the honest cost of replacing a face at all.
2. **Background preservation is close, not decisive.** 32.6 dB against
   31.2 dB. The intuition that "replacement mode keeps the real scene and LTX
   re-renders it" is *directionally* right and *practically* marginal on this
   fixture. The superseded raw-photo anchor's 6.1 dB is the number that shows
   what a real regression looks like — that arm opened on the reference
   photograph's grey studio backdrop, which is the "the photo appears in my
   video" report.
3. **The cost gap is 7.5×.** 1642 s against 218 s for the same 15 seconds of
   output. Caveats in both directions: Wan ran with `--offload_model True
   --t5_cpu` and without a FlashAttention build (SDPA fallback), so this is a
   pessimistic upper bound for Wan; but it is also a fair reading of what a
   *first* integration would cost.

**Audio.** Wan-Animate emits a video-only file (`ffprobe`: one h264 stream, no
audio). It therefore cannot generate replacement audio, restart it, or
duplicate it — the source's own track would still be laid down exactly once by
`_deliver_restyle`.

**Frame rate.** Wan-Animate resamples to its own 30 fps (450 frames for a 24 fps
15 s source, delivered as 15.000 s). Any integration must resample back to the
source's rate; ZolexAI's delivery contract is the source's length *and* its
timebase.

### D — animation mode, for completeness

Animation mode was run on the same speaking fixture. It does what it says: the
reference woman performs the man's speech and expressions — **against the
reference photograph's own grey studio backdrop**. The kitchen is gone
(background PSNR 6.2 dB). Wall time 1510 s.

This settles which mode is relevant. The brief asks to "replace the source actor
while retaining the source scene", and that is **replacement** mode. Animation
mode is a different product.

### T2 — full-body dance: the case that actually fails

This is where the audit earned its keep, and the first attempt at these cells
was wrong in a way worth recording: `ltx_smoke.py` constructs `execution` as
`{"runtime": "ltx"}` and does **not** read the workflow YAML, so a run that
looks like V2V silently exercises the old still-conditioned restyle with a weak
0.3 reference hint. Those results were discarded and the cells re-run with
`v2v_engine` and `v2v_reference_identity` passed explicitly.

| Arm | Result on the full-body dance | Wall time |
| --- | --- | --- |
| **B** — LTX identity, shipped defaults | **Broken.** Two people: the dancer, *plus* a motionless woman's head-and-shoulders standing in the road for the entire video | 199 s |
| **C** — LTX identity, `subject_attention` 0.2 | **Broken identically.** Loosening the edge map's grip changes nothing | 176 s |
| **E2** — Wan-Animate, replacement | **Works.** One dancer, replaced, choreography and street preserved — but wearing an invented pink dress, not the reference's leather jacket | 1109 s |
| **B′** — LTX identity, **after the fix below** | **Works.** One dancer, the reference woman, **in her black leather jacket** | 176 s |

### Root cause of the full-body failure — found, and it is a bug

Cell C failing exactly like cell B is the clue: if loosening the control signal
changes nothing, the control signal is not what is wrong. Building the anchor
directly and **looking at it** found the fault immediately.

`scripts/person_anchor.py` composites the reference person into the source
person's bounding box and **bottom-aligns** them, so "their feet meet the same
ground":

```python
scale = min(FIT_WIDTH * box_w / cutout.width, FIT_HEIGHT * box_h / cutout.height)
paste_y = sy1 - size[1]          # the source person's feet
```

That is correct only if the reference photo *shows feet*. For a
head-and-shoulders photo the cutout's lowest row is a **crop edge across the
chest**, so the bust is scaled to the dancer's width and planted where the
dancer's feet are. The generated anchor for the dance fixture is a street with
a grey smudge where the dancer was inpainted out, and **a disembodied bust
sitting on the asphalt**. The model then rendered exactly that, faithfully, for
ten seconds.

A headshot against a full-body clip is the commonest thing a customer uploads.
This was the common case, not an edge case, and it is the true content of the
report "it is not even close… we need to enforce that it must take the image
from the reference" — the reference *was* being used, at full strength, in the
wrong place.

### The fix

Align by the end of the person the photo actually shows:

- Detect truncation — the reference's matte reaching its own photo's bottom
  edge means the subject is a crop, not a whole figure.
- A **whole** figure keeps today's behaviour: fitted inside the box,
  bottom-aligned, feet to the same floor.
- A **truncated** one is scaled **across the shoulders** (width is the only
  correspondence a bust shares with a standing figure) and hung from the
  **top** of the source person's box, putting its head where the head already
  is. The body continues past the box; the control signal states the body's
  pose for every frame anyway.

Measured after the change, on the same fixtures:

| Fixture | Before | After |
| --- | --- | --- |
| Full-body dance | bust standing in the road all video, dancer unreplaced | one dancer, reference identity, correct wardrobe |
| Speaking 15 s (regression check) | r 0.947 / bg 31.23 dB | **r 0.943 / bg 31.14 dB** — unchanged within run-to-run variance |

The close-up case is undisturbed because a bust against a close-up source is
width-bound under both the old and new arithmetic, so its **scale never
changed**; only its vertical placement did, and there the old code was hanging
the reference's head below the source person's head anyway.

### The residual limit neither system beats

Cropping into the dancer's face at the same scale in all four arms shows the
same thing: **the face is roughly 20 × 25 pixels in the source.** Wan-Animate's
own driving material makes this explicit — its `src_face.mp4` for this fixture
is a blurred upscale, because there is nothing sharper to extract.

Sex, hair, build and wardrobe transfer at this framing. **A recognisable face
does not, in any arm, and this is an information limit rather than a model
limit.** Any promise of "the reference person's face" on a full-body wide shot
would be a promise the physics does not support.

### Scores, kept separate

Out of 10. Objective where a number exists; a stated human judgement where not.
No overall column, deliberately — the trade-offs are the finding.

| | LTX shipped | **LTX fixed** | Wan replace | Wan animate |
| --- | --- | --- | --- | --- |
| Identity, close-up | 8 | 8 | 8 | 8 |
| Identity, full-body | **1** | 6 | 6 | n/a |
| Temporal identity stability | 7 | 7 | 8 | 8 |
| Source motion preservation | 8 | 8 | **9** | 8 |
| Source camera / scene preservation | 8 (31.2 dB) | 8 (31.1 dB) | **9** (32.6 dB) | **0** (6.2 dB) |
| Lip-sync | **9** (r .947, 0 ms) | **9** (r .943, 0 ms) | 9 (r .930, 0 ms) | 7 (r .723, 0 ms) |
| Face quality, full-body | 2 | 3 | 3 | n/a |
| Clothing / appearance fidelity | 3 | **8** (leather jacket) | 5 (invented dress) | 7 |
| Long-form stability (30 s, 4 sections) | — | 6 — holds across seams, drifts after a hard cut | untested >15 s | untested |
| Runtime, 15 s clip | **218 s** | ~172 s | 1642 s | 1510 s |
| Runtime, 10 s clip | 199 s | **176 s** | 1109 s | — |
| Peak VRAM (whole GPU, incl. ~24 GB worker) | 91 GB | 91 GB | 81 GB | 77 GB |

Wan's runtimes carry `--offload_model True --t5_cpu` and an SDPA fallback (no
FlashAttention build), so they are a pessimistic upper bound — but also a fair
estimate of what a first integration costs.

---

## 6. Winning architecture

**The source video keeps supplying structure and timing; the reference supplies
who; and the reference must be placed in the frame where a person of that crop
belongs.** The third clause is the one that was missing, and it was missing in
forty lines of compositing arithmetic rather than in the model.

Everything the brief asks to protect is protected by machinery that already
exists and that no provider can bypass. `_deliver_restyle` assembles sections at
planned frame counts, normalises each to the source's own frame rate, lays the
source's audio down **exactly once** via `mux_audio`, and verifies duration and
dimensions against `OutputExpectation`. Any future engine that returns video
sections inherits all of it. That is the seam a second provider would attach to
— not a rewrite.

---

## 7. Decision

**DECISION A — fix the current LTX path. Do not add a second model now.**

The brief allows all three outcomes and warns against forcing B. The evidence
points at A without much ambiguity:

1. **The reported failure was a bug, not a capability gap.** A headshot was
   being planted at a dancer's feet. Fixing the placement turned the worst
   fixture from "two people, one a floating bust" into a correct replacement,
   at **176 s** and with the reference's **wardrobe** carried across — which
   the purpose-built 14B model got wrong on the same clip.
2. **On close framing, LTX was already at parity and 7.5× cheaper.** Lip-sync
   0 ms in both, marginally better in LTX; background within 1.4 dB.
3. **Lip-sync never justified a new model.** It measures 0 ms on the shipping
   path. The "minor V2V lip-sync issue" was the fractional-section timing
   defect already fixed, not the model's facial motion.
4. **The remaining full-body gap is resolution, not architecture.** A 20-pixel
   face does not become a recognisable person in any arm.

**Wan2.2-Animate is nevertheless a validated option, and the work to prove it is
done.** It is installed, patched to run without FlashAttention, benchmarked, and
licence-audited. It genuinely does one thing LTX cannot: a pose skeleton carries
no appearance, so it cannot leak the source person's hair or clothing silhouette
into the render (§2d) — the mechanism behind the "braided hair on both ears"
report. If that class of artifact persists after the anchor fix, the provider
seam in §6 is where it goes. **Recommendation: revisit only on evidence, and
not before the anchor fix has been seen against real customer footage.**

Adopting it today would mean 68 GB of weights, a second environment, an AGPL
detector to replace, 30 fps resampling, single-person-only masks, and ~6× the
GPU cost — to fix a bug that forty lines of arithmetic fixed instead.

---

## 8. What changed

Two files. Additive, isolated, and confined to the composited anchor.

| File | Change |
| --- | --- |
| [apps/worker/scripts/person_anchor.py](../../apps/worker/scripts/person_anchor.py) | `REFERENCE_TRUNCATION_TOLERANCE`, `is_truncated()`, `place_cutout()`; the composite now aligns a cropped reference by the head and scales it across the shoulders. Logs which branch it took. |
| [apps/worker/tests/test_person_anchor.py](../../apps/worker/tests/test_person_anchor.py) | New. 8 tests over the placement geometry — the module keeps its heavy imports inside functions, so this runs without torch or a GPU. |

Nothing else was touched. No workflow definition, no adapter, no API, no
frontend. The no-reference V2V path does not reach this code at all.

**Not changed, on purpose:** `workflow-definitions/video-to-video.yaml`
documents `v2v_identity_anchor_strength` as "default 0.65" and
`v2v_identity_refresh_strength` as "default 0.2", while the code says **1.0**
and **0.0**. The refresh knob was retired to 0.0 after it flashed a studio
portrait into a customer's video twice in one evening; an operator trusting that
comment could re-enable the exact failure. It should be corrected — but that
file carries permanent local edits on the VPS and editing it here creates a
stash-pop conflict at the next deploy (runbook §44.1), so it is reported rather
than quietly changed.

---

## 9. Answers to the brief's remaining questions

**Reference propagation.** Reaches the API as a declared role
(`inputs: dict[RoleName, uuid.UUID]` — the API special-cases nothing), survives
serialization, arrives at the worker as a staged file, is decode-verified, and
reaches LTX as one conditioning frame. Verified in code and by the fixture runs.

**Source audio.** Attached exactly once, by `mux_audio`, after silent picture is
assembled — structurally, not by convention. Wan-Animate outputs a video-only
file (`ffprobe`: one h264 stream), so even that path could not have generated,
restarted or duplicated audio.

**Source duration and motion.** The 10 s dance returned 10.005 s and the 15 s
speaker 15.0 s; choreography and camera track the source in every arm.

**Long-form and cuts.** Measured after the fix on a 30 s clip that contains
several hard cuts (a woman by a car on a desert highway, cutting to an in-car
shot), rendered in **4 sections in 334 s**, delivered at **29.958 s** against a
29.973 s source.

| Point in the clip | Identity |
| --- | --- |
| 0 % | reference identity present; the anchor's ghost is also visible as a second face low in frame |
| 25 %, 50 % | holds — the reference woman, black leather jacket, across two pass seams |
| 75 %, 100 % (after a hard cut, in-car) | **drifts** — hair reads auburn rather than black, and the face is no longer clearly the reference |

So identity survives **pass seams** but degrades across a **hard cut**, which is
consistent with §2(c): after frame 0, identity travels on the continuity frame
and the caption, and a cut is precisely the moment the continuity frame stops
describing the new shot. This is a real, reproducible limit and it is *not*
fixed by the anchor change. The obvious next experiment — re-anchoring a freshly
composited frame at each detected cut rather than only at frame 0 — is now cheap
to try, because the compositing step knows how to place a person correctly.

Caveat on this fixture: the source's own performer is already a dark-haired
woman, so it is a weak identity-contrast test; the hair-colour shift is the
clearest signal available in it.

**Multi-person.** LTX mattes *all* people, so identity assignment is undefined;
Wan-Animate's own guide states its mask extraction is "designed for
single-person videos ONLY". Both are single-primary-person features. The
existing YAML help line already sets that expectation.

---

## 10. Remaining limitations

- **Full-body wide shots deliver appearance, not a recognisable face** — a
  ~20 px face carries no identity. Honest phrasing for the product: the result
  *looks like* the reference person; it is not a face swap.
- **Frame 0 carries the anchor's inpaint ghost.** TELEA inpainting of a
  whole-body region leaves a smudge that is visible for one or two frames and
  would be visible in a thumbnail taken at 0 s. Poster frames should come from
  ~0.5 s; a background-plate inpaint would remove it properly.
- **Wardrobe below the crop is invented** when the reference is a headshot —
  there is no information about it in the upload. The describer caption is what
  keeps it plausible.
- **Multi-person sources**: undefined assignment (above).
- **Identity drifts across a hard cut** (measured, §9). Pass seams are fine;
  cuts are not. Re-anchoring at detected cuts is the next experiment.
- **60 s and longer is still unmeasured** — 30 s is the longest run here.
- **The consent question is still open.** Replacing a person's identity from an
  uploaded photograph, on a commercial service, with no versioned consent gate,
  remains the recorded recommendation from 19 August and is unchanged by this
  work.

---

## 11. Regression status

Worker suite, `-p no:randomly`, run on a clean tree before the change and again
after it:

| | Passed | Failed | Skipped |
| --- | --- | --- | --- |
| Before | 772 | 1 | 1 |
| After | **780** | 1 | 1 |

The +8 are the new `test_person_anchor.py`. The single failure is the same one
in both runs —
`test_music_video.py::test_a_track_longer_than_one_pass_becomes_several_scenes`,
a pre-existing LAME/environment issue that reproduces on an unmodified tree and
is unrelated to this work.

Workflows exercised by that suite and unaffected: text-to-video,
image-to-video, video-to-video with and without a reference, Extend, Music,
Music Video and Director modes. The **no-reference V2V path never reaches the
changed code** — `build_identity_anchor` is only called under
`v2v_reference_identity` with a reference present.

Beyond the suite, the two GPU regression checks that matter were run on real
renders: the 15 s speaking fixture (unchanged within run-to-run variance) and a
30 s four-section job (correct duration, correct section count).

Nothing was deployed. Nothing was committed. The production checkout on the GPU
box was briefly modified during testing and **restored** (`git checkout --`,
verified clean); the measured runs reach the patched script through
`PERSON_ANCHOR_COMMAND`, which is why that seam is a command in the first place.


---

## 12. Addendum — the four-pass decay, and two attempts at it

A customer's 29-second job put the question precisely: **frame 0 was the
reference man exactly, and the closing passes were a stranger who merely
matched his description** — short cropped hair, clean-shaven, maroon jacket,
every attribute correct and the person wrong. At 8 s per pass that clip is four
passes, and §2(c) says only the first ever sees the reference.

Two things were ruled out by measurement before anything was built:

| Hypothesis | Test | Result |
| --- | --- | --- |
| The edge map's grip on the face | `v2v_identity_subject_attention` at 0.5 / 0.3 / 0.15 | **Not the lever** — all three transfer identity equally |
| Decay across a single seam | frames either side of the pass seam, 15 s clip | **Holds** — it takes four passes to degrade |

A third factor is worth recording because it shaped the earlier benchmark:
**every reference image in §5 was generated by LTX itself**, and the model
reproduces its own output easily. It also had a large appearance gap (a woman
reference against a man source), which forces the face to change. A real
photograph of a man against a source man of similar build gives the model no
such pressure — it can satisfy every word of the caption while keeping the
source's face. The fixtures were flattering the pipeline on exactly the axis
that matters.

### Attempt 1 — re-anchor every pass. Measured, failed, dropped.

Re-compositing the reference into each seam frame, so every pass restates the
real face. It kept one conditioning image per pass (two are safe only at the
measured decoder cells), preserved the seam's background, cost +16 s, and
passed 19 unit tests.

On the GPU it made things **worse**, twice:

1. Passes 2–4 became giant close-up portraits with the scene destroyed. The
   rebuilt anchor showed why — see below.
2. After fixing the scale (§ below), the rebuild still **inpainted out
   everything the matte had caught**: the car vanished from the seam frame and
   the next pass inherited a car-less desert.

**The mechanism is unsafe with a salient-object matte.** Every rebuild both
pastes a person and erases whatever else the matte swept up, on exactly the
frames where the matte is least trustworthy. Not shipped.

A methodology note worth keeping: the first two GPU runs of this experiment
silently executed **production** code. The worker tree had been copied to test
in isolation, but the copy's venv carries an editable-install finder pointing
back at the original path, so `sys.path` never got a look in and the patched
adapter was never loaded. `PYTHONPATH=<copy>` overrides it. A debug print that
failed to appear is what caught it.

### Attempt 2 — match heads, not bounding boxes. Measured, shipped.

The giant head exposed a real defect in the anchor, independent of
re-anchoring: **BiRefNet mattes the salient OBJECT, not a person.** On a seam
frame of a woman standing beside a car it returned the two as one region, so
"the source person's box" spanned most of the frame, and scaling a bust to that
width produced a head half the frame high.

The bounding box is therefore not a trustworthy handle. The **top of the matte
is the person's head** in any framing where a head is visible at all, so a
cropped reference is now scaled by matching its head band to the source's, and
positioned so the two heads line up — indifferent to whatever else the matte
included.

| Case | Result |
| --- | --- |
| The seam frame that produced the giant head | correctly-scaled figure, no giant head |
| Full-body dance (must stay fixed) | still one dancer, reference identity, leather jacket — 171 s |
| Close-up speaker (must not regress) | unchanged — 172 s |

Shipped with the box-width path kept as the fallback for when either matte is
not shaped like a person.

### What this leaves for the four-pass decay

Unfixed. The honest workaround today is framing the job: **a single-pass clip
(~8 s or less) never decays**, because the anchor is frame 0 and there is no
seam to lose the person across. Longer jobs still drift.

The next candidate is a **person-restricted** matte rather than a salient-object
one — which would make both the frame-0 anchor and any future re-anchoring
trustworthy. Until that exists, re-anchoring should stay out: it multiplies the
matte's mistakes by the number of passes.
