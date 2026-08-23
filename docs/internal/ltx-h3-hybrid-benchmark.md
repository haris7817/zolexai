# LTX → H3: the hybrid, and why it is shaped this way

**Date:** 24 August 2026 · **Branch:** `dual-engine-benchmark-prep` · **No GPU.**

A third benchmark strategy: run LTX, decode its output to ordinary RGB, and
hand that to H3 as one reference among the customer's own. It exists to answer
one question.

> **Does running LTX first improve H3's result enough to justify a second
> generation pass?**

Not "can it be made to work" — of course it can. The honest default answer is
**no**: a hybrid pays both engines' inference, both model loads and a switch
between them, and it has to earn that back in quality, identity or adherence
before it is worth anything to anyone. Nothing in the implementation assumes
it does, and **hybrid is not a provider**: it has no entry in the routing
table, `auto` cannot reach it, and no customer can be served by it.

---

## 1. Why decoded RGB, and why not latents

The tempting version of this idea is to feed LTX's latents into H3's sampler
and skip a decode. It is rejected, and not on grounds of effort.

LTX and H3 are different models with different VAEs. LTX's video VAE
compresses 32×32 spatially and 8× temporally (which is where its 8k+1 frame
rule comes from); H3 ships its own visual and audio VAEs with their own
geometry. There is no shared space, no published mapping between them, and no
reason for one model's latent channels to mean anything in the other's. A
tensor that happens to have a compatible *shape* is not a compatible
*representation* — it is noise with the right dimensions, and the failure mode
is confident garbage rather than an error.

Decoded RGB is the only interface the two genuinely share. It is what H3's
reference conditioning is documented to take, and it is what our own chain
already produces at every seam. So the handoff boundary is pixels:

```text
LTX  →  decoded RGB video / selected frames  →  H3 reference conditioning
```

## 2. The rule that makes the comparison mean anything

The interesting hypothesis is *not* "LTX made a video, give it to H3". It is:

```text
the customer's photograph   owns  WHO
the LTX draft               owns  approximately HOW IT MOVES
the prompt                  owns  WHAT HAPPENS
H3 regenerates from all three
```

Collapse that into "hand H3 the draft" and the original identity asset is
gone. H3 is then conditioned on our own invention, and an identity score
measures how faithfully H3 reproduced an LTX hallucination — a number that
looks like a result and is not one.

So the implementation enforces provenance rather than trusting it:

- every reference carries `origin`: `user_asset` or `generated_intermediate`;
- the handoff manifest lists `original_references` and `generated_references`
  as separate fields, so a reader can never confuse them;
- a hybrid that would drop an original asset is **refused**, not compiled;
- the customer's own track is always the audio reference on the final pass —
  the draft's generated audio is discarded, because a music video measured
  against invented audio measures nothing.

## 3. Where hybrid is tested, and where it is not

19 of 41 cases carry a third cell. Each has a stated reason, and the omissions
are decisions rather than oversights.

| Workflow | Hybrid? | Why |
|---|---|---|
| **Image to video** (B1–B6, B8) | yes | the cleanest separation of the two engines' strengths: the photograph owns identity, the draft may own motion |
| **Reference-person V2V** (D1–D5) | yes, priority | LTX has *no* identity input at all and composites the person into a frame; H3 takes them as a subject reference. Whether LTX's structural draft adds anything to H3's own reading of the source is the open question |
| **Music video** (E1–E3) | yes | H3 caps a generation *and* its audio at 15 s, so a song is already a chain; a draft may carry performance staging that H3 regenerates against the real vocal |
| **Premium T2V** (A2, A3, A5, A6) | yes, four only | a cinematic human scene, fast action, a difficult camera move, a multi-shot sequence — where a draft could supply motion structure H3 would otherwise invent from text |
| **Extend** | **no** | H3 documents continuation as a native Ref2VA task type; the source clip already *is* the reference, and an LTX draft of a continuation is a draft of the thing under test |
| **Standard V2V** | **no** | a plain restyle already hands H3 the source video. A draft would be a restyle of a restyle and neither engine's contribution would be attributable |
| **Everything else** | no | testing hybrid everywhere would spend the GPU budget on cells whose answer is already "no" |

## 4. The handoff form is a variable, not a setting

What of the draft H3 actually sees is one of the things the benchmark is for.
More references are **not** obviously better: H3 counts every file against its
12-file ceiling, a draft clip spends one of only three video slots, and a
draft carrying the whole shot may pin H3 to LTX's mistakes as firmly as to its
virtues.

| Form | What it hands over |
|---|---|
| `full_video` | the whole decoded window (≤ 15 s, H3's per-clip limit) |
| `first_frame` / `last_frame` / `first_and_last` | cheap structural anchors; H3's FL2VA takes exactly these |
| `keyframes` | motion as a sequence of positions rather than a clip |
| `video_plus_original_image` | the draft for motion, the customer's photograph for identity |

Defaults are **per workflow**, because the hypothesis differs by workflow —
image-to-video wants the draft *and* the photograph, while a reference-person
restyle already holds its identity asset and needs only the draft's structure.
(A single global default was the first thing the dry run proved wrong.)

## 5. Section mapping

The two engines section differently — LTX at 30 s, H3 at 15 s — so the handoff
does not assume they line up. Each **final** section takes its own window of
the decoded draft:

```text
60s image-to-video
  LTX draft :  [0–30]           [30–60]              2 sections
  H3 final  :  [0–15] [15–30]   [30–45] [45–60]      4 sections
  handoff   :  each H3 section takes its own window of the decoded draft
```

## 6. Cost accounting

A hybrid records **both** passes: LTX generation, LTX decode, handoff
preparation, H3 reference preparation, H3 generation, H3 decode, and the model
switch. Comparing only the H3 half of a hybrid against a whole LTX run is the
easiest way to reach a wrong conclusion, so `RunRecord` carries every leg and
the total is their sum.

Two derived numbers do the arguing:

- `incremental_quality_gain(baseline, candidate)` — overall, or one metric
- `incremental_cost_ratio(baseline, candidate)` — wall-clock multiple

```text
H3 direct 9.1 quality / 8 min   ·  hybrid 9.2 / 14 min   → +0.1 for 1.75x: no
H3 direct 7.4 identity          ·  hybrid 9.1 identity   → may be worth 1.75x
```

**The threshold is deliberately not defined.** It is a product decision, and
hard-coding one now would make it look measured.

## 7. What is implemented, and what is not

Implemented, exercisable today with no GPU: the strategy enum, scope and
rationale tables, the handoff manifest with provenance, per-workflow default
forms, section mapping, H3-limit inheritance (a hybrid ending in H3 inherits
every one of H3's refusals), the benchmark cells, cost/quality helpers, and
the dry-run output in `scripts/dual_engine_bench.py`.

Not implemented, on purpose: any inference, any decode, any frame extraction,
any fake result, and any latent path between the two models.

**Verified, 24 August.** Full worker suite **879 passed / 1 failed / 1
skipped** — exactly 35 more passes than the pre-hybrid baseline of 844, which
is the 35 new tests, with the same single pre-existing failure (this machine's
LAME encoder makes a 4.0 s MP3 probe at exactly 4.0 s). The LTX golden argv
snapshot is green, and `git diff` against the baseline commit touches no
adapter, longform, director or workflow file: the hybrid is additive in the
strict sense.

## 8. Unresolved — for the GPU

1. Does hybrid beat direct H3 anywhere, and by enough?
2. Which handoff form wins, per workflow — and does the answer differ between
   identity-critical and motion-critical cases?
3. What does the model switch actually cost on one card, and does it change
   the answer for short clips?
4. Does an LTX draft *harm* H3 by pinning it to LTX's failure modes (drift,
   presence blindness) rather than only its structure?
5. On music video: does H3 `fully_copy` beat LTX a2vid's measured goal-B sync,
   and does a draft help or merely cost?
