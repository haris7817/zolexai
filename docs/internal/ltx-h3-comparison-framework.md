# LTX 2.5 vs MiniMax H3 — the comparison framework

**Date:** 22 August 2026 · **Branch:** `dual-engine-benchmark-prep` · **No GPU.**

This document is the method, agreed before the hardware exists. Its companion,
[h3-pre-gpu-integration.md](h3-pre-gpu-integration.md), records what H3
officially is; the LTX baseline it is measured against is
[research-ltx25-zolexai-audit.md](research-ltx25-zolexai-audit.md).

**The question is not which model is better.** It is which model is better for
each ZolexAI workflow, and the honest answer may well be a mix. Nothing here
selects an engine; `auto` routes every workflow to LTX until this benchmark
has been run.

---

## 1. Provider architecture

```text
                        ZolexAI request
                              │
                    Director / semantic plan        ← ONE plan, engine-agnostic
                              │
                     Long-form orchestrator         ← sections, seams, windows
                              │
                       Provider router              ← auto | ltx | h3
                   ┌──────────┴──────────┐
              LTX provider            H3 provider
              (subprocess CLI)        (served model)
```

Three properties this shape exists to guarantee:

**One semantic plan, two compilations.** Characters, actions, dialogue, story
events, timing, camera intent, exits and scene outcome are decided once and
must be identical across engines. Only the *wording* differs, because the two
engines document incompatible prompt formats (§2.5 of the H3 document). A
benchmark that fed each engine a differently-planned scene would be measuring
our prose.

**H3 lives beside LTX, never inside it.** `worker/providers/` is a new package;
`worker/adapters/ltx.py` keeps owning execution. The LTX provider is a *reader*
of the shipped adapter — it asks the adapter for section counts, frame counts
and seeds rather than recomputing them, so the manifest can never drift from
what the GPU is actually sent.

**A manifest exists for every run.** `compile()` is pure and produces the plan
without running anything: provider, pipeline, duration, section plan,
references, audio windows, effective settings, and explicit notes where an
input the customer supplied never reaches the model. Every benchmark result
carries the manifest it ran under, so a score is always traceable to a plan.

### 1.1 Override

`provider=auto|ltx|h3`, from `parameters.provider` (per request, what QA uses)
or `execution.provider` (per workflow). The request wins. Internal only — not
exposed publicly. An unknown or unavailable engine is **refused, never
substituted**: a silent fallback would make an A/B compare LTX against LTX and
call it a tie.

### 1.2 Where the honesty checks are

- The H3 provider refuses what H3 does not document (4:5 frames, clips under
  4 s, more than 9 images / 3 videos / 3 audio / 12 files).
- The H3 manifest reports `steps`, `guidance` and `quantization` as
  `UNKNOWN — GPU validation required` rather than carrying a plausible number.
- The LTX manifest names inputs that never reach the model on the chosen path
  (a supplied track on a non-a2vid path is muxed, not conditioned).
- `structural_winner` is only declared where one engine *cannot* do something.
  `EMULATED` never loses to `NATIVE` by default — which of two chains looks
  better is a measurement.

## 2. Capability comparison

Generated from `worker/providers/capabilities.py` (28 rows; 20 marked as
requiring a GPU test). Condensed here to the rows that drive routing:

| Capability | LTX 2.5 | H3 | Structural winner | GPU test |
|---|---|---|---|---|
| Text to video | native | native | — | ✓ |
| Image to video | native | native | — | ✓ |
| First frame | native | native | — | ✓ |
| Last frame | native | native | — | ✓ |
| First **and** last frame | emulated | native | — | ✓ |
| Structural V2V | native | native | — | ✓ |
| Reference image (identity) | emulated (pixels in a frame) | native (subject reference) | — | ✓ |
| Multiple image references | emulated | native (≤ 9) | — | ✓ |
| Reference video | native (1 control channel) | native (≤ 3 clips) | — | ✓ |
| Reference audio | native (frozen latent) | native (2 modes) | — | ✓ |
| **Multimodal references in one request** | none | native | **H3** | |
| Audio-conditioned video | native | native (`fully_copy` only) | — | ✓ |
| Native audio generation | native | native | — | ✓ |
| Dialogue generation | native (5 validated languages) | native (11 stated stable) | — | ✓ |
| Lip-sync to supplied audio | native (goal B measured) | native (unmeasured) | — | ✓ |
| **Single pass over 15 s** | native | none | **LTX** | |
| 60-second long form | emulated (1 seam) | emulated (3 seams) | — | ✓ |
| Extend / continuation | emulated | native task type | — | ✓ |
| Structured camera control | none | none | — | ✓ |
| Person identity transfer | emulated | native | — | ✓ |
| Music video | native | native | — | ✓ |
| **Prompt timestamps** | none (forbidden) | native (prescribed) | **H3** | |
| **4:5 aspect** | native | none | **LTX** | |
| 4:3 aspect | emulated | native | — | |
| **2K output** | emulated (delivery upscale) | none in open release | **LTX** | |

Five structural winners out of 28 rows. Everything else is a measurement.

## 3. Benchmark cases

41 cases across 10 groups, defined in `worker/providers/benchmark.py`,
listed and dry-run by `scripts/dual_engine_bench.py`.

| Group | Cases | What it separates |
|---|---|---|
| **A** Text to video | A1–A9 | quality, adherence, action, multi-character, camera, multi-shot, 30 s and 60 s long form |
| **B** Image to video | B1–B8 | source fidelity, identity, action, camera, dialogue, 30 s and 60 s continuation |
| **C** Standard V2V | C1–C6 | style transformation vs source preservation, camera, dialogue timing, 30 s |
| **D** Reference-person V2V | D1–D5 | identity at four framings and across hard cuts — the strongest a-priori H3 case |
| **E** Music video | E1–E4 | 15 s / 30 s / 60 s / 2 min: singer identity, mouth timing, seams, cost |
| **F** Dialogue | F1–F4 | one speaker, turn-taking, emotion, reaction shots |
| **G** Camera | G1 × 20 terms | which camera concepts each engine actually honours |
| **H** Long form | H1–H2 | one global plan at 30 s and 60 s, including a mid-video departure |
| **I** Extend | I1 | our seam-frame construction vs H3's documented continuation task |
| **J** Multimodal | J1 | H3 only — recorded as a capability, not scored as a contest |

Every case names what it measures and how many runs it needs. Launch-critical
paths take five runs; nothing takes fewer than three.

### 3.1 Fairness rules

- Same source assets, same seed policy, same duration, same semantic plan.
- Each engine renders at its own documented native shape (LTX 1024×576, H3
  768-short-edge). Forcing one into the other's grid would measure the
  resampling, not the model.
- Section counts will differ — that *is* the finding, not a confound. Record
  sections, seams and per-section length with every run.
- A case one engine cannot express is marked single-engine rather than faked.

## 4. Scoring

Overall, 1–10 per component, weights summing to 100:

| Metric | Weight |
|---|---:|
| Visual quality | 15 % |
| Prompt adherence | 15 % |
| Temporal consistency | 15 % |
| Identity consistency | 15 % |
| Motion quality | 10 % |
| Camera adherence | 10 % |
| Source / reference fidelity | 10 % |
| Long-form continuity | 5 % |
| Seam quality | 5 % |

**Scored separately and never folded in:** `lip_sync` and `audio_response`. A
music video that looks lovely and does not follow the vocal has failed at the
thing it was for, and an average would hide exactly that.

Lip-sync uses this project's existing ladder so new numbers are comparable
with the ones already on record:

- **A** — audio exists in the output.
- **B** — mouth timing responds to the vocal. *(LTX a2vid measures here:
  −125…−208 ms, r≈0.43–0.49.)*
- **C** — word- and phoneme-level articulation is synchronised. *(Never
  demonstrated on any path here. Do not call B "perfect lip-sync".)*

A partial score card yields no overall — `overall_score()` returns `None`
rather than averaging an incomplete result.

## 5. Performance and reliability

Recorded for **every** run (`RunRecord`): provider, pipeline, GPU, duration,
resolution, frames, steps, reference count, audio input, cold/warm, model load
seconds, generation seconds, decode seconds, total wall seconds, peak VRAM,
peak host RAM, success/failure with a failure class, retries.

Failure classes are the ones this project has actually seen: `oom`, `cublas`,
`corrupt_output`, `duration_mismatch`, `audio_mismatch`, `identity_failure`,
`other`.

**Repeats are not optional.** 481 frames passed six times in isolation and
failed five of fifteen when the card was shared — a single beautiful sample
would have hidden that. Three runs minimum, five on launch-critical paths, and
report the success rate beside the score.

## 6. Cost and switching

Once GPU pricing is known, compute cost per 5 s, per 30 s, per 60 s and per
minute of music video, for both engines. **Keep quality and cost in separate
columns** — a cheaper engine that fails a workflow is not cheaper.

Model-switch cost is its own measurement (LTX loaded → H3 load → LTX reload),
because it decides whether one card can serve both engines, whether jobs should
be batched by provider, or whether launch needs two cards. H3's bf16 build
(~110 GB with its text encoder) does not fit our card at all, so the quantized
build is part of this measurement, not a footnote.

## 7. The decision, when the evidence exists

Fill in `result_skeleton()` and produce:

| Workflow | LTX score | H3 score | Speed | Quality | Cost | Final provider |
|---|---|---|---|---|---|---|
| T2V short · T2V long · I2V short · I2V long · standard V2V · reference V2V · music video · dialogue · camera-heavy · extend · multimodal reference · fast mode · quality mode | | | | | | |

Every decision must cite quality, speed, VRAM, reliability and cost evidence.
A mixed outcome is the expected and preferred result; "model X wins
everything" should be distrusted unless the table actually says so.

Product routing (fast / quality / reference+ / music video) is chosen *after*
that table exists, and raw model names stay out of the public surface unless
they are useful to a customer.

## 8. Rules that hold regardless of outcome

1. No routing change without benchmark evidence in the commit message.
2. LTX's golden argv snapshot (`tests/test_ltx_golden.py`, 11 shapes) must stay
   green through every change. If it fails, the change is wrong — not the
   snapshot.
3. H3 cannot generate until its licence is authorised and a node carries the
   weights; `health()` refuses and says so.
4. Nothing enables an experimental LTX flag globally to make a comparison look
   better; the audit's flags stay off and are A/B'd on their own terms.
