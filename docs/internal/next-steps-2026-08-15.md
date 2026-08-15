# Next steps — client findings and engineering fix plan

**Updated:** 15 August 2026
**Scope:** Milestone 2 optimisation after client acceptance testing
**Deployment:** local worktree only; **do not deploy production from this work**

Related evidence: [`issue-triton-na-kernel.md`](./issue-triton-na-kernel.md) and
[`gpu-worker-runbook.md`](./gpu-worker-runbook.md).

---

## 0. Current state

Production uses the RTX PRO 6000 with `LTX_MAX_SECONDS=10`. That avoids the
dimension-dependent fallback-neighbourhood-attention crash by chaining every
long output. The duration/aspect matrix passes, but client testing exposed
visible, identity, action and audio defects at those same 10-second boundaries.

The kernel bug and the long-form bugs are separate:

1. the kernel bug forces more passes;
2. the old chaining implementation repeated prompt/audio state per pass and
   made continuity depend on one predecessor still;
3. model limitations remain after deterministic pipeline faults are removed.

No production routing, secrets, infrastructure or deployment state was changed
in this worktree.

---

## 1. Confirmed root causes

| Client issue | Evidence-backed cause | State |
|---|---|---|
| Dialogue/action loops every ~10s | The same master prompt reached every LTX pass | Fixed locally with deterministic section prompts |
| T2V/I2V audio restarts | Each pass produced audio and pass files were concatenated | Replay reduced locally by section-specific dialogue; true master audio still required |
| Some T2V/I2V outputs have no audio | Completion required video only; a missing audio stream passed validation | Fixed locally; one decodable non-zero-duration audio stream is required per pass and final file |
| Final speech can be cut | Per-section generated audio has no full-duration dialogue timeline | Not fully fixed; requires master audio or an audio-conditioned visual path |
| Music Video track restarts | Code does not support this diagnosis: visuals are stripped and the uploaded track is muxed once | Existing architecture is correct; RTX artifact verification still mandatory |
| V2V source audio restarts | Code strips generated audio and muxes source audio once | Existing architecture is correct; RTX artifact verification still mandatory |
| I2V person changes after section one | Original uploaded image was used only in pass one, then replaced by predecessor-final-frame conditioning | Fixed locally: later passes receive temporal context plus the original identity anchor |
| Scene/object identity drifts | No persistent section constraints and only one-frame temporal context | Partially fixed with persistent continuity instructions; model/reference benchmark remains |
| Visible seam/freeze | Sections were not normalized before concat; exact duplicate/static-frame cause is not yet measured | Normalization fixed locally; seam-window RTX analysis remains |
| Repeated Extend fails/reuses old source | Same-route `?source=` changes did not reset React Hook Form defaults | Fixed locally |
| V2V strength controls appear inert | Public quality/motion/adherence parameters were never read by the distilled adapter | Fake controls hidden locally; private tested conditioning remains |
| V2V style is weak | Current path is sparse source-still conditioning, not a true video-conditioned restyle path | Model/runtime limitation; benchmark required |
| High/Ultra appears identical | All levels selected the same distilled runtime | Unsupported levels hidden locally pending guided model work |

The visible seam itself is **not yet root-caused**. FPS/timebase normalization
removes one deterministic risk, but the pause could still be a duplicated
predecessor frame or static settling at the start of a conditioned generation.

---

## 2. Local implementation completed (not deployed)

### 2.1 Shared section prompt planner

`worker/longform/prompts.py` now builds one prompt per section. It:

- preserves user-authored fragments verbatim rather than paraphrasing names,
  colours, numbers, counts, dialogue or camera directions;
- recognises explicit `Persistent:` and `Section N / start-end:` lines;
- distributes dialogue lines and `then` / `next` / `finally` sequences once;
- tells later sections to continue directly and never replay prior action or
  dialogue;
- keeps ambiguous visual/style prose persistent instead of inventing a story
  split.

For client goldens, explicit prompt structure is still the most deterministic:

```text
Persistent: exactly two cars; CAR A black; CAR B pearl-white; same road
Section 1 / 0-10: both cars enter; low tracking camera
Section 2 / 10-20: black car overtakes; do not repeat the entrance
Section 3 / 20-30: both cars stop; do not replay the overtake
```

### 2.2 Audio modes and verification

The media layer now names the ownership mode explicitly:

```text
SOURCE_AUDIO
GENERATED_MASTER_AUDIO
GENERATED_PER_SECTION_AUDIO
NO_AUDIO
```

Current workflow mapping:

| Workflow | Current mode | Assembly rule |
|---|---|---|
| T2V / I2V | `GENERATED_PER_SECTION_AUDIO` | Section-specific prompts; normalized sections; validate every section and final stream |
| Music Video | `SOURCE_AUDIO` | Strip generated audio; attach uploaded track once after visual assembly |
| V2V | `SOURCE_AUDIO` or `NO_AUDIO` | Strip generated audio; restore first source stream once |
| Extend | Source timeline plus optional generated tail | Never repeat source audio; add silence only where stream-layout normalization requires it |

Verification now checks stream presence, exactly one audio stream, non-zero audio
duration, approximate A/V duration agreement and full decode.

This does **not** turn T2V/I2V dialogue into master audio. The distilled LTX
entry point cannot produce audio-only output. Exact dialogue/singing timing needs
a separate provider or a guided runtime that accepts one full master track and
corresponding per-section audio windows.

### 2.3 Seam-safe assembly

Every generated LTX section is re-encoded to one explicit dimension, FPS,
timebase, pixel format and audio layout before concat. V2V and Music Video then
attach source audio once after the visual timeline is complete.

No long crossfade was added. The remaining seam must be measured before choosing
whether to trim duplicated/static lead-in frames or improve temporal conditioning.

### 2.4 Identity and controls

- I2V pass one uses the uploaded image at frame zero/full strength.
- Later I2V passes use predecessor context at frame zero and the original image
  at one-third of the section with default strength `0.2`.
- A user-supplied seed now reaches each section deterministically (`seed + index`).
- Public quality/motion/prompt-adherence controls are hidden for current video
  runtimes because no invoked model flag consumes them.
- V2V private `v2v_keyframes`, structure, continuity and reference strengths
  remain wired and command-level tested.

### 2.5 Progress and repeated Extend

Worker progress reports can now carry:

```json
{
  "phase": "generating",
  "section_index": 2,
  "section_total": 3,
  "section_start_seconds": 10,
  "section_end_seconds": 20
}
```

The payload is stored in the durable event and reaches SSE. Customer copy is
dynamic: `Generating section 1 of 3`, `2 of 3`, etc.; no total is hard-coded.
Assembly, audio, verification and upload have separate machine-readable phases.

The Extend form now watches a changed `?source=` value and resets its required
source asset, allowing an Extend result to become the next Extend input.

---

## 3. Tests added or strengthened

- dialogue lines occur in exactly one planned section;
- inline sequential actions are not replayed;
- single-pass prompts remain byte-for-byte unchanged;
- section progress includes index, total and exact time window;
- worker → API → durable SSE event preserves section payload;
- original I2V image remains in later-pass conditioning without owning frame 0;
- user seed reaches section invocations deterministically;
- silent T2V/I2V model pass fails before completion;
- audio probe reports one stream and a usable duration;
- existing Music Video and V2V tests continue to assert source audio is attached
  once and generated audio is removed.

Local results:

| Check | Result |
|---|---|
| Worker full suite | **304 passed, 90 skipped** |
| Skips | FFmpeg/media cases; this Windows environment has no usable FFmpeg binaries |
| Web TypeScript | passed |
| Web ESLint | passed |
| Catalogue parity QA | not run; it requires a live API on port 8000 |
| API Python compile | passed |
| API schema + workflow catalogue smoke | passed |
| API PostgreSQL/Redis integration suite | not run; local fixtures waited on unavailable services and were stopped |
| RTX/model smoke | not run |

The skipped media tests and unrun DB/RTX tests are release blockers, not assumed
passes.

---

## 4. Required P0 validation before any production deployment

1. Run API integration tests with isolated PostgreSQL and Redis.
2. Run the full worker suite on a host with FFmpeg/ffprobe so audio duration,
   mux, normalization and decode tests execute rather than skip.
3. On RTX, generate 30s and 60s T2V/I2V goldens and inspect 09.5-10.5 and
   19.5-20.5 second windows.
4. Record frame hashes/differences, FPS, timebase, frame count, video duration,
   audio duration and `freezedetect` output.
5. Run Music Video and V2V with source audio and prove the final artifact has
   exactly one audio stream with correlation to the source across section
   boundaries.
6. Run original → Extend → Extend → Extend and verify 10 → 20 → 30 → 40 seconds,
   new asset selection, signed URLs, lineage and no source replay.
7. Verify structured progress for 30s (3 sections) and 60s (6 sections).
8. Do not declare exact dialogue/singing fixed until a complete master audio
   timeline plays through without section correlation peaks/restarts.

---

## 5. P1 continuity and V2V work

### 5.1 Golden subject bibles

Run the two-cars, woman/robot, boxing and I2V prompts using explicit persistent
facts. Measure count, colour and identity at every section, not only the first
and final frame. The deterministic planner preserves facts; whether the current
distilled model obeys them is an RTX/model question.

### 5.2 True V2V restyle path

Trace and benchmark an actual supported video-conditioning/restyle entry point.
The acceptance target is strong style change while preserving source timing,
motion, framing and cuts. Sparse stills are a fallback, not proof of shot-for-shot
transformation.

Only expose a public transformation-strength control after a test proves two
values produce different runtime configuration and measurably different output.

### 5.3 Better temporal context

If seam-window analysis shows model settling rather than timestamp faults,
replace one-frame continuation with the strongest temporal context the runtime
supports. Do not add broad crossfades that hide a pause with ghosting.

---

## 6. P2 runtime, quality and speed

Do not start these before P0 artifact validation:

1. timebox NATTEN installation/build for `sm_120`; a successful optimized kernel
   may remove the forced 10-second ceiling and most seams;
2. measure model-load time per pass, then build a supervised persistent LTX
   service if weight reload dominates;
3. benchmark distilled NVFP4 versus guided/non-distilled FP8;
4. test BF16 only if FP8/BF16 quality differs enough to justify its cost;
5. expose Standard versus High/Ultra only when they invoke different, tested
   configurations with real CFG/guidance, negative prompt and step controls;
6. remeasure resolution and conditioned per-shape ceilings on the 96 GB card.

Measure prompt accuracy, count, camera adherence, identity, style, generation
time, peak VRAM and host RAM. BF16 by itself is not the prompt-adherence fix.

---

## 7. Files changed in the local fix

- long-form planning/progress: `worker/longform/{prompts,chain,progress}.py`;
- LTX orchestration: `worker/adapters/ltx.py`;
- media/audio verification: `worker/media/{audio,probe,validate}.py`;
- worker progress transport: `worker/adapters/base.py`, `worker/core/client.py`,
  `worker/jobs/runner.py`;
- API progress contract/event storage: `schemas/internal.py`,
  `api/v1/internal.py`, `services/generation.py`;
- repeated Extend UI: `CreatorWorkspace.tsx`;
- public control truth: all five video workflow YAML definitions;
- regression coverage: worker long-form/LTX/media/V2V/Music Video/runner tests,
  API event/definition request tests, and smoke callbacks;
- internal evidence: this file and `issue-triton-na-kernel.md`.

---

## 8. Recommended commit split

1. `fix(worker): plan long-form sections and preserve I2V identity`
2. `fix(worker): normalize section media and enforce audio verification`
3. `feat(progress): carry structured long-form section events end to end`
4. `fix(web): allow generated extensions to be extended repeatedly`
5. `fix(workflows): hide controls unsupported by the distilled runtime`
6. `test,docs: client regressions, RTX checklist and confirmed root causes`

Keep production-only routing changes and infrastructure files out of these
commits.

---

## 9. Product decisions still required

- Should every T2V/I2V output require audio? This local fix says yes because the
  client reported missing audio as a defect. If silent video is valid, expose an
  explicit `NO_AUDIO` choice rather than accepting accidental silence.
- Which provider/runtime owns full-duration dialogue and singing master audio?
- Should the UI teach explicit `Persistent:` / `Section:` prompt structure, or
  should a server-side planner service create it?
- Is strong shot-for-shot V2V a launch requirement if the current LTX entry point
  cannot provide true video conditioning?
- What latency/cost targets define Standard, High and Ultra?
- If NATTEN cannot run on `sm_120`, is permanent 10-second chaining acceptable
  after seam improvements, or must the runtime/model change?
