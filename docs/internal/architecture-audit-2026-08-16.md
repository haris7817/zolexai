# Architecture audit — 16 August 2026

**Commit audited:** `a582afa`
**Method:** read out of the repository. Every finding below is mechanism, not
inference from output.
**Companion documents:** [`issue-triton-na-kernel.md`](./issue-triton-na-kernel.md),
[`next-steps-2026-08-15.md`](./next-steps-2026-08-15.md).

Section numbers in the right-hand column refer to the master baseline document's
open inspection queue.

| Audit | Master § | Verdict |
|---|---|---|
| Prompt enhancer and text encoder | new | **Lever exists, never switched on** |
| Text encoder / conditioning | §59 | **No guidance, no negative prompt, no step count** |
| Frontend parameter survival | §53 | Clean, except one surviving fake control |
| Video-to-video conditioning | §67 | Sparse stills; the weak-restyle knob is identified |
| Long-form remaining unknowns | §56 | Closed |

---

## 1. The prompt enhancer is implemented, tested, and disabled

LTX 2.5 ships a prompt enhancer that expands a terse prompt into a detailed
cinematic instruction, and a Gemma-4 12B text encoder. **Both are already our
configuration.**

| Component | Our value |
|---|---|
| Text encoder | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| Enhancer flag | `--enhance-prompt`, appended by `_command` |

`apps/worker/worker/adapters/ltx.py:1045`:

```python
if job.execution.get("enhance_prompt"):
    cmd.append("--enhance-prompt")
```

`grep enhance_prompt workflow-definitions/` returns **no match in any of the six
workflow definitions.** The flag has never been set, therefore never run.

It is off deliberately, not accidentally — `test_ltx.py:146` pins the default —
because the enhancer **rewrites the user's prompt**, and the codebase holds a
standing rule that what the user typed reaches the model verbatim.

That rule is correct for a user who wrote three careful paragraphs. It is
actively harmful for a user who typed *"a car driving in rain"*, which is the
case behind the client's prompt-adherence complaint.

### The trap: do not simply set the flag

`--enhance-prompt` is appended inside `_command`, which runs **once per pass**.
Combined with the per-section prompts introduced in `2873150`, a 60-second video
would send six different section prompts to the enhancer and receive **six
independently invented cinematic paraphrases** — different lighting language,
different camera vocabulary, different adjectives per section.

That is a mechanism for *worse* boundary drift. It would plausibly degrade
identity and clothing/colour consistency at seams while improving adherence
within each section.

### Correct ordering

```
enhance ONCE on the master prompt
  → derive section prompts from the enhanced text
  → each pass renders its section of one consistent description
```

This inverts the current order (`prompts.py` splits, then `_command` would
enhance). The enhancement must move out of `_command` and up into the planner.
Contained, but a change rather than a flag flip.

It also amortises the enhancer's model-load cost across all passes instead of
paying it N times — see §4 below.

### Unverified

1. **Does the pipeline accept the flag?** We pass it; nothing has ever exercised
   it. If `ltx_pipelines.distilled` rejects it, every render exits non-zero.
2. **What does it cost?** The enhancer runs a language model. Given that ~28 s of
   every ~30 s pass is already model loading, a per-pass enhancer load could be
   significant.

Both answered by `python -m ltx_pipelines.distilled --help` on the box.

---

## 2. §59 — the runtime exposes no quality dial at all

| Question | Answer |
|---|---|
| Encoder | Gemma-4 12B with projection, **bf16** — the transformer is NVFP4, the encoder is not quantised |
| Loaded when | Per pass, from disk, in a fresh subprocess |
| Embedding caching | **None.** Every pass re-encodes from scratch |
| Whole-narrative vs per-section | Per-section since `2873150`; needs runtime confirmation |
| **Negative prompt** | **Zero references in the entire worker** |
| **Guidance / CFG** | **Zero references** |
| **Step count** | **Zero references** |

The complete argument surface of `_command`:

```
--quantization --prompt --num-frames --height --width
--frame-rate --seed --output-path
--transformer-path --text-encoder-path --video-vae-path
--audio-vae-path --duration-head-path --spatial-upsampler-path
[--enhance-prompt]  [--image PATH FRAME_IDX STRENGTH]...
```

### Consequence: quality tiers cannot be built on this runtime

Standard / High / Ultra require a dial that produces a measurably different
invocation. There is no dial. Not unwired — **absent**. All three levels looked
identical to the client because they *were* identical.

### Consequence: there are exactly three adherence levers today

1. **the prompt text** — `2873150`, done, unvalidated on GPU;
2. **conditioning frames** — the temporal-context experiment;
3. **`--enhance-prompt`** — never switched on.

That is the entire inventory. Guidance scale, negative prompt and step count all
require the **guided/dev checkpoint** — 42 GB, on the box, never run.

### Limit of this finding

This establishes what *our code passes*, not what the *pipeline accepts*. The
module is `ltx_pipelines.distilled`, and a distilled model is unguided by
construction — CFG is baked in during distillation, so guidance genuinely should
not exist there. But `ltx_pipelines` may expose other entry points for the dev
checkpoint.

**This is now the highest-value GPU question**, and it is one command:

```bash
python -m ltx_pipelines.distilled --help
ls /workspace/ltx2-benchmark/packages/*/src/ltx_pipelines/
```

If a guided entry point exists, the runtime-quality benchmark becomes a real
experiment rather than a hope.

---

## 3. §53 — frontend parameter survival

### 3.1 Result players are not force-muted — this explanation is ruled out

`MediaPreview.tsx` renders generated results with `controls` and **no `muted`
attribute**. The only `muted` in the codebase is
`PreviewVideo.tsx:111` — `muted={interaction === "hover"}` — the marketing
landing-page hover preview, which is correct behaviour.

Every other `muted` hit is the Tailwind class `text-zx-text-muted`.

**Therefore "voice too quiet / unintelligible" is not a frontend muting bug.**
That was the cheap explanation and it is dead. The cause is genuinely in the
model output or the media pipeline, and must be classified as Case A or Case B
before anything is changed.

### 3.2 Fake controls — five of six workflows are clean

`2873150` hid every control the runtime does not consume:

| Workflow | quality | motion | adherence | seed | quality levels |
|---|---|---|---|---|---|
| text-to-video | false | false | false | **true** | `[]` |
| image-to-video | false | false | false | **true** | `[]` |
| extend-video | false | false | false | false | `[]` |
| video-to-video | false | false | false | false | `[]` |
| music-video | false | false | false | false | `[]` |
| **music** | false | false | **true** | **true** | `[]` |

Seed is exposed exactly where it is honoured (`_seed_for_step`, and ACE-Step's
`use_random_seed`/`seed`). Correct.

### 3.3 One surviving fake control — Music `prompt_adherence`

`music.yaml` sets `prompt_adherence: true`. The API validates it
(`schemas/generation.py:34`, default 75, range 0–100). **No runtime reads it.**

```
grep -rn "prompt_adherence" apps/worker/worker/  →  no matches
```

`MusicRequest` has no such field, and `AceStepProvider.build_payload` sends only
`caption`, `lyrics`, `audio_duration`, `bpm`, `key_scale`, `use_random_seed`,
`seed`, `reference_audio`.

**The customer moves a slider that changes nothing.** Every video workflow was
cleaned; music was missed.

Two options, both one line:

- set `prompt_adherence: false` in `music.yaml` — matches what the video
  workflows did, honest immediately;
- map it onto a real ACE-Step parameter — larger, and no obvious target exists
  in the provider's payload.

Recommend the first. **Not applied — this changes a deployed public contract and
deployment is currently gated.**

### 3.4 Music parameters supported by the provider but not offered

`MusicRequest` carries `bpm`, `key` and the instrumental/lyrics equivalence, and
`build_payload` forwards them. None are exposed in the public API. Real controls
that exist and are hidden — the mirror image of §3.3.

---

## 4. §67 — V2V is sparse-still conditioning, and the weak-restyle knob is identified

Current V2V is **not** video-conditioned. Per pass it supplies:

| Signal | Default | Meaning |
|---|---|---|
| `v2v_keyframes` | **3** | source stills spread across the window |
| `v2v_structure_strength` | **0.7** | how hard those stills pull |
| `v2v_continuity_strength` | **0.85** | frame 0 of every pass after the first |
| `v2v_reference_strength` | **0.3** | optional reference image, first pass only |

All four are overridable per workflow through the private `execution` block.

The client reported *"source preserved, restyle too weak."* The code names the
cause in its own docstring:

> *"At 1.0 the source frame IS the output frame and the prompt does nothing;
> near 0 the prompt wins and the source is a suggestion. This is the dial
> between 'restyled' and 'unrelated'."*

**0.7 is high on that dial.** Sweeping `v2v_structure_strength` down
(0.7 → 0.5 → 0.3) is the direct, no-code-change answer to the client's
complaint, and it can be done per workflow.

Sweep with `v2v_keyframes` held at 3, then vary keyframes second — fewer
keyframes also loosens structure, and moving both at once confounds the result.

### Output ceiling

`_MAX_OUTPUT_LONG_SIDE = 1920`, `_MAX_OUTPUT_SHORT_SIDE = 1080` — output is
already capped at 1080p, consistent with the resolution probe finding that 1080p
generates successfully.

---

## 5. §56 — long-form unknowns, closed

### 5.1 Uneven final pass — yes, and it is already in production

```python
duration = min(max_segment_seconds, total_seconds - start)
```

A 15-second request at ceiling 10 plans **10 s + 5 s**. Since 15 s is an offered
duration, uneven final passes occur on live traffic today, and the duration ×
aspect matrix passed with them.

Because the kernel failure is shape- and size-dependent, a *shorter* final pass
is safer, not riskier. Low concern — but note `_frame_count = round(seconds ×
24)`, so a 5 s pass is 120 frames and a different latent shape. Given that
`704 = 11 × 64` fails while much larger grids pass, a pathological frame count
is theoretically possible. Not observed.

### 5.2 Overlap is structurally zero

`plan_segments` accepts `overlap_seconds`, defaulting to `0.0`. **`render_chain`
never passes it.** The parameter exists and is dead.

This independently confirms from the media layer what the chain layer already
showed: segments butt-join, there is no overlap and no crossfade anywhere. Any
ghosting explanation based on blending is ruled out from both directions.

### 5.3 Music Video boundaries

`plan_musical_boundaries()` supplies cut points; `_plan` validates every window
against the pass ceiling and raises rather than widening one. A timing layer
cannot produce a pass the GPU will not survive. Sections are deliberately
**unequal** — log the real values on the next run rather than assuming even
windows.

### 5.4 Extend's first conditioning frame

`extract_final_frame(source, …)` at `ltx.py:867` — the source's last frame, the
same helper the chain uses between passes. Consistent.

---

## 6. What changed in the picture

| Previously believed | Now established |
|---|---|
| Quality tiers are "not wired up" | There is **nothing to wire** — the runtime has no quality parameter |
| Prompt adherence needs a better model | One unused lever exists first: the built-in enhancer |
| "Voice too quiet" might be a muted player | **Ruled out** — result players are unmuted with controls |
| All fake controls were removed in `2873150` | One remains: Music `prompt_adherence` |
| V2V restyle weakness is a runtime limitation | It is a **strength value** — 0.7 on a documented dial, overridable per workflow |
| Ghosting might come from overlap or crossfade | **Ruled out twice** — no overlap, no crossfade, confirmed in two layers |

## 7. Ordered next actions

**No GPU required:**

1. Decide Music `prompt_adherence` — hide it or map it (§3.3).
2. Move prompt enhancement from `_command` to the section planner, so one
   enhanced master prompt feeds all passes (§1).

**One GPU trip, cheap, high information:**

3. `python -m ltx_pipelines.distilled --help` — does `--enhance-prompt` exist,
   and does any guided entry point expose guidance / negative prompt / steps?
   This single answer decides whether the quality roadmap is real (§2).

**GPU experiments, in value order:**

4. Temporal context: multi-frame continuation vs the current single still.
5. `v2v_structure_strength` sweep 0.7 → 0.5 → 0.3 (§4).
6. `i2v_reference_strength` sweep 0.0 / 0.1 / 0.2 / 0.3.
7. Audio classification — Case A vs Case B — now that frontend muting is ruled
   out (§3.1).
