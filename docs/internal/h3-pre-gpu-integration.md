# MiniMax H3 — what it officially is, and what we built around it

**Date:** 22 August 2026 · **Branch:** `dual-engine-benchmark-prep` (from
`ltx25-alignment-audit` @ `c649c78`) · **No GPU.** Nothing was deployed, no
weights were downloaded, no model was run, and no routing decision was made.

Every capability statement below is sourced from official MiniMax material,
read 22 August 2026:

- model card — <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- announcement — <https://www.minimax.io/news/minimax-h3-open-source>
- prompt guides — `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` and
  `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` in the model repo
- licence FAQ — `docs/QA-about-License.md` in the model repo
- code/deployment — <https://github.com/MiniMax-AI/MiniMax-H3> and the
  official ComfyUI integration

Community write-ups were read but are cited only where they agree with
official material. One did not, and that is recorded in §2.3 because it is the
kind of error that would have decided a workflow wrongly.

---

## 1. What H3 is

An omni-modal generative system: text, images, video and audio in; video with
**native 32 kHz stereo audio** out, generated in the same pass. The
H3-Omni-Transformer is described as an approximately 33B-parameter dense
single-stream transformer, roughly 13B of which sits in AdaLN branches that
can be cached for inference-only deployment. Open-sourced 3 August 2026.

Two task-specific checkpoints, and the distinction runs through everything
below:

| Head | Takes | Purpose |
|---|---|---|
| **FL2VA** | text + zero, one or two images | text-to-video, first-frame, last-frame, first-and-last-frame |
| **Ref2VA** | text + up to 9 images, 3 video clips, 3 audio clips (max 12 files) | consistent characters, video editing, motion reference, clip continuation |

### 1.1 Hard limits (official)

| Property | Value |
|---|---|
| Duration, one generation | **4–15 seconds** |
| Frame rate | 24 fps |
| Aspect ratios | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 |
| Resolution | short edge 768 px by default; 2K only via **H3-Regenerate-2K, which is not in the open-source release** |
| Audio out | 32 kHz stereo |
| Reference video clips | ≤ 3, each 2–15 s, total ≤ 15 s |
| Reference audio clips | ≤ 3, each 2–15 s, total ≤ 15 s |
| Reference images | ≤ 9 |
| Files across all types | ≤ 12 |
| Dialogue languages stated stable | 11 (ar, zh, en, fr, de, it, ja, ko, pt, ru, es) |
| Attention | full attention only; sparse attention "will be released in a future update" |

**The 15-second ceiling is the single most consequential fact in this
document.** It is not a memory limit we could buy our way past — it is the
documented output range of the model. Every duration the product sells above
15 seconds is, on H3, a multi-generation chain.

### 1.2 Weights and what they imply for our card

From the official ComfyUI repackaging:

| File | bf16 | int8_convrot | pruned/nvfp4 |
|---|---:|---:|---:|
| DiT (per head, fl2va or ref2va) | 61.7 GB | 31.7 GB | 19.5 GB |
| Text encoder (Qwen3-VL-32B) | 48.0 GB | 25.3 GB | 14.6 GB (nvfp4-awq) |
| Video VAE | 4.9 GB (fp16) | — | — |
| Audio VAE | 0.6 GB (fp32) | — | — |

bf16 DiT + bf16 text encoder is roughly **110 GB — more than the RTX PRO
6000's 95.6 GB**. H3 on our hardware therefore means a quantized build from
day one, and *which* build is a quality variable the benchmark has to control
for. This is unlike LTX, where nvfp4-prequant has served production
throughout. Community VRAM guidance quotes 24 GB for int8-pruned + nvfp4 text
encoder and 12–16 GB for Q4_K_M GGUF, with explicit quality loss at IQ1_S.

### 1.3 Deployment

Official paths are **SGLang** (`sglang serve --model-path … --model-variant
fl2va|ref2va`, examples using `--num-gpus 4 --ulysses-degree 4`), **vLLM**,
**diffusers** (`ModularPipeline.from_pretrained`), and **ComfyUI**. There is
no single-command CLI equivalent to the way we shell into LTX.

That difference is architectural, not cosmetic: LTX is a subprocess per pass,
H3 is a served model. We already run one engine that way — ACE-Step is a
long-lived HTTP service the worker talks to like a database — so the pattern
exists in this codebase, and the H3 provider is written to be filled in
against a service rather than a CLI.

References travel as a `conditions[]` array of `{type, uri, role,
frame_index}` entries, `file://` or `http(s)://`.

## 2. The parts that decide workflows

### 2.1 Long-form is a chain on both engines, with different arithmetic

Upstream LTX ships **no extension pipeline** either (audit §2), so neither
engine renders a minute in one pass. What differs is the seam count, and our
dry run prints it:

```text
60-second text-to-video, same request, both engines
  ltx  ltx_pipelines.distilled   1024x576   2 sections, 1 seam   [30s, 30s]
  h3   MiniMax-H3/FL2VA          1366x768   4 sections, 3 seams  [15s, 15s, 15s, 15s]
```

LTX's 30-second sections are a *story-coherence* measurement (a single 60 s
pass returned a departed character); the GPU sustains 60 s. H3's 15 seconds is
a documented model limit. Three seams against one is not automatically worse —
H3 may hold identity across a seam better than LTX does — but it is the
comparison group H (long form) exists to settle.

### 2.2 H3 has a real continuation task; ours is a construction

H3 documents `video continuation` as a Ref2VA task type: "new content
continues, extends, resumes, or transitions from an existing source video."
Our LTX extension conditions on the source's final frame — a construction over
a model that has no such task. Benchmark group I.

### 2.3 Audio: two modes, and only one of them is a music video

The official Ref2VA guide distinguishes them explicitly:

- **`fully_copy`** — "The complete source audio serves as the target video's
  complete final audio track", and the guide states lip-sync occurs **when
  audio is directly copied**.
- **reference only** — "the signal is not copied directly; only timbre,
  rhythm, music style, dialogue content, or sound texture is referenced", and
  "the target speaker follows … voice timbre and measured delivery without
  copying the original signal". This mode does **not** lip-sync, and the
  supplied audio is not the output track.

A widely-circulated third-party write-up describes only the second mode and
concludes H3 "does not lip-sync the generated character to supplied audio".
Had we taken it, we would have written off H3 for music video on a false
premise. The compiler therefore pins `fully_copy` for any customer-supplied
track, and the manifest records the mode so a reviewer can see which one a
result came from.

The guide also covers the singing case directly: where the vocal is part of a
directly reused soundtrack and no on-screen character produces it, the audio
asset is the audible source and no speaker id is invented for it.

**Still unmeasured by us:** whether H3's `fully_copy` sync is better or worse
than LTX a2vid's, which measures at goal-B (mouth follows vocal energy,
−125…−208 ms, r≈0.45). Neither is demonstrated at goal C. Benchmark group E.

### 2.4 Identity is a first-class input on H3 and does not exist on LTX

There is no identity input anywhere in the LTX family — a reference person can
only enter as pixels in a conditioned frame, which is why our V2V path
composites the reference into the source's opening frame and anchors it. H3
takes the person as a subject reference image, carried in every generation.
This is the strongest a-priori case for H3 in the product, and it is still a
benchmark (group D), not a decision.

### 2.5 Prompt format differs, and both are documented

| | LTX | H3 |
|---|---|---|
| Shape | one flowing chronological paragraph | three named fields: `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` |
| Timestamps | **forbidden** by its own enhancer prompts | **prescribed**: `[Shot 2] At 00:03.500, …` |
| Dialogue | quoted inline with a delivery description | `<d>[Language]…</d>` with stable `(S1)`/`(S2)` speaker ids |
| Camera | prose, "think like a cinematographer", ≤ 200 words | a **closed motion vocabulary** (Zoom/Push/Pull/Pan/Truck/Tilt/Pedestal/Arc/Tracking/Static/Shake/POV/Roll) with amplitude and speed modifiers |

Writing one prompt for both would hand at least one engine a format its own
documentation rejects. `worker/providers/h3_prompt.py` compiles the same
Director plan into H3's format, including a conservative mapping from our
camera vocabulary into H3's closed list (unmapped phrases pass through as
prose and are recorded as unmapped rather than dropped).

### 2.6 Aspect ratios do not line up

H3 has 4:3 and 3:4, which we do not offer on LTX. H3 has **no 4:5**, which we
*do* offer. A 4:5 request is refused by the H3 provider rather than reshaped —
quietly changing a customer's frame would make the benchmark compare two
different products.

## 3. Licensing — an open blocker, not a formality

The open weights are limited to the **EU, UK, South Korea and the US**, and an
organisation in those regions must **apply** for a formal licence: MiniMax
reviews the deployment scenario and confirms compliance controls before
authorising use. The licence also carries safety guardrails requiring
automated moderation of user-submitted content.

For a SaaS that generates video for paying customers this is a gating
question, and it is the same class of gate as LTX Attachment A #20 (which is
itself still open). **Two consequences, both concrete:**

1. Where the GPU is rented from matters. The H3 provider's `health()` refuses
   until this is resolved, and says why.
2. The moderation obligation is a product requirement, not a legal footnote —
   it needs an owner before H3 could serve a customer.

## 4. What was built, and what deliberately was not

Built, and exercisable today without a GPU:

| Piece | Where |
|---|---|
| Capability matrix, 28 rows, cited | `worker/providers/capabilities.py` |
| Provider protocol (`capabilities/validate/compile/generate/health`) | `worker/providers/base.py` |
| LTX provider — reads the shipped adapter, never re-derives | `worker/providers/ltx.py` |
| H3 provider — compile only; `generate` refuses | `worker/providers/h3.py` |
| H3 prompt compiler (official format + camera mapping) | `worker/providers/h3_prompt.py` |
| Router with `provider=auto\|ltx\|h3` override | `worker/providers/router.py` |
| Dry-run manifest (sections, seams, references, audio windows, settings) | `worker/providers/manifest.py` |
| Benchmark: 41 cases across 10 groups, scoring, result schema | `worker/providers/benchmark.py` |
| Harness CLI | `scripts/dual_engine_bench.py` |
| LTX golden argv snapshot, 11 shapes | `tests/test_ltx_golden.py` |

Deliberately **not** decided, because each is a measurement:

- H3 steps, guidance/CFG, quantization build, offload strategy, section length
- H3 resolution beyond the documented 768 short edge
- any change to `auto` routing — every workflow still resolves to LTX
- whether H3 is better at anything

## 5. Unresolved questions for the GPU session

1. Which quantized H3 build we run, and what it costs in quality — bf16 does
   not fit on the card.
2. H3's real steps/CFG defaults (CFG-distilled weights exist; the open
   material does not state the numbers).
3. Whether SGLang-as-a-service co-exists with LTX subprocesses and ACE-Step on
   one card, and what a provider switch costs in load time (benchmark part 13).
4. Whether `fully_copy` audio sync beats LTX a2vid (group E) — and whether
   either reaches goal C.
5. Whether H3's three seams at 60 s beat LTX's one (group H).
6. Whether H3's native subject reference beats our composited anchor (group D).
7. Licence authorisation, and the moderation obligation's owner.
