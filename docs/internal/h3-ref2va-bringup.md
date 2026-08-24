# H3 Ref2VA bring-up on the RTX PRO 6000 WS

**Date:** 24 August 2026 · Instance `C.48538452` (Israel) · **Nothing deployed,
`auto` still resolves to LTX, H3 reachable only by explicit benchmark code.**

---

## 1. What was downloaded, and one wrong turn

| | |
|---|---|
| Repository | `MiniMaxAI/MiniMax-H3` |
| Revision | **`42ed227ee7df40d41602854ae760620d6eb651fe`** (2026-08-13) |
| Whole repo | **464 GB** (not the 354 GB estimated pre-GPU) |

The repository holds **two different formats side by side**, which is easy to
miss and cost us a download:

- `FL2VA/` and `Ref2VA/` — the **original checkpoint** format, for SGLang/vLLM.
  134 GB each, self-contained, duplicating every shared component.
- Root-level `transformer/`, `transformer_ref/`, `text_encoder/`, `vae/`,
  `audio_vae/` — the **diffusers** format, where `transformer/` serves
  `t2va`/`fl2va` and `transformer_ref/` serves `ref2va`, sharing one encoder
  and one pair of VAEs.

I fetched `Ref2VA/*` first (135 GB, verified complete: 81 files, every shard
present, on-disk bytes equal to the index totals) and then had to fetch the
diffusers-format components as well once the runtime decision landed on
diffusers. The `Ref2VA/` copy is not wasted — it is exactly what an SGLang
deployment would need — but it is currently idle.

**FL2VA has deliberately NOT been downloaded.** Ref2VA answers the
highest-value questions (reference V2V, music video, supplied audio,
continuation); FL2VA only matters for direct T2V/I2V comparison and has not
earned another 134 GB yet.

## 2. Runtime selection — diffusers, and why not SGLang

The pre-GPU research picked SGLang. **Measurement overturned that.**

| Runtime | Verdict | Evidence |
|---|---|---|
| **diffusers** | **selected** | First-class official integration. `diffusers 0.40.0.dev0` ships `MiniMaxH3ModularPipeline`, `MiniMaxH3Blocks`, both VAEs, `MiniMaxH3Scheduler` and all three reference classes. Documented single-card memory recipes. Exact `ref2va` reference semantics published. |
| SGLang | rejected | The **pip release has no H3 model at all** — `sglang 0.5.18` carries `minimax_m2`, `minimax_m3`, `minimax_m3_vl` and nothing else, and none of the cookbook's flags (`--model-variant`, `--performance-mode`, `--layerwise-offload-components`, `--ulysses-degree`) exist in its 676-flag `serve` interface. The cookbook's install is `pip install -e "/sgl-workspace/sglang/python[diffusion]"` — a source tree inside **their Docker image**, which cannot be nested here. |
| vLLM | not attempted | Recipes exist, but no official reproducible request script is published against it. |
| ComfyUI | rejected | Graph tool, not an API; wrong shape for a benchmark harness. |

Also decisive: the official deployment example is **four GPUs**
(`--num-gpus 4 --ulysses-degree 4`). We have one. diffusers is the runtime
whose documentation actually covers a single card.

## 3. Provider-native facts, taken from source

Read from the checkpoint and the official docs, **not invented**.

| Setting | Provider value | Source |
|---|---|---|
| Precision | **BF16** | README checkpoint table |
| Guidance | **none — guidance-distilled** | "guidance is baked into the weights, there is no guider, no `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass" |
| Video scheduler shift | **12.0** | `model_index.json` `sigma_shift_scales.video`, and the scheduler default |
| Audio scheduler shift | **3.0** | `model_index.json` `sigma_shift_scales.audio` |
| FPS | **24, fixed** | docs |
| Duration | **5 to 15 s** | docs |
| Frame lattice | **`17 * n + 5`**, snapped up | docs |
| Canvas | short edge **768**, multiples of 32 | `canvas_short_edge` config |
| Max pixels | 1032192 | `canvas_max_pixels` config |
| Reference short edge | 2048 | `reference_image_short_edge` config |
| Output audio | **32 kHz stereo**, from the same denoising loop | docs; measured 32000 Hz |
| Reference limits | ≤9 images, ≤3 videos, ≤3 audio, **12 total**; audio may never be the only reference | docs |
| `num_inference_steps` | **GPU TUNING REQUIRED** | **Not specified anywhere** — absent from the blocks, the checkpoint, the README and the model docs. The official SGLang request omits it and lets the server decide. |

**No value was invented.** Where the provider is silent, the row says so.

### 3.1 The Ref2VA request shape

Two distinct things, easy to conflate:

**The API request** (official `reproducible-768p-ref2va-request.sh`) posts to
`/v1/videos` with `task`, `prompt`, `conditions[]` of `{type, uri, role}`,
`target: {short_edge, aspect_ratio, duration_seconds}` and `seed`.

**The prompt is a structured IR**, not free text. Named sections in order:
`subject_definitions:` · `summary:` · `retention_analysis:` ·
`detailed_description:` · `overall_soundscape:` · `non_diegetic_music:`.
Entities are declared as `<Subject 1>`, `<Video 1>`, `<Audio 1>`; dialogue is
wrapped `<d>[English] ... </d>`; speakers are tagged `(S1)`.

**Audio mode lives inside `retention_analysis`,** not as an API field:
`fully_preserved`, `partially_copy`, `reference`. So "fully_copy" is a
*retention value*, not a flag — which is a semantic our compiler must express,
not approximate.

The provider's own example writes mouth behaviour explicitly, including the
stop: *"Exactly as his voice stops, his lips meet in a relaxed, peaceful smile,
and his jaw ceases speaking motion."* That is the presence-blindness failure
described as a prompt instruction.

In **diffusers**, references are passed as an ordered `references=[...]` list of
`MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` / `MiniMaxH3AudioReference`,
each built with `from_file` so the media's real rates travel with it. **Order is
semantic** — it labels them in the prompt presentation and advances the shared
rotary clock, so reordering is a different request.

## 4. Memory — measured, and it corrects my own prediction

I predicted bf16 would not fit: 61.7 GB transformer + 62.1 GB conditioner +
10.3 GB VAEs = **134.1 GB against 125 GB of host RAM**. That arithmetic assumed
`enable_auto_cpu_offload` holds every weight resident in host RAM.

**It does not.** diffusers memory-maps the checkpoint and streams blocks, so
host residency settles far below the total.

```text
load_components(bf16)      11.1 s   host 5.2 GB   VRAM 0.0 GB   <- lazy/mmap, not a real load
                                                                   (load-only proves nothing here)

B1 generation, measured:
  peak VRAM               81.7 GB torch-reserved  /  84.3 GB nvidia-smi
  peak host RAM           71.0 GB of 125 GB
  peak swap                0.2 GB          <- no thrashing
  minimum free RAM        53 GB            <- guard threshold was 8 GB, never approached
```

**Verdict: 95.6 GB VRAM + 128 GB host RAM is SUFFICIENT for Ref2VA at bf16.**
~13 GB of VRAM headroom and ~54 GB of RAM headroom at the provider-native
canvas. No quantization deviation is required, so the provider-native baseline
is the one we can actually run — the §16 exception does not need to be invoked.

A load-only test cannot establish this, because the load is lazy. Residency has
to be measured during generation.

## 5. B1 — subject-image Ref2VA: **PASS**

Provider's own example asset, provider-native canvas, no ZolexAI code involved.

```text
prompt      "The astronaut looks around slowly, then speaks a short greeting to camera."
references  [MiniMaxH3ImageReference.from_file(<the diffusers docs astronaut>)]
num_frames  124   (17*7+5, the smallest the VAE decodes at >= 5 s)
steps       30 requested -> 29 evaluations   PROVISIONAL, not provider-specified
seed        42

output      1344x768, 124 frames, 5.175 s, h264 + AAC 32 kHz stereo, 910 KB
generation  580.9 s
```

Inspected, not merely validated: mean luma 74.65 (black would be ~16),
interframe delta 3.486 (frozen would be 0), audio RMS -15.3 dB (real generated
soundtrack, not silence). The frame shows the reference astronaut's identity
preserved — suit, gold visor, proportions — relocated into a new lunar scene.
Video and audio came out of one denoising loop, as documented.

## 6. The number that reframes the benchmark

```text
LTX 2.5   5.00 s of video in  28.2 s  =    5.6x real time
H3        5.17 s of video in 580.9 s  =  112.3x real time
                                         ------------------
                                         H3 is ~20x slower
```

Both provider-native, both on this card, both inspected.

**This is a benchmark-economics fact, not a quality judgement** — H3's output
quality has not been scored, and 17.4 s/step was measured at the trained
1344x768 canvas. The docs state 960x544 runs about 2.3x faster per step, so
there is a real lever, and it is a lever the benchmark should pull deliberately
and record rather than apply silently.

But the 406-run plan was costed on LTX-like speeds. At 112x real time the H3
and hybrid cells are the dominant cost of the entire pack, and the frozen plan
needs re-costing before anyone commits to running it whole.

## 7. State

- **FL2VA: not downloaded.** Deliberate.
- Disk 426 GB of 879 GB used.
- `auto` routing unchanged, production untouched, H3 reachable only from
  benchmark code.
- `num_inference_steps` remains **GPU TUNING REQUIRED**; every measurement above
  carries the provisional value it was taken at.
