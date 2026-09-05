# LTX 2.5 GPU benchmark — 5 September 2026

**Every number here was measured on the node the same day.** Raw files:
`benchmarks/results/ltx25/2026-09-05-gpu-validation/` (per-cell JSON, the
official run's 1 Hz VRAM trace, the frame-comparison JSONs, the weight
hashes). Video outputs stay on the node under `/workspace/results/` and
`/workspace/zolexai/benchmarks/results/ltx25/` (not committed).

## GPU

NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB VRAM,
compute capability 12.0, driver 595.71.05 (CUDA 13.2), 48 CPUs, 503 GB RAM.
Vast.ai container `163.182.37.67:20577`, no persistent volume.

## Models loaded

| | Official pipeline (Step 7) | Client graphs through ZolexAI (Steps 8, 9) |
|---|---|---|
| Transformer | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (18.72 GB, `nvfp4-prequant`, sm_120 ltx-kernels) | T2V / First-Last: `LTX-2.5-Distilled-Q8_0.gguf` (23.60 GB, Abiray) · Character Replacement: `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` (21.50 GB, official) |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (26.26 GB) | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (15.37 GB, official) |
| VAE | `ltx-2.5-video-vae-bf16` (1.47 GB), `ltx-2.5-audio-vae-bf16` (0.36 GB) | same files, plus `taeltx2_3` for sampler previews only |
| Upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` (1.00 GB) | same file |
| LoRAs | none | T2V / First-Last: OmniNFT-RL @0.4, transition @0.8, LTX-2 19B detailer @0.3 · Character Replacement: Ripple v11 @1.35 |
| Duration head | `ltx-2.5-duration-head-bf16` (loaded, unused: `--num-frames` given) | not in the graphs |

Sizes and sha256 for all of them: `ltx25-gpu-model-validation.md` §2 and §5.

## Step 7 — official Lightricks pipeline, text to video

`ltx_pipelines.distilled` @400fd31, seed 42, 1280x704, 121 frames, 24 fps,
prompt "A cinematic shot of a futuristic city at sunset, flying vehicles
moving between skyscrapers, realistic lighting, smooth camera movement".

| Runtime (cold, model load included) | VRAM peak | VRAM mean | RAM max RSS | Output | Errors |
|---|---|---|---|---|---|
| **41.5 s** | 24,093 MiB | 13,133 MiB | 26,567 MiB | 1280x704 · 121 frames · 24.0 fps · 5.042 s · stereo audio | none |

Frames 0/60/119 inspected: photoreal sunset skyline, a flying vehicle
mid-frame, camera in motion.

## Step 8 — ZolexAI output versus ComfyUI

Same client graph (`ltx25_text_to_video.json`), same seed (42), same 5 s,
same 16:9, on the same server.

| Pair | Container | Frames (PSNR / SSIM over 121 frames) | Audio (normalised cross-correlation) | Verdict |
|---|---|---|---|---|
| A: graph submitted straight to ComfyUI with the brief's prompt · B: the same request through the ZolexAI adapter | identical (1280x704, 121 f, 24 fps, 48 kHz stereo) | mean 19.9 dB / 0.585 | 0.52 | DIFFERENT — B's positive text carries the adapter's soundtrack-owner sentence ("No one speaks. The only sounds are the ones the scene itself makes."), a product rule from 28 Aug; the seeds were identical |
| **A2: graph submitted straight to ComfyUI with B's exact text, after `/free` (cache cleared, models unloaded and reloaded) · B** | identical | **∞ dB / 1.000 on every frame** | **1.000, zero lag** | **MATCH — bit-identical video and audio** |
| Official CLI (Step 7) · A | identical container | 16.2 dB | — | different by construction: different transformer (NVFP4 vs GGUF Q8), no LoRAs, different sampler chain |

Conclusion: the ZolexAI path submits exactly what the client's graph
produces; the only thing ZolexAI adds is text in the prompt box, and with
that text held equal the result is the same file. Frame tiles of A and B
(both photoreal, same scene family) are in the session record.

## Step 9 — generation ladder through ZolexAI (client T2V graph)

All 1280x704 (16:9 at the graph's 0.9 MP), 24 fps, seed 42 unless noted,
the brief's prompt, one submission each, ComfyUI 0.34.5, models resident
after the first run.

| Length | Runtime | Output | VRAM peak | VRAM mean | ComfyUI RSS |
|---|---|---|---|---|---|
| **5 s** (cold: models loading) | 54.7 s | 5.042 s, 121 f | 26,056 MiB | 19,244 MiB | — |
| **5 s** (warm, seed 43, no cache) | **48.8 s** | 5.042 s, 121 f | 26,926 MiB | 20,711 MiB | 50,546 MiB |
| **10 s** | **76.3 s** | 10.042 s, 241 f | 27,910 MiB | 21,911 MiB | — |
| **15 s** | **106.3 s** | 15.042 s, 361 f | 29,414 MiB | 23,978 MiB | — |
| **30 s** | **215.2 s** | 30.042 s, 721 f | 34,456 MiB | 29,204 MiB | 57,093 MiB (sampled) |

Roughly 20 s fixed cost plus 6.5 s per output second; the 30 s cell runs at
7.2× real time. (The ladder's own 5 s cell hit ComfyUI's execution cache
from Step 8B and returned in 0.4 s; the warm number above is the honest one.)

### The other two graphs, and the extension

| Cell | Runtime | Output | VRAM peak | ComfyUI RSS | Notes |
|---|---|---|---|---|---|
| First/Last Frame, first still only, 5 s, 9:16 | 63.8 s | 704x1280, 121 f | 30,728 MiB | 75,060 MiB | identity held through frame 119 (one-image conditioning node) |
| First/Last Frame, first + last (same still), 5 s, 9:16 | 66.9 s | 704x1280, 121 f | 30,700 MiB | 70,205 MiB | identity held; head turn and return |
| Character Replacement, the ZIP source clip (8.6 s) + the ZIP still | **163.9 s** | 736x1280, 193 f, 8.042 s, source audio | **75,944 MiB** | 102,207 MiB | container identical to the ZIP sample; same four-frame handoff (frame 0 = photo, frame 4 onward = source motion); audio correlation with the sample 1.000 |
| Extend Video, +5 s on the 5 s T2V output | 65.3 s (pass 63.7 s) | 1280x704, 241 f, 10.063 s (promised 10.042) | as T2V | — | seam at 5.04 s clean at 24 fps; overlap frame dropped (121 rendered, 120 kept) |

## VRAM

| | Peak | Average (during the run) |
|---|---|---|
| Official CLI, NVFP4, 5 s | 24,093 MiB | 13,133 MiB |
| Client T2V graph, 5–30 s | 26,056–34,456 MiB | 19,244–29,204 MiB |
| First/Last Frame, 5 s | 30,728 MiB | 21,573–22,285 MiB |
| Character Replacement, 8 s | 75,944 MiB | 43,887 MiB |

Character Replacement is the heavy one: the int8 transformer, the int8
text encoder, the IC-LoRA guide on a 193-frame source and the Ripple LoRA
peak at 76 GB on this card. It fits alone; it will not fit beside ACE-Step's
~24 GB resident. RAM: the ComfyUI process holds 50–102 GB RSS with models
loaded; the box has 503 GB.

## Quality notes

* Official CLI and the client T2V graph both produce photoreal, coherent
  5 s clips from the brief's prompt; the client graph's look (LoRAs,
  two-stage schedule) is warmer and more detailed at the cost of 48.8 s vs
  41.5 s.
* First-frame-only First/Last Frame: the first implementation (bypassing the
  two-image conditioning node) matched the still at frame 0 and then showed a
  different person somewhere else — stage 1 had no image conditioning. Fixed
  on the spot: the node runs with one image (its own counter), and the
  identity now holds. Commit `24037b6`.
* Character Replacement matches the delivered sample's behaviour frame for
  frame in structure (the photo, then the source's motion on the photo's
  setting) with the same person visible throughout.
* The extension seam is invisible at frame level; by the end of a 5 s
  continuation the prompt's camera glide has visibly moved the framing —
  expected, prompt-driven, not drift.

## Settings changed from the measurements

`ltx_comfy_expected_wall_per_output_second` 8 → **7.5** (progress pacing:
48.8 s for 5 s, 215 s for 30 s).

## Verdict

```text
✓ Same official LTX 2.5 models installed   — 14 files, sizes and hashes recorded
✓ Client ZIP workflow validated             — 3 graphs compile, every node/file offered, all 3 rendered
✓ GPU generation successful                 — official CLI and all three client graphs
✓ ZolexAI output matches ComfyUI            — bit-identical with equal text and seed
✓ Benchmarks recorded                       — 5/10/15/30 s ladder, VRAM, RAM, fps, durations

LTX 2.5 GPU VALIDATION: PASS
```
