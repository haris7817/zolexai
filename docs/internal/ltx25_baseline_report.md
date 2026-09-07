# LTX 2.5 FAST 1080 — baseline freeze

*8 Sep 2026, RTX PRO 6000 Blackwell Max-Q (`ltx-6000-2`), 97 GB VRAM,
241.5 GiB container memory ceiling. The node was idle and clean before the
measurement (16.3 GiB of the ceiling in use, no OOM pressure).*

This is the frozen "before" for the speed work. Nothing in it was changed to
produce these numbers: the client's exported graph was compiled through the
production compiler (`compile_fast_1080`) and queued, exactly as a Text to
Video job does.

## The workflow

| | |
| --- | --- |
| file | `LTX2.5_ACTUAL_WORKFLOW_ONLY_FAST_1080_8s_AUDIO.json` |
| sha256 | `19480e7432539bed3e478ef1ee998df906857cf5103e4104daa73989b1f6e712` |
| size | 151,017 bytes, 53 nodes across 5 subgraphs |
| frozen copy | `benchmarks/client-pack/ltx25/speed/LTX2.5_FAST_1080_original.json` (byte-identical) |

## The pipeline, node by node

| stage | node | value |
| --- | --- | --- |
| transformer | `UNETLoader` | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, `weight_dtype=default` |
| precision | — | **NVFP4** (4-bit weights, 18.7 GB on disk) |
| LoRAs | — | **none — the graph has no LoRA loader** |
| detailer | — | **none — the graph has no detailer stage** |
| text encoder | `CLIPLoader` ×2 | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`, type `ltxv` |
| video VAE | `VAELoader` | `ltx-2.5-video-vae-conv-bf16.safetensors` |
| audio VAE | `VAELoader` | `ltx-2.5-audio-vae-bf16.safetensors` |
| canvas | `EmptyLTXVLatentVideo` | 1920 × 1088 |
| delivery | `ImageScale` | 1920 × 1080, lanczos, `crop=center` |
| frames | `ComfyMathExpression` | `1 + floor(fps*seconds/8)*8` — 193 at 8 s, 361 at 15 s, 721 at 30 s |
| fps | `PrimitiveFloat` / `CreateVideo` | 24 |
| audio latent | `LTXVEmptyLatentAudio` | length-matched, 25 |
| sampler | `KSamplerSelect` | `euler_ancestral` |
| guider | `CFGGuider` | **cfg = 1.0** (no classifier-free guidance — already one forward pass per step) |
| schedule | `ManualSigmas` | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` |
| steps | — | **8** (9 sigmas) |
| decode | `VAEDecodeTiled` | tile 1536, overlap 256, temporal 256/32 |
| preprocess | `LTXVPreprocess` | 18 |

### What the schedule actually does

The eight steps are not evenly spaced. Their sigma deltas are:

    0.00625  0.00625  0.00625  0.00625  0.065625  0.184375  0.303125  0.421875

**The first four steps together move 2.5% of the noise range; the last four
move 97.5%.** Half the compute is spent in the top 2.5%. That is the single
largest structural target for the speed work, and it is a number the graph
already exposes (Phase 4 tests it).

## Runtime as found

| | |
| --- | --- |
| ComfyUI | v0.34.5 (`7fd919f0`), port 8189, own venv |
| launch flags | `--listen 127.0.0.1 --port 8189 --disable-auto-launch --preview-method none` |
| torch | 2.14.0+cu130, CUDA 13.0, device capability 12.0 |
| attention | **`Using pytorch attention`** — plain SDPA |
| accel libraries | **none installed**: no SageAttention, no FlashAttention, no NATTEN, no xformers (triton 3.8.0 present) |
| `--fast` | **not enabled** (fp16_accumulation, cublas_ops, autotune all off) |
| VRAM policy | `NORMAL_VRAM` — the 18 GB transformer is *streamed*: "Model LTXAV prepared for dynamic VRAM loading, 17831MB Staged" |
| kernels | `comfy_kitchen` cuda backend active (nvfp4 gemm, na3d); triton backend present but disabled |

Three things in that table are headroom, not settings: no accelerated
attention on a Blackwell card, no fast-math flags, and weight streaming on a
97 GB card whose peak use is 47 GB.

## Baseline measurement — 15 s

Seed 987654, 16:9, automatic dialogue off, timed by ComfyUI's own
`execution_start`→`execution_success` (not the adapter wall, which would
include queue wait).

| | |
| --- | --- |
| ComfyUI execution | **310.3 s** |
| adapter wall | 313.1 s |
| delivered | 1920×1080, 361 frames, 24 fps, with audio |
| peak VRAM | 46,690 MiB (46.7 GB of 97 GB) |
| peak container RAM | 66.0 GiB (of 241.5 GiB) |

### Where the 310 s goes

| phase | time | share |
| --- | --- | --- |
| model init (first step) | 33.3 s | 11% |
| sampling, 8 steps | 271 s total, **~34.0 s/step** | 87% |
| text encode + VAE decode + mux | ~39 s | 13% |

**Sampling is 87% of the render.** Any optimization that does not make a
step cheaper, or remove a step, cannot move the total much.

## The published ladder (7 Sep, same graph, same node)

| length | frames | time | peak VRAM |
| --- | --- | --- | --- |
| 5 s | 121 | 58 s | — |
| 8 s | 193 | 110–121 s | 45.9 GB |
| 15 s | 361 | 304–314 s | 34.3–46.7 GB |
| **30 s** | **721** | **~1050 s (17.5 min)** | 56.9 GB |

30 s costs 3.4× a 15 s render, not 2× — attention cost grows faster than
frame count. **The 30 s baseline for this work is 1050 s; the target is
300–360 s, a 2.9–3.5× speed-up.**

## Rollback

Nothing was changed to produce this report. The only node-side change so far
is a launcher that reads extra ComfyUI flags from `/workspace/comfy_extra_args`
(empty = the launch line above); the original launcher is kept at
`/opt/supervisor-scripts/zolexai-ltx-comfy.sh.orig`.
