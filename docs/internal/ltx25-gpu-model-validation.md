# LTX 2.5 on the GPU — model and workflow validation

**Internal. 5 September 2026. Measured on the node; every number below was
read from the machine, not from a document.** Node: Vast.ai container
`163.182.37.67:20577`, not a persistent volume (recycle/destroy wipes it).

## 1. GPU

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| VRAM | 97,887 MiB |
| Driver | 595.71.05 (CUDA 13.2 driver); image toolkit CUDA 12.8.1 |
| Compute capability | 12.0 (sm_120) |
| Host | Ubuntu 24.04.4, kernel 6.8.0-138, 48 CPUs, 503 GB RAM, 8.0 TB free |
| Torch, official LTX env | 2.13.0+cu132 (repo pin), NATTEN 0.21.7, `ltx-kernels` built for sm_120 — `import ltx_kernels` OK |
| Torch, ComfyUI env | 2.14.0+cu130 |

## 2. Official Lightricks LTX 2.5 stack (Step 2)

`/workspace/ltx2-benchmark` = `github.com/Lightricks/LTX-2` at
`400fd31054597515f47125691032c04b1c3ee24e` (the audited pin), `uv sync
--extra natten --group dev --group kernels`; the kernels built with the
venv's own nvcc 13.2 (`.venv/lib/python3.11/site-packages/nvidia/cu13`) and
`TORCH_CUDA_ARCH_LIST=12.0`. The system nvcc (12.8) cannot build them
against a cu132 torch — same finding as the previous node.

Every file below lives once, under `/workspace/ltx2-benchmark/models/ltx-2.5/`,
and is mirrored into the ComfyUI tree by symlink. Sizes and hashes were
computed on the node after download.

| File | Bytes | sha256 (first 16) | Role |
|---|---:|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 | `4b94231e734c1950` | **primary production transformer** (Blackwell, `--quantization nvfp4-prequant`) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 | `31eb3cad89b9e54e` | bf16 fallback; any LoRA tier |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26,263,858,182 | `ef7243612fdae7a7` | official text conditioning |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | 1,472,223,346 | `847e14ca7f3355de` | video DiffVAE |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | 364,866,540 | `c52733d37f6a7fb7` | audio VAE + vocoder |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 995,778,752 | `eb5a71fe4068ee87` | x2 spatial upscaler |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 3,843,690 | `2ec71e4206ed365d` | auto-duration head |
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` (optional, unused) | 42,018,190,584 | `792a2bad501ca032` | guided/audio tiers |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (optional, unused) | 8,899,889,568 | `86370bbf79a9eb4e` | dev-tier stage 2 |
| `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | 654,465,352 | `a1b888a87f661d27` | the CLI runtime's V2V transform engine |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | `c4279eeff115cbea` | the client's character graph (ComfyUI only) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15,372,969,374 | `6ce688a0aa98a5fa` | the client's three graphs (ComfyUI only) |

Every byte count equals the Hugging Face tree listing recorded in
`ltx25-model-inventory.md`; the three hashes that listing carried (nvfp4,
int8 transformer, int8 text encoder) match.

## 3. Step 7 — the official pipeline test

`ltx_pipelines.distilled`, split layout, `--quantization nvfp4-prequant`,
seed 42, 1280x704 (the pack's own 16:9 canvas; the CLI needs multiples of
64), 121 frames at 24 fps, the brief's prompt.

| | |
|---|---|
| Wall clock (model load included, cold) | 41.5 s |
| VRAM peak / mean (nvidia-smi, 1 Hz) | 24,093 / 13,133 MiB |
| RAM max RSS | 26,567 MiB |
| Output | 1280x704, 121 frames, 24.0 fps, 5.042 s, 1 audio stream |
| Errors | none |
| Frames 0 / 60 / 119 | photoreal sunset skyline, a flying vehicle mid-frame, camera moving; not green, not black |

Files: `/workspace/results/step7/official_t2v_5s_nvfp4-prequant_seed42_*.{mp4,json,vram.csv,stdout.log,stderr.log}`.

## 4. ComfyUI environment (Step 5)

`/workspace/ComfyUI-ltx` = ComfyUI **v0.34.5**, own venv (Python 3.12,
torch 2.14.0+cu130, comfy-kitchen 0.2.31, comfyui-frontend-package 1.49.6,
gguf 0.19.0, kornia **0.8.1**), port 8189 under supervisord
(`zolexai-ltx-comfy`), `extra_model_paths.yaml` pointing at the official
tree. Node packs, at the commits that serve:

| Pack | Commit | Versus the graphs' stamp |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` 2026-01-12 | = stamp |
| ComfyUI-KJNodes | `e8e88f7` 2026-08-29 | = newest stamp |
| rgthree-comfy | `35c9f1e` 2026-08-27 | = newest stamp |
| ComfyUI-VideoHelperSuite | `115de7a` 2026-08-25 | = newest stamp |
| ComfyUI-Easy-Use | `005c578` 2026-09-01 | = stamp |
| ComfyMath | `c011772` 2025-03-08 | = stamp |
| RES4LYF (drozbay fork) | `26036f6` 2026-08-06 | = stamp (on a non-default branch; fetched explicitly) |
| ComfyUI-mxToolkit | `7f7a0e5` 2025-05-07 | HEAD (no tag matches "0.9.92") |
| ComfyUI-Custom-Scripts | `609f3af` 2026-02-12 | HEAD (no tag matches "1.2.5") |
| **ComfyUI-LTXVideo** | **`15d09ab` 2026-08-20 (master)** | **deviates from stamp `229437c` (2026-05-11): that commit fails to import against 0.34.5 core (`interleaved_freqs_cis` moved). Master imports once kornia is pinned to 0.8.1 (0.8.3 removed `pad` from `kornia.geometry.transform.pyramid`). The one node the graphs use, `LTXAddVideoICLoRAGuideAdvanced`, exists with the exact input names the character graph sets.** |

All **64** node classes the three graphs use are present on the server;
`ResolutionSelector` offers the three product labels (`16:9 (Widescreen)`,
`9:16 (Portrait Widescreen)`, `1:1 (Square)`) — verified live, and the
compiler now reads this ComfyUI's newer combo schema (`bd28200`).

## 5. Client ZIP workflow compatibility (Step 3)

`scripts/ltx_comfy_health.py --deep` on the node: **HEALTHY** — "3 graphs
compile and every node class, combo value and model file they name is
offered by the server". Per graph:

### 5.1 Text to Video (+ Audio) — `ltx25_text_to_video.json`

| Check | Result |
|---|---|
| Transformer | `LTX-2.5-Distilled-Q8_0.gguf` via `UnetLoaderGGUF` — present, 23,604,666,816 B, sha256 `524a1410c6476d94…` (= Abiray/LTX-2.5-Distilled-GGUF). A community GGUF Q8_0 of the official distilled transformer, **not** an official file; kept as the client shipped it |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` via `CLIPLoader` type `ltxv` — official int8 file, present, hash matches |
| VAEs | `ltx-2.5-video-vae-bf16` / `ltx-2.5-audio-vae-bf16` via `VAELoaderKJ` — official files, same bytes the CLI runtime uses |
| Preview VAE | `taeltx2_3.safetensors` (23,531,296 B, `f0773b4e…`, madebyollin/taehv via Kijai) — feeds only `ModelPreviewOverrideKJ`; never on the output path |
| Upscaler | official spatial x2 — present |
| LoRAs | `LTX-2.3-OmniNFT-RL-Lora_bf16` @0.4 (`204b491f…`), `ltx2.3-transition` @0.8 (`ba420d6f…`) through `Power Lora Loader (rgthree)`; `ltx-2-19b-ic-lora-detailer` @0.3 (`05efdae9…`) through `LoraLoaderModelOnly` — all present; the first two are LTX-2.3 adapters (Lightricks: "the large majority … run on LTX-2.5 without changes"), the third an LTX-2 19B adapter with no compatibility statement from any source. **The graph loads and renders with all three** (Step 8A) |
| Result | RAN on the GPU: 1280x704, 121 frames, 24 fps, 5.04 s with audio (Step 8A, seed 42) |

### 5.2 First/Last Frame Video (+ Audio) — `ltx25_first_last_frame.json`

Same loaders, LoRAs and files as 5.1; adds `LoadImage` ×2 →
`ResizeImageMaskNode` → `LTXVImgToVideoInplace` (strength 0.8) and
`LTXVImgToVideoInplaceKJ` (indices 0 / −1). Compiles and validates on the
server with both stills and with the first still only (last-frame node
bypassed). Render: see §7 when run.

### 5.3 Character Replacement — `ltx25_character_replacement.json`

| Check | Result |
|---|---|
| Transformer | `LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` via `UNETLoader` — the official int8 file (symlinked under the client's folder convention), hash matches |
| Text encoder / VAEs / upscaler | official files, as 5.1 |
| LoRA | `LTX/LTX-2.5/LTX25_Ripple_v11.safetensors` @1.35 (`bde54361…`, WepeNerd/LTX-Ripple, LTX-2.x derivative) — present |
| Guide node | `LTXAddVideoICLoRAGuideAdvanced` from ComfyUI-LTXVideo `15d09ab` — present, inputs identical to the graph's |
| Patches | `ModelSamplingSD3`, `LTX2AttentionTunerPatch`, `LTXVChunkFeedForward`, `ModelAttentionBackend` — all present |
| Result | compiles and validates; render: see §7 when run |

### 5.4 What was NOT changed

No graph edited, no model substituted, no node renamed. The two dependency
deviations (LTXVideo pack commit, kornia pin) are environment pins forced by
ComfyUI 0.34.5 compatibility and are recorded above; the graphs themselves
are the ZIP's bytes (`benchmarks/client-pack/ltx25/README.md`).

## 6. Directory structure (Step 6)

```
/workspace/ltx2-benchmark/models/ltx-2.5/   (official, one copy)
├── diffusion_models/      nvfp4 · distilled-bf16 · dev-bf16 · distilled-comfy-int8-convrot
├── text_encoders/         gemma4 bf16 · gemma4 comfy-int8-convrot
├── vae/                   video-vae-bf16 · audio-vae-bf16
├── latent_upscale_models/ spatial-upscaler-x2
├── model_patches/         duration-head
└── loras/                 distilled-lora-450 · ic-lora-union-control-ref0.5
/workspace/ComfyUI-ltx/models/              (client tree: symlinks to the above + the client's own files)
├── diffusion_models/      LTX-2.5-Distilled-Q8_0.gguf · LTXVideo/v2/…int8-convrot (→ official)
├── text_encoders/ vae/ latent_upscale_models/ model_patches/  (→ official) + vae/taeltx2_3
└── loras/                 OmniNFT · transition · detailer · LTX/LTX-2.5/LTX25_Ripple_v11 (+ → official LoRAs)
```

Every file exists (deep health check, `weights_official.txt`,
`weights_public.txt` under `/workspace/provision/`).

## 7. Steps 8 and 9

See `docs/internal/ltx25-gpu-benchmark.md`.
