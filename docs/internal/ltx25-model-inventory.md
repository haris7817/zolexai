# LTX 2.5 model inventory — what runs on the GPU for video generation

**Internal. 5 September 2026.** Produced by a five-source research sweep
(official Lightricks/LTX-2.5 repository, ComfyUI documentation, community
quantisations, the four LoRAs in the client pack, the official
`ltx-pipelines` repository), each sweep adversarially re-verified against
its primary source, then consolidated. Byte counts come from the Hugging
Face tree API unless marked "(unverified)". **Nothing here was measured on a
GPU** (the node is unavailable); the VRAM figures in §5 are sums of file
sizes, not resident memory.

Why this document exists: the platform now has two LTX 2.5 execution paths
— the CLI runtime already on the node and the client's three ComfyUI graphs
(`docs/internal/ltx-client-workflow-audit.md`) — and they load different
files of the same model. §3 and §4 map both to the official release; §5 is
the download and VRAM arithmetic for GPU day.

## 0. Resolved from the repository after the sweep

* **The tiny preview VAE (`taeltx2_3`) is preview-only in all three client
  graphs.** In `ltx25_text_to_video.json` and `ltx25_first_last_frame.json`
  it is loaded by `VAELoaderKJ` ("Tiny VAELoader KJ") and consumed only by
  `ModelPreviewOverrideKJ`; in `ltx25_character_replacement.json` by
  `VAELoader` ("Tiny VAE (sampler previews)") into `LTX2SamplingPreviewOverride`.
  The output path decodes with the full video VAE. A missing file would
  fail node validation, so it must be present; its quality never reaches the
  customer (settles item 2 of §7 below).
* **LoRA on the NVFP4 pre-quantised checkpoint** is recorded in project
  memory as known-bad on the CLI runtime (the worker forces
  `quantization none` + CPU offload for every LoRA tier); item 8 of §7 is
  therefore answered for our node by measurement, not documentation.
* **The GGUF the client shipped is not in the ZIP** — only the graph names
  it. Item 1 of §7 (Abiray vs realrebelai) is settled on GPU day by hashing
  the file the client provides, and `scripts/ltx_comfy_health.py --record`
  is where that hash goes.
* The character graph's INT8 transformer is the same official file
  `docs.comfy.org` uses for its LTX-2.5 template; the T2V/FLF graphs' GGUF
  Q8 is a community quantisation the client chose. Both stay as shipped
  (client decision, 5 Sep 2026); §4 lists the official INT8 file as the
  documented alternative if the GGUF loader misbehaves (ComfyUI-GGUF issue
  #477), which would be a recorded deviation, not a default.

---

Scope: consolidated from the five research sweeps (official HF repo, Comfy docs, community quants, LoRAs, official pipelines) plus their adversarial verification passes. Every byte count below was read from the Hugging Face tree API by a verifier unless marked **(unverified)**. Tensor dtypes were read from safetensors/GGUF headers only for non-gated files; for the gated Lightricks/LTX-2.5 files "precision" is what the filename and model card say. Nothing was measured on a GPU.

---

## 1. How LTX 2.5 generates a video

LTX-2.5 is described by Lightricks as "an open world model with open weights" that "generates synchronized, high-fidelity video and audio from text, image, and video inputs". It ships as a split, "Comfy-aligned" pack — one safetensors per component (LTX-2.3 was a monolith; "Files are not interchangeable between the two models").

| Component | What the sources say it is | What it does at inference |
|---|---|---|
| **22B DiT transformer** (`dev` or `distilled`) | ltx-core README: an "asymmetric dual-stream diffusion transformer" — "14B-parameter video stream (spatiotemporal dynamics) + 5B-parameter audio stream (1D temporal), sharing 48 transformer blocks but differing in width", 3D RoPE for video, 1D temporal RoPE for audio, bidirectional audio<->video cross-attention, cross-modality AdaLN. **Caveat:** the 14B/5B split text describes LTX-2 (19B); the 2.5 22B split is not stated on any fetched page. Kijai's 2.3 note ("blocks 0-1 and 46-47 kept in bf16") and comfy-quants confirm 48 blocks for the 22B family. | One model denoises video and audio latents jointly; there is no separate audio transformer. DFR docstring: "stage 2 still runs an audio pass because video needs the cross-modal attention". **dev** = "Full / trainable DiT", needs CFG/STG guidance; used by TI2Vid two-stage, Keyframe, A2Vid. **distilled** = "Fixed 8-step schedule, CFG=1" ("8 predefined sigmas: 8 steps stage 1, 4 steps stage 2"); used by DistilledPipeline, DFR, ICLora, DubIt. Card: the 2.5 distilled "retains much more of the full model's visual quality, prompt adherence, and motion consistency". 2.5 adds native multishot in one pass, generated keyframe slots, `keyframes_abs_pos_embedding`, 9-row `scale_shift_table`, `learnable_registers` in the embeddings connectors. |
| **Gemma 4 12B text encoder "with-proj"** | "Gemma 4 12B, fine-tuned for LTX, with the text projection bundled in; required by every pipeline". Loader checks version tag `gemma4-12b-ltx-v1`; stock Google Gemma 4 "is not a substitute". | Encodes the prompt and emits separate video and audio contexts through two connectors (`text_embedding_projection`, `audio_projector`). In ComfyUI: CLIPLoader type `ltxv`. The optional **prompt enhancer** is a different model (Gemma 4 E2B from Comfy-Org/gemma-4) that rewrites the prompt before encoding. |
| **Video VAE** (DiffVAE or Conv) | Encoder `[B,3,F,H,W] -> [B,128,F',H/32,W/32]`, `F' = 1+(F-1)/8`: 32x32 spatial, 8x temporal compression, 128 latent channels. Two decoders: "DiffVAE — higher quality, heavier" (NADiffusionDecoder, neighborhood attention, 2 Euler steps for distilled; fastest with natten, Triton/eager fallback, `blackwell_dsl` B200-only) and "Conv VAE — faster, lighter" (single forward pass). | Encodes conditioning images/video into latents; decodes final latents to pixels. Card: the diffusion decoder "replaces the VAE reconstruction stage; sharper faces, textures, and on-screen text". The latent grid is why `num_frames % 8 == 1` and width/height must be divisible by 32. |
| **Audio VAE + vocoder** (one file) | Stereo 16 kHz mel input; encoder `-> [B,8,T/4,16]` (4x temporal, 8 channels, 16 mel bins, ~1/25 s per token); HiFi-GAN-style stereo vocoder 16 kHz mel -> 24 kHz waveform (`output_sampling_rate` 24000, `upsample_rates [6,5,2,2,2]`). | Decodes the audio stream's latents to a waveform. "Missing the audio one is the most common reason a render comes out silent" (realrebelai). |
| **Latent spatial upscaler x2** | "x2 spatial upscaler required for multi-stage pipeline"; in_channels 128, mid 512, spatial_scale 2.0. | Stage 1 renders at half resolution; this upsamples latents 2x; stage 2 re-denoises at full size (4 distilled steps). Single-stage FLF2V does not use it. |
| **Latent temporal upscaler x2** | "x2 temporal upscaler"; DFR `--temporal-upscalings {0,1,2}`: "0->base fps, 1->2x with 2 tiles, 2->4x with 4 tiles". | Doubles playback fps per round (24 -> 48 -> 96): "extra temporal rounds add frames, not seconds". DFR only; the transformer snaps conditioning fps to 60 above 30 because "rates the model never saw (48, 50, 120, ...)" break RoPE time. |
| **Duration head** (model patch) | Consumes the Gemma caption embedding via the audio and video connectors, projects to a 256-dim pooler, attention-pools with a learned query, MLP -> log-duration. "LTX-2.5+". | When `--num-frames` is omitted (`--auto-duration MIN MAX`, defaults 1-20 s) predicts clip length, "snapped to the VAE's causal temporal grid (8k + 1)". |
| **Distilled LoRA `lora-450`** | "Distilled LoRA (dev-transformer workflows)". | Applied to the **dev** transformer in stage 2 of TI2Vid two-stage/HQ, Keyframe, A2Vid so the refinement pass runs on the distilled sigma schedule. Not used by distilled-checkpoint pipelines. What "450" denotes is not stated anywhere fetched. |
| **Pixel spatial upscaler IC-LoRA** (separate gated repo) | "Detailing IC-LoRA — required by DFRPipeline's refinement stage (--detailing-lora)"; strength "is ignored and hardcoded to 0.5". | DFR (Diffusion Fidelity Rendering) = distilled transformer, half-res stage 1 with generated keyframes, full-res re-denoise under this IC-LoRA, optional temporal rounds, optional tiled 4K epilogue (3840x2176). Not wired into ZolexAI. |

Native defaults from `constants.py`: 121 frames, 24.0 fps, stage-1 512x768 (=> 1024x1536 output); the Diffusers example runs stage 1 at 960x544 and stage 2 at 2x. Quantization in ltx-pipelines (`--quantization`): `fp8-cast` (bf16 downcast on load), `fp8-scaled-mm` (expects an fp8 checkpoint; Hopper+), `nvfp4-cast` (online, Blackwell), `nvfp4-prequant` (loads the shipped NVFP4 file, Blackwell + ltx-kernels).

LoRA compatibility statement (card): "the large majority of LoRAs and IC-LoRAs trained on LTX-2.3 run on LTX-2.5 without changes. A small number of exceptions exist — validate your adapters before production use." Lightricks' own 2.5 example graphs load the 2.3 Union-Control and Deblur IC-LoRAs on the 2.5 distilled transformer. No source extends this to LTX-2 (19B) adapters.

---

## 2. The official release: Lightricks/LTX-2.5

Repo facts (HF API): gated=`auto` (click-through), 17 files, 14 weight files summing to 200,853,641,398 B (~200.9 GB), lastModified 2026-09-01T06:29:03Z, sha `5e6e7101…`, cardData `license_name: ltx-2.x-community-license-agreement`, `license_link: github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x`. The raw README, file blobs and commit log return HTTP 401. No fp8, GGUF, int4 or mxfp8 file is shipped officially. The card body links `LICENSE.md`, which 404s; the live licence file is `LICENSE-2_x`.

| Folder / file | Precision | Bytes (GB) | Role | Runtime |
|---|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | bf16 | 42,018,190,584 (42.0) | "Full / trainable DiT (bf16)"; guided two-stage pipelines | ltx-pipelines; ComfyUI (Lightricks node-pack graphs) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | bf16 | 42,018,190,584 (42.0) | "Distilled DiT (bf16). Fixed 8-step schedule, CFG=1." | ltx-pipelines (`--quantization fp8-cast` downcasts on the fly); ComfyUI |
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | int8-convrot | 21,504,034,224 (21.5) | "Full DiT (Comfy int8 + convrot). ComfyUI only" | ComfyUI only |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | int8-convrot | 21,504,034,224 (21.5); sha256 `c4279eef…` | "Distilled DiT (Comfy int8 + convrot). ComfyUI only — not for ltx-pipelines / PyTorch" | ComfyUI only (the docs.comfy.org template file) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | nvfp4 | 18,721,548,408 (18.7); sha256 `4b94231e…` | "Distilled DiT (NVFP4). ComfyUI, or ltx-pipelines with --quantization nvfp4-prequant (Blackwell / ltx-kernels)" | ltx-pipelines (Blackwell) or ComfyUI (Blackwell). Only distilled exists in nvfp4. |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | bf16 | 26,263,858,182 (26.3) | "Gemma4 TE + projections (bf16)" | ltx-pipelines; ComfyUI |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | int8-convrot | 15,372,969,374 (15.4); sha256 `6ce688a0…` | "Same TE, Comfy int8 — ComfyUI only" | ComfyUI only |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | bf16 | 1,472,223,346 (1.47) | "DiffVAE — higher quality, heavier" | both |
| `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | bf16 | 1,452,269,922 (1.45) | "Conv VAE — faster, lighter" | both |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | bf16 | 364,866,540 (0.365) | "Audio VAE + vocoder" | both |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | bf16 | 995,778,752 (0.996) | "x2 spatial upscaler required for multi-stage pipeline" | both |
| `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | bf16 | 261,944,000 (0.262) | "x2 temporal upscaler" (DFR temporal rounds) | ltx-pipelines DFR only; no official ComfyUI workflow uses it |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | bf16 | 8,899,889,568 (8.9) | "Distilled LoRA (dev-transformer workflows)" | ltx-pipelines dev two-stage; no ComfyUI template uses it |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | bf16 | 3,843,690 (0.0038) | "Auto duration when --num-frames omitted" | ltx-pipelines; ComfyUI PR #15499 added duration nodes but no template loads it |

Companion Lightricks repo: `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` — 327,322,640 B, gated, precision unreadable, licence field `ltx-2-community-license`; required by DFR.

Dev vs distilled: identical byte counts per precision (same tensor set). Guided pipelines run dev in stage 1 with CFG/STG (typical cfg 2.0-5.0, stg 0.5-1.5) then apply lora-450 for stage 2; distilled pipelines run 8+4 steps at CFG 1 with no LoRA. DFR: "Do not pass the full (dev) transformer".

### Quantization formats and what each needs

| Format | Official file? | What it is | Hardware / runtime |
|---|---|---|---|
| **bf16** | yes (all components) | plain weights | any GPU; ltx-pipelines path; docs.ltx.io system requirements: "minimum 32GB+ VRAM", 32 GB RAM, 100 GB disk, CUDA 12.7+, Python 3.12+; recommended A100 80GB / H100 |
| **fp8** | **no** | ltx-pipelines runtime policies only: `fp8-cast` (downcast bf16 on load, any FP8-capable GPU), `fp8-scaled-mm` (needs an fp8 checkpoint, Hopper+ per optimization.md). Community fp8 files exist (guillaume127 23,485,111,216 B for ComfyUI; vonkaiser 21,025,119,068 B for ltx-pipelines FP8_SCALED_MM) | Ada SM 8.9+ for ComfyUI fp8 compute |
| **comfy-int8-convrot** | yes (both transformers, TE) | ComfyUI stock `int8_tensorwise` + ConvRot: int8 weights + fp32 per-row scale, regular Hadamard rotation group 256 applied when `in_features % 256 == 0` (true for all LTX-2 widths), activations dynamically row-quantized and rotated in the comfy-kitchen kernel (W8A8); marker `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`. **The mapping to `int8_tensorwise` was read from the non-gated Comfy-Org gemma4 e2b file, not from the gated Lightricks files (unverified for them).** rockerBOO's header inspection: attention INT8+ConvRot, norms/gate-logits/adaln/patchify/proj_out/scale-shift left unquantized, "all 48 blocks uniformly" (third-party, unverified) | ComfyUI >= 0.27.0 for int8 (2026-06-30), **>= 0.32.0 for LTX-2.5** (PR #15499, 2026-08-11; 0.33.1 fixes a float64 device bug in the diffusion decoder); comfy-kitchen >= 0.2.12; SM >= 7.5 (Turing+), "not Blackwell-gated"; comfy/quant_ops.py disables the CUDA backend below torch cu130 ("on nvidia 20 series and above it is required that you update your pytorch to cu130 or higher"); unsupported setups "fall back to dequantized matmul". ltxworkflow.com's "RTX 30xx cannot run this format" is contradicted by the primary sources. |
| **nvfp4** (pre-quantized) | yes (distilled only) | "FP4 E2M1 data + FP8 E4M3 per-16 block scales + FP32 per-tensor scale"; `nvfp4-prequant` loads "packed uint8 weights + block scales + weight_scale_2 + calibrated input_scale", static activation scale, `hi_first` nibble order; K % 32 == 0, N % 8 == 0 | ltx-pipelines: Blackwell SM >= 10 + ltx-kernels `nvfp4_cpp` (built for sm_100a/110a/**120a** — consumer/RTX PRO Blackwell included; `uv sync --group kernels`); cuBLASLt FP4 block-scaled GEMM. ComfyUI: `supports_nvfp4_compute()` requires `major >= 10`; on other GPUs "silently dequantizes". ComfyUI loadability disputed: HF discussion #16 initially failed with a mat1/mat2 shape error, LTX team replied "We fixed the issue"; BennyDaBall/rockerBOO still report the file lacks `.comfy_quant` markers (third-party, unverified). rockerBOO: 56 of 224 FF tensors (blocks 42-47 + embeddings_connector) are plain bf16 (unverified). The official file has changed at least once (vonkaiser's mirror is 18,721,432,024 B; commit "Restore keyframes support on the distilled NVFP4 transformer"). |
| **GGUF** | **no** — community only | llama.cpp block quants of the distilled (and, at vantagewithai, dev) transformer, dequantized per layer by city96 ComfyUI-GGUF (`IMG_ARCH_LIST` includes `ltxv`). realrebelai: a naive conversion drops the safetensors `__metadata__` `config` blob so ComfyUI builds a 2.3-shaped model and fails on shape mismatches; their files embed the 61-key config as a GGUF KV. **No GGUF of the text encoder loads in stock ComfyUI-GGUF** (`TXT_ARCH_LIST` has gemma3, not gemma4; issue #450 open). Q8_0 (23.6 GB) is larger than the official int8-convrot (21.5 GB). | any GPU ComfyUI-GGUF runs on; city96 README: "LoRA loading is experimental"; upstream last commit 2026-01-12; issue #477 (2026-08-29, open): LTX-2.5 GGUFs can leave three bf16 non-layer tensors as GGMLTensor with packed dims -> "size of tensor a (4096) must match tensor b (8192)" (AMD, elix3r Q5_K_M) |

Third-party VRAM claims (all **unverified**, none official): ltxworkflow.com "24 GB VRAM" for the int8-convrot transformer; smeltcore.com 22.67 GiB resident on a 4090 (calculated); BennyDaBall "10s @ 1280x736 with synced audio in ~50s on an RTX 5090"; VentureBeat quoting launch materials "minimum of 16GB of VRAM". comfy-quants measured **LTX-2.3** dev on an RTX PRO 6000: int8_tensorwise+ConvRot 45.4 dB PSNR vs bf16, 1.15x bf16 speed, -28% VRAM; FP8 0.96x / 38.5 dB; NVFP4 0.66x / 32.1 dB — 2.3 numbers, not 2.5, no GGUF column.

---

## 3. What runs on ZolexAI's GPU node today (ltx-pipelines CLI, pinned @400fd31)

The pinned commit `400fd31054597515f47125691032c04b1c3ee24e` is the merge of PR #276 (2026-08-16, "Add LTX-2.x Community License Agreement in txt format") — after v1.2.0 (11 Aug, the 2.5 release) and before v1.3.0 (26 Aug), which switched DFR to `--distilled-checkpoint-path`, added `--spatial-upscalings 2` for 4K, keyframe-aware DiffVAE decode and auto decode tiling.

| Node file (`models/ltx-2.5/…`) | Official file | Bytes | Status |
|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | identical name in `diffusion_models/` | 18,721,548,408 | **Primary transformer.** Needs `--quantization nvfp4-prequant`, Blackwell SM >= 10, ltx-kernels `nvfp4_cpp` built with `TORCH_CUDA_ARCH_LIST=10.0`/`uv sync --group kernels` ("skipped on hosts without SM >= 10"); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recommended. Which revision of the file the node holds is unknown (the official file has changed; see §2). |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | identical | 42,018,190,584 | **bf16 fallback** for the distilled tiers (DistilledPipeline / ICLora); `fp8-cast` can downcast it at load. |
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` (optional) | identical | 42,018,190,584 | Guided (TI2VidTwoStages, CFG/STG) and audio (A2VidPipelineTwoStage: "video-only denoising, audio frozen" in stage 1) tiers. No nvfp4 dev exists. |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (optional) | identical | 8,899,889,568 | Required by `default_2_stage_arg_parser` for the dev tiers (stage-2 refinement; README example strength 0.8, default 1.0). |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | identical | 26,263,858,182 | The only TE ltx-pipelines can load (int8-convrot is "ComfyUI only"). |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | identical | 1,472,223,346 | DiffVAE; `--diffvae-optimization chunked_eager` default; natten pin `natten==0.21.7+torch2130cu132` / torch 2.13.0 for the fast path; `blackwell_dsl` is B200-only ("Not used on consumer Blackwell (sm_120)"). The conv VAE is not present on the node. |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | identical | 364,866,540 | Audio VAE + vocoder. |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | identical | 3,843,690 | `--duration-head-path`; DistilledPipeline built with `supports_auto_duration=True`. |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | identical | 995,778,752 | Required by every two-stage parser. |
| `loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (optional) | `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` (not in the 2.5 repo) | 654,465,352; header: 960 tensors BF16, `model_version "2.3.0"`, `reference_downscale_factor "2"` | ICLoraPipeline ("Can only be used with a distilled model"); Lightricks' own 2.5 Union Control example graph loads this exact 2.3 file at strength 1.0. Repo memory: stage 2 needs even latent dims (`--skip-stage-2` path). |

Not present on the node: the temporal upscaler (262 MB), the conv VAE, the DFR detailing IC-LoRA (327 MB) — so DFR, the card's "production-quality" T2V/I2V path, cannot run today. Whether `nvfp4-prequant` accepts `--lora-paths` on top of a pre-quantized checkpoint is not addressed by any fetched doc (project memory records LoRA+FP8 as known-bad).

---

## 4. What the client's three ComfyUI graphs load

"Same file" = byte-identical to a file the CLI runtime already holds (symlink/copy suffices). Local paths such as `LTXVideo/v2/` and `LTX/LTX-2.5/` are the client's folder conventions, not upstream paths.

| File (graph, node, strength) | Role | Precision | Bytes | Source repo | Licence (as declared) | Verification | Same as CLI? |
|---|---|---|---|---|---|---|---|
| `LTX-2.5-Distilled-Q8_0.gguf` (T2V + FLF; `UnetLoaderGGUF`) | distilled transformer | GGUF Q8_0 (header: `general.architecture ltxv`, `file_type 7`, 4349 tensors) | **Abiray**: 23,604,666,816, sha256 `524a1410…`; **realrebelai**: 23,632,744,448, sha256 `82476f2e…` — identical filename, different files | Abiray/LTX-2.5-Distilled-GGUF (lastModified 2026-08-22) or realrebelai/LTX-2.5_GGUFs (2026-08-12) | Abiray: `ltx-2-community-license-agreement` (link 404), README "original LTX-2.x Community License"; realrebelai: `see-base-model` | sizes/hashes verified; **which one the client has is unresolved** (hash the local file) | **NEW** (no GGUF on the node) |
| `LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` (character replacement; `UNETLoader`) | distilled transformer | int8-convrot (filename/card; header gated) | 21,504,034,224; sha256 `c4279eef…` | Lightricks/LTX-2.5 (gated) | LTX-2.x Community License | size verified; format internals unverified | **NEW** (node has nvfp4 + bf16 only) |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (all three; `CLIPLoader` type `ltxv`) | text encoder + LTX projections | int8-convrot (filename/card) | 15,372,969,374; sha256 `6ce688a0…` | Lightricks/LTX-2.5 (gated) | LTX-2.x Community License | size verified | **NEW**. Alternative: the node's bf16 TE (26,263,858,182 B) is what the Lightricks node-pack 2.5 graphs and docs.ltx.io load via the same CLIPLoader — usable instead at +10.9 GB. |
| `ltx-2.5-video-vae-bf16.safetensors` (`VAELoader`) | video VAE (DiffVAE) | bf16 | 1,472,223,346 | Lightricks/LTX-2.5 | LTX-2.x | verified | **SAME** |
| `ltx-2.5-audio-vae-bf16.safetensors` (`VAELoader`, `LTXVAudioVAEDecode`) | audio VAE + vocoder | bf16 | 364,866,540 | Lightricks/LTX-2.5 | LTX-2.x | verified | **SAME** |
| `taeltx2_3.safetensors` (preview VAE) | tiny AE for previews | **F16** (header: 128 tensors) — not bf16; any bf16 is a load-time cast | 23,531,296; sha256 `f0773b4e…` (identical in Kijai/LTX2.3_comfy/vae and madebyollin/taehv) | madebyollin/taehv (GitHub) via Kijai/LTX2.3_comfy | taehv: MIT; Kijai repo tag: LTX-2 community | verified; taehv README: "For LTX-2.3 and LTX-2.5, load the taeltx2_3 weights"; author: "blurry outputs" | **NEW** (tiny) |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (`LatentUpscaleModelLoader`) | x2 latent upscaler | bf16 | 995,778,752 | Lightricks/LTX-2.5 | LTX-2.x | verified | **SAME** |
| `LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` @0.4 | RL LoRA (OmniNFT, arXiv 2605.12480: AV quality, cross-modal alignment, AV sync); no trigger word | **BF16** (header: 2688 tensors; downcast from the 1,233,512,728 B F32 PEFT adapter, r=32, alpha=64) | 616,948,520; sha256 `204b491f…` | Kijai/LTX2.3_comfy/loras (repack of zghhui/OmniNFT `LTX-2.3-RL-Lora`) | **conflicting**: Kijai repo `ltx-2-community-license-agreement`; upstream HF card `apache-2.0`; upstream GitHub README "Research use only" | verified; Kijai: "strength should be 2.0 in ComfyUI to match the original config alpha 64, rank 32" — client uses 0.4 | **NEW** |
| `ltx2.3-transition.safetensors` @0.8 (FLF) | first/last-frame transition LoRA; trigger word `zhuanchang`; card recommends strength 1.0, embedded guidance 1.0, CFG 4.0 | **BF16** (header: 1152 tensors) | 390,229,424; sha256 `ba420d6f…` | joyfox/LTX-2.3-Transition-LORA (valiantcat/… is a 307 redirect to it, same repo) | `apache-2.0` declared by uploader (LTX-2.3 derivative) | verified | **NEW** |
| `ltx-2-19b-ic-lora-detailer.safetensors` @0.3 | "video detailer IC-LoRA trained on top of LTX-2-19b" | **BF16** (header: 960 tensors; `__metadata__` embeds the LTX-2 Community License, Jan 5 2026) | 2,617,401,920; sha256 `05efdae9…` | Lightricks/LTX-2-19b-IC-LoRA-Detailer (lastModified 2026-03-28) | `ltx-2-community-license` (LTX-2, Jan 5 2026 text) | verified | **NEW** |
| `LTX/LTX-2.5/LTX25_Ripple_v11.safetensors` @1.35 (character replacement) | "First Frame All Frames (FFAF) IC-LoRA for LTX-2.5" — edit the first frame, propagate through the video; recommended strength 1.35 (matches client) | **BF16** (header: 960 tensors) | 654,443,392; sha256 `bde54361…` | WepeNerd/LTX-Ripple (community; base_model Lightricks/LTX-2.5; lastModified 2026-08-30) | `ltx-2-community-license-agreement`, ships `LICENSE-2_x` (LTX-2.x, Aug 11 2026): "LTX Ripple is an LTX-2.x derivative" | verified (two sweeps missed it; two located it via HF search / the WepeNerd demo Space) | **NEW** |

Not in the client graphs but in the official template set: `gemma4_e2b_it_int8_convrot.safetensors` (prompt enhancer, 5,199,997,904 B, Comfy-Org/gemma-4, apache-2.0, header marker `int8_tensorwise`+convrot verified) — the bf16 sibling (10,278,774,160 B) is the model the node's Director/Idea mode already runs.

### Cross-version adapters on the 2.5 22B transformer — what the sources say

- **LTX-2.3 -> 2.5 (OmniNFT, Transition, Union-Control):** covered by Lightricks' general statement ("large majority … run without changes … validate"). Lightricks' own 2.5 example graphs load the 2.3 Union-Control and Deblur IC-LoRAs on the 2.5 distilled bf16 transformer, so the mechanism is sanctioned; the specific OmniNFT and Transition adapters have no 2.5 statement from their authors. Several 2.3 beta IC-LoRAs (HDR, Dub-It, Relight) are explicitly "LTX-2.3 only — LTX-2.5 support in development", i.e. Lightricks re-validates per adapter.
- **LTX-2 19B -> 2.5 22B (Detailer):** no source claims compatibility; the card's statement covers 2.3-trained adapters only; Lightricks shipped no 2.3/2.5 detailer, and its 2.5 "detail" adapter is the Pixel Spatial Upscaler IC-LoRA (327 MB). The dossier notes the client loads it with `LoraLoaderModelOnly` inside the T2V/FLF subgraphs; the official IC-LoRA path uses `LTXICLoRALoaderModelOnly` (+ `LTXAddVideoICLoRAGuide` for guide frames). Whether it loads cleanly or does anything at 0.3 is untested.
- **LoRA on quantized weights:** ComfyUI v0.27.0 notes mention int8 LoRA fixes (#14650/#14685), implying support, but LTX is not named; city96: "LoRA loading is experimental" for GGUF. Nothing official covers the client's Q8_0 GGUF + official int8 TE pairing (Abiray's bundled workflow does exactly this, using `UnetLoaderGGUF` + `CLIPLoader`).
- **Strengths vs recommendations:** OmniNFT 0.4 vs Kijai's 2.0; Transition 0.8 vs card's 1.0; Ripple 1.35 = card's 1.35 (docs.ltx.io's generic 0.9-1.6 LoRA band is not the IC-LoRA guidance); Detailer has no recommended strength.

---

## 5. VRAM / disk arithmetic

Sums of the verified byte counts only. **Activations, latents, KV/attention workspace, decode buffers, CUDA context and framework overhead are not included; nothing here was measured.** GGUF weights are dequantized per layer at run time, ComfyUI offloads/patches LoRAs, and ltx-pipelines can `--offload cpu|disk`, so resident VRAM will differ from these figures in both directions.

| Path | Files summed | Total bytes | ≈ GB |
|---|---|---|---|
| **CLI distilled nvfp4** (today's primary) | nvfp4 transformer 18,721,548,408 + bf16 TE 26,263,858,182 + video VAE 1,472,223,346 + audio VAE 364,866,540 + spatial upscaler 995,778,752 + duration head 3,843,690 | 47,822,118,918 | **47.8** (+ union-control LoRA 654,465,352 -> 48,476,584,270 ≈ 48.5) |
| **CLI distilled bf16 fallback** | bf16 transformer 42,018,190,584 + TE + VAEs + upscaler + duration head | 71,118,761,094 | **71.1** |
| **CLI dev bf16 + lora-450** (guided/audio tiers) | dev bf16 42,018,190,584 + lora-450 8,899,889,568 + TE 26,263,858,182 + video VAE + audio VAE + upscaler + duration head | 80,018,650,662 | **80.0** |
| **Client T2V / FLF** (GGUF Q8 + int8 TE + VAEs + upscaler + preview VAE + 3 LoRAs) | Q8_0 23,604,666,816 (Abiray; realrebelai +28,077,632) + int8 TE 15,372,969,374 + video VAE 1,472,223,346 + audio VAE 364,866,540 + spatial upscaler 995,778,752 + taeltx 23,531,296 + OmniNFT 616,948,520 + transition 390,229,424 + detailer 2,617,401,920 | 45,458,615,988 | **45.5** (45.49 with realrebelai's Q8). The official FLF2V template is single-stage and omits the upscaler (-0.996 GB) — whether the client's FLF graph does was not established. |
| **Client character replacement** (int8 transformer + int8 TE + VAEs + Ripple) | int8 transformer 21,504,034,224 + int8 TE 15,372,969,374 + video VAE 1,472,223,346 + audio VAE 364,866,540 + Ripple 654,443,392 | 39,368,536,876 | **39.4** (+ upscaler + taeltx -> 40,387,846,924 ≈ 40.4) |

Disk for the client pack: new downloads = Q8_0 GGUF 23,604,666,816 + int8 transformer 21,504,034,224 + int8 TE 15,372,969,374 + taeltx 23,531,296 + four LoRAs 4,279,023,256 = **64,784,224,966 B ≈ 64.8 GB**; reusable from the node = video VAE + audio VAE + spatial upscaler = 2,832,868,638 B ≈ 2.83 GB. Official guidance is only "32GB+ VRAM, 100GB+ disk" (docs.ltx.io); no primary source publishes per-variant VRAM. Only the third-party figures in §2 exist, all **(unverified)**; vonkaiser's ltx-pipelines FP8 pack budgets "~32.5 GB resident" for transformer ~21 + TE ~7.4 + VAEs/upscaler ~4 **(unverified, different files)**.

---

## 6. Licence summary per file family

| Family | Licence | Notes |
|---|---|---|
| Lightricks/LTX-2.5 (all 14 files) and the Pixel-Spatial-Upscaler IC-LoRA | **LTX-2.x Community License Agreement, License date August 11, 2026** (`LICENSE-2_x`, 30,399 B; "applicable to all LTX-2.5 versions released since August 11, 2026") | §2.1: entities with annual revenue ≥ $10,000,000 need a paid Commercial Use Agreement ("measured across the whole entity, including subsidiaries and affiliates"); Attachment A #18 (no training other ML models except Derivatives), #19 (no circumventing watermark/provenance), **#20 (no product "that directly competes with Licensor's commercial products or services" without a separate licence)**; gated click-through with marketing consent; "Transfer of fine-tunes may require a paid license". `LICENSE.md` links on several cards 404. |
| Lightricks LTX-2 / 2.3 adapters (Union-Control, Deblur, 19B Detailer) and the ComfyUI-LTXVideo node pack | **LTX-2 Community License Agreement, January 5, 2026** (`LICENSE-2`; text embedded in the safetensors headers) | Root `LICENSE` index: applies to "all LTX-2 versions released since January 5, 2026, including LTX-2.3 until August 11, 2026". |
| ltx-pipelines / ltx-core / ltx-kernels code | No package-level LICENSE; GitHub reports "Other"/NOASSERTION; root index -> LICENSE-2_x; §1.9 defines "LTX-2.x" to include "inference-enabling code … at github.com/Lightricks/LTX-2" | Not stated for ltx-kernels directly. |
| Community GGUFs (Abiray, realrebelai, etc.) | Uploader-declared `ltx-2-community-license-agreement` / `see-base-model`; realrebelai: "All original licensing terms and usage restrictions carry over from the base model" | Derivatives of gated weights; Abiray's tag names the LTX-2 licence while its prose says LTX-2.x. |
| WepeNerd/LTX-Ripple | LTX-2.x Community License (ships `LICENSE-2_x`) | "LTX Ripple is an LTX-2.x derivative". |
| OmniNFT-RL LoRA | **Conflicting**: Kijai repack under LTX-2 community tag; upstream HF card `apache-2.0`; upstream GitHub README "Research use only. See individual submodule licenses" | Unresolved which governs the converted file and whether "research only" blocks commercial use. |
| Transition LoRA (joyfox/valiantcat) | `apache-2.0` declared by uploader on an LTX-2.3 derivative | Whether the LTX licence's Derivative terms override the Apache tag was not analysed. |
| taeltx2_3 | MIT (madebyollin/taehv, "Copyright (c) 2025 Ollin Boer Bohan") | Kijai's redistribution repo is tagged LTX-2 community; Karam98 mirror tags MIT. |
| Comfy-Org/gemma-4 prompt enhancer | `apache-2.0` (repackager field; base google/gemma-4-E2B-it; Google's licence page displayed the Apache 2.0 text per one verifier) | Google's Gemma terms not independently analysed. |
| Tooling | comfy-quants GPL-3.0; comfy-kitchen Apache-2.0; city96 ComfyUI-GGUF Apache-2.0 | Tool licences; the checkpoints carry the Lightricks licence regardless. |

---

## 7. Unresolved / unverified items

1. **Which `LTX-2.5-Distilled-Q8_0.gguf` the client has** — Abiray (23,604,666,816 B, `524a1410…`) vs realrebelai (23,632,744,448 B, `82476f2e…`); a local sha256 settles it. Abiray documents neither the conversion tool nor whether the 61-key `config` KV is embedded; its "nearly indistinguishable from bf16" is prose, not a measurement. ComfyUI-GGUF issue #477 (open) may affect any 2.5 GGUF.
2. **taeltx2_3 in the client graphs** — ComfyUI core previews cannot auto-load it (`VIDEO_TAES` lists only `taeltx_2`; LTXV latent formats set no `taesd_decoder_name`); the dossier's internal audit says `VAELoaderKJ` loads it from `vae/`; the node actually used in the three graphs was not read from the graphs. Author documents blurry output; the `taeltx2_3_wide` fix is `.pth`-only on a branch and needs ComfyUI-bleh code.
3. **Gated file internals** — dtype/quant markers of every Lightricks/LTX-2.5 file, the int8-convrot recipe (whether blocks 0,1,46,47 stay bf16 as in comfy-quants' 2.3 recipe), the nvfp4 file's `.comfy_quant` markers and its 56 bf16 FF tensors, the 1.47 GB DiffVAE composition (the Diffusers pack's diffusion_decoder alone is 834,313,048 B), and what changed in the repo on 2026-09-01 (commit log 401).
4. **`450` in `lora-450`** — meaning not stated anywhere.
5. **LTX-2 19B detailer on the 22B 2.5 transformer** — no source; needs a load test and an A/B at 0.3.
6. **OmniNFT** — 0.4 vs Kijai's stated 2.0 equivalence; licence conflict (apache-2.0 vs research-only vs LTX-2).
7. **Transition** — 0.8 vs recommended 1.0; whether client prompts include `zhuanchang`; Apache tag on an LTX derivative.
8. **nvfp4-prequant + LoRA stacking** in ltx-pipelines (union-control on the pre-quantized checkpoint) — undocumented; memory records LoRA+FP8 as known-bad.
9. **nvfp4 in ComfyUI** on SM 12.0 — native kernel vs silent dequant, and the marker dispute (LTX team "fixed" vs BennyDaBall/rockerBOO).
10. **No official per-variant VRAM numbers** for bf16/fp8/int8/nvfp4/GGUF; all GB claims beyond "32GB+ minimum" are third-party or computed, and none of the §5 sums is measured.
11. **docs.comfy.org's "up to 50 FPS" and "Native 4K HDR"** — no counterpart in Lightricks docs (24 fps native, 48/96 via DFR temporal rounds; 4K = 3840x2176 via DFR spatial epilogue; HDR via EXR IC-LoRA).
12. **ComfyUI-LTXVideo node pack** README (master, 2026-08-20) contains no LTX-2.5 content; 2.5 support lives in core ComfyUI >= 0.32.0 (templates `minComfyUIVersion 0.32.0`); the pack may still be needed only for IC-LoRA nodes. No official ComfyUI DFR / temporal-upscaler / duration-head workflow was found.
13. **Repo README vs pipelines.md conflict** on HDRICLoraPipeline (Dev vs Distilled).
14. **Pinned tree vs v1.3.0** — confirm which v1.3.0 changes (DFR args, PipelineOutput, 4K epilogue, decode tiling) the node's @400fd31 lacks before upgrading.
15. **22B video/audio stream split** — the "14B + 5B" text is LTX-2 (19B); not stated for 2.5.
16. **Whether the client's local files are byte-identical to upstream** and were obtained under the Lightricks gate — no local hashes compared.
17. **Repos that do not exist** (as of 2026-09-05): Kijai/LTX2.5_comfy, QuantStack/LTX-2.5-GGUF, unsloth/LTX-2.5-GGUF, Comfy-Org/ltx-2.5 — do not cite them.