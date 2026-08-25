# H3 client runtime — the frozen configuration

**Frozen 25 August 2026.** This document is the source of truth for the H3
INT8 ComfyUI runtime the client-test build ships. Every value below was
verified on the RTX PRO 6000 Blackwell WS, not carried from chat or memory.
Changing any pinned value is a new compatibility test, not a drop-in edit —
the pack's own guide says so about its Extender, and the rule generalises.

---

## 1. Service stack

| Component | Pin | Verified how |
|---|---|---|
| ComfyUI core | tag **`v0.33.3`** = commit `4da9e2db` | checked out and served the runs |
| ComfyUI frontend (graph metadata) | `1.49.6` | serialized in the frozen graphs |
| `ComfyUI_MiniMax_H3_Extender` | **`6a3583d0840116978f739600f482c03176ce9972`** (v1.9.0) | full-SHA checkout; upstream is already at 1.9.2 — do NOT upgrade without a compatibility pass |
| `ComfyUI-Easy-Use` | **`4de1ab3b66e48da916b6f263bacd001df53a2720`** | full-SHA checkout; provides the lazy duration selector |
| Python / torch | 3.12 / `torch 2.11.0+cu128` | sm_120 requires CUDA ≥ 12.8 wheels |
| Media | ffmpeg + imageio-ffmpeg | final H.264/AAC assembly |

## 2. Weights — official Comfy-Org, SHA256-verified

Repo: `Comfy-Org/MiniMax-H3`. Local SHA256 of every file matched the repo's
LFS records byte-for-byte (`ALL OFFICIAL-VERIFIED`, 25 Aug).

| File | Bytes | SHA256 |
|---|---:|---|
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c…` (API-verified at download) |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1,956,193,000 | `5b9ab5ad…` — **on disk, NOT wired**: rejected on quality 25 Aug |

Never substitute the community mirrors: the Abiray copies of the same
filenames have different SHA256s, and the only other repos carrying the
encoder's exact name are uncensored/abliterated variants.

Health checks (`worker/adapters/h3_comfy.py::h3_comfy_health`) verify
existence + exact published byte size on every poll; full SHA256 is a
provisioning-time check.

## 3. Workflow graphs

The client's delivered graphs, byte-transcribed and structurally validated
(every one of 270 links resolves on both ends), committed at
`benchmarks/client-pack/`:

| Graph | Nodes/links | Serves |
|---|---|---|
| `minimax_h3_r2v_extender.json` | 35 / 47 | Reference video (R2V/Ref2VA) — also T2V-capable, unused |
| `minimax_h3_i2v_extender.json` | 77 / 180 | Image animation (FL2VA) |
| `minimax_h3_t2v_extender.json` | 31 / 43 | not routed — T2V stays LTX |

**These files are the only durable copy of the client's deliverable.**

## 4. Sampling and segment logic — the pack's pins

| Setting | Value | Origin |
|---|---|---|
| steps | **20** | client graphs (adapter never changes it) |
| sampler / scheduler | `res_multistep` / `beta` | client graphs |
| denoise | 1.0 | client graphs |
| guidance | none — BasicGuider, guidance-distilled | client graphs |
| fps | 24 | client graphs |
| output | H.264 CRF 17 `fast`, AAC 192k | Final Decode widgets |
| video context between segments | **22 frames (0.917 s)** | Extender default, confirmed in source |
| audio context between segments | **0 — disabled** | client graphs; see the audio-seam check |
| frame lattice | `17k+5` | Extender `_align_frame_count` |

Duration presets (index 0–4) and their exact segment plans:

```text
index 0   5 s   1 segment            124 f = 5.167 s
index 1  10 s   1 segment            243 f = 10.125 s
index 2  15 s   1 segment            362 f = 15.083 s
index 3  30 s   2 segments 15+15     362+362−22        = 702 f  = 29.250 s
index 4  60 s   5 segs 12.5×3+12×2   311×3+294×2−4×22  = 1433 f = 59.708 s
```

Fixed seeds (repeatability): R2V segments `731003121–125`, I2V noise seeds
`410620260911–915`, T2V `731003101–105`. The adapter does not touch them.

## 5. Tiers — measured, ZolexAI-internal

| Tier | Canvas | Measured (per 5 s warm) | Realtime |
|---|---|---:|---:|
| **Draft** (R2V) | 544×320 | ~55 s | ~11× |
| **Quality** (R2V) | 960×544 | ~171 s | ~33× |
| I2V (single proven canvas) | 1280×736 | ~300 s warm | ~58× |

Cost scales linearly in pixels. Tier selection: `execution.h3_tier`
(`draft`/`quality`), default quality. Not exposed: Turbo (quality collapse at
v0.1 — wrong subject, streak artifacts), BF16 diffusers (economics),
`h3_comfy` runtime name itself (frontend says Draft/Quality only).

## 6. The prompt discipline (mandatory for 30 s / 60 s)

The pack's shipped placeholder prompts drift (measured: wardrobe change at
segment 2, portrait collapse at segment 4, scene reset at segment 5). The
integration generates every segment prompt via
`worker/longform/h3_prompts.py`: each segment re-states subject, wardrobe,
environment, props, camera and reference labels, names its handoff, and
states permanent departures negatively so they cannot resurrect. Proven: the
same 60 s plan with disciplined prompts held one subject/coat/room/camera
across all five segments (`client-h3-comfyui-results.md` §8).

## 7. Known limits carried into client-test

- **Audio seams**: `audio_context_length = 0`; audio continuity across 30/60 s
  seams is checked separately (see the audio-seam report) — video context does
  not cover it.
- **R2V does not re-enact source motion.** The graph consumes images; the
  integration maps reference→Picture 1, source-video first frame→Picture 2.
  Motion re-enactment remains LTX transform's job.
- **Seam-exact continuation is FL2VA's property, not Ref2VA's** — which is why
  Extend stays on LTX.
