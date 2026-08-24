# Client H3 ComfyUI workflow pack — audit

**Date:** 25 August 2026 · Source: `MiniMax_H3_Extended_Workflow_Pack_Guide.docx`
(ComfyHelper client delivery, order `FO61C09B98582`), read in full from the
document itself, not from chat summaries.

**Why this exists.** Our measured H3 numbers are all from the provider-native
BF16 diffusers path (112–326x real time). The client's pack describes a
different execution architecture — INT8 ConvRot weights, ComfyUI, segment
continuation — and the routing decision must not be made against the slow path
alone. This audit records what the pack actually says, what could be verified
against source, and what is missing.

---

## 1. What the delivery contains — and what we actually have

Per the guide's own delivery scope:

> Three separately importable workflow JSON files, one sample MP4, this guide,
> and one gallery image. **Model files, installation, cloud deployment, and API
> wrapping are not included.**

| Item | Status on our side |
|---|---|
| The guide (docx) | **HAVE** — `E:\Downloads\MiniMax_H3_Extended_Workflow_Pack_Guide.docx` |
| `minimax_h3_t2v_extender.json` | **MISSING** |
| `minimax_h3_i2v_extender.json` | **MISSING** |
| `minimax_h3_r2v_extender.json` | **MISSING** |
| 59.7 s sample MP4 (1280x736) | **MISSING** |
| Gallery image | missing (embedded copies exist inside the docx) |
| Model files | not part of the delivery by design |

Searched: `E:\Downloads`, `E:\zolexai`, `C:\Users\Hp\{Downloads,Desktop,Documents,OneDrive}`.
Nothing matching `minimax_h3_*extender*.json` exists on this machine, and no
59.7 s / 1280x736 MP4 is present.

> **CLIENT JSON REQUIRED.** The workflows themselves are the deliverable — the
> guide describes them but does not contain them. Reconstructing graphs from
> the description and calling the result "the client's workflow" is exactly
> what the ground rules forbid. The three JSONs (and ideally the sample MP4)
> must come from the client before any reproduction run is honest.

## 2. Environment pins — verified against source

| Component | Guide pin | Verified |
|---|---|---|
| ComfyUI core | `0.33.3` (metadata baseline) | not yet installed |
| ComfyUI frontend | `1.49.6` | not yet installed |
| `ComfyUI_MiniMax_H3_Extender` | **v1.9.0 @ `6a3583d0840116978f739600f482c03176ce9972`** | **VERIFIED** — commit exists in [tritant/ComfyUI_MiniMax_H3_Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender), "Bump version from 1.7.0 to 1.9.0", 21 Aug 2026 |
| `ComfyUI-Easy-Use` | `4de1ab3b66e48da916b6f263bacd001df53a2720` | repo exists; commit not yet fetched |
| Media | imageio-ffmpeg / working FFmpeg | ffmpeg present on the box |

Upstream Extender is already at **1.9.2** (also 21 Aug 2026 — three versions in
one day). The guide itself warns that upgrading is "a new compatibility test,
not a drop-in change", and the hard rules pin us to v1.9.0. **Do not upgrade.**

### 2.1 Claims verified at code level

| Guide claim | Source finding | Result |
|---|---|---|
| 22 frames (~0.917 s) of continuation context | `motion_context_disk.py`: `trim_frames` defaults to **22** (`int(trim_frames if trim_frames is not None else 22)`); 22/24 fps = 0.9167 s | **VERIFIED** (as a default, overridable per graph) |
| 17k+5 frame alignment | `extender.py`: `_align_frame_count` loops `while n % 17 != 5: n += 1` | **VERIFIED** |
| Preset frame counts 124 / 243 / 362 | 124 = 17·7+5, 243 = 17·14+5, 362 = 17·21+5 | **VERIFIED** arithmetic |
| Disk Join / RAM / Final Decode nodes | `MiniMaxH3MotionContextDiskJoin`, `motion_context_ram`, disk cache with `CACHE_VERSION` | **VERIFIED** to exist |
| Ref2VA references + audio refs | `MAX_VIDEO_REFS = 3`, `MAX_STANDALONE_AUDIO_REFS = 3`, reference upload/encode helpers | **VERIFIED** |
| Lazy duration routing via Easy-Use | guide claim; plausible (`lazy: True` inputs seen) | **partially verified** — full check needs the JSON |
| 30 s plan: 2×362 − 22 = **702 frames** | arithmetic works exactly | **CONSISTENT** |
| 60 s plan: 5 segments → **1,433 frames** | does **not** decompose as 5×362 − 4×22 = 1722; the per-segment plan must be non-uniform | **CANNOT VERIFY without the JSON** — the "duration-plan JSON" lives inside the workflow |

### 2.2 What the guide explicitly does NOT claim

Worth stating because it bears on the client's expectations:

- **No performance numbers anywhere.** The guide's validation section says:
  *"No fresh T2V or R2V render, queue/history log, GPU benchmark, or
  cloud-cost benchmark is included."* The **"~1 minute of video in ~6
  minutes"** figure exists only in conversation — it has no documentary basis
  in the pack, and testing it is measuring a rumour, not verifying a spec.
- **The sample proves only I2V-60s.** The guide's own evidence boundary:
  *"The included 1280x736, 59.7-second sample strongly matches the I2V
  60-second configuration. It is not evidence that all three modes were
  rendered during this packaging pass."* T2V and R2V are unproven even by the
  vendor.

## 3. Workflow architecture (from the guide)

| | T2V | I2V | R2V / Ref2VA |
|---|---|---|---|
| File | `minimax_h3_t2v_extender.json` | `minimax_h3_i2v_extender.json` | `minimax_h3_r2v_extender.json` |
| Canvas | **544x320** | **1280x736** | **544x320** |
| Inputs | Prompts 1–5, fixed seeds | 1 source image + prompts | 1–3 reference images + prompts |
| Diffusion model | ref2va INT8 (same file as R2V) | fl2va INT8 | ref2va INT8 |
| R2V picture roles | — | — | P1 identity · P2 environment/composition · P3 optional secondary/style |

Duration routing (index 0–4, **opens at 4 = the expensive 60 s path**):

| Index | Nominal | Plan | Assembled |
|---|---|---|---|
| 0 | 5 s | 1 segment, Prompt 1 | 124 f / 5.167 s |
| 1 | 10 s | 1 segment | 243 f / 10.125 s |
| 2 | 15 s | 1 segment | 362 f / 15.083 s |
| 3 | 30 s | 2 continued segments, Prompts 1–2 | 702 f / 29.250 s |
| 4 | 60 s | 5 continued segments, Prompts 1–5 | 1,433 f / 59.708 s |

Output: 24 fps, H.264 CRF 17 fast, AAC 192k, prefixes `minimax_h3_{t2v,i2v,r2v}`.
Continuation carries 22 frames of motion/audio context between segments; each
later prompt is meant to start from the prior final frame and end in a "stable
handoff pose".

**Contrast with our BF16 baseline, for fairness accounting:**
R2V at 544x320 is **174,080 px/frame against our 1,032,192** (1344x768) — a
**5.93x** smaller pixel workload before quantization, pruning, steps or runtime
enter the picture. Phase 13's warning is quantitative: most of any speedup may
be resolution, not INT8.

## 4. Model files — sourcing (Phase 3)

The guide names five files and their ComfyUI folders. Four of five are
**exactly located** in
[Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot)
(HF, last modified 2026-08-15), with API-published SHA256:

| File | Bytes | SHA256 (prefix) | Licence |
|---|---:|---|---|
| `MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors` | 20,970,379,689 | `ed7e9aa5…` | MiniMax H3 Community |
| `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors` | 20,970,379,688 | `f07a5427…` | MiniMax H3 Community |
| `minimax_h3_video_vae_fp16.safetensors` | 5,207,808,557 | `88539d72…` | MiniMax H3 Community |
| `minimax_h3_audio_vae_fp32.safetensors` | 605,254,869 | `83043bf3…` | MiniMax H3 Community |

Notes that matter:

- **"Pruned" is a quality variable on top of INT8.** The BF16 transformer is
  61.7 GB; this file is ~19.5 GiB. INT8 alone would halve BF16, so pruning has
  removed a substantial fraction of weights as well. The author publishes **no
  conversion method, no source revision, and no quality evaluation**. Filename
  case differs from the guide (`MiniMax_H3_…` vs `minimax_h3_…`) — cosmetic.
- The repo also carries a **`ref2.json` (53.2 kB)** — an example workflow that
  may be the ancestor of the pack's R2V graph. Useful context; not the client's
  graph.
- Licence territory: same MiniMax H3 Community licence as the base model —
  fine on the Israel box, same constraints as before.

### 4.1 The text encoder — STOP, cannot be sourced reliably

`qwen3vl_32b_minimax_h3_int8_convrot.safetensors` is a problem:

1. The Abiray README says it lives in this repo's `/text_encoders` folder
   (27.1 GB, "recommended for 24 GB GPUs") — **but the actual repository tree
   contains no `text_encoders` folder and no such file.** The README is stale
   or the file was removed.
2. HF search for the filename finds it only in **modified variants**:
   `linjian257/qwen3vl_32b_minimax_h3_int8_convrot_uncensored-by-linjian257`
   (explicitly *uncensored*) and
   `OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`
   (*Heretic* = abliterated), plus assorted NVFP4/GGUF/FP8 relatives.

Substituting an abliterated encoder silently would be a quality, licence-risk
and reproducibility failure all at once — and we could no longer attribute any
quality difference to INT8 vs BF16, because the conditioning model itself would
differ. **Per the ground rules: STOP on this file and ask.**

The question for the client: *which repository did your `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
actually come from, and what is its SHA256?* (Windows:
`certutil -hashfile <file> SHA256`.) If it is one of the uncensored variants,
that is a fact we need to know before it conditions anything.

## 5. Blockers and what can proceed

**Blocked until the client answers:**

1. **The three workflow JSONs** (+ sample MP4). Without them, any run is a
   reconstruction, not a reproduction — `CLIENT JSON REQUIRED`.
2. **The text encoder provenance** (§4.1).

**Can proceed now (exactly pinned, no substitution involved):**

- Download the four verified weight files (~44 GB total) into the ComfyUI
  layout the guide specifies.
- Build the isolated pinned environment: `/workspace/comfyui-h3-client`,
  ComfyUI at the 0.33.3 baseline, Extender at `6a3583d`, Easy-Use at
  `4de1ab3b…` — recording any sm_120/CUDA-13.2 deviation as
  `Client pin → actual → reason → effect`.

**Deliberately not started:** FL2VA-first testing (R2V is the priority per the
D-group question), any T2V/I2V run, hybrid, Tier 2, the 406 pack, any Music
Video rebuild (LTX A2V at 14.7x already holds that workflow), and the MiniMax
API (paused, not cancelled).

**Standing follow-up:** `P0 LTX CODE FOLLOW-UP` — the opening-frame
person-anchor composite artifact in `v2v_reference_identity`. The final
Reference V2V routing decision requires `LTX_FIXED vs H3_INT8`, and neither
half exists yet.
