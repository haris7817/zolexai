# LTX 2.5 client workflows — Phase 0 audit and migration plan

**Internal. 5 September 2026. GPU unavailable — nothing here was executed on a
model.** Companion to `client-final-audit.md` (the repository audit of the same
day); this document is the ZIP-facing half and the design the later phases
implement.

Source of truth: `LTX2.5 Pipeline-2.zip`
(sha256 `fb1e7772921dcf093165aa1e57614c42bb1ffb9af03591dd1e095535a7a970a1`).
The three graphs inside it are frozen, byte-identical, under
`benchmarks/client-pack/ltx25/`. They are used as shipped: same models, same
LoRAs, same samplers, same sigma schedules, same canvas logic, same
conditioning. The worker changes only prompt, seed, duration, input media and
output location — the sanctioned-edit discipline the H3 compiler already
follows.

---

## 1. Current architecture, in one screen

Three services, one repository. The public catalogue is `workflow-definitions/
*.yaml`; the API validates it and serves the public projection; the web app
builds every tool surface from that response; the worker receives each
workflow's private `execution:` block with the claim and picks an adapter from
`execution.runtime`.

```
web ──▶ api ──▶ postgres (queue) ◀── worker ──▶ adapters/{mock, harness, ltx (CLI), music, h3_comfy}
                                                       │
                                     ComfyUI :8188 (H3, frozen v0.33.3)   ACE-Step :8001   LTX CLI env
```

Facts that shape the plan (details in `client-final-audit.md` §0):

* The LTX runtime in production is the **CLI** (`ltx_pipelines.*` subprocesses),
  not ComfyUI. The ZIP graphs are ComfyUI graphs, so this milestone adds a
  second LTX execution path rather than tuning the first.
* ComfyUI exists on the node only for H3, pinned at **v0.33.3**. The ZIP graphs
  use nodes stamped `comfy-core 0.34.0` (`ComfySwitchNode`,
  `ComfyMathExpression`, `ModelAttentionBackend`, `PrimitiveBoolean`) and
  KJNodes/LTXVideo commits newer than the H3 freeze. They need their **own
  ComfyUI instance** (own venv, own port); H3's stays frozen and unused.
* H3 is live today on Text to Video "Best" through `deploy/vps-local.sh`.
  Removing Best removes the route; `ENABLE_H3=false` makes the removal
  structural.
* `worker/comfy/client.py` (submit, poll, cancel, interrupt, free) and the
  `widgets_values_named`-driven UI→API conversion in `worker/comfy/graph.py`
  are reusable. The ZIP graphs add three things that compiler never met:
  subgraphs, KJNodes Set/Get virtual links, and HTTP upload/download.

---

## 2. The ZIP, decoded

### 2.1 Files

| File | sha256 | Probe |
|---|---|---|
| `01 - Text to Video (+ Audio).json` → `ltx25_text_to_video.json` | `2dcd9661…b914c` | 39 root nodes, 1 subgraph (30 nodes), 31 root links |
| `01_Text_to_Video.mp4` (sample output, not committed, 37 MB) | `70b9d622…76cca` | 30.04 s, 721 frames, 1280x704, 24 fps, AAC 48 kHz stereo |
| `02- First-Last Frame to Video (+ Audio).json` → `ltx25_first_last_frame.json` | `1926bd6d…4da967` | 51 root nodes, 1 subgraph (35 nodes), 41 root links |
| `output1.mp4` (sample output, not committed, 44 MB) | `b8c10d86…3330d9` | 30.04 s, 721 frames, 832x1088, 24 fps, audio |
| `Pasted image.png` → `samples/first_last_frame_input.png` | `975320c1…24ee3f0b7` | 603x900 still |
| `LTX2.5 Character-Replacement.json` → `ltx25_character_replacement.json` | `2ea75472…88adc9` | 119 root nodes, 6 subgraph definitions (one nested), 104 root links |
| `Reference_video.mp4` → `samples/character_replacement_source.mp4` | `d8944670…1abfaf` | 8.61 s, 258 frames, 576x1024, 29.97 fps, audio |
| `character_replacement-output` (sample output, not committed, 5.8 MB) | `4927a738…d6a7` | 8.04 s, 193 frames, 736x1280, 24 fps, source audio 44.1 kHz |

The three large sample outputs stay in the ZIP; their hashes and probes above
are the reference for the GPU-day comparison (§6).

### 2.2 Serialization facts the compiler relies on

* Frontend format `0.4`, `frontendVersion 1.49.6`, exported from a ComfyUI
  fork (`comfy_fork_version feature/av_inference`). Every node with widgets
  carries **`widgets_values_named`** (name → value) — the same field the H3
  graphs carry and the H3 compiler reads. Zero nodes lack it.
* **Subgraphs** live in `definitions.subgraphs[]`; an instance node's `type` is
  the subgraph UUID. Boundary links run to/from virtual nodes `-10` (input)
  and `-20` (output); `inputs[k].linkIds` / `outputs[k].linkIds` map boundary
  slots. Instance nodes can carry widget values for subgraph inputs
  (`width`, `height`, `value`, `sampler_name`), all of which are also linked
  in these files. The character graph nests one subgraph inside another.
* **KJNodes `SetNode` / `GetNode`** are frontend-only virtual links keyed by a
  `Constant` name. Root `SetNode`s are read by `GetNode`s inside subgraphs
  (the character graph has 25 Set / 28 Get pairs). They must be resolved
  globally before flattening.
* No `Reroute`, no muted or bypassed nodes, no "Use Everywhere" broadcast
  nodes. `MarkdownNote` nodes are documentation only.
* Dotted input names (`values.a`, `variables.b`, `num_images.image_1`,
  `images.image0`) are ComfyUI dynamic inputs and are used verbatim as API
  keys.
* Unlinked optional inputs (`VHS_VideoCombine.vae`, `.meta_batch`,
  `MathExpression.c`, `ImageResizeKJv2.mask`, …) are simply omitted.

### 2.3 Graph 01 — Text to Video (+ Audio)

| | |
|---|---|
| Transformer | `UnetLoaderGGUF` → `LTX-2.5-Distilled-Q8_0.gguf` |
| Text encoder | `CLIPLoader` → `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`, type `ltxv` |
| VAEs | `VAELoaderKJ` ×3: video `ltx-2.5-video-vae-bf16`, audio `ltx-2.5-audio-vae-bf16`, preview `taeltx2_3` (bf16, main device) |
| Upscaler | `LatentUpscaleModelLoader` → `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` |
| LoRAs | `Power Lora Loader (rgthree)`: `LTX-2.3-OmniNFT-RL-Lora_bf16` @ 0.4, `ltx2.3-transition` @ 0.8; inside the subgraph `LoraLoaderModelOnly` → `ltx-2-19b-ic-lora-detailer` @ 0.3 |
| Canvas | `ResolutionSelector` (core): `16:9 (Widescreen)`, 0.9 MP, multiple 32 → width/height; stage 1 runs at 0.5× (`ImageScaleBy`), stage 2 at full |
| Length | `mxSlider` "Clip Length ( in seconds )" = 30; frames = `fps*seconds+1` (`MathExpression\|pysssss` "a*b+1"), fps `INTConstant` 24 → 721 frames; audio latent from the same count |
| Sampling | `SamplerCustomAdvanced` ×2: stage 1 `ManualSigmas` `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`; `LTXVLatentUpsampler`; stage 2 `ManualSigmas` `0.85, 0.7250, 0.4219, 0.0`; `KSamplerSelect` `euler_ancestral`; `LTXVDualCFGGuider` 1/1 (distilled, CFG 1); `RandomNoise` randomize |
| Decode | `VAEDecodeTiled` 512/64/512/4 + `LTXVAudioVAEDecode` |
| Output | `VHS_VideoCombine` "Final-Output": h264-mp4, yuv420p, crf 10, 24 fps, prefix `LTX2.5/01_Text_to_Video` |
| Memory | `easy cleanGpuUsed` ×6 between stages |

Runtime inputs the worker sets: positive `text`, negative `text`, slider
`Xi`/`Xf`, `aspect_ratio` label, `noise_seed`, `filename_prefix`.

### 2.4 Graph 02 — First-Last Frame to Video (+ Audio)

Same loaders, LoRAs, upscaler, two-stage schedule and output node as graph 01.
Differences:

| | |
|---|---|
| Inputs | `LoadImage` "Load Image1" / "Load Image2" → `ResizeImageMaskNode` (scale to the selector's width/height, lanczos, center) → subgraph `image` / `images` |
| First frame | `LTXVPreprocess` 18 → `LTXVImgToVideoInplace` strength 0.8 (on the stage-2 latent) |
| First + last | `LTXVImgToVideoInplaceKJ`: two images, strengths 1/1, indices 0 / −1 (on the stage-1 latent) |
| Sampler | `euler_ancestral_cfg_pp` |
| Length | same slider mechanism (30 s → 721 frames) |

The still is **resized to the selected canvas** (the pack's behaviour, kept —
client decision "same as in zip"). Last frame optional = the inner
`LTXVImgToVideoInplaceKJ` node bypassed exactly as a ComfyUI user would bypass
it (latent passes straight through); the stage-2 first-frame conditioning
remains.

### 2.5 Graph 03 — Character Replacement

| | |
|---|---|
| Transformer | `UNETLoader` → `LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| LoRA | `LoraLoaderModelOnly` → `LTX/LTX-2.5/LTX25_Ripple_v11.safetensors` @ **1.35** (rgthree loader present but empty) |
| Patches | `ModelSamplingSD3` shift 13, `LTX2AttentionTunerPatch`, `LTXVChunkFeedForward` 2/4096, `ModelAttentionBackend` "comfy kitchen attention" |
| Inputs | `VHS_LoadVideoFFmpeg` (source, `force_rate` = fps 24, `frame_load_cap` = planned frames, format LTXV) and `LoadImage` (the reference/first frame) |
| Motion guide | source frames → `LTXAddVideoICLoRAGuideAdvanced` (frame_idx 0, strength 1) |
| First-frame edit | subgraph "Replace first frame": `BatchImagesNode`(reference image, source frames) → `Frames Slice` 0..frames |
| Canvas | `INTConstant` "Set Width" 736 / "Set Height" 1280; both `ImageResizeKJv2` lanczos, crop, center |
| Length | `INTConstant` "Set Length (seconds)" 8; frames = `((round((a*b-1)/8))*8)+1` (`SimpleCalculatorKJ`) → 193; latent length = min(frames, loaded frames) ("Max Frames" subgraph) |
| Sampling | single pass (`PrimitiveBoolean` true): `KSamplerSelect` `lcm`, `ManualSigmas` 8-step schedule, `CFGGuider` 1, `LTXVDualCFGGuider` 1/1; a `BasicScheduler linear_quadratic 15` alternative and a 2-pass branch are wired but switched off |
| Audio | `PrimitiveBoolean` "Use Audio from Video Input" true → source audio passes through (`easy ifElse`) |
| Output | `VHS_VideoCombine`: h264-mp4, crf 12, 24 fps, prefix `video/LTX-2.5/inpaint_Edit` |

**What the sample proves** (`client-final-audit.md` §7.6): the loaded image
was a plain photo of a different person in a different place; frame 0 of the
output is that photo, frame 4 onward is the source's motion performed by the
new person in the photo's environment. Client decision 5 Sep: that is the
product. Runtime inputs the worker sets: `video`, `image`, `Set Length`,
`Set Width`/`Set Height` (orientation from the source), positive/negative
`text`, `noise_seed` ×2, `filename_prefix`.

### 2.6 Node dependencies (from `properties.cnr_id` / `ver` in the graphs)

| Pack | Pin seen in the graphs | Nodes used |
|---|---|---|
| ComfyUI core | ≥ **0.34.0** (newest stamp) | CLIPLoader, CLIPTextEncode, EmptyImage, EmptyLTXVLatentVideo, GetImageSize, ImageScaleBy, KSamplerSelect, LTXV* (Conditioning, ConcatAVLatent, SeparateAVLatent, LatentUpsampler, EmptyLatentAudio, AudioVAEDecode/Encode, ImgToVideoInplace, Preprocess, CropGuides, DualCFGGuider), LatentUpscaleModelLoader, LoadImage, LoraLoaderModelOnly, ManualSigmas, RandomNoise, ResizeImageMaskNode, ResolutionSelector, SamplerCustomAdvanced, VAEDecode, VAEDecodeAudio, VAEDecodeTiled, VAELoader, UNETLoader, BasicScheduler, CFGGuider, BatchImagesNode, ComfySwitchNode, ComfyMathExpression, ModelAttentionBackend, ModelSamplingSD3, PrimitiveBoolean, PrimitiveFloat, SetLatentNoiseMask, SolidMask |
| ComfyUI-GGUF (city96) | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | UnetLoaderGGUF |
| ComfyUI-KJNodes (kijai) | newest of `e8e88f7c…`, `5b38397a…`, `71578cf4…`, `67643150…`, `c6ce76d0…`, `37659859…`, `5219cd17…` (1.4.0 era) | INTConstant, VAELoaderKJ, ImageResizeKJv2, ModelPreviewOverrideKJ, SimpleCalculatorKJ, LTXVImgToVideoInplaceKJ, LTX2AttentionTunerPatch, LTX2SamplingPreviewOverride, LTXVChunkFeedForward; SetNode/GetNode (frontend-only) |
| rgthree-comfy | `6b76ee6f…` / `35c9f1e1…` | Power Lora Loader (rgthree) |
| ComfyUI-LTXVideo (Lightricks) | `229437c6b65796d6a7a63ae34be2bd5ba31fa543` | LTXAddVideoICLoRAGuideAdvanced |
| ComfyUI-VideoHelperSuite | `115de7a9…` / `3234937f…` (1.7.9) | VHS_VideoCombine, VHS_LoadVideoFFmpeg, VHS_VideoInfo |
| ComfyUI-Easy-Use | 1.3.4 / `005c5783…` | easy cleanGpuUsed, easy ifElse |
| ComfyMath (evanspearman) | `c01177221c31b8e5fbc062778fc8254aeb541638` | CM_IntToFloat, CM_FloatToInt |
| ComfyUI-mxToolkit (Smirnov75) | 0.9.92 | mxSlider (backend node: `Xi`, `Xf`, `isfloatX`) |
| ComfyUI-Custom-Scripts (pythongosssss) | 1.2.5 | MathExpression\|pysssss |
| RES4LYF (`drozbay/RES4LYF` fork stamped) | `26036f647ca15d3048a193daf99a40cecfc3820d` | Frames Slice |

The H3 stack pins ComfyUI v0.33.3, older than the 0.34.0 core nodes above:
the LTX pack **cannot** share that instance without upgrading it, and upgrading
it is a compatibility pass on H3 nobody has run. Second instance.

### 2.7 Model manifest

| File (as named in the graphs) | ComfyUI folder | Source | Licence | Status |
|---|---|---|---|---|
| `LTX-2.5-Distilled-Q8_0.gguf` (23.6 GB) | `diffusion_models/` (or `unet/`) | `Abiray/LTX-2.5-Distilled-GGUF` (quant of Lightricks/LTX-2.5) | LTX-2.x Community | **not on node** — community quant, record sha on download |
| `LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` (21.5 GB) | `diffusion_models/LTXVideo/v2/` | `Lightricks/LTX-2.5` (gated) | LTX-2.x Community | not on node |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `text_encoders/` | `Lightricks/LTX-2.5` (gated) | LTX-2.x Community | not on node |
| `ltx-2.5-video-vae-bf16.safetensors`, `ltx-2.5-audio-vae-bf16.safetensors` | `vae/` | `Lightricks/LTX-2.5` | LTX-2.x Community | LTX CLI env has equivalents under other names; copy/symlink |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `latent_upscale_models/` | `Lightricks/LTX-2.5` | LTX-2.x Community | likely present in the CLI env |
| `taeltx2_3.safetensors` | `vae_approx/` (VAELoaderKJ reads `vae/`) | `madebyollin/taehv` | MIT | preview only; graphs load it with `VAELoaderKJ`, so it must exist |
| `LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` | `loras/` | `Kijai/LTX2.3_comfy` (repo licence: LTX-2 community) | as repo | not on node |
| `ltx2.3-transition.safetensors` | `loras/` | `joyfox/LTX-2.3-Transition-LORA` | **Apache-2.0** | not on node |
| `ltx-2-19b-ic-lora-detailer.safetensors` (2.62 GB) | `loras/` | `Lightricks/LTX-2-19b-IC-LoRA-Detailer` | LTX-2 community | not on node |
| `LTX/LTX-2.5/LTX25_Ripple_v11.safetensors` | `loras/LTX/LTX-2.5/` | `WepeNerd/LTX-Ripple` | LTX-2.x Community (stated on the card) | not on node |

Licence position: every weight is under the LTX-2.x Community Licence or
Apache-2.0; nothing non-commercial. The Attachment A #20 question on LTX
itself (`ltx-2.5-licensing-review.md`) is unchanged by this pack. The OmniNFT
LoRA's own licence is inherited from the hosting repo and should be
re-checked on download.

---

## 3. Migration plan

### 3.1 Two new runtimes, one shared service layer

```
worker/comfy/ltx_graphs.py        flatten (subgraphs, Set/Get) + compile + sanctioned edits + object_info check
worker/providers/ltx_comfy.py     LtxComfyService: health · upload · generate · progress · cancel · collect
worker/adapters/ltx_comfy.py      runtime "ltx_comfy": text-to-video, image-to-video (first/last), extend-video
worker/adapters/character_replacement.py  runtime "character_replacement": character-replacement only
worker/longform/continuation.py   the reusable extension engine (Phase 4)
```

* Both adapters talk to one ComfyUI instance (`LTX_COMFY_BASE_URL`, default
  `http://127.0.0.1:8189`), over HTTP only: inputs go up through
  `/upload/image`, outputs come back through `/view`. No shared filesystem
  assumption, unlike the H3 path.
* Graph → API conversion: resolve Set/Get globally, inline subgraphs
  recursively with namespaced ids, translate links, emit
  `widgets_values_named` as inputs, drop note/virtual nodes, then apply the
  per-graph edit set. Every class type in the compiled prompt is checked
  against the live `/object_info` by the health check, and the
  `ResolutionSelector` label is resolved against the live option list.
* The CLI adapter (`adapters/ltx.py`) is untouched. `runtime: ltx` remains
  the rollback for every workflow the new runtime takes over.

### 3.2 Per-workflow mapping

| Workflow | Runtime | Graph | Duration | Notes |
|---|---|---|---|---|
| text-to-video | `ltx_comfy` | 01 | 5/10/15/30 s, one pass each | quality toggle and Director removed; aspect 16:9 / 9:16 / 1:1 (the selector has no 4:5) |
| image-to-video ("First/Last Frame Video") | `ltx_comfy` | 02 | 5/10/15/30 s | `source_image` required (first frame), `last_frame` optional; with a last frame the pass is single by construction |
| extend-video | `ltx_comfy` | 02 (first frame only) | +5/10/15/30 s | source's final frame → first frame; source + continuation concatenated (Phase 4) |
| character-replacement (new) | `character_replacement` | 03 | source length, capped by `max_seconds` (default 20, the graph note's own ceiling) | 736x1280 budget oriented to the source; source audio passes through |
| video-to-video | `ltx` | — | — | **unchanged**, regression tests only |
| music-video, music | `ltx` / `music` | — | — | unchanged |

### 3.3 H3 fencing (`ENABLE_H3`, default false)

Worker: `supports()` false, `run()` refuses, node registration omits
`h3_comfy`, the benchmark router refuses `provider=h3`. API: the registry
refuses to boot on any YAML whose `runtime`/`runtime_by_quality` names
`h3_comfy` while the flag is off. Deploy overlay: no H3 lines. Nothing
deleted; `ENABLE_H3=true` restores the 28 Aug behaviour exactly.

### 3.4 Catalogue changes (YAML, hence UI)

* `text-to-video`: `supported_durations` 5/10/15/30; no quality levels; no
  prompt modes; aspect 16:9, 9:16, 1:1.
* `image-to-video`: name "First/Last Frame Video"; inputs `source_image`
  (FIRST FRAME) + `last_frame` (optional); durations 5/10/15/30.
* `extend-video`: durations 5/10/15/30.
* `character-replacement` (new): `duration_mode: source`, inputs
  `source_video` + `reference_image`, prompt optional, no seed/quality; a
  new `hidden` catalogue flag lets production carry the file without
  offering the tool until the GPU validation clears it.
* `video-to-video`, `music`, `music-video`: byte-identical.

### 3.5 Deployment shape

* A second supervisord program `ltx-comfyui` (own venv, port 8189, the pins
  in §2.6, the weights in §2.7) documented in a new runtime-freeze file.
* `deploy/vps-local.sh` gains profiles: `production` (today's LTX CLI
  routing, H3 lines removed, new tool hidden) and `client-test` (the §3.2
  routing). Nothing is applied automatically.
* `.env` additions on the node: `ENABLE_H3=false`, `LTX_COMFY_BASE_URL`,
  `LTX_COMFY_MODELS_DIR`, `RUNTIMES=ltx,ltx_comfy,character_replacement,music`.

---

## 4. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Widget→input names come from `widgets_values_named`; a node pack revision that renames an input breaks submission. | Health check validates every compiled prompt against `/object_info` before any job; `scripts/ltx_comfy_health.py` prints the diff on GPU day. |
| R2 | `ResolutionSelector` labels or `mxSlider` semantics differ on the installed version. | Labels verified against docs.comfy.org (§2.3); resolved against live options at runtime; `mxSlider` confirmed a backend node (`Xi`/`Xf`/`isfloatX`). |
| R3 | Subgraph flattening bugs (wrong slot mapping) produce a prompt ComfyUI rejects. | Offline invariants: every link resolves, no UUID/Set/Get types remain, link count and node count match the graph's own totals; fixture tests on all three graphs. |
| R4 | VRAM: LTX ComfyUI (~45 GB warm) beside ACE-Step and idle H3 on a 96 GB card. | Lazy eviction extended to both ComfyUI instances; H3 instance stopped when the flag is off; ACE-Step eviction remains the Phase 5 lever. |
| R5 | Progress is coarse (queue state only). | Elapsed-time pacing as for H3, with an expected-rate setting that is marked unmeasured until GPU day. |
| R6 | The "optional last frame" bypass is a structural edit the pack did not ship. | It is the ComfyUI bypass semantics applied to one node, documented, and covered by a fixture test; a customer who supplies both frames runs the graph unmodified. |
| R7 | Character replacement on a landscape source: the pack pins a portrait canvas. | Same pixel budget, orientation follows the source; the pack's own note lists both orientations as user settings. |
| R8 | LoRA availability/licence on download. | Manifest §2.7 with sources; transition LoRA is Apache-2.0, the rest LTX community; record sha256 at provisioning. |
| R9 | Production routing drift (the 64-video PNG incident class). | Overlay script with `--check`; the API refuses mock-output lines on real runtimes and `h3_comfy` when disabled. |
| R10 | Everything below the API is unmeasured on a GPU. | Each phase carries STATUS: WAITING FOR GPU VALIDATION; readiness report ends NO until §6 is executed. |

---

## 5. Dependencies

* GPU node with ≥ 48 GB free VRAM for the LTX ComfyUI instance; ~75 GB disk
  for the weights in §2.7.
* ComfyUI ≥ 0.34.0 plus the nine custom-node packs in §2.6, pinned at the
  commits the graphs stamp (or newer, after a compatibility pass).
* Hugging Face access to the gated `Lightricks/LTX-2.5` repo (already held
  for the CLI runtime).
* ffmpeg on the node (present).

---

## 6. GPU validation checklist (to execute when the GPU returns)

1. Load required models — `scripts/ltx_comfy_health.py --deep` (existence,
   size, sha256 against the manifest).
2. Load required ComfyUI nodes — same script: every class type of all three
   compiled prompts present in `/object_info`; `ResolutionSelector` options
   include the three product ratios.
3. Execute Text-to-Video — `scripts/ltx_comfy_bench.py t2v --seconds 5 10 15 30
   --aspect 16:9 9:16 1:1`.
4. Execute First/Last Frame — `… flf --first samples/first_last_frame_input.png
   [--last …] --seconds 5 10 15 30`.
5. Execute Character Replacement —
   `… cr --video samples/character_replacement_source.mp4 --image <photo>`.
6. Test the extension system — a 30 s T2V result extended by 5/10/15/30 s;
   inspect the seam frame and the audio join.
7. Measure runtime, VRAM peak, RAM peak, fps, output duration — the bench
   script samples `nvidia-smi` and `psutil` and writes
   `benchmarks/results/ltx25/<date>.json` + markdown.
8. Compare against the ZIP samples — duration/resolution/fps/audio layout
   against §2.1, and a side-by-side viewing of the character-replacement
   sample.

STATUS: WAITING FOR GPU VALIDATION.
