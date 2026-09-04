# Client final milestone — Phase 0 audit

**Internal. 5 September 2026. Read-only: no code was changed for this document.**

Repository commit `926d2e3` on branch `dual-engine-benchmark-prep`, working
tree clean. Inputs to the audit: the repository, the deployed-state runbooks
under `docs/internal/`, and the client's `LTX2.5 Pipeline-2.zip` (three
ComfyUI graphs with sample outputs).

The client brief is reproduced in the task, so this document does not repeat
it. It answers four questions: what exists, what production actually runs
today, what the client's new graphs contain, and how each phase maps onto the
architecture without replacing it.

---

## 0. Findings that change the plan

Read these before anything else. Each one is a fact the brief did not assume.

1. **H3 is in production today, on Text to Video "Best".** The committed YAML
   says `runtime: mock` everywhere; production routing is applied by
   `deploy/vps-local.sh`, and its `text-to-video` block reads
   `runtime_by_quality: {fast: ltx, best: h3_comfy}`. Phase 1 (hide H3) and
   Phase 3 (remove Best/Standard) therefore change a live customer path, not
   a dormant one. Removing the quality toggle removes the only route to H3.

2. **The current LTX runtime is not ComfyUI.** `apps/worker/worker/adapters/ltx.py`
   shells out to the LTX repository's own `uv` environment and runs
   `ltx_pipelines.distilled` / `ic_lora` / `a2vid_two_stage` /
   `ti2vid_two_stages` as subprocesses. ComfyUI exists on the GPU node only
   for H3, pinned at `v0.33.3`. The client's new graphs are ComfyUI graphs.
   Phases 3, 4 and 6 are therefore an **engine-path addition** (LTX via
   ComfyUI) beside the existing CLI path, not a tuning change to it. The
   ComfyUI client and graph-to-API compiler written for H3
   (`worker/comfy/client.py`, `worker/comfy/graph.py`) are reusable.

3. **The character-replacement graph does not take a reference photo.** It
   takes the source video plus an **edited first frame** in which the
   replacement character has already been painted in, and the Ripple LoRA
   propagates that edit through the clip. The graph's own note says: "Edit
   and replace its first frame. Use an image editor or LLM." Phase 6's
   stated input (source video + reference image) needs a first-frame editing
   step that is not in the pack. Options are in §7.6.

4. **The three graphs share one model set that the node does not have.**
   GGUF Q8 distilled transformer (T2V, FLF), INT8 distilled transformer
   (replacement), the Gemma-4-12B INT8 text encoder, the 2.5 VAEs, the x2
   spatial upscaler, and five LoRAs. Provisioning is a Phase 2 task and each
   LoRA needs a licence check (§8).

5. **`main` is 86 commits behind the working branch.** Every client fix since
   21 Aug lives on `dual-engine-benchmark-prep`, and the GPU node deploys by
   `git pull` from it. The "commit every phase separately" rule must name
   which branch is the release line; the audit assumes the current branch
   and flags a merge to `main` as part of Phase 13.

6. **The dev machine cannot run Python.** Every interpreter on this box,
   including the worker venv, dies at import with `SRE module mismatch`
   (a stray `PYTHONHOME` pointing at `E:\ZKBioTime\Python311`). Backend and
   worker suites run on the GPU node or in Docker until this is fixed. Node,
   ffmpeg and the web toolchain work.

---

## 1. Current architecture

Three independently deployable services in one repository, with YAML as the
single source of truth for what the product offers.

```
browser ──▶ apps/web (Next.js 15)          reads workflow-definitions/*.yaml
   │              │                        at build time (catalog.server.ts)
   │  REST + SSE  ▼
   └────────▶ apps/api (FastAPI) ──▶ PostgreSQL (jobs = the queue, events, assets)
                  ▲                 ──▶ Redis     (wake-up doorbell, rate limits)
                  │ internal API    ──▶ MinIO/S3  (presigned PUT/GET only)
                  │ (service token, SSH tunnel from the GPU node)
          apps/worker (pull-based, one GPU node: RTX PRO 6000 96 GB, `ltx-6000-1`)
                  ├── adapters/ltx.py       subprocess → /workspace/ltx2-benchmark (LTX CLI)
                  ├── adapters/h3_comfy.py  HTTP → ComfyUI :8188 (H3 INT8 pack)
                  ├── adapters/music.py     HTTP → ACE-Step :8001
                  └── adapters/mock.py, harness.py
```

### 1.1 Request flow

1. Web posts `POST /api/v1/generations` with `workflow_id`, prompt,
   `parameters` (duration, aspect ratio, quality, seed, sound, prompt_mode…)
   and uploaded input asset ids.
2. `WorkflowRegistry.validate_request` checks everything against the YAML
   (durations, per-quality durations, aspect ratios, quality levels, input
   roles). Extend Video additionally resolves Director lineage from the
   source asset's own job.
3. The job is a row; the worker claims it with `FOR UPDATE SKIP LOCKED` and a
   lease. The claim carries the workflow's private `execution:` block, so the
   worker never reads YAML.
4. `worker/workflows/resolver.py` picks the adapter: `execution.runtime`,
   overridden per quality by `runtime_by_quality`, with `execution_by_quality`
   overlaying settings. An engine that does not `supports()` the workflow
   falls back to the base runtime with a warning.
5. The adapter renders, reports progress (renews the lease), uploads to a
   presigned URL, and confirms. SSE streams the durable event log to the
   browser.

### 1.2 Routing layers, in precedence order

| Layer | Where | Who edits it |
|---|---|---|
| Committed YAML `execution:` | `workflow-definitions/*.yaml` | repo (all say `runtime: mock`) |
| Production overlay | `deploy/vps-local.sh` (applied on the VPS after pull) | deploy |
| Per-quality runtime | `execution.runtime_by_quality` (worker resolver) | YAML/overlay |
| Provider override | `parameters.provider` / `execution.provider` (`providers/router.py`) | QA only; `auto` = LTX everywhere |
| Node capability | `RUNTIMES=ltx,h3_comfy,music` env on the worker; API intersects at claim | GPU node `.env` |

Note the API's claim-time intersection (`ids_for_runtimes`) reads only the
base `execution.runtime`, not `runtime_by_quality`. That is why a node
declaring only `ltx` still receives a "Best" job destined for `h3_comfy` and
the worker-side fallback exists.

---

## 2. Existing modules

### 2.1 Workflows (the product surface)

| id | duration_mode | durations | quality toggle | prompt modes | committed runtime | production runtime (vps-local.sh) |
|---|---|---|---|---|---|---|
| text-to-video | fixed | 5/10/15/20/30/60s (Best: 5–30) | **fast / best** | Standard / Director | mock | `ltx`, best → `h3_comfy` (h3_max_seconds 30, h3_steps 12), prompt_structuring_v2 |
| image-to-video | fixed | 5/10/15/20/30/60s | none | none | mock | `ltx`, prompt_structuring_v2 |
| video-to-video | source | — | **fast / best** | none | mock | `ltx` at both levels; Best adds `v2v_reference_identity` |
| extend-video | fixed | 5s…5m | none | none | mock | `ltx` |
| music-video | source | — | none | none | mock | `ltx`, `audio_conditioning` + `require_audio_conditioning`, 15 steps |
| music | minutes | 1–5m | none | none | mock | `music` (ACE-Step) |

The README under `workflow-definitions/` states the catalogue is frozen at
six; a seventh workflow (Character Replacement) is a change request by that
rule, and the client has now made it.

### 2.2 Worker (`apps/worker/worker`, ~21k lines)

| Package | Role | Size |
|---|---|---|
| `adapters/ltx.py` | The LTX CLI runtime. All five video workflows; distilled default, IC-LoRA transform (V2V), a2vid (music video), guided tier (off). Frame-count guards, identity anchor, Director integration. | 3,619 lines |
| `adapters/h3_comfy.py` + `comfy/` | H3 through ComfyUI: graph load, sanctioned edits, submit, poll, interrupt, health, VRAM eviction. | 699 + 520 |
| `adapters/music.py`, `music/` | ACE-Step provider, Cerebras lyric writing, language detection. | ~3,300 |
| `longform/` | The chain: segment planning, seam conditioning, prompt structuring, section prompts, music-video shot director, H3 prompt discipline. | ~2,100 |
| `director/` | Idea → plan → per-section prompts (Cerebras or local Gemma); vision captions of the source still. | ~2,900 |
| `media/` | ffmpeg/ffprobe, frames, masks (BiRefNet person matte), edge-map control, audio, vocals, validation. | ~1,900 |
| `providers/` | Benchmark-only abstraction over LTX and H3 (manifests, capabilities, router with `auto` = LTX, hybrid strategy). Not on the production job path. | ~2,400 |
| `jobs/runner.py` | Claim → workspace → adapter → upload; lease keeper; cancel/timeout handling. | 363 |

Tests: 748 test functions in `apps/worker/tests` (last recorded run 909
passed / 1 environmental failure / 1 skipped on 25 Aug).

### 2.3 API (`apps/api/app`)

Stateless FastAPI. Routes: `/workflows`, `/generations` (create, list, get,
cancel, SSE events), `/assets` (presigned upload/confirm), `/internal`
(worker register/heartbeat/claim/progress/complete/fail). One Alembic
migration. 116 tests; the suite must run `-p no:randomly` (order-dependent
TRUNCATEs). The public projection strips `execution:` by allowlist and
`test_no_provider_or_infrastructure_names_anywhere` fails on `ltx`, `comfy`,
`gpu`, etc. in any public field.

### 2.4 Web (`apps/web`, `packages/`)

Next.js 15 / React 19. Every tool surface is built from the API's workflow
list; there is no per-workflow page code. Quality is rendered generically
from `supported_quality_levels`; the label text "Fast/Best" comes from
`qualityLabel()`. Three readers of the YAML: the API, the Zod contract in
`packages/workflow-contracts`, and `features/workflows/catalog.server.ts`
(hand-projects fields for the landing page and shell; a new YAML field must
be added there or it vanishes client-side). `config/feature-flags.ts` has one
flag today (`previewBuild`). QA scripts: parity, e2e (Playwright), flow,
responsive, contrast.

### 2.5 Benchmarks and documentation

`benchmarks/client-pack/` holds the only durable copy of the client's H3
graphs; `benchmarks/frozen/cases.json` the 41-case golden pack; `benchmarks/
review/` the measured outputs. Fifty-plus internal documents; the ones this
milestone depends on are `gpu-worker-runbook.md` (§34 build a GPU box, §37
supervision, §39–45 per-feature deploys), `production-runbook.md`
(VPS, rollback §21), `client-final-validation-2026-08-28.md` (the last deploy
order of operations), and `ltx-2.5-licensing-review.md`.

---

## 3. H3 footprint (input to Phase 1)

Nothing is deleted in Phase 1. This is the complete list of what a feature
flag has to fence.

| Location | What | Phase 1 action |
|---|---|---|
| `worker/adapters/h3_comfy.py` | The adapter. `supports()` already gates V2V on `h3_comfy_video_to_video`. | `supports()` returns False for every workflow when `ENABLE_H3` is false; `run()` refuses with a non-retriable error naming the flag. |
| `worker/adapters/registry.py` | `"h3_comfy": H3ComfyAdapter()` | Keep registered (so a mis-routed job fails clearly, and the resolver fallback serves it on LTX). |
| `worker/core/config.py` | `h3_comfy_*` settings | Add `enable_h3: bool = False` (env `ENABLE_H3`). |
| `worker/workflows/resolver.py` | `runtime_by_quality` → fallback when unsupported | Already falls back to base runtime when `supports()` is False; add a log line that says the flag is the reason. |
| `worker/providers/router.py`, `h3.py`, `hybrid.py`, `strategy.py` | Benchmark-only; `auto` already = LTX | Unchanged; `get_provider("h3")` may additionally refuse when the flag is off so a QA override cannot reach it by accident. |
| `worker/longform/h3_prompts.py`, `providers/h3_prompt.py` | Prompt discipline | Unchanged (pure functions). |
| `worker/main.py` registration | Reports `RUNTIMES` to the API | Strip `h3_comfy` from the advertised list when the flag is off, so the API never routes a base-runtime `h3_comfy` workflow to this node. |
| API | No H3 code. `schemas/workflow.py` docstring mentions "Best runs the H3 engine". | Add `ENABLE_H3` to API settings and refuse at startup any YAML whose `runtime` / `runtime_by_quality` names `h3_comfy` while the flag is off (same pattern as `_reject_mock_output_on_a_real_runtime`). Prevents the overlay from re-enabling it silently. |
| `deploy/vps-local.sh` | `best: h3_comfy` for text-to-video, `h3_max_seconds`, `h3_steps` | Rewrite the block (Phase 3 removes the quality toggle anyway). Keep the old block in a comment for rollback. |
| Web | No H3 references (only `<h3>` tags). The "Best" level is the only user-visible trace. | Nothing H3-specific; Phase 11 removes Best/Standard. Optionally mirror `NEXT_PUBLIC_ENABLE_H3` for symmetry, but there is nothing to hide today. |
| Tests | `test_h3_comfy` (32), `test_h3_prompts` (25), `test_hybrid` (17), `test_client_test_routing` (9), `test_golden_pack` (18), plus mentions in `test_longform`, `test_providers`, `test_worker` | Keep. Tests that exercise the adapter must set `ENABLE_H3=true` via the settings fixture; add tests for the off state (supports False, run refuses, registration omits). |
| `benchmarks/client-pack/*.json`, `docs/internal/h3-*.md`, `.env` on the GPU node | Graphs, docs, ComfyUI service | Untouched. ComfyUI can stay running; with the flag off nothing submits to it. |

Behaviour with `ENABLE_H3=false` (the default): a "Best" T2V request routes
to LTX with a warning; no node advertises `h3_comfy`; the API refuses to boot
on YAML that names it; the benchmark router refuses `provider=h3`.

---

## 4. Production state, as deployed

| Item | Value | Source |
|---|---|---|
| VPS | Docker compose: api, web, postgres, redis, minio; CloudPanel/nginx | `production-runbook.md` |
| VPS YAML | Six files edited by `deploy/vps-local.sh` after each pull (commit 0a1cf05) | `deploy/vps-local.sh` |
| GPU node | RTX PRO 6000 96 GB, `ltx-6000-1`, supervisord, SSH tunnel to the VPS internal API | `gpu-worker-runbook.md` §33, §37 |
| Services on the node | worker (`RUNTIMES=ltx,h3_comfy,music`), ACE-Step :8001, ComfyUI v0.33.3 :8188 (H3 INT8) | `client-test-deployment-plan.md` §3 |
| LTX runtime | `/workspace/ltx2-benchmark` @ `400fd31`, NVFP4 prequant for distilled, unquantized + CPU offload for LoRA tiers | `ltx.py`, `research-ltx25-zolexai-audit.md` |
| Known VRAM condition | Card measured at 95.2/95.6 GB during ordinary passes while ACE-Step holds its reservation; ~1 in 3 audio-tier passes hit `CUBLAS_STATUS_INTERNAL_ERROR` | memory note, `research-2026-08-21…` §7b |
| Last validation | 28 Aug deploy plan; GPU checks V1–M2 listed, not all recorded as run | `client-final-validation-2026-08-28.md` §3 |

---

## 5. The client's LTX 2.5 pack

Three ComfyUI graphs (frontend format 0.4 with subgraphs). What they contain,
read from the JSON, and what their sample outputs measure.

### 5.1 Shared model set

| File in graph | Role | Present on node? |
|---|---|---|
| `LTX-2.5-Distilled-Q8_0.gguf` | Transformer, T2V and FLF (loaded by `UnetLoaderGGUF`) | No (node has the LTX repo's NVFP4/bf16 safetensors) |
| `LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | Transformer, character replacement | No |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | Text encoder (Comfy-repacked Gemma) | No (node has `gemma-4-e2b-it` for Director only) |
| `ltx-2.5-video-vae-bf16`, `ltx-2.5-audio-vae-bf16`, `taeltx2_3` | VAEs + preview VAE | Probably (LTX repo ships VAEs, but under different names/paths) |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` | Stage-2 upscaler | Likely, under the LTX repo layout |
| `LTX-2.3-OmniNFT-RL-Lora_bf16` @ 0.4 | Quality/RL LoRA (T2V, FLF) | No — community LoRA, licence unknown |
| `ltx2.3-transition` @ 0.8 | Transition/motion LoRA (T2V, FLF) | No — community LoRA, licence unknown |
| `ltx-2-19b-ic-lora-detailer` @ 0.3 | Detailer (inside the T2V/FLF subgraphs) | No — Lightricks IC-LoRA, LTX licence |
| `LTX25_Ripple_v11` @ 1.35 | First-frame-edit propagation (replacement) | No — `WepeNerd/LTX-Ripple`, licence unknown |

### 5.2 Graph 01 — Text to Video (+ Audio)

* Two-stage distilled sampling: stage 1 at half resolution with an 8-sigma
  manual schedule (`1.0 … 0.4219, 0`), `LTXVLatentUpsampler` x2, stage 2 with
  a 3-sigma refine (`0.85, 0.725, 0.4219, 0`). Sampler `euler_ancestral`,
  CFG 1 (distilled), `LTXVDualCFGGuider` 1/1. Audio latent generated in the
  same pass.
* Canvas from `ResolutionSelector` (16:9, scale 0.9, multiple of 32) →
  1920x1088 nominal; the shipped sample is **1280x704**. Frames = `fps ×
  seconds + 1` (the 8k+1 lattice this project already enforces).
* Duration slider 30 s. **Sample output: 30.04 s, 721 frames, 24 fps, stereo
  AAC 48 kHz**, single pass. This is the client's evidence that a 30 s
  single pass is fine on their card; our own 30 s ceiling (`max_segment_seconds:
  30`) matches.
* Prompt style: a long structured prompt with a scene block, per-character
  blocks, an identity-preservation sentence, and **timed beats
  ("0.0–4.0 seconds: …")**. A long negative prompt. Both are what
  `longform/prompts.py` and the Director compiler already produce in prose
  form; the timed-paragraph format is already parsed by `parse_timed_sections`.
* Custom nodes: ComfyUI-GGUF, KJNodes, rgthree, ComfyMath, mxToolkit,
  pysssss custom-scripts, Easy-Use (`cleanGpuUsed`), VideoHelperSuite, plus
  core LTXV nodes (`LTXVConditioning`, `LTXVConcatAVLatent`, `LTXVLatentUpsampler`,
  `LTXVEmptyLatentAudio`, `EmptyLTXVLatentVideo`).

### 5.3 Graph 02 — First-Last Frame to Video (+ Audio)

* Same two-stage core as 01. Adds two `LoadImage` → `ResizeImageMaskNode`
  (1920x1088, lanczos, centre) → `LTXVImgToVideoInplace` (first frame,
  strength 0.8) and `LTXVImgToVideoInplaceKJ` (last frame, index −1).
  Sampler `euler_ancestral_cfg_pp`. `LTXVPreprocess` 18 on both stills.
* **Sample output: 30.04 s, 832x1088 portrait, 721 frames, audio.** The
  input still is 603x900; the graph resizes to the canvas, so aspect is
  decided by the `ResolutionSelector`, not by the image. Our I2V today does
  the opposite (`canvas_from_source_image: true`, commit e292c75). Decide
  which wins (§10).
* Last frame is optional by disconnecting node 5437; the graph as shipped
  has both connected.

### 5.4 Graph 03 — Character Replacement

* Inputs: `VHS_LoadVideoFFmpeg` (the source clip) and one `LoadImage` (the
  **edited first frame**). Subgraph "Replace first frame" batches the image
  in front of frames 1..N of the source; `LTXAddVideoICLoRAGuideAdvanced`
  uses the source video as the IC-LoRA motion guide; Ripple LoRA @ 1.35 on
  the INT8 distilled transformer; single pass (`Set Single Pass: true`),
  `lcm` sampler, manual 8-sigma schedule (a `BasicScheduler linear_quadratic
  15` alternative is wired for "more steps"); audio passthrough from the
  source (`Use Audio from Video Input: true`). `ModelSamplingSD3` 13,
  `LTX2AttentionTunerPatch`, `LTXVChunkFeedForward` 2/4096.
* Canvas 736x1280 portrait, 8 s → 193 frames. **Sample output: 8.04 s,
  736x1280, 193 frames, source audio.** The reference video is 576x1024 at
  29.97 fps, 8.6 s; the graph resamples to 24 fps and caps frames.
* The positive prompt describes the *replacement* person in detail and asks
  the model to follow the reference video's motion; the negative prompt
  lists the *source* person's traits to suppress leakage. Both are
  per-job prompts the worker must generate (the Director vision captioner
  already produces identity descriptions from a photo).
* Graph note: "Width & height divisible by 32 + 1; frame count 8k+1;
  invalid values are silently rounded", "Length: try 5, 10 or 20".

### 5.5 What this pack is not

* No music-video graph, no extension graph, no V2V restyle graph. Phases 5
  and 7 stay on the existing CLI runtime.
* No node-version pins, no model download manifest, no measured timings.
  The frozen-stack discipline used for H3 (`h3-client-runtime-freeze.md`)
  has to be repeated for this pack in Phase 2.
* ComfyUI **v0.33.3** (the H3 pin) predates several nodes used here
  (`ComfySwitchNode`, `ComfyMathExpression`, `LTX2AttentionTunerPatch`,
  `LTXVChunkFeedForward`, `LTXVImgToVideoInplace`). Either one newer ComfyUI
  serves both packs after an H3 compatibility pass, or the LTX pack runs as a
  second ComfyUI instance on its own port and venv. The second option keeps
  H3 frozen exactly as documented and is the recommendation.

---

## 6. Runtime comparison: current LTX CLI vs the client's ComfyUI graphs

| Property | Current (`ltx_pipelines.distilled` via CLI) | Client pack (ComfyUI) |
|---|---|---|
| Weights | NVFP4 prequant transformer (LoRA tiers: bf16 + CPU offload) | GGUF Q8 (T2V/FLF), INT8 convrot (replacement) |
| Text encoder | LTX repo Gemma | Comfy-repacked Gemma-4-12B INT8 |
| Schedule | Pipeline default (steps not exposed on distilled; `inference_steps` on a2vid) | Manual sigmas: 8 + 3 (two-stage), explicit |
| LoRAs | None on the default tier | Detailer 0.3 + OmniNFT-RL 0.4 + transition 0.8 |
| Negative prompt | Not available on distilled | Wired (CFG 1, so effect is via the dual guider only) |
| First/last frame | `--image PATH IDX STRENGTH` (first frame at 0; last frame possible at index N−1, unmeasured) | Native inplace nodes, strength 0.8 |
| Audio | Same-pass audio; `soundscape` clause | Same-pass audio |
| Measured speed | 34 s per 5 s clip at 1024x576 (RTX PRO 6000) | Unmeasured on our node; sample outputs only |
| Long form | `longform/chain.py`: 30 s sections, seam frame conditioning, identity anchor, Director sections | None; single pass per graph |
| Cancellation | Subprocess kill | ComfyUI `/interrupt` (already implemented for H3) |
| Progress | Log-marker parsing | Queue state only; elapsed-time pacing (as for H3) |

The two paths can coexist in one adapter family: a new `ltx_comfy` runtime
that renders one pass through a graph, plugged into the **same**
`render_chain` for extensions, so 30 s stays a single graph submission and
longer durations become chained submissions with the seam frame fed to the
FLF graph's first-frame input. That reuses every continuity mechanism the
CLI path has and is how "unlimited extensions" is delivered honestly.

---

## 7. Replacement plan, phase by phase

Each phase is one commit series on the release branch. "Unchanged" means no
diff in that area.

### Phase 1 — H3 flag
§3 above. Worker `ENABLE_H3` (default false) fences `supports()`, `run()`,
registration and the benchmark router; API `ENABLE_H3` refuses YAML that
names `h3_comfy` while off; `vps-local.sh` T2V block loses `best: h3_comfy`.
Tests for both states. No deletions.

### Phase 2 — Deploy the current LTX system
Two deliverables: (a) a verified deployment of what exists (worker start,
ComfyUI start for the LTX pack, adapters' health, create/progress/cancel/
storage checks, in the style of `client-final-validation-2026-08-28.md` §3),
and (b) provisioning of the pack: second ComfyUI instance with pinned core
and custom nodes, the model set in §5.1 with SHA256s recorded, a health
check in the worker mirroring `h3_comfy_health`. Output:
`docs/internal/ltx-comfy-runtime-freeze.md` and a deployment doc.

### Phase 3 — Text to Video
* YAML: `supported_quality_levels: []`, `settings.quality: false`, remove
  `supported_durations_by_quality`; `supported_durations: ["5s","10s","15s","30s"]`
  (20 s and 60 s leave the ladder; 60 s becomes "generate 30 s, then
  Extend"). Keep Director mode (client did not ask to remove it).
* Worker: new `adapters/ltx_comfy.py` + `comfy/ltx_graphs.py` (graph 01 and
  02 compiled to API prompts with sanctioned edits: prompt, negative,
  seconds, canvas, seed, first/last image, output prefix). One graph
  submission per chain step through `render_chain`.
* Extensions: Extend Video keeps its ladder and now submits the FLF graph
  with the source's final frame as first frame; the identity anchor and
  Director lineage stay as they are.
* Benchmark table (resolution, wall time, VRAM peak, RAM, output length)
  for 5/10/15/30 s at 16:9 and 9:16 on the node, recorded before the routing
  flips.
* Prompt structuring: the CLI path's `structure_prompt` and Director output
  are reused unchanged; the graph's negative prompt gets a fixed house
  default plus the Director's per-job exclusions.

### Phase 4 — Image to Video
* YAML: `supported_durations: ["5s","10s","15s","30s"]`; second input
  `last_frame` (image, optional); rename to "First/Last Frame Video" in
  `name`/descriptions (id stays `image-to-video` so history, Extend
  lineage and the `identity_image` mechanism keep working).
* Worker: graph 02 through the same `ltx_comfy` adapter; last frame wired
  only when supplied (disconnect node 5437 otherwise, the same
  "disconnect unused loader" edit the H3 compiler already performs).
* Decide canvas policy (§10 Q3).
* Web: `catalog.server.ts` and the Zod contract need nothing for a second
  input role (inputs pass through as a list); Dropzone renders per input.

### Phase 5 — Video to Video
No diff under `adapters/ltx.py::_run_restyle`, `media/control.py`,
`media/masks.py`, the V2V YAML, or the V2V UI. Phase 12 adds/asserts the
regression tests (`test_video_to_video.py`, `test_reference_identity.py`,
`test_transform.py`, `test_ltx_golden.py` argv shapes) and the deploy doc
records that `runtime_by_quality` for V2V stays `ltx/ltx`. The Fast/Best
toggle on V2V is **not** in the client's removal list; it stays.

### Phase 6 — Character Replacement (new module)
* YAML: `character-replacement.yaml`, category video, `duration_mode:
  source`, inputs `source_video` (required) + `reference_image` (required),
  aspect from source, no quality, no seed. Public copy must not name the
  model. The README's "frozen at six" note is updated with the change
  request.
* API: no route code; the registry loads the seventh file. `capabilities.
  extend: false` initially.
* Worker: `adapters/character_replacement.py` (runtime `ltx_comfy` shares
  the service; a separate adapter keeps the module "completely separate" as
  asked). Steps: probe source → 24 fps resample and 8k+1 frame cap → **build
  the edited first frame** → compile graph 03 with the frame, the clip, a
  generated positive prompt (replacement identity from the reference photo
  via the existing vision captioner) and a generated negative prompt (source
  identity via the same captioner on the source's first frame) → submit →
  collect. Long sources chain in windows the graph tolerates (5/10/20 s per
  its note), each window's edited first frame being the previous output's
  last frame — the identity then propagates without re-editing.
* **What the client's own sample shows (frame inspection, 5 Sep).** The
  "first frame" fed to the graph was NOT a pixel edit of the source frame.
  It was a standalone photo of a different man in a different place (a
  yacht interior), framed as a close-up. Frame 0 of the output is that
  photo verbatim; by frame 4 the video has snapped to the source's
  composition with the new man performing the source's motion, and it
  holds for the full 8 s. Two consequences: (a) a plain reference photo
  works as the input, so the module can ship with "reference photo only"
  and a one-frame trim at the start; (b) the photo's **background replaced
  the source's** (yellow wall → yacht), so "preserve background" is not
  what this graph does with a raw photo. Preserving the background needs
  the photo composited onto the source frame first — candidate (1) below.
* First-frame editing, when the background must survive: three candidates,
  to be measured in that order. (1) The existing person-matte composite
  (`media/masks.py`, BiRefNet) placing the reference photo into frame 0 —
  cheapest, already on the node, and the 20 Aug audit found its
  bottom-alignment bug (fixed locally, uncommitted); it only works when the
  photo's pose resembles the source's first frame. (2) An open image-edit
  model driven by a mask and the reference (FLUX.1 Kontext dev is
  non-commercial; Qwen-Image-Edit is Apache-2.0 and the licence-clean
  candidate; VRAM ~20–40 GB, must share the card). (3) Client supplies the
  edited frame in the UI as an optional third input — the graph's literal
  contract, zero model risk, worst UX. Recommendation: ship (1) as the
  default with (3) as an optional override, and benchmark (2) as a
  follow-up.
* Web: nothing bespoke; the tool appears from the API list. Icon: add one
  glyph to `Icon.tsx` and the `workflowIconSchema` enum.
* Consent: `masks.py` records that replacement "requires recorded consent —
  a product gate". A checkbox and a stored acknowledgement on the job are
  part of this phase, not an afterthought.

### Phase 7 — Music Video
Stays on the CLI a2vid path (audio conditioning is the lip-sync). Research
items already open in `music-video-director-audit.md` §5: the five
before/after benchmark cases, image conditioning of the performer per
section (the "higher-value follow-up"), per-section trim of the upscaler
tail artefacts, beat-level cutting. Two new items from this audit: use the
pack's T2V prompt shape (scene block + per-character block + timed beats)
as the section-prompt template, and evaluate whether a ComfyUI a2vid graph
exists upstream that would let music video share the new engine path.
Every change ships behind an execution key with a before/after render.

### Phase 8 — Voice cloning (research only)
Candidates to evaluate: XTTS-v2 (Coqui, CPML licence — non-commercial),
F5-TTS (MIT, CC-BY-NC weights per training data — check), Fish-Speech
(CC-BY-NC-SA), OpenVoice v2 (MIT), CosyVoice 2 (Apache-2.0), Chatterbox
(MIT). Pipeline into LTX: generated speech → `a2vid_two_stage` audio
conditioning, which already lip-syncs to a supplied track. Deliverable is a
document with licence, latency, VRAM and a recommendation; no code.

### Phase 9 — Music reference (research only)
ACE-Step is already integrated and has a reference-audio/style path in its
API surface to verify; alternatives MusicGen (CC-BY-NC weights), Stable
Audio Open (community licence), YuE (Apache-2.0). Style extraction, never
melody copying: tempo, key, genre tags, instrumentation → prompt, with
the reference audio itself never conditioning the generator unless the
model documents style-only conditioning. Document with licence and GPU
cost; no code.

### Phase 10 — Speed
Only measured changes. Candidates, cheapest first: `--offload none` on
the unquantized tiers once VRAM headroom exists (already measured 23–30 %
faster); 15 steps on a2vid (already in production); the pack's Q8/INT8 vs
our NVFP4 on the same clip; sigma-schedule length on the new graphs;
attention backend on the new ComfyUI (`ModelAttentionBackend` node,
sage/flash); moving ACE-Step off the card during video jobs (removes the
1-in-3 CUBLAS failure and unlocks offload none). Each with before/after
runtime and a quality note against the same seed.

### Phase 11 — Frontend
Quality toggle removal is a YAML change (the panel is generic). Rename
copy for Image to Video. New tool card appears from the API. Duration
chips come from YAML. Verify `catalog.server.ts` parity for the new
input role and the new workflow id, and run `qa:parity`, `qa:e2e`.

### Phase 12 — Testing
Backend and worker suites (on the node or in Docker, §0.6), golden argv
suite for the CLI path, a new golden set for compiled ComfyUI API prompts
(same pattern as `test_h3_comfy`), web build + lint + typecheck, Playwright
e2e for T2V/I2V(FLF)/V2V/Character Replacement/Extend, infrastructure
tests (queue claim, cancel mid-render, failure retry, upload confirm).

### Phase 13 — Client package
`docs/client-readiness-report.md` with the benchmark tables from Phases 3,
4, 6 and 10, the deployment doc from Phase 2, env-var list, rollback, and
the merge of the release branch to `main`.

---

## 8. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | LoRA licences unverified (OmniNFT-RL, transition, Ripple). LTX Attachment A #20 already flags production use as needing written clarification. | High that at least one is non-commercial | Blocks shipping a graph | Check each repo's licence before download; ship graphs with the LoRA disabled if unclear and measure the quality delta. |
| R2 | ComfyUI version split: LTX pack needs newer core than the H3 pin. | Certain | Breaks H3 if upgraded in place | Second instance, own venv/port; H3 stays frozen. |
| R3 | VRAM: a second ComfyUI holding ~40 GB of LTX weights beside ACE-Step and (idle) H3 on a 96 GB card. | High | OOM, the 1-in-3 CUBLAS failure gets worse | Lazy eviction already exists (`evict_comfy_vram`); extend it to both instances; consider stopping ACE-Step during video jobs. |
| R4 | Character replacement's first-frame edit is unsolved by the pack. | Certain | Phase 6 quality depends on it | §7.6 options; ship the matte composite first, measure. |
| R5 | Removing 20 s and 60 s from T2V/I2V changes a customer-visible ladder the client approved on 27 Aug. | Medium | Complaint | Confirm the ladder (§10 Q1) before the YAML lands. |
| R6 | Dropping the Best level silently demotes any customer flow that used H3 for its identity strength on I2V (28 Aug: I2V is already on LTX, so only T2V Best is affected). | Low | Quality regression on T2V | Benchmark the new graph against a Best render of the same prompt before flipping. |
| R7 | Canvas policy conflict (§5.3): the pack resizes the still to the selected aspect; the current I2V lets the picture decide, which was a deliberate 28 Aug fix. | Medium | Stretched or cropped stills | Keep "picture decides"; feed the graph the source's own grid. |
| R8 | Frame lattice: graph silently rounds invalid sizes/counts. | Medium | Off-by-a-few-frames durations, audio drift at seams | Compute width/height (÷32) and frames (8k+1) worker-side before submission, as `conforming_frames` does today; assert on the output probe. |
| R9 | Two YAML overlays (`vps-local.sh` + committed) already caused the 64-video PNG incident. Adding a seventh workflow and removing keys widens the surface. | Medium | Wrong content type or runtime in production | Extend `vps-local.sh --check` to assert the full expected runtime block per workflow, including absence of `h3_comfy`. |
| R10 | `main` is 86 commits behind; two branches carry uncommitted local fixes noted in memory (V2V anchor, 60 s continuity). | Certain | Lost work, unclear release line | Decide the release branch now (§10 Q5); commit or discard local edits before Phase 1. |
| R11 | Dev machine cannot run Python. | Certain | Backend tests only runnable remotely | Fix `PYTHONHOME` or use Docker for the suites; not a blocker for the audit. |
| R12 | Progress fidelity through ComfyUI is coarse (queue state only). | Certain | Customers see a slow bar | Same elapsed-time pacing used for H3; measure per-second rates in Phase 3 and record them. |

---

## 9. Rollback strategy

Every phase must be reversible by configuration, then by git.

* **Routing rollback (minutes).** All engine selection is in the execution
  block that `deploy/vps-local.sh` writes. Keep the pre-change block for
  each workflow in the script as a commented `block_*_previous`; rollback is
  swapping the function body, `bash deploy/vps-local.sh`, rebuild api+web
  (the API image bakes the YAML — `gpu-worker-runbook.md` §14). The worker
  needs no restart for a routing change.
* **H3 rollback.** `ENABLE_H3=true` on the worker and API plus the previous
  T2V block restores the 28 Aug behaviour exactly. Nothing is deleted, so
  this stays possible indefinitely.
* **Engine-path rollback.** The `ltx_comfy` adapter is a new runtime name.
  Setting a workflow's `runtime` back to `ltx` returns it to the CLI path
  with no code change; the CLI adapter and its golden tests are not modified
  by Phases 3–4.
* **Schema rollback.** No database migration is planned: a new workflow is a
  YAML file, a new input role is a JSON field, consent is a parameter.
  Removing `character-replacement.yaml` makes existing jobs of that type
  unloadable in history; keep the file and set it hidden rather than delete
  it if the module is withdrawn (a `ui.hidden` field would be a small
  contract addition; note it in Phase 6).
* **Git.** Each phase is one commit series with a tag (`phase-1-h3-flag` …).
  `git revert` of a phase range is clean because phases do not share files
  except YAML and `vps-local.sh`, which are re-applied by the script.
* **Node.** The second ComfyUI instance is a separate supervisord program;
  stopping it leaves H3, ACE-Step and the CLI runtime untouched.
* **Production runbook §21** remains the VPS-level procedure (image
  rollback + `up -d`).

---

## 10. Decisions needed from the client before the phases start

1. **Duration ladder.** Brief says 5/10/15/30 for T2V and I2V. Today both
   offer 20 s and 60 s. Confirm both are dropped (60 s becomes Extend).
2. **Extend Video** keeps its 5 s…5 m ladder and stays a separate tool, or
   becomes the "Extensions" button only. Assumed: unchanged tool, and the
   new engine behind it.
3. **Canvas for First/Last Frame:** picture decides the aspect (current) or
   the selected aspect ratio decides and the still is resized (the pack).
4. **Character Replacement input:** the pack's sample proves a plain
   reference photo works (§7.6). Recommendation: reference photo required,
   edited first frame as an optional advanced input. Open sub-question:
   must the source background survive? The sample replaces it.
5. **Release branch:** `dual-engine-benchmark-prep` is what production
   pulls; confirm it, or merge to `main` first.
6. **LoRA policy:** ship community LoRAs only with a verified licence, or
   ship the graphs without them if a licence cannot be confirmed.
7. **Director mode** stays on Text to Video (assumed yes).
8. **Video to Video Fast/Best** stays (assumed yes; only T2V/I2V lose it).

---

## 11. Phase 0 checklist

- [x] Repository structure, providers, adapters, workers, API routes,
      frontend modules, ComfyUI integration, current LTX implementation read.
- [x] Client pack unpacked and every graph parsed; sample outputs probed.
- [x] Production routing reconstructed from `deploy/vps-local.sh`.
- [x] H3 footprint enumerated for Phase 1.
- [x] Replacement plan, risks, rollback written.
- [ ] Client answers to §10.
- [ ] Phase 1 begins only after this document is committed.
