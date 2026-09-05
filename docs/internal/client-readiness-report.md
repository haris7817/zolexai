# Client readiness report — LTX 2.5 client workflows

**5 September 2026. Integration complete; GPU validation PASSED the same
evening on an RTX PRO 6000 Blackwell** (`ltx25-gpu-model-validation.md`,
`ltx25-gpu-benchmark.md`). What remains before a customer can test is
deployment, not validation: the client-test environment (the VPS API/web
rebuilt with the `client-test` routing profile and this node's worker
registered against it) has not been applied yet.

```text
LTX 2.5 GPU VALIDATION: PASS
CLIENT TEST READY: NO — pending the client-test deployment (routing profile + worker registration)
```

---

## 1. Implemented modules

| Module | Status | Where |
|---|---|---|
| Text to Video on the client's LTX 2.5 graph (5/10/15/30 s; 16:9, 9:16, 1:1; no Best/Standard/Director) | INTEGRATED, tested against a fake service | `workflow-definitions/text-to-video.yaml`, `apps/worker/worker/adapters/ltx_comfy.py` |
| First/Last Frame Video (replaces Image to Video; first frame required, last frame optional; same ladder) | INTEGRATED, tested | `workflow-definitions/image-to-video.yaml`, same adapter |
| Character Replacement (new module, own runtime and adapter; photo = character + setting, video = motion/camera/timing) | INTEGRATED, tested; ships `hidden: true` | `workflow-definitions/character-replacement.yaml`, `apps/worker/worker/adapters/character_replacement.py` |
| Extension engine (chained continuation: overlap frame dropped, audio edge-faded, source stitched in front, metadata sidecar) and Extend Video on it (5/10/15/30 s per step) | INTEGRATED, tested with real ffmpeg work | `apps/worker/worker/longform/continuation.py`, `workflow-definitions/extend-video.yaml` |
| Graph compiler (subgraph flattening, Set/Get resolution, sanctioned per-job edits, live `/object_info` verification) | INTEGRATED, tested on all three graphs | `apps/worker/worker/comfy/ltx_graphs.py` |
| ComfyUI service layer: health · generate · progress · cancel · collect, HTTP only | INTEGRATED, tested | `apps/worker/worker/providers/ltx_comfy.py` |
| H3 hidden: `ENABLE_H3=false` on API and worker; adapter declines, node never advertises, router refuses, API refuses YAML that routes to it; nothing deleted | DONE, tested | `apps/worker/worker/core/config.py`, `apps/api/app/core/config.py`, `apps/api/app/services/workflow_registry.py` |
| Video to Video, Music Video, Music | UNTOUCHED — pinned by sha256 against the starting commit, guarded by tests | `apps/api/tests/test_untouched_workflows.py`, `apps/worker/tests/test_untouched_runtimes.py` |
| `hidden` catalogue flag (loaded, readable by id, not listed, refused at creation) | DONE, tested | API schema/registry, web catalogue reader, parity check |
| Frontend | DONE: catalogue-driven; icon, contract and preview mapping for the new tool; landing chip renamed; build, lint, typecheck, parity green | `packages/workflow-contracts`, `apps/web` |
| Deployment preparation | DONE: profiles `production` / `client-test`, supervisord program for the second ComfyUI, runtime freeze doc, GPU validation checklist, runbook §46 | `deploy/vps-local.sh`, `deploy/gpu/`, `docs/internal/ltx-comfy-runtime.md`, `docs/internal/gpu-validation-checklist.md` |
| GPU-day scripts | WRITTEN, not run | `apps/worker/scripts/ltx_comfy_health.py`, `apps/worker/scripts/ltx_comfy_bench.py` |
| Speed optimisation | CHECKLIST ONLY (GPU required) | `docs/internal/ltx-comfy-optimization-checklist.md` |

## 2. Workflows used

The three graphs from `LTX2.5 Pipeline-2.zip`, byte-identical (sha256 in
`benchmarks/client-pack/ltx25/README.md`), used as shipped: same models,
LoRAs, samplers, schedulers, canvas logic and conditioning. The worker
changes only prompt, seed, duration, input media and output location, plus
the two selections the graphs themselves expose as user settings (the
aspect-ratio label; the width/height/length constants of the character
graph, oriented to the source). The one structural edit — bypassing the
last-frame conditioning node when no last frame is supplied — is ComfyUI's
own bypass gesture applied to one node, covered by a test, and absent when
both frames are supplied.

| Product workflow | Graph | Runtime | Per-job inputs the worker sets |
|---|---|---|---|
| Text to Video | `ltx25_text_to_video.json` | `ltx_comfy` | positive/negative text, Clip Length slider, aspect label, `noise_seed`, `filename_prefix` |
| First/Last Frame Video | `ltx25_first_last_frame.json` | `ltx_comfy` | the above + `Load Image1`, `Load Image2` (or bypass) |
| Extend Video | `ltx25_first_last_frame.json` (first frame only), once per section | `ltx_comfy` | as above, the previous part's last frame as the first frame |
| Character Replacement | `ltx25_character_replacement.json` | `character_replacement` | text, `video`, `image`, `Set Length (seconds)`, `Set Width`/`Set Height`, seeds, prefix |
| Video to Video, Music Video, Music | unchanged | `ltx` / `music` | unchanged |

## 3. Models

All under the LTX-2.x Community Licence or Apache-2.0 (details and sources
in `docs/internal/ltx-comfy-runtime.md` §3). None is on the node yet.

| File | Used by | Source |
|---|---|---|
| `LTX-2.5-Distilled-Q8_0.gguf` | T2V, FLF | Abiray/LTX-2.5-Distilled-GGUF |
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | Character Replacement | Lightricks/LTX-2.5 (gated) |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | all | Lightricks/LTX-2.5 |
| `ltx-2.5-video-vae-bf16`, `ltx-2.5-audio-vae-bf16`, `taeltx2_3` | all | Lightricks/LTX-2.5, madebyollin/taehv |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` | all | Lightricks/LTX-2.5 |
| `LTX-2.3-OmniNFT-RL-Lora_bf16` @0.4, `ltx2.3-transition` @0.8, `ltx-2-19b-ic-lora-detailer` @0.3 | T2V, FLF | Kijai/LTX2.3_comfy, joyfox/LTX-2.3-Transition-LORA (Apache-2.0), Lightricks |
| `LTX25_Ripple_v11` @1.35 | Character Replacement | WepeNerd/LTX-Ripple |

## 4. Dependencies

* A second ComfyUI instance (≥ 0.34.0) with ten node packs at the commits
  the graphs stamp (`ltx-comfy-runtime.md` §2), on port 8189, under
  supervisord (`deploy/gpu/zolexai-ltx-comfy.{conf,sh}`).
* Worker env: `RUNTIMES=ltx,ltx_comfy,character_replacement,music`,
  `LTX_COMFY_BASE_URL`, `LTX_COMFY_MODELS_DIR`, `ENABLE_H3=false`
  (`.env.example` documents each).
* API env: `ENABLE_H3=false`.
* ffmpeg on the node (present). Hugging Face access to the gated LTX-2.5
  repo (held).

## 5. Benchmarks

Measured 5 Sep 2026 on the RTX PRO 6000 (full tables in `ltx25-gpu-benchmark.md`):

| | |
|---|---|
| Text to Video 5 / 10 / 15 / 30 s (client graph, 1280x704) | 48.8 / 76.3 / 106.3 / 215.2 s; VRAM peak 27–34 GB |
| First/Last Frame 5 s (704x1280) | 63.8 s first-only, 66.9 s both stills; VRAM peak 30.7 GB; identity held |
| Character Replacement on the ZIP inputs (736x1280, 8 s) | 163.9 s; VRAM peak 75.9 GB; same handoff behaviour as the delivered sample |
| Extend +5 s | 65.3 s; promised 10.042 s, measured 10.063 s; clean seam |
| Official Lightricks CLI, 5 s, NVFP4 | 41.5 s cold; VRAM peak 24.1 GB |
| ZolexAI vs ComfyUI direct, same text and seed | bit-identical (PSNR ∞, SSIM 1.0, audio 1.0) |

```text
Speed optimisation before/after pairs:  not started (Phase 5 checklist; baseline now exists)
```

## 6. What was verified without a GPU

| Suite | Result |
|---|---|
| Worker — graph compiler (`test_ltx_graphs.py`) | 27 passed |
| Worker — ltx_comfy adapter/service/routing (`test_ltx_comfy.py`) | 28 passed |
| Worker — First/Last Frame adapter | 6 passed |
| Worker — Character Replacement | 11 passed |
| Worker — extension engine | 9 passed |
| Worker — untouched-runtime guards | 4 passed |
| Worker — full suite | 1056 passed, 10 failed, 1 skipped — every failure reproduces at the starting commit (§7.7) |
| API — full suite (real PostgreSQL/Redis) | 131 passed, 3 skipped (the skips: Director-mode tests, unreachable since Director left the product) |
| Web — lint, typecheck, production build | clean; six create pages prerendered |
| Web — `qa:parity` against the live API | PASS (6 served, 1 hidden) |
| Web — `qa:e2e` (Playwright, real API + mock worker + built web) | 22 / 25 checks pass; see §7 |
| Deploy overlay dry run (`deploy/vps-local.sh`, both profiles, `--check`) | PASS on a copy of the definitions |

The fake ComfyUI in the worker tests answers the real HTTP conversation
(catalogue, upload, submit, poll, view, cancel) and serves real MP4s, so
compile, upload, submit, progress pacing, collection, validation, muting,
cancellation and every failure class are exercised. The model is the only
thing not exercised.

## 7. Limitations and findings

1. **GPU validation is done; deployment is not.** The client-test
   environment still needs `deploy/vps-local.sh --profile client-test` on the
   VPS, an api+web rebuild, and this node's worker pointed at it
   (`RUNTIMES=ltx,ltx_comfy,character_replacement`, `ENABLE_H3=false`). The
   node is a non-persistent Vast.ai container: recycle or destroy wipes the
   180 GB of weights (`deploy/gpu/provision/` rebuilds it in about an hour).
2. **Character Replacement is shipped hidden.** The definition, adapter,
   runtime, icon and tests exist; `hidden: true` keeps it out of the
   catalogue until the GPU validation clears it (`--profile client-test`
   flips it).
3. **Character Replacement windows the source to whole seconds, at most
   20 s** (the graph's own note lists 5/10/20 s). Longer sources are cut;
   the delivered length is logged. A chained variant is unbuilt.
4. **Progress is elapsed-time paced** (ComfyUI exposes queue state only);
   the rate is unmeasured.
5. **Audio at extension seams is edge-faded, not crossfaded**, so the
   delivered length is exact; a change of ambience between passes is a
   model property the metadata records.
6. **Pre-existing, unrelated to this milestone:** the Extend hand-off
   (`/app/create/extend-video?source=<asset>`) does not fill the source
   under a production `next start`, because the create page reads
   `searchParams` on the server while being statically prerendered
   (`apps/web/src/app/app/create/[workflowId]/page.tsx`, last changed 13
   Aug). It works under the dev server, which is what earlier e2e runs used.
   Two e2e checks fail on it. Fix: read the param client-side
   (`useSearchParams`) or mark the route dynamic.
7. **Pre-existing worker-suite failures (10)** reproduce at the milestone's
   starting commit: three "never touches the planner" tests (the soundscape
   clause the 28 Aug fix appends), five H3 video-to-video tests (H3's V2V
   default went off on 28 Aug), two music-video tests. None involves the
   new code.
8. **The LTX Attachment A #20 question** (production use of LTX in a
   competing SaaS) is unchanged by this pack; see `ltx-2.5-licensing-review.md`.
9. **Routing is not applied anywhere.** Production still runs the CLI
   runtime; the client-test profile exists and is verified as a script only.

## 8. Rollback plan

* **Routing:** `bash deploy/vps-local.sh --profile production` then rebuild
  api and web (`gpu-worker-runbook.md` §14) — every workflow back on the CLI
  runtime, Character Replacement hidden. The CLI adapter and its golden
  tests were not modified this milestone (pinned by hash).
* **Services:** `supervisorctl stop zolexai-ltx-comfy`; remove
  `ltx_comfy,character_replacement` from `RUNTIMES`; restart the worker.
* **H3:** `ENABLE_H3=true` on API and worker restores the 28 Aug behaviour
  exactly; the code, graphs and docs are untouched.
* **Git:** each phase is one commit (`git log 926d2e3..HEAD`); a revert
  of a phase range is clean, and the deploy overlay re-applies routing.
* **Data:** no database migration was needed; a new workflow is a YAML
  file, a new input role is a JSON field.

## 9. When the GPU becomes available

Execute `docs/internal/gpu-validation-checklist.md`: load the models and
node packs, run the health script, execute Text-to-Video, First/Last Frame,
Character Replacement and the extension test through the benchmark script,
record runtime / VRAM / RAM / fps / output duration, compare against the
ZIP samples, then apply the client-test profile and rerun parity and e2e.
Only then:

```text
CLIENT TEST READY: YES
```

Until then:

```text
CLIENT TEST READY: WAITING FOR GPU VALIDATION
```
