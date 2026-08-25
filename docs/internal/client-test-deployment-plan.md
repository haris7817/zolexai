# Client-test deployment plan

**Scope: the client-test environment ONLY.** Production is untouched, shipped
workflow YAML is untouched, `auto → ltx` is untouched. Everything here is the
per-environment configuration the architecture was designed for: routing is a
`runtime:` line per workflow, exactly the M2 mechanism.

---

## 1. Branch and commit

Branch `dual-engine-benchmark-prep`, at the commit this plan ships in (see
`git log docs/internal/client-test-deployment-plan.md`). All integration code
is additive; `git revert` of the integration commits restores the previous
state exactly.

## 2. The client-test routing (Phase 14)

Applied by editing the **environment's** copy of `workflow-definitions/` —
never the shipped files. One line per workflow, plus the music guard pair:

```yaml
# text-to-video.yaml      — unchanged path, real runtime
execution:
  runtime: ltx

# image-to-video.yaml     — H3 INT8 (I2V graph, FL2VA)
execution:
  runtime: h3_comfy
  # h3_tier is R2V-only; I2V has one proven canvas (1280x736)

# video-to-video.yaml     — H3 INT8 (R2V graph) for reference jobs
execution:
  runtime: h3_comfy
  h3_tier: quality          # 960x544 delivery; draft = 544x320
  # duration_mode: source is honoured — the adapter maps the source length
  # to the nearest pack preset (5/10/15/30/60 s)

# music-video.yaml        — LTX native A2V, guarded
execution:
  runtime: ltx
  audio_conditioning: true
  require_audio_conditioning: true   # the guard: the unconditioned
                                     # prompt-only + post-mux route REFUSES

# extend-video.yaml       — unchanged path, real runtime
execution:
  runtime: ltx

# music.yaml              — unchanged (ACE-Step)
execution:
  runtime: music
```

The worker must be started with `RUNTIMES` including `ltx,h3_comfy,music`.

## 3. GPU node services (RTX PRO 6000 class, ≥64 GB VRAM, ≥64 GB RAM)

### 3.1 ComfyUI (H3 INT8) — the pinned stack

Build once per node (already scripted on the current box):

```bash
/workspace/build_client_stack.sh     # ComfyUI v0.33.3 + pinned nodes + venv
/workspace/fetch_weights.sh          # official Comfy-Org weights, SHA-verified
# plus the FL2VA file for I2V:
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --local-dir /workspace/models/h3-comfy
ln -sf /workspace/models/h3-comfy/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  /workspace/comfyui-h3-client/ComfyUI/models/diffusion_models/MiniMax/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

Start:

```bash
cd /workspace/comfyui-h3-client && source .venv/bin/activate
cd ComfyUI && python main.py --listen 127.0.0.1 --port 8188 \
  > /workspace/logs/comfy-server.log 2>&1 &
```

### 3.2 ACE-Step (music workflow)

```bash
cd /workspace/acestep-benchmark
ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B \
  uv run acestep-api --host 127.0.0.1 --port 8001 &
# ACESTEP_CONFIG_PATH is NOT optional — unset, the wrong model loads silently.
```

**Scheduling note:** ACE-Step holds ~24 GB resident. On a 96 GB card it
coexists with LTX (~28 GB peak) but NOT with an H3 INT8 run (peaks 52–63 GB).
The client-test worker should serialise music jobs against H3 jobs, or the
environment should stop ACE-Step during H3-heavy testing — co-tenancy is the
measured 1-in-3 failure mode.

### 3.3 Worker

```bash
cd /workspace/src/zolexai/apps/worker && source .venv/bin/activate
export LTX_REPO_DIR=/workspace/src/LTX-2
export LTX_MODEL_DIR=/workspace/models/ltx25
export H3_COMFY_BASE_URL=http://127.0.0.1:8188
export H3_COMFY_INPUT_DIR=/workspace/comfyui-h3-client/ComfyUI/input
export H3_COMFY_MODELS_DIR=/workspace/models/h3-comfy
export ACESTEP_BASE_URL=http://127.0.0.1:8001
# plus the platform's own worker settings (API URL, storage credentials,
# WORKER_NAME, RUNTIMES=ltx,h3_comfy,music) per the existing GPU worker
# runbook (docs/internal/gpu-worker-runbook.md).
python -m worker
```

## 4. Health checks

```bash
# ComfyUI up + pinned nodes + weights at exact published sizes:
python - <<'PY'
import asyncio
from worker.adapters.h3_comfy import h3_comfy_health
print(asyncio.run(h3_comfy_health()))
PY

# ACE-Step:  curl -s http://127.0.0.1:8001/docs -o /dev/null -w '%{http_code}\n'
# LTX:       python -m pytest tests/test_ltx_golden.py -q   (11 argv shapes)
# GPU:       nvidia-smi
```

Health is refusal-shaped: if any pinned piece is missing, H3 reports
unavailable and jobs routed to it FAIL as H3 failures. There is no silent
fallback to LTX in client-test mode — a wrong-provider success would be a
worse outcome than an honest error.

## 5. Ports and secrets

| Service | Port | Auth |
|---|---|---|
| ComfyUI | 8188 loopback | none (loopback only; never expose) |
| ACE-Step | 8001 loopback | none (loopback) |
| API / storage | per existing platform config | existing secrets |

New secrets required: **none**. The HF token is only needed at provisioning
time for weight download.

## 6. Storage and history

No new path: the adapter returns a standard `AdapterResult` (MP4 in the job
workspace) and the existing runner uploads it to storage and records history
exactly as for LTX jobs. Provider/runtime metadata stays internal
(`execution.runtime`), and the frontend continues to show its normal result
view. Verified by the worker suite's storage/output contract tests.

## 7. Job progress

The adapter reports the house vocabulary (`preparing → generating →
post_processing → uploading`) through the standard `StageReporter`, with
section counters on 30/60 s runs paced by measured rates. ComfyUI does not
expose intra-node progress; the bar therefore moves on elapsed-vs-expected and
never claims completion it has not observed.

## 8. Rollback

1. Revert the environment's `workflow-definitions/` copies to `runtime: ltx`
   (or `mock`) — one line per workflow; no code change needed.
2. Stop the ComfyUI service. LTX and music paths are unaffected.
3. Full code rollback: `git revert` the integration commits; the adapter and
   provider changes are additive and isolated
   (`worker/comfy/`, `worker/adapters/h3_comfy.py`,
   `worker/longform/h3_prompts.py`, one guard block in `worker/adapters/ltx.py`,
   config keys, registry entry).

## 9. Known-limits card (hand to the client with the build)

- Durations are the pack's presets: 5/10/15/30/60 s (I2V and reference V2V).
- Reference V2V composes from the reference photo + the source's first frame;
  it does not re-enact the source's motion (LTX transform does that).
- 60 s runs take ~12 min (quality tier ~2.5x longer); the progress bar is
  honest about it.
- One of four audio seams in the 60 s validation showed a loudness step
  (9.8 dB); three were clean. `execution.h3_audio_context` exists as the
  recorded experiment if listening tests object — default stays the pack's
  own behaviour.
