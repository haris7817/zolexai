# LTX 2.5 ComfyUI runtime — the second instance

**Internal. Written 5 September 2026, before the GPU node was available.**
Every pin below is what the client's graphs stamp, the ZIP's own notes say,
or a public source states; nothing has been installed or run yet.
STATUS: WAITING FOR GPU VALIDATION. When the node returns, provision in the
order of §3, run `scripts/ltx_comfy_health.py --deep`, and replace the
"stamped" pins here with the commits that actually served the validation.

Why a second instance: the H3 runtime is frozen at ComfyUI **v0.33.3**
(`h3-client-runtime-freeze.md`). The client's LTX graphs use nodes stamped
`comfy-core 0.34.0` and KJNodes/LTXVideo commits newer than that freeze.
Upgrading the H3 instance would be a compatibility pass on an engine that is
hidden anyway; a separate instance costs a venv and a port and changes
nothing about H3.

---

## 1. Service

| | |
|---|---|
| Base URL | `http://127.0.0.1:8189` (`LTX_COMFY_BASE_URL`) |
| Checkout | `/workspace/ComfyUI-ltx` (own venv; never the H3 checkout) |
| Launch | `python main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch` |
| Supervisor | `/etc/supervisor/conf.d/zolexai-ltx-comfy.conf` → `/opt/supervisor-scripts/zolexai-ltx-comfy.sh` (`stopasgroup`, `killasgroup`, same pattern as §37 of the GPU runbook) |
| Log | `/tmp/zolexai-ltx-comfy.log` |
| Worker env | `RUNTIMES=ltx,ltx_comfy,character_replacement,music`, `LTX_COMFY_BASE_URL`, `LTX_COMFY_MODELS_DIR=/workspace/ComfyUI-ltx/models`, `ENABLE_H3=false` |

The worker talks to it over HTTP only (`/upload/image`, `/prompt`,
`/history`, `/view`, `/queue`, `/interrupt`, `/free`, `/object_info`). No
shared directory is required.

## 2. Pins (stamped by the graphs; verify on install)

| Component | Pin | Source of the pin |
|---|---|---|
| ComfyUI core | ≥ **0.34.0** — the newest `comfy-core` stamp in the graphs | `properties.ver` on ComfySwitchNode, ComfyMathExpression, ModelAttentionBackend, PrimitiveBoolean |
| ComfyUI-GGUF (city96) | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | UnetLoaderGGUF |
| ComfyUI-KJNodes (kijai) | the newest of `e8e88f7c88e3f6205b122f5de87e69a09fbce5ac`, `5b38397a6430fdb16c7bd14a6bd64c2b0e69a5f0`, `71578cf49e48978cf1c6714494b669b1e571777b`, `676431504394217d8e0992e740370dbeec5e8dc1` (1.4.0 era) | LTX2AttentionTunerPatch, LTXVChunkFeedForward, LTXVImgToVideoInplaceKJ, SimpleCalculatorKJ, VAELoaderKJ, ImageResizeKJv2, INTConstant, ModelPreviewOverrideKJ, LTX2SamplingPreviewOverride |
| rgthree-comfy | `6b76ee6f2c5a007710b5a16f97c94330d6ecc871` | Power Lora Loader |
| ComfyUI-LTXVideo (Lightricks) | `229437c6b65796d6a7a63ae34be2bd5ba31fa543` | LTXAddVideoICLoRAGuideAdvanced |
| ComfyUI-VideoHelperSuite | `115de7a9d9e34410cffb9ecfd268e993b11a50fb` (1.7.9 era) | VHS_VideoCombine, VHS_LoadVideoFFmpeg, VHS_VideoInfo |
| ComfyUI-Easy-Use | `005c57839c5bee88f1a0a41970ca965ab470a4c8` (1.3.4) | easy cleanGpuUsed, easy ifElse |
| ComfyMath (evanspearman) | `c01177221c31b8e5fbc062778fc8254aeb541638` | CM_IntToFloat, CM_FloatToInt |
| ComfyUI-mxToolkit (Smirnov75) | 0.9.92 | mxSlider |
| ComfyUI-Custom-Scripts (pythongosssss) | 1.2.5 | MathExpression\|pysssss |
| RES4LYF (`drozbay/RES4LYF` fork stamped; upstream ClownsharkBatwing) | `26036f647ca15d3048a193daf99a40cecfc3820d` | Frames Slice |

`SetNode`/`GetNode` (KJNodes) are frontend-only and are resolved by the
compiler; they need no server support.

## 3. Weights

Place under `/workspace/ComfyUI-ltx/models/…` exactly as named — the graphs
reference these strings and the health check reads the loader nodes' combo
lists to confirm the server sees them.

| Path under `models/` | Source | Licence | Size |
|---|---|---|---|
| `diffusion_models/LTX-2.5-Distilled-Q8_0.gguf` (`UnetLoaderGGUF` may also read `unet/`) | `Abiray/LTX-2.5-Distilled-GGUF` | LTX-2.x Community | 23.6 GB |
| `diffusion_models/LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` (gated) | LTX-2.x Community | 21.5 GB |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` (gated) | LTX-2.x Community | record on download |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | `Lightricks/LTX-2.5` | LTX-2.x Community | record |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | `Lightricks/LTX-2.5` | LTX-2.x Community | record |
| `vae/taeltx2_3.safetensors` | `madebyollin/taehv` | MIT | small |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `Lightricks/LTX-2.5` | LTX-2.x Community | record |
| `loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` | `Kijai/LTX2.3_comfy` | repo: LTX-2 community | record |
| `loras/ltx2.3-transition.safetensors` | `joyfox/LTX-2.3-Transition-LORA` | Apache-2.0 | record |
| `loras/ltx-2-19b-ic-lora-detailer.safetensors` | `Lightricks/LTX-2-19b-IC-LoRA-Detailer` | LTX-2 community | 2.62 GB |
| `loras/LTX/LTX-2.5/LTX25_Ripple_v11.safetensors` | `WepeNerd/LTX-Ripple` | LTX-2.x Community | record |

Record sha256 and byte size of every file at download into
`benchmarks/results/ltx25/weights.json`; the health check's deep mode reads
presence, and the readiness report cites the hashes.

## 4. Order of operations on GPU day

1. Clone ComfyUI, create the venv, install torch for the card, install the
   node packs at the pins above, start the service under supervisord.
2. Download the weights (§3). Hugging Face access to the gated LTX-2.5 repo
   is already held for the CLI runtime.
3. `cd apps/worker && .venv/bin/python scripts/ltx_comfy_health.py --deep`
   — must report HEALTHY. Every "not installed" is a node pack; every "not
   among the offered values" is a missing file or a renamed option.
4. `scripts/ltx_comfy_bench.py t2v --seconds 5` — the first render. Look at
   it before running the rest of the matrix (§6 of the workflow audit).
5. Set `RUNTIMES` on the worker, restart it, confirm `worker_ready` lists
   `ltx_comfy` and `character_replacement`.
6. Apply the client-test routing (`deploy/vps-local.sh --profile client-test`)
   in the client-test environment only.

## 5. Co-tenancy

The card is shared with ACE-Step (~24 GB resident) and, when `ENABLE_H3` is
on, the H3 ComfyUI (~52 GB warm). With H3 hidden its instance should not be
running at all. The worker evicts every other ComfyUI's models before an
LTX ComfyUI job and vice versa (`evict_comfy_vram`); the CLI LTX runtime
does the same on its way in. VRAM headroom for the pack is unmeasured — the
GGUF transformer alone is 23.6 GB, the text encoder and VAEs add to it, and
the character graph loads the INT8 transformer instead. Measure before
deciding whether ACE-Step and this instance can be resident together.

## 6. Rollback

* Routing: `deploy/vps-local.sh --profile production` restores the CLI
  runtime for Text to Video, First/Last Frame and Extend, and hides
  Character Replacement.
* Service: `supervisorctl stop zolexai-ltx-comfy`; nothing else depends on it.
* Worker: remove `ltx_comfy,character_replacement` from `RUNTIMES`.
