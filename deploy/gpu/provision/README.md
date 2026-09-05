# GPU node provisioning — the scripts that built the validated node (5 Sep 2026)

Run in this order on a fresh Vast.ai `base-image_cuda-12.8.1` container with
an RTX PRO 6000 (sm_120). Logs go beside each script under `/workspace/provision/`.

| Order | Script | What it does |
|---|---|---|
| 1 | `ltx_env.sh` | clones `Lightricks/LTX-2` at `400fd31`, `uv sync --extra natten --group dev` (torch 2.13 cu132, NATTEN, nvcc 13.2 in-venv) |
| 2 | `ltx_kernels.sh` | builds `ltx-kernels` for sm_120 with the venv toolchain (absolute `CUDA_HOME`); verifies `import ltx_kernels` |
| 3 | `comfy_env.sh` | ComfyUI v0.34.5, own venv (torch cu130), the node packs at their pins, `extra_model_paths.yaml`. **Then by hand:** `uv pip install kornia==0.8.1` and `git -C custom_nodes/ComfyUI-LTXVideo checkout 15d09ab` — bake these in before the next node |
| 4 | `downloads_public.sh` | the client's non-gated files (GGUF Q8, four LoRAs, taeltx) |
| 5 | `downloads_gated.sh` | the fourteen official Lightricks files + the Union-Control LoRA; needs `HF_HOME/token`; a read timeout is recovered by re-running the script (completed files are skipped) |
| 6 | `link_official_into_comfy.sh` | mirrors the official tree into the ComfyUI tree by symlink; client folder conventions (`LTXVideo/v2/`) |
| 7 | `worker_env.sh` | the ZolexAI worker venv on the node (`git archive HEAD | ssh … tar -x -C /workspace/zolexai` first) |
| 8 | `../zolexai-ltx-comfy.{conf,sh}` | supervisord program for the ComfyUI service on :8189 |

Validation runners used on 5 Sep: `step7_official_t2v.sh` (official CLI with
VRAM/RAM sampling), `step8_9.sh` (deep health, direct-vs-adapter, the ladder),
`step10_more.sh`, `step11_cr_extend.sh`, `step12_rerender.sh` (same-text match,
warm 5 s, First/Last both ways, Character Replacement on the ZIP inputs,
Extend). `hash_public.sh` records sha256 of the client files.

Never commit a token; the scripts read it from `HF_HOME/token`.
