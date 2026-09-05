#!/usr/bin/env bash
# Official Lightricks LTX-2.5 files (gated: needs a token from an account that
# accepted the licence at huggingface.co/Lightricks/LTX-2.5) into the ONE tree
# both runtimes read: /workspace/ltx2-benchmark/models/ltx-2.5.
set -euxo pipefail
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/workspace/.hf_home
HF=/venv/main/bin/hf
M=/workspace/ltx2-benchmark/models/ltx-2.5
# Required (Step 2), positional to avoid the --exclude trap (runbook 34.1)
$HF download Lightricks/LTX-2.5 --local-dir "$M" \
  diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors
# Optional (download, keep disabled)
$HF download Lightricks/LTX-2.5 --local-dir "$M" \
  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
# The client graphs use the official ComfyUI int8 variants (same repo, same tree)
$HF download Lightricks/LTX-2.5 --local-dir "$M" \
  diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
# The CLI runtime transform-engine LoRA (a different Lightricks repo)
$HF download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control --local-dir /workspace/provision/dl/union
find /workspace/provision/dl/union -name "*.safetensors" -exec cp --update=none {} "$M/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \;
# Client folder conventions inside the ComfyUI tree: symlinks to the official files
C=/workspace/ComfyUI-ltx/models
mkdir -p "$C/diffusion_models/LTXVideo/v2"
ln -sfn "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" "$C/diffusion_models/LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
for f in "$M"/diffusion_models/*.safetensors "$M"/text_encoders/*.safetensors "$M"/vae/*.safetensors "$M"/latent_upscale_models/*.safetensors "$M"/model_patches/*.safetensors "$M"/loras/*.safetensors; do
  printf "%s %s %s\n" "$(sha256sum "$f" | cut -c1-64)" "$(stat -c %s "$f")" "$f"
done > /workspace/provision/weights_official.txt
find "$M" -type f -name "*.safetensors" -printf "%s %p\n" | sort -k2
echo GATED_DL_DONE
