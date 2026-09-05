#!/usr/bin/env bash
# Client-graph files that are NOT behind the Lightricks gate.
set -euxo pipefail
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/workspace/.hf_home
HF=/venv/main/bin/hf
M=/workspace/ComfyUI-ltx/models
mkdir -p "$M"/{diffusion_models,vae,loras/LTX/LTX-2.5} /workspace/provision/dl
$HF download Abiray/LTX-2.5-Distilled-GGUF LTX-2.5-Distilled-Q8_0.gguf --local-dir "$M/diffusion_models"
$HF download Kijai/LTX2.3_comfy loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors vae/taeltx2_3.safetensors --local-dir /workspace/provision/dl/kijai
cp -n /workspace/provision/dl/kijai/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors "$M/loras/"
cp -n /workspace/provision/dl/kijai/vae/taeltx2_3.safetensors "$M/vae/"
$HF download joyfox/LTX-2.3-Transition-LORA --local-dir /workspace/provision/dl/transition
find /workspace/provision/dl/transition -name "*.safetensors" -exec cp -n {} "$M/loras/ltx2.3-transition.safetensors" \;
$HF download WepeNerd/LTX-Ripple --local-dir /workspace/provision/dl/ripple
find /workspace/provision/dl/ripple -name "*Ripple_v11*.safetensors" -exec cp -n {} "$M/loras/LTX/LTX-2.5/LTX25_Ripple_v11.safetensors" \;
# The detailer is a Lightricks repo and may be gated: try, tolerate 401.
$HF download Lightricks/LTX-2-19b-IC-LoRA-Detailer --local-dir /workspace/provision/dl/detailer || echo "DETAILER_NEEDS_TOKEN"
find /workspace/provision/dl/detailer -name "*.safetensors" -exec cp -n {} "$M/loras/ltx-2-19b-ic-lora-detailer.safetensors" \; 2>/dev/null || true
ls -la "$M"/diffusion_models "$M"/loras "$M"/loras/LTX/LTX-2.5 "$M"/vae
echo PUBLIC_DL_DONE
