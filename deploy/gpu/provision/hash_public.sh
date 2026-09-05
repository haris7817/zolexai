#!/usr/bin/env bash
set -euo pipefail
M=/workspace/ComfyUI-ltx/models
for f in "$M/diffusion_models/LTX-2.5-Distilled-Q8_0.gguf" "$M/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors" "$M/loras/ltx2.3-transition.safetensors" "$M/loras/ltx-2-19b-ic-lora-detailer.safetensors" "$M/loras/LTX/LTX-2.5/LTX25_Ripple_v11.safetensors" "$M/vae/taeltx2_3.safetensors"; do
  printf "%s %s %s\n" "$(sha256sum "$f" | cut -c1-64)" "$(stat -c %s "$f")" "$f"
done > /workspace/provision/weights_public.txt
echo HASH_DONE
