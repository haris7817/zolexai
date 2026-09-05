#!/usr/bin/env bash
# After the gated download: mirror the official tree into the ComfyUI models
# tree by symlink (one copy on disk), so both the extra_model_paths view and
# the plain models/ view show every file, and the deep health check passes.
set -euo pipefail
M=/workspace/ltx2-benchmark/models/ltx-2.5
C=/workspace/ComfyUI-ltx/models
for sub in diffusion_models text_encoders vae latent_upscale_models model_patches loras; do
  mkdir -p "$C/$sub"
  for f in "$M/$sub"/*.safetensors; do
    [ -e "$f" ] || continue
    ln -sfn "$f" "$C/$sub/$(basename "$f")"
  done
done
mkdir -p "$C/diffusion_models/LTXVideo/v2"
ln -sfn "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" "$C/diffusion_models/LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
echo "=== official tree ==="
find "$M" -type f \( -name "*.safetensors" -o -name "*.gguf" \) -printf "%12s  %p\n" | sort -k2
echo "=== comfy tree ==="
find "$C" \( -type f -o -type l \) \( -name "*.safetensors" -o -name "*.gguf" \) -printf "%12s  %p -> %l\n" | sort -k2
echo LINK_DONE
