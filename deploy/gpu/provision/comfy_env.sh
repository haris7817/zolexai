#!/usr/bin/env bash
# Second ComfyUI instance for the client graphs: ComfyUI v0.34.5 + pinned node packs.
set -euxo pipefail
export UV_LINK_MODE=copy
cd /workspace
[ -d ComfyUI-ltx ] || git clone -q --branch v0.34.5 https://github.com/comfyanonymous/ComfyUI ComfyUI-ltx
cd ComfyUI-ltx && git describe --tags
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
python -c "import torch;print(\"comfy torch\",torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())"
python -c "import comfy_kitchen;print(\"comfy_kitchen\",comfy_kitchen.__version__)" || true
cd custom_nodes
newest() { d=$1; shift; best=""; bd=0; for s in "$@"; do git -C "$d" cat-file -e "$s^{commit}" 2>/dev/null || git -C "$d" fetch -q origin "$s" 2>/dev/null || true; t=$(git -C "$d" log -1 --format=%ct "$s" 2>/dev/null || echo 0); if [ "$t" -gt "$bd" ]; then best=$s; bd=$t; fi; done; [ -n "$best" ] || best=HEAD; echo "$best"; }
clone_at() { name=$1; url=$2; shift 2; [ -d "$name" ] || git clone -q "$url" "$name"; sha=$(newest "$name" "$@"); git -C "$name" checkout -q "$sha"; echo "$name @ $(git -C "$name" log -1 --format='%h %ci')"; if [ -f "$name/requirements.txt" ]; then uv pip install -r "$name/requirements.txt"; fi; }
clone_tag_or_head() { name=$1; url=$2; tagpat=$3; [ -d "$name" ] || git clone -q "$url" "$name"; t=$(git -C "$name" tag -l "$tagpat" | tail -1); if [ -n "$t" ]; then git -C "$name" checkout -q "$t"; fi; echo "$name @ $(git -C "$name" log -1 --format='%h %ci') tag=${t:-HEAD}"; if [ -f "$name/requirements.txt" ]; then uv pip install -r "$name/requirements.txt"; fi; }
clone_at ComfyUI-GGUF https://github.com/city96/ComfyUI-GGUF 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
clone_at ComfyUI-KJNodes https://github.com/kijai/ComfyUI-KJNodes e8e88f7c88e3f6205b122f5de87e69a09fbce5ac 5b38397a6430fdb16c7bd14a6bd64c2b0e69a5f0 71578cf49e48978cf1c6714494b669b1e571777b 676431504394217d8e0992e740370dbeec5e8dc1 c6ce76d00bb8177d1b0286cad891df08eff5226e 37659859825cea55940a58110525795ce5deb8be 5219cd171cb44e2edce9e4daad6cc42c41eded5c
clone_at rgthree-comfy https://github.com/rgthree/rgthree-comfy 6b76ee6f2c5a007710b5a16f97c94330d6ecc871 35c9f1e186603ba312d3b15350e89aa50b860ee6
clone_at ComfyUI-LTXVideo https://github.com/Lightricks/ComfyUI-LTXVideo 229437c6b65796d6a7a63ae34be2bd5ba31fa543
clone_at ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite 115de7a9d9e34410cffb9ecfd268e993b11a50fb 3234937ff5f3ca19068aaba5042771514de2429d
clone_at ComfyUI-Easy-Use https://github.com/yolain/ComfyUI-Easy-Use 005c57839c5bee88f1a0a41970ca965ab470a4c8
clone_at ComfyMath https://github.com/evanspearman/ComfyMath c01177221c31b8e5fbc062778fc8254aeb541638
clone_at RES4LYF https://github.com/drozbay/RES4LYF 26036f647ca15d3048a193daf99a40cecfc3820d
clone_tag_or_head ComfyUI-mxToolkit https://github.com/Smirnov75/ComfyUI-mxToolkit "*0.9.92*"
clone_tag_or_head ComfyUI-Custom-Scripts https://github.com/pythongosssss/ComfyUI-Custom-Scripts "*1.2.5*"
cd ..
mkdir -p models/{diffusion_models/LTXVideo/v2,text_encoders,vae,latent_upscale_models,loras/LTX/LTX-2.5,model_patches,unet}
cat > extra_model_paths.yaml <<'YAML'
# The official LTX-2.5 tree the CLI runtime uses is the same tree ComfyUI reads:
# one copy of every official file, no duplicates.
ltx25_official:
  base_path: /workspace/ltx2-benchmark/models/ltx-2.5
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  latent_upscale_models: latent_upscale_models
  loras: loras
  model_patches: model_patches
YAML
echo COMFY_ENV_DONE
