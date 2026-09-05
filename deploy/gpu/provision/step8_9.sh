#!/usr/bin/env bash
# Steps 6 (verify), 8 (ComfyUI direct vs ZolexAI adapter, same seed) and 9 (T2V ladder benchmark).
set -euo pipefail
PROMPT="A cinematic shot of a futuristic city at sunset, flying vehicles moving between skyscrapers, realistic lighting, smooth camera movement"
SEED=42
cd /workspace/zolexai/apps/worker
source .venv/bin/activate
export LTX_COMFY_MODELS_DIR=/workspace/ComfyUI-ltx/models
export LTX_COMFY_BASE_URL=http://127.0.0.1:8189
R=/workspace/results/step8
mkdir -p "$R" /workspace/results/step9

echo "=== STEP 6: layout + deep health ==="
bash /workspace/provision/link_official_into_comfy.sh | tail -40
python scripts/ltx_comfy_health.py --deep | head -30

echo "=== STEP 8A: client graph submitted directly to ComfyUI (seed $SEED) ==="
python scripts/ltx_comfy_direct.py t2v --prompt "$PROMPT" --seed $SEED --seconds 5 --aspect 16:9 \
  --out "$R/A_comfy_direct_t2v_5s_seed$SEED.mp4" --dump-prompt "$R/A_prompt.json" | tee "$R/A.log"

echo "=== STEP 8B: the same job through the ZolexAI adapter (seed $SEED) ==="
python scripts/ltx_comfy_bench.py t2v --seconds 5 --aspect 16:9 --seed $SEED --prompt "$PROMPT" --stamp step8-B | tee "$R/B.log"
cp /workspace/zolexai/benchmarks/results/ltx25/step8-B/t2v-5s-16x9-r0.mp4 "$R/B_zolexai_t2v_5s_seed$SEED.mp4"

echo "=== STEP 8: compare ==="
python scripts/compare_outputs.py "$R/A_comfy_direct_t2v_5s_seed$SEED.mp4" "$R/B_zolexai_t2v_5s_seed$SEED.mp4" \
  --label-a comfy_direct --label-b zolexai --json "$R/compare_A_vs_B.json" || true
echo "--- official (Step 7) vs client graph (A): different pipelines, expected to differ ---"
OFF=$(ls /workspace/results/step7/*.mp4 | head -1)
python scripts/compare_outputs.py "$OFF" "$R/A_comfy_direct_t2v_5s_seed$SEED.mp4" \
  --label-a official_cli --label-b comfy_direct --json "$R/compare_official_vs_A.json" || true

echo "=== STEP 9: T2V ladder through the ZolexAI adapter ==="
python scripts/ltx_comfy_bench.py t2v --seconds 5 10 15 30 --aspect 16:9 --seed $SEED --prompt "$PROMPT" --stamp step9-t2v | tee /workspace/results/step9/t2v.log
cp -r /workspace/zolexai/benchmarks/results/ltx25/step9-t2v /workspace/results/step9/ 2>/dev/null || true
echo STEP8_9_DONE
