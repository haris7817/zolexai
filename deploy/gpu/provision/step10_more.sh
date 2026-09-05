#!/usr/bin/env bash
# Step 8 (settled): same text, same seed, fresh execution — must match.
# Plus: a warm 5 s number, the FLF graph both ways, the character graph on the
# ZIP sample inputs, and one extension pass.
set -euo pipefail
PROMPT="A cinematic shot of a futuristic city at sunset, flying vehicles moving between skyscrapers, realistic lighting, smooth camera movement"
BTEXT="$PROMPT No one speaks. The only sounds are the ones the scene itself makes."
SEED=42
cd /workspace/zolexai/apps/worker
source .venv/bin/activate
export LTX_COMFY_MODELS_DIR=/workspace/ComfyUI-ltx/models LTX_COMFY_BASE_URL=http://127.0.0.1:8189
R=/workspace/results/step8; S=/workspace/results/step9; mkdir -p "$R" "$S"
PID=$(supervisorctl pid zolexai-ltx-comfy)
S8=/workspace/zolexai/benchmarks/client-pack/ltx25/samples

echo "=== STEP 8 (A2): cache cleared, models unloaded, same text as ZolexAI, seed $SEED ==="
curl -s -X POST http://127.0.0.1:8189/free -H "Content-Type: application/json" -d '{"unload_models":true,"free_memory":true}'; echo
python scripts/ltx_comfy_direct.py t2v --prompt "$BTEXT" --seed $SEED --seconds 5 --aspect 16:9 \
  --out "$R/A2_comfy_direct_sametext_seed$SEED.mp4" --dump-prompt "$R/A2_prompt.json" | tee "$R/A2.log"
python scripts/compare_outputs.py "$R/A2_comfy_direct_sametext_seed$SEED.mp4" "$R/B_zolexai_t2v_5s_seed$SEED.mp4" \
  --label-a comfy_direct_sametext --label-b zolexai --json "$R/compare_A2_vs_B.json" || true

echo "=== STEP 9 (warm 5 s, seed 43, models resident, no cache hit) ==="
python scripts/ltx_comfy_bench.py t2v --seconds 5 --aspect 16:9 --seed 43 --prompt "$PROMPT" --comfy-pid "$PID" --stamp step9-warm5 | tee "$S/warm5.log"

echo "=== FLF: first frame only, 5 s ==="
python scripts/ltx_comfy_bench.py flf --first "$S8/first_last_frame_input.png" --seconds 5 --aspect 9:16 --seed $SEED --comfy-pid "$PID" \
  --prompt "The man looks at the camera, then slowly turns his head to look out over the harbour; soft afternoon light, gentle handheld camera." --stamp step9-flf-first | tee "$S/flf_first.log"

echo "=== FLF: first + last frame (the same still), 5 s ==="
python scripts/ltx_comfy_bench.py flf --first "$S8/first_last_frame_input.png" --last "$S8/first_last_frame_input.png" --seconds 5 --aspect 9:16 --seed $SEED --comfy-pid "$PID" \
  --prompt "The man looks at the camera, glances to the side, and returns to the same pose; soft afternoon light, gentle handheld camera." --stamp step9-flf-both | tee "$S/flf_both.log"

echo "=== CHARACTER REPLACEMENT: the ZIP sample source + the ZIP still ==="
python scripts/ltx_comfy_bench.py cr --video "$S8/character_replacement_source.mp4" --image "$S8/first_last_frame_input.png" --seed $SEED --comfy-pid "$PID" \
  --prompt "A man with short black curls, a trimmed beard, dark rectangular sunglasses, a white open-collar shirt under a grey suit jacket, a thin silver necklace, on the bright deck of a luxury yacht in a harbour." --stamp step9-cr | tee "$S/cr.log"
CR=$(ls /workspace/zolexai/benchmarks/results/ltx25/step9-cr/cr-r0.mp4)
python scripts/compare_outputs.py /workspace/results/zip_sample_character_replacement.mp4 "$CR" --label-a zip_sample --label-b zolexai_cr --json "$S/compare_cr_vs_zip_sample.json" || true

echo "=== EXTEND: +5 s on the ZolexAI 5 s output ==="
python - <<'PY'
import asyncio, json, time
from pathlib import Path
from worker.adapters.base import AdapterInput, AdapterJob
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.media import probe_media
src = Path("/workspace/results/step8/B_zolexai_t2v_5s_seed42.mp4")
ws = Path("/workspace/results/step9/extend-ws"); ws.mkdir(parents=True, exist_ok=True)
job = AdapterJob(job_id="extend-5s", workflow_id="extend-video", workflow_version="1",
    prompt="The camera keeps gliding forward between the towers as the sun sinks lower; more vehicles cross the skyline.",
    parameters={"duration": "5s", "aspect_ratio": "16:9", "seed": 42},
    inputs=[AdapterInput(role="source_video", kind="video", content_type="video/mp4", download_url="file://x", path=src)],
    execution={"runtime": "ltx_comfy"}, output_content_type="video/mp4", workspace=ws)
async def prog(status, progress, message, details=None): print(f"    [{status:>15}] {progress:3d}% {message}", flush=True)
async def main():
    t = time.monotonic(); r = await LtxComfyAdapter().run(job, prog); wall = time.monotonic() - t
    info = await probe_media(r.path)
    print(json.dumps({"output": str(r.path), "wall_seconds": round(wall, 1), "duration": info.duration_seconds, "width": info.width, "height": info.height, "fps": info.fps, "frames": info.frame_count, "has_audio": info.has_audio}, indent=1))
    print((ws / "continuation.json").read_text())
asyncio.run(main())
PY
echo STEP10_DONE
