#!/usr/bin/env bash
set -euo pipefail
cd /workspace/zolexai/apps/worker; source .venv/bin/activate; export LTX_COMFY_MODELS_DIR=/workspace/ComfyUI-ltx/models LTX_COMFY_BASE_URL=http://127.0.0.1:8189; S=/workspace/results/step9; S8=/workspace/zolexai/benchmarks/client-pack/ltx25/samples; SEED=42; PID=$(supervisorctl pid zolexai-ltx-comfy)
echo "=== CHARACTER REPLACEMENT: the ZIP sample source + the ZIP still ==="
python scripts/ltx_comfy_bench.py cr --video "/workspace/zolexai/benchmarks/client-pack/ltx25/samples/character_replacement_source.mp4" --image "$S8/first_last_frame_input.png" --seed $SEED --comfy-pid "$PID" \
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
