#!/usr/bin/env bash
set -euo pipefail
cd /workspace/zolexai/apps/worker; source .venv/bin/activate
export LTX_COMFY_MODELS_DIR=/workspace/ComfyUI-ltx/models LTX_COMFY_BASE_URL=http://127.0.0.1:8189
S=/workspace/results/step9; S8=/workspace/zolexai/benchmarks/client-pack/ltx25/samples; PID=$(supervisorctl pid zolexai-ltx-comfy)
echo "=== FLF first-only, one-image node (fixed) ==="
python scripts/ltx_comfy_bench.py flf --first "$S8/first_last_frame_input.png" --seconds 5 --aspect 9:16 --seed 42 --comfy-pid "$PID" \
  --prompt "The man looks at the camera, then slowly turns his head to look out over the harbour; soft afternoon light, gentle handheld camera." --stamp step12-flf-first-fixed | tee "$S/flf_first_fixed.log"
echo "=== EXTEND +5 s (fixed) ==="
python - <<'PY'
import asyncio, json, time
from pathlib import Path
from worker.adapters.base import AdapterInput, AdapterJob
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.media import probe_media
src = Path("/workspace/results/step8/B_zolexai_t2v_5s_seed42.mp4")
ws = Path("/workspace/results/step9/extend-ws-fixed"); ws.mkdir(parents=True, exist_ok=True)
job = AdapterJob(job_id="extend-5s-fixed", workflow_id="extend-video", workflow_version="1",
    prompt="The camera keeps gliding forward between the towers as the sun sinks lower; more vehicles cross the skyline.",
    parameters={"duration": "5s", "aspect_ratio": "16:9", "seed": 42},
    inputs=[AdapterInput(role="source_video", kind="video", content_type="video/mp4", download_url="file://x", path=src)],
    execution={"runtime": "ltx_comfy"}, output_content_type="video/mp4", workspace=ws)
async def prog(status, progress, message, details=None): pass
async def main():
    t = time.monotonic(); r = await LtxComfyAdapter().run(job, prog); wall = time.monotonic() - t
    info = await probe_media(r.path)
    print(json.dumps({"output": str(r.path), "wall_seconds": round(wall, 1), "duration": info.duration_seconds, "width": info.width, "height": info.height, "fps": info.fps, "frames": info.frame_count, "has_audio": info.has_audio}, indent=1))
asyncio.run(main())
PY
echo STEP12_DONE
