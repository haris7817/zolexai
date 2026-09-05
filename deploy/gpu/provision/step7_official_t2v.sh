#!/usr/bin/env bash
# Step 7: the OFFICIAL Lightricks LTX 2.5 text-to-video test (ltx_pipelines.distilled),
# with VRAM (nvidia-smi, 1 Hz) and RAM (max RSS) sampled for the whole run.
#
#   bash step7_official_t2v.sh [SEED] [QUANT]      QUANT: nvfp4-prequant (default) | bf16
set -euo pipefail
SEED=${1:-42}
QUANT=${2:-nvfp4-prequant}
R=/workspace/ltx2-benchmark
M=$R/models/ltx-2.5
OUT=/workspace/results/step7
mkdir -p "$OUT"
PROMPT="A cinematic shot of a futuristic city at sunset, flying vehicles moving between skyscrapers, realistic lighting, smooth camera movement"
STAMP=$(date +%Y%m%d-%H%M%S)
TAG="official_t2v_5s_${QUANT}_seed${SEED}_${STAMP}"
if [ "$QUANT" = "bf16" ]; then
  TRANSFORMER=$M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors; QARGS=()
else
  TRANSFORMER=$M/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors; QARGS=(--quantization "$QUANT")
fi
# VRAM sampler
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > "$OUT/$TAG.vram.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
cd "$R"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
START=$(date +%s.%N)
/usr/bin/time -v uv run python -m ltx_pipelines.distilled \
  --transformer-path "$TRANSFORMER" \
  --text-encoder-path "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$M/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --duration-head-path "$M/model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  --spatial-upsampler-path "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  "${QARGS[@]}" \
  --prompt "$PROMPT" \
  --seed "$SEED" \
  --width 1280 --height 704 --num-frames 121 --frame-rate 24 \
  --output-path "$OUT/$TAG.mp4" 2> "$OUT/$TAG.stderr.log" | tee "$OUT/$TAG.stdout.log"
END=$(date +%s.%N)
kill $SMI 2>/dev/null || true
WALL=$(python3 -c "print(round($END-$START,1))")
MAXRSS_KB=$(grep "Maximum resident set size" "$OUT/$TAG.stderr.log" | awk '{print $NF}')
VRAM_PEAK=$(cut -d, -f2 "$OUT/$TAG.vram.csv" | sort -n | tail -1)
VRAM_MEAN=$(cut -d, -f2 "$OUT/$TAG.vram.csv" | awk '{s+=$1;n++} END {if(n) printf "%d", s/n}')
PROBE=$(ffprobe -v error -show_entries stream=width,height,avg_frame_rate,nb_frames,codec_type:format=duration -of json "$OUT/$TAG.mp4")
python3 - "$OUT/$TAG.json" "$TAG" "$WALL" "$MAXRSS_KB" "$VRAM_PEAK" "$VRAM_MEAN" "$SEED" "$QUANT" <<'PY' "$PROBE"
import json, sys
out, tag, wall, rss, vpeak, vmean, seed, quant, probe = sys.argv[1:10]
p = json.loads(probe)
v = next(s for s in p["streams"] if s["codec_type"] == "video")
a = [s for s in p["streams"] if s["codec_type"] == "audio"]
num, den = v["avg_frame_rate"].split("/")
rec = {
    "tag": tag, "pipeline": "ltx_pipelines.distilled @400fd31", "quantization": quant, "seed": int(seed),
    "prompt_words": 20, "requested": {"width": 1280, "height": 704, "num_frames": 121, "fps": 24},
    "wall_seconds": float(wall), "max_rss_mib": round(int(rss) / 1024) if rss else None,
    "vram_peak_mib": int(vpeak) if vpeak else None, "vram_mean_mib": int(vmean) if vmean else None,
    "output": {"width": v["width"], "height": v["height"], "fps": round(int(num) / int(den), 3),
               "frames": int(v.get("nb_frames") or 0), "duration": float(p["format"]["duration"]),
               "audio_streams": len(a)},
}
json.dump(rec, open(out, "w"), indent=2)
print(json.dumps(rec, indent=2))
PY
echo "STEP7_DONE $TAG"
