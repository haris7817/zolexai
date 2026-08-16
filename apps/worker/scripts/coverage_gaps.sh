#!/usr/bin/env bash
# The coverage `ltx_matrix.sh` structurally cannot reach.
#
# Three gaps, each real:
#
#   1. MUSIC GENERATION runs on a different runtime entirely — an HTTP service
#      holding its own weights, not the LTX subprocess. The LTX matrix never
#      touches it, so "the matrix passed" says nothing about music.
#
#   2. CHAINING IS NEVER EXERCISED by the matrix. Its fixtures are 8s and 20s,
#      both under the 60s ceiling, so video-to-video and music-video both run
#      single-pass there. The client's actual use case — a 2-4 minute song — is
#      the chained path, and it is the one where seams, section prompts and
#      identity drift still exist. Passing single-pass proves nothing about it.
#
#   3. VIDEO-TO-VIDEO IS TESTED ON ONE ASPECT. Its grid follows the SOURCE, not
#      the request, so a portrait upload renders at a different shape — and
#      shapes are exactly how the kernel failure presents.
#
# Long fixtures are built from the box's own output, so nothing is uploaded.
#
#   ./scripts/coverage_gaps.sh
#
# ~20 minutes. Run it after ltx_matrix.sh, not instead of it.

set -uo pipefail

OUT="${OUT:-/tmp/coverage-gaps-$(date +%Y%m%d-%H%M).md}"
LOGDIR="${LOGDIR:-/tmp/coverage-gaps-logs}"
PY="${PY:-/workspace/zolexai/.venv-worker/bin/python}"
FIX="${FIX:-/workspace/fixtures}"

export LTX_REPO_DIR="${LTX_REPO_DIR:-/workspace/ltx2-benchmark}"
export LTX_QUANTIZATION="${LTX_QUANTIZATION:-nvfp4-prequant}"
export LTX_MAX_SECONDS="${LTX_MAX_SECONDS:-60}"
export ACESTEP_BASE_URL="${ACESTEP_BASE_URL:-http://127.0.0.1:8001}"

mkdir -p "$LOGDIR" "$FIX"

row () { echo "| $1 | $2 | $3 | $4 | $5 |" >> "$OUT"; }

{
  echo "# Coverage gaps — $(date -Iseconds)"
  echo
  echo "What ltx_matrix.sh cannot cover: the music runtime, the chained path,"
  echo "and video-to-video on a non-16:9 source."
  echo
  echo "| area | case | result | wall | detail |"
  echo "|---|---|---|---:|---|"
} >> "$OUT"

# ── Long fixtures, built from the box's own renders ──────────────────────
echo "=== building long fixtures ==="

# A >60s video forces chaining. Today's 60s probe renders are the raw material;
# concatenating one with itself gives 120s, which is two passes at the ceiling.
if [ ! -f "$FIX/long.mp4" ]; then
  SRC=$(ls -t /tmp/grid-1024x576.mp4 /tmp/ltx-smoke-*/output.mp4 2>/dev/null | head -1)
  if [ -n "${SRC:-}" ]; then
    printf "file '%s'\nfile '%s'\n" "$SRC" "$SRC" > /tmp/concat.txt
    ffmpeg -v error -y -f concat -safe 0 -i /tmp/concat.txt -c copy "$FIX/long.mp4" \
      && echo "  long.mp4    $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$FIX/long.mp4")s"
  else
    echo "  long.mp4    SKIPPED — no 60s render found under /tmp"
  fi
fi

# A portrait source. The 9:16 grid probe left exactly this.
if [ ! -f "$FIX/portrait.mp4" ] && [ -f /tmp/grid-576x1024.mp4 ]; then
  ffmpeg -v error -y -i /tmp/grid-576x1024.mp4 -t 8 -c:v libx264 -preset veryfast \
    -crf 18 -c:a aac -movflags +faststart "$FIX/portrait.mp4" \
    && echo "  portrait.mp4 $(ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x "$FIX/portrait.mp4" | head -1)"
fi

# A >60s track, so music-video chains the way a real song does.
if [ ! -f "$FIX/long.mp3" ] && [ -f "$FIX/long.mp4" ]; then
  ffmpeg -v error -y -i "$FIX/long.mp4" -t 100 -vn -c:a libmp3lame -b:a 192k "$FIX/long.mp3" \
    && echo "  long.mp3    $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$FIX/long.mp3")s"
fi
echo

# ── 1. Music generation, every offered length ────────────────────────────
# Cost tracks step count rather than duration, so a 5m song is barely slower
# than a 1m one. All five are affordable and the range is the product's.
echo "### music generation"
for d in 1m 2m 3m 4m 5m; do
  T="$LOGDIR/music-$d.log"; printf '%-22s' "music $d"
  S=$(date +%s)
  if DURATION=$d "$PY" scripts/music_smoke.py "an upbeat pop song about summer in Lahore" > "$T" 2>&1; then
    W=$(( $(date +%s)-S ))
    echo "✅ PASS ${W}s  $(grep -m1 -oE 'duration: *[0-9.]+' "$T")"
    row music "$d" "✅ PASS" "${W}s" "$(grep -m1 -oE 'duration: *[0-9.]+s?' "$T")"
  else
    W=$(( $(date +%s)-S ))
    echo "❌ FAIL ${W}s"
    row music "$d" "❌ FAIL" "${W}s" "$(grep -ioE 'Error[^|]{0,80}' "$T" | tail -1)"
  fi
done
echo

# ── 2 & 3. The chained path, and V2V on a portrait source ────────────────
cell () {
  local label=$1 mode=$2 tag=$3; shift 3
  local T="$LOGDIR/$tag.log"
  printf '%-22s' "$tag"
  local S; S=$(date +%s)
  if env "$@" MODE="$mode" "$PY" scripts/ltx_smoke.py \
       "a koi pond at dawn, slow cinematic push in" > "$T" 2>&1; then
    local W=$(( $(date +%s)-S ))
    local F; F=$(grep -m1 '^frame:' "$T" | awk '{print $2}')
    local D; D=$(grep -m1 '^duration:' "$T" | awk '{print $2}')
    echo "✅ PASS ${W}s  $F  $D"
    row "$label" "$tag" "✅ PASS" "${W}s" "$F $D"
  else
    local W=$(( $(date +%s)-S ))
    local E; E=$(grep -ioE 'CUBLAS_STATUS_[A-Z_]+|illegal memory access|invalid argument|no audio stream' "$T" | head -1)
    echo "❌ FAIL ${W}s  ${E:-see $T}"
    row "$label" "$tag" "❌ FAIL" "${W}s" "${E:-see $T}"
  fi
}

echo "### chained path (source longer than one pass)"
[ -f "$FIX/long.mp4" ] && cell chaining restyle     v2v-120s-chained   VIDEO="$FIX/long.mp4"
[ -f "$FIX/long.mp3" ] && cell chaining music-video mv-100s-chained    AUDIO="$FIX/long.mp3" ASPECT_RATIO=16:9
echo
echo "### video-to-video on a non-16:9 source"
[ -f "$FIX/portrait.mp4" ] && cell aspect restyle   v2v-portrait       VIDEO="$FIX/portrait.mp4"

echo
echo "=========================================="
echo "done — $OUT"
grep -c '✅' "$OUT" | xargs printf 'passed: %s\n'
grep -c '❌' "$OUT" | xargs printf 'failed: %s\n'
echo
echo "The chained cells are the ones that matter: they are the only place"
echo "seams, section prompts and boundary drift can still occur."
