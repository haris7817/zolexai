#!/usr/bin/env bash
# Exercise the product's ACTUAL parameter matrix against the real GPU.
#
# Why this exists
# ---------------
# The RTX PRO 6000 migration was validated with ONE 10-second 16:9 clip. It
# passed, the box was declared good, and customers hit failures for hours. Each
# subsequent fix was also wrong: LTX_MAX_SECONDS=20 failed on chained passes, 15
# failed on portrait, only 10 survived. The failure is shape-dependent by
# construction and no single sample can see it.
#
# Worse, the shapes that fail follow no rule we can predict. Measured 16 Aug on
# 60s renders: 1024x576 PASSES, 896x512 FAILS, 1152x640 FAILS. A larger grid
# passing where a smaller one fails rules out every simple "budget" model. So
# every cell is measured. Nothing is extrapolated.
#
# Usage
# -----
#   cd /workspace/zolexai/apps/worker
#   TIER=1 ./scripts/ltx_matrix.sh      # risky cells only, ~40 min
#   TIER=2 ./scripts/ltx_matrix.sh      # everything, ~2-3 h
#
# Fixtures (only needed for the conditioned modes):
#   FIXTURE_IMAGE=/workspace/fixtures/still.png
#   FIXTURE_VIDEO=/workspace/fixtures/clip.mp4
#   FIXTURE_AUDIO=/workspace/fixtures/song.mp3
#
# Results are appended to $OUT as a markdown table as each cell finishes, so a
# run that is interrupted still leaves everything measured up to that point.
# Re-running skips cells whose log already exists — kill it and resume freely.

set -uo pipefail

TIER="${TIER:-1}"
OUT="${OUT:-/tmp/ltx-matrix-$(date +%Y%m%d-%H%M).md}"
LOGDIR="${LOGDIR:-/tmp/ltx-matrix-logs}"
PY="${PY:-/workspace/zolexai/.venv-worker/bin/python}"
PROMPT="${PROMPT:-a koi pond at dawn, slow cinematic push in}"

export LTX_REPO_DIR="${LTX_REPO_DIR:-/workspace/ltx2-benchmark}"
export LTX_QUANTIZATION="${LTX_QUANTIZATION:-nvfp4-prequant}"
# High enough that the adapter never chains. We are measuring the SINGLE-PASS
# ceiling; chaining is measured separately and deliberately.
export LTX_MAX_SECONDS="${LTX_MAX_SECONDS:-60}"

FIXTURE_IMAGE="${FIXTURE_IMAGE:-/workspace/fixtures/still.png}"
FIXTURE_VIDEO="${FIXTURE_VIDEO:-/workspace/fixtures/clip.mp4}"
FIXTURE_AUDIO="${FIXTURE_AUDIO:-/workspace/fixtures/song.mp3}"

mkdir -p "$LOGDIR"

ASPECTS=(16:9 9:16 1:1 4:5)
if [ "$TIER" = "1" ]; then
  DURATIONS=(60s)            # the cell most likely to fail
else
  DURATIONS=(5s 10s 15s 30s 60s)
fi

have () { [ -f "$1" ]; }

header () {
  {
    echo "# LTX matrix — $(date -Iseconds)"
    echo
    echo "- repo: \`$LTX_REPO_DIR\`"
    echo "- quantization: \`$LTX_QUANTIZATION\`"
    echo "- per-pass ceiling: \`$LTX_MAX_SECONDS\` (single-pass unless a cell says otherwise)"
    echo "- natten: \`$(uv --directory "$LTX_REPO_DIR" run python -c 'import natten;print(natten.__version__)' 2>/dev/null || echo ABSENT)\`"
    echo "- grids: \`$($PY -c 'from worker.adapters.ltx import _DIMENSIONS;print(_DIMENSIONS)' 2>/dev/null)\`"
    echo
    echo "| mode | aspect | duration | result | wall | frame | measured |"
    echo "|---|---|---|---|---:|---|---|"
  } >> "$OUT"
}

# cell MODE ASPECT DURATION [extra env assignments...]
cell () {
  local mode=$1 aspect=$2 duration=$3; shift 3
  local tag="${mode}-${aspect//:/x}-${duration}"
  local log="$LOGDIR/$tag.log"

  if [ -f "$log" ] && grep -q "SMOKE TEST PASSED" "$log"; then
    printf '%-34s %s\n' "$tag" "⏭  already passed, skipping"
    return
  fi

  printf '%-34s' "$tag"
  local start; start=$(date +%s)

  env "$@" MODE="$mode" ASPECT_RATIO="$aspect" DURATION="$duration" \
    "$PY" scripts/ltx_smoke.py "$PROMPT" > "$log" 2>&1
  local rc=$? ; local wall=$(( $(date +%s) - start ))

  local frame measured result
  if [ $rc -eq 0 ]; then
    frame=$(grep -m1 '^frame:'    "$log" | awk '{print $2}')
    measured=$(grep -m1 '^duration:' "$log" | awk '{print $2}')
    result="✅ PASS"
    printf '%s  %ss  %s\n' "$result" "$wall" "$frame"
  else
    frame='—'
    measured=$(grep -ioE 'CUBLAS_STATUS_[A-Z_]+|illegal memory access|invalid argument|out of memory' "$log" | head -1)
    [ -z "$measured" ] && measured=$(grep -oE '[A-Za-z_.]*Error[^|]{0,60}' "$log" | tail -1)
    result="❌ FAIL"
    printf '%s  %ss  %s\n' "$result" "$wall" "$measured"
  fi

  echo "| $mode | $aspect | $duration | $result | ${wall}s | $frame | ${measured:-—} |" >> "$OUT"
}

header
echo "writing results to $OUT"
echo "logs in $LOGDIR"
echo

# ── text to video — the only mode with no conditioning ───────────────────
echo "### text-to-video"
for a in "${ASPECTS[@]}"; do for d in "${DURATIONS[@]}"; do cell text "$a" "$d"; done; done

# ── image to video — conditioned from frame zero ─────────────────────────
if have "$FIXTURE_IMAGE"; then
  echo; echo "### image-to-video"
  for a in "${ASPECTS[@]}"; do for d in "${DURATIONS[@]}"; do
    cell image "$a" "$d" IMAGE="$FIXTURE_IMAGE"
  done; done
else
  echo "SKIP image-to-video — FIXTURE_IMAGE not found at $FIXTURE_IMAGE" | tee -a "$OUT"
fi

# ── extend — conditioned on the source's final frame ─────────────────────
if have "$FIXTURE_VIDEO"; then
  echo; echo "### extend-video"
  for a in "${ASPECTS[@]}"; do for d in "${DURATIONS[@]}"; do
    cell extend "$a" "$d" VIDEO="$FIXTURE_VIDEO"
  done; done
else
  echo "SKIP extend-video — FIXTURE_VIDEO not found at $FIXTURE_VIDEO" | tee -a "$OUT"
fi

# ── restyle and music-video take their length from the SOURCE ────────────
# Aspect and duration are ignored by these two, so one cell each is the whole
# matrix — pass a longer fixture to test a longer result.
if have "$FIXTURE_VIDEO"; then
  echo; echo "### video-to-video (duration from source)"
  cell restyle 16:9 source VIDEO="$FIXTURE_VIDEO"
fi
if have "$FIXTURE_AUDIO"; then
  echo; echo "### music-video (duration from track)"
  for a in "${ASPECTS[@]}"; do cell music-video "$a" source AUDIO="$FIXTURE_AUDIO"; done
fi

echo
echo "=========================================="
echo "done — $OUT"
grep -c '✅' "$OUT" | xargs printf 'passed: %s\n'
grep -c '❌' "$OUT" | xargs printf 'failed: %s\n'
