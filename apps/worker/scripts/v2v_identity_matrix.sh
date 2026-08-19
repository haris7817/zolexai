#!/usr/bin/env bash
# The reference-identity benchmark matrix — the measurement that is allowed to
# turn `v2v_reference_identity` on, and to set its three defaults.
#
# Nothing in the identity mode is enabled from theory: the conditioning
# strengths, the attention weighting, and the answer to "is this good enough
# to ship" are all judgements to be made against real footage on the real
# card. This script runs the controlled comparisons, keeps every clip, and
# prints the timing/preservation tables; identity itself still needs an eye,
# which is what the kept files and exported stills are for.
#
# Fixtures are env vars; tests whose fixture is unset are skipped, so a first
# session with one talking-head clip is still useful:
#
#   REFERENCE=/fixtures/new-person.png \
#   SPEECH=/fixtures/talking-10s.mp4 \        # A, B and the sweep
#   SILENT=/fixtures/moving-nospeech.mp4 \    # C
#   LONG=/fixtures/talking-45s.mp4 \          # D  (30-60s)
#   CUTS=/fixtures/hard-cuts.mp4 \            # E
#   PROFILE=/fixtures/head-turn.mp4 \         # F
#   OCCLUSION=/fixtures/hand-over-face.mp4 \  # G
#     ./scripts/v2v_identity_matrix.sh "keep the performance, use the person from the reference"
#
# Per take: wall time, then `av_offset_probe.py` (lip-sync timing: constant
# offset vs cumulative drift, video vs mux) and `drift_check.sh` (motion/
# camera preservation). Stills at 0/25/50/75/100% land next to each clip for
# identity, flicker and long-form-reset review. SWEEP=1 adds the
# subject-attention x refresh-strength grid on the SPEECH clip.

set -uo pipefail

REFERENCE="${REFERENCE:?REFERENCE=/path/to/person.png is required}"
PROMPT="${*:-keep the original performance and camera movement, use the person from the reference image}"
PY="${PY:-/workspace/zolexai/.venv-worker/bin/python}"
OUTDIR="${OUTDIR:-/tmp/v2v-identity-matrix}"
export LTX_REPO_DIR="${LTX_REPO_DIR:-/workspace/ltx2-benchmark}"

[ -f "$REFERENCE" ] || { echo "reference not found: $REFERENCE" >&2; exit 2; }
mkdir -p "$OUTDIR"
cd "$(dirname "$0")/.."

IDENTITY_EXEC='"v2v_engine": "transform", "v2v_reference_identity": true'
BASELINE_EXEC='"v2v_engine": "transform"'

vram_peak() {  # background sampler; prints peak MiB when killed
  local peak=0 used
  while sleep 2; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -gt "$peak" ] && peak=$used && echo "$peak" > "$1"
  done
}

stills_of() {  # output.mp4 dest-dir — 0/25/50/75/100% for identity review
  local clip=$1 dir=$2 dur
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$clip")
  mkdir -p "$dir"
  for pct in 0 25 50 75 99; do
    ffmpeg -v error -y -ss "$(echo "$dur * $pct / 100" | bc -l)" -i "$clip" \
      -frames:v 1 "$dir/at-${pct}pct.png"
  done
}

take() {  # label source with_reference extra_exec_json
  local label=$1 source=$2 with_ref=$3 extra=${4:-}
  local exec_json="{$BASELINE_EXEC}"
  [ "$with_ref" = "yes" ] && exec_json="{$IDENTITY_EXEC${extra:+, $extra}}"

  printf '%-28s' "$label"
  local log="$OUTDIR/$label.log" started peakfile="$OUTDIR/$label.vram"
  started=$(date +%s)
  vram_peak "$peakfile" & local sampler=$!

  local ref_env=()
  [ "$with_ref" = "yes" ] && ref_env=(REFERENCE="$REFERENCE")
  if env "${ref_env[@]}" EXECUTION="$exec_json" MODE=restyle VIDEO="$source" \
       "$PY" scripts/ltx_smoke.py "$PROMPT" > "$log" 2>&1
  then
    kill "$sampler" 2>/dev/null
    local src dest="$OUTDIR/$label"
    src=$(grep -m1 '^file:' "$log" | awk '{print $2}')
    mkdir -p "$dest" && cp "$src" "$dest/output.mp4"
    stills_of "$dest/output.mp4" "$dest/stills"
    echo "✅ $(( $(date +%s) - started ))s  peak $(cat "$peakfile" 2>/dev/null || echo '?') MiB  $dest/"
    echo "--- $label: lip-sync timing vs source ---"
    "$PY" scripts/av_offset_probe.py "$source" "$dest/output.mp4" || true
    echo
  else
    kill "$sampler" 2>/dev/null
    echo "❌ $(( $(date +%s) - started ))s  see $log"
  fi
}

echo "reference: $REFERENCE"
echo "prompt:    $PROMPT"
echo

# ── The matrix ───────────────────────────────────────────────────────────
if [ -n "${SPEECH:-}" ]; then
  take "A-baseline-no-reference"  "$SPEECH" no
  take "B-speech-with-reference"  "$SPEECH" yes
fi
[ -n "${SILENT:-}" ]    && take "C-silent-with-reference"    "$SILENT" yes
[ -n "${LONG:-}" ]      && take "D-longform-with-reference"  "$LONG" yes
[ -n "${CUTS:-}" ]      && take "E-cuts-with-reference"      "$CUTS" yes
[ -n "${PROFILE:-}" ]   && take "F-profile-with-reference"   "$PROFILE" yes
[ -n "${OCCLUSION:-}" ] && take "G-occlusion-with-reference" "$OCCLUSION" yes

# ── The strength sweep (opt-in — 9 extra renders) ────────────────────────
if [ -n "${SWEEP:-}" ] && [ -n "${SPEECH:-}" ]; then
  for attention in 0.35 0.5 0.65; do
    for refresh in 0.2 0.35 0.5; do
      take "sweep-att${attention}-ref${refresh}" "$SPEECH" yes \
        "\"v2v_identity_subject_attention\": $attention, \"v2v_identity_refresh_strength\": $refresh"
    done
  done
fi

# ── Motion/camera preservation, side by side ─────────────────────────────
if [ -n "${SPEECH:-}" ] && [ -d "$OUTDIR/A-baseline-no-reference" ]; then
  echo "=== motion preservation (SSIM shape vs source; compare B against A) ==="
  ./scripts/drift_check.sh "$SPEECH" "$OUTDIR"/*/output.mp4 || true
fi

cat <<'EOF'

How to read this:
  * A vs B av_offset tables — if B's timing is no worse than A's, reference
    conditioning does not cost lip-sync; if both show a GROWING offset the
    stitching fix has regressed; a flat nonzero line on both is a mux offset.
  * B/D stills at 0..99% — the identity question itself: same person as the
    reference throughout, no return to the source person at late sections,
    no flicker between neighbouring stills.
  * drift_check — B should hold a line comparable to A; a cliff means the
    identity conditioning is displacing the source's structure.
  * The sweep — pick the LOWEST subject attention that still tracks motion
    and the LOWEST refresh strength that holds identity to the last section;
    those become the shipped defaults, recorded in video-to-video.yaml.
EOF
