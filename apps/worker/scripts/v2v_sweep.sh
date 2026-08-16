#!/usr/bin/env bash
# How hard should the source pull, now that it pulls more often?
#
# Conditioning density and conditioning strength are not independent. The
# restyle used three source stills at 0.7; raising the density to one still
# every four seconds fixed the drift (the subject stopped leaving the video)
# and immediately cost the style — eight stills at 0.7 is far more total pull
# than three at 0.7, and the prompt loses. Distributing the same influence
# means each anchor has to pull less.
#
# There is no arithmetic for "the same influence" — an anchor is not a blend —
# so it is measured. Every take here uses the SAME source, prompt and density,
# and varies only the strength.
#
#   VIDEO=/workspace/fixtures/client-clip.mp4 ./scripts/v2v_sweep.sh "as a vivid oil painting"
#
# ~90s per take. Prints the drift table across all of them at the end; the
# style itself still needs an eye, which is what the kept file paths are for.

set -uo pipefail

VIDEO="${VIDEO:?VIDEO=/path/to/source.mp4 is required}"
PROMPT="${*:-the same scene as a vivid oil painting, thick visible brushstrokes}"
STRENGTHS="${STRENGTHS:-0.7 0.55 0.4 0.25}"
PY="${PY:-/workspace/zolexai/.venv-worker/bin/python}"
OUTDIR="${OUTDIR:-/tmp/v2v-sweep}"

export LTX_REPO_DIR="${LTX_REPO_DIR:-/workspace/ltx2-benchmark}"

mkdir -p "$OUTDIR"
[ -f "$VIDEO" ] || { echo "source not found: $VIDEO" >&2; exit 2; }

echo "source: $VIDEO"
echo "prompt: $PROMPT"
echo "strengths: $STRENGTHS"
echo

outputs=()
for strength in $STRENGTHS; do
  label="strength-$strength"
  printf '%-18s' "$label"
  started=$(date +%s)

  log="$OUTDIR/$label.log"
  if EXECUTION="{\"v2v_structure_strength\": $strength}" \
     MODE=restyle VIDEO="$VIDEO" "$PY" scripts/ltx_smoke.py "$PROMPT" > "$log" 2>&1
  then
    src=$(grep -m1 '^file:' "$log" | awk '{print $2}')
    # Copied out of the smoke test's throwaway workspace, so the whole sweep
    # is still on disk to look at once the table below says which to look at.
    dest="$OUTDIR/$label.mp4"
    cp "$src" "$dest" && outputs+=("$dest")
    echo "✅ $(( $(date +%s) - started ))s  $dest"
  else
    echo "❌ $(( $(date +%s) - started ))s  see $log"
  fi
done

echo
[ ${#outputs[@]} -gt 0 ] || { echo "no takes to compare" >&2; exit 1; }

# drift_check labels each column by its file's PARENT directory, which would
# make every take here read "v2v-sweep". One directory per take fixes that.
staged=()
for output in "${outputs[@]}"; do
  name=$(basename "$output" .mp4)
  mkdir -p "$OUTDIR/$name"
  cp "$output" "$OUTDIR/$name/output.mp4"
  staged+=("$OUTDIR/$name/output.mp4")
done

./scripts/drift_check.sh "$VIDEO" "${staged[@]}"

echo "  Read it with the clips, not instead of them: the highest mean is the"
echo "  take that followed your upload most closely, which at some point stops"
echo "  being a restyle at all. Look for the LOWEST strength that still holds."
echo
