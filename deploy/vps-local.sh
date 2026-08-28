#!/usr/bin/env bash
#
# The production VPS's local workflow-definition edits, applied by script.
#
# ## Why this file exists
#
# The shipped YAML ships `runtime: mock` plus a pair of mock-output lines, and
# production has always edited six files by hand to say otherwise. Those edits
# lived nowhere but in the deployed working tree, which made every deploy a
# `git stash` / `git pull` / `git stash pop` with conflicts resolved from
# memory at whatever hour the deploy happened.
#
# It failed. On or before 28 Aug 2026 a stash pop restored
# `output_content_type: image/png` to files whose runtime was real, so the API
# presigned every upload as a PNG: 64 finished videos went out with the wrong
# content type and would not play. The customer's report was "I can't download
# anything", and it had been true for weeks. The same evening two more pops
# failed outright and left the checkout with unmerged paths.
#
# So the edits are written down. `git reset --hard && git pull && bash
# deploy/vps-local.sh` replaces the dance, resolves nothing by hand, and is
# idempotent — running it twice changes nothing the second time.
#
# ## What it does NOT do
#
# It does not touch `.env`, compose files, or anything untracked. It edits only
# `workflow-definitions/*.yaml`, and only the two things production changes:
# the runtime block, and the deletion of the mock-output lines.
#
# Usage, from the repository root:
#
#     bash deploy/vps-local.sh          # apply
#     bash deploy/vps-local.sh --check  # verify only, exit 1 if wrong
#
set -euo pipefail

cd "$(dirname "$0")/.."
DEFS=workflow-definitions
CHECK_ONLY=${1:-}

if [ ! -d "$DEFS" ]; then
  echo "error: run from the repository root (no $DEFS/)" >&2
  exit 1
fi

# ── The runtime block each workflow gets, replacing `  runtime: mock` ─────
#
# Anything not named here keeps the committed value. Keys that exist only in
# production (engine routing, timeouts, tuning) live in this file and nowhere
# else, which is the whole point.

block_extend_video() {
  cat <<'YAML'
  runtime: ltx
YAML
}

block_image_to_video() {
  # No `runtime_by_quality`: Image to Video dropped its quality toggle on
  # 28 Aug 2026 at the client's request, so no quality parameter is sent and
  # the base runtime serves every job. `h3_max_seconds` went with it — it
  # bounded a lattice nothing routes to any more.
  cat <<'YAML'
  runtime: ltx
  timeout_seconds: 5400
  prompt_structuring_v2: true
YAML
}

block_music_video() {
  # `require_audio_conditioning` is a guard, not a setting: it refuses the
  # prompt-only + post-mux route outright, so a config slip cannot ship a
  # music video whose mouth was never told about the song.
  cat <<'YAML'
  runtime: ltx
  audio_conditioning: true
  require_audio_conditioning: true
  inference_steps: 15
YAML
}

block_music() {
  cat <<'YAML'
  runtime: music
YAML
}

block_text_to_video() {
  cat <<'YAML'
  runtime: ltx
  runtime_by_quality:
    fast: ltx
    best: h3_comfy
  timeout_seconds: 5400
  h3_max_seconds: 30
  h3_steps: 12
  prompt_structuring_v2: true
YAML
}

block_video_to_video() {
  # Both levels on LTX since 28 Aug 2026: the other engine reads stills and
  # returns a different video, which is not what this workflow sells. See the
  # note in the committed YAML. `h3_tier` is gone with it — dead config for a
  # workflow that no longer reaches that engine.
  cat <<'YAML'
  runtime: ltx
  runtime_by_quality:
    fast: ltx
    best: ltx
YAML
}

# ── Mechanics ────────────────────────────────────────────────────────────

apply_one() {
  local name="$1" file="$DEFS/$1.yaml" tmp
  [ -f "$file" ] || { echo "error: missing $file" >&2; exit 1; }

  tmp=$(mktemp)
  "block_${name//-/_}" > "$tmp"

  # Replace the single `  runtime: mock` line with the block. Idempotent: a
  # file already carrying a real runtime has no such line and is left alone.
  if grep -q '^  runtime: mock$' "$file"; then
    sed -i -e "/^  runtime: mock\$/{r $tmp" -e 'd}' "$file"
  fi

  # The mock runtime's output declaration. Deleting these is what makes the
  # API presign uploads for the media the worker actually produces; leaving
  # them on a real runtime is the 64-video incident.
  sed -i -e '/^  output_content_type: image\/png$/d' \
         -e '/^  output_kind: image$/d' "$file"

  rm -f "$tmp"
}

check_one() {
  local file="$DEFS/$1.yaml" bad=0
  if grep -q '^  runtime: mock$' "$file"; then
    echo "  $1: still on the mock runtime"; bad=1
  fi
  if grep -q '^  output_content_type:\|^  output_kind:' "$file"; then
    echo "  $1: mock-output lines present — uploads will be signed as PNG"; bad=1
  fi
  return $bad
}

WORKFLOWS=(extend-video image-to-video music-video music text-to-video video-to-video)

if [ "$CHECK_ONLY" = "--check" ]; then
  echo "checking $DEFS/ ..."
  failed=0
  for w in "${WORKFLOWS[@]}"; do check_one "$w" || failed=1; done
  if [ "$failed" -eq 0 ]; then echo "all six workflows carry their production runtime"; fi
  exit "$failed"
fi

for w in "${WORKFLOWS[@]}"; do apply_one "$w"; done

echo "applied. verifying:"
failed=0
for w in "${WORKFLOWS[@]}"; do check_one "$w" || failed=1; done
if [ "$failed" -ne 0 ]; then
  echo "VERIFY FAILED — do not build" >&2
  exit 1
fi

echo
grep -n '^  runtime:' "$DEFS"/*.yaml
echo
echo "mock-output lines remaining (must be none):"
grep -n 'output_content_type\|output_kind' "$DEFS"/*.yaml || echo "  none"
