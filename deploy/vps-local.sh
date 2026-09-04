#!/usr/bin/env bash
#
# The deployment's local workflow-definition edits, applied by script.
#
# ## Why this file exists
#
# The shipped YAML ships `runtime: mock` plus a pair of mock-output lines, and
# a deployment edits the files to say otherwise. Those edits used to live
# nowhere but in the deployed working tree, which made every deploy a
# `git stash` / `git pull` / `git stash pop` with conflicts resolved from
# memory. It failed: on or before 28 Aug 2026 a stash pop restored
# `output_content_type: image/png` to files whose runtime was real, so the
# API presigned every upload as a PNG — 64 finished videos would not play.
#
# So the edits are written down. `git reset --hard && git pull && bash
# deploy/vps-local.sh --profile <name>` replaces the dance, resolves nothing by
# hand, and is idempotent.
#
# ## Profiles (final milestone, 5 Sep 2026)
#
#   production   What production runs today: the LTX CLI runtime for Text to
#                Video, First/Last Frame Video and Extend Video, Video to Video
#                untouched at both quality levels, music on ACE-Step, Character
#                Replacement present but hidden. NO H3 line anywhere — H3 is
#                hidden by client decision and the API refuses to boot on a
#                definition that routes to it while ENABLE_H3 is off.
#
#   client-test  The routing the GPU validation clears: Text to Video, First/
#                Last Frame Video and Extend Video on the client's LTX 2.5
#                ComfyUI graphs (`ltx_comfy`), Character Replacement visible on
#                its own runtime, everything else identical to production.
#                Apply this ONLY in the client-test environment, and only after
#                `scripts/ltx_comfy_health.py --deep` reports HEALTHY on the node.
#
# Rollback from client-test is `--profile production` plus the api/web
# rebuild; nothing else changes.
#
# ## What it does NOT do
#
# It does not touch `.env`, compose files, or anything untracked. It edits only
# `workflow-definitions/*.yaml`, and only three things: the runtime block, the
# `hidden:` line of Character Replacement, and the deletion of the mock-output
# lines.
#
# Usage, from the repository root:
#
#     bash deploy/vps-local.sh --profile production          # apply
#     bash deploy/vps-local.sh --profile client-test         # apply
#     bash deploy/vps-local.sh --profile production --check  # verify only
#
set -euo pipefail

cd "$(dirname "$0")/.."
DEFS=workflow-definitions

PROFILE=production
CHECK_ONLY=
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    *) echo "usage: $0 [--profile production|client-test] [--check]" >&2; exit 2 ;;
  esac
done
case "$PROFILE" in production|client-test) ;; *) echo "error: unknown profile '$PROFILE'" >&2; exit 2 ;; esac

if [ ! -d "$DEFS" ]; then
  echo "error: run from the repository root (no $DEFS/)" >&2
  exit 1
fi

# ── The runtime block each workflow gets, replacing `  runtime: mock` ─────
#
# Anything not named here keeps the committed value. Keys that exist only in
# a deployment (engine routing, timeouts, tuning) live in this file and
# nowhere else, which is the whole point.

video_runtime() {
  # The engine behind Text to Video, First/Last Frame Video and Extend Video.
  case "$PROFILE" in
    client-test) echo "ltx_comfy" ;;
    *) echo "ltx" ;;
  esac
}

block_extend_video() {
  cat <<YAML
  runtime: $(video_runtime)
YAML
}

block_image_to_video() {
  # `soundscape` carries the client's audio safety rule: no supplied dialogue
  # means no speech at all, and the soundtrack is the sounds the scene MAKES.
  # `prompt_structuring_v2` is the revised continuity block (CLI runtime).
  cat <<YAML
  runtime: $(video_runtime)
  timeout_seconds: 5400
  prompt_structuring_v2: true
YAML
}

block_music_video() {
  # UNCHANGED this milestone. `require_audio_conditioning` is a guard: it
  # refuses the prompt-only + post-mux route outright, so a config slip cannot
  # ship a music video whose mouth was never told about the song.
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
  # No quality levels since 5 Sep 2026, so no `runtime_by_quality` — and no H3.
  cat <<YAML
  runtime: $(video_runtime)
  timeout_seconds: 5400
  prompt_structuring_v2: true
YAML
}

block_video_to_video() {
  # UNTOUCHED (client rule): both levels on the CLI runtime since 28 Aug 2026;
  # Best adds reference identity through the committed execution_by_quality.
  cat <<'YAML'
  runtime: ltx
  runtime_by_quality:
    fast: ltx
    best: ltx
YAML
}

block_character_replacement() {
  cat <<'YAML'
  runtime: character_replacement
YAML
}

character_replacement_hidden() {
  case "$PROFILE" in
    client-test) echo "false" ;;
    *) echo "true" ;;
  esac
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

  if [ "$name" = "character-replacement" ]; then
    sed -i -e "s/^hidden: \(true\|false\)$/hidden: $(character_replacement_hidden)/" "$file"
  fi

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
  if grep -q 'h3_comfy' "$file"; then
    echo "  $1: routes to h3_comfy — H3 is hidden; the API will refuse to boot"; bad=1
  fi
  if [ "$1" = "character-replacement" ] && ! grep -q "^hidden: $(character_replacement_hidden)$" "$file"; then
    echo "  $1: hidden line does not match profile '$PROFILE'"; bad=1
  fi
  if [ "$1" = "video-to-video" ] && ! grep -q '^  v2v_engine: transform$' "$file"; then
    echo "  $1: the committed transform engine line is missing — this file must not be edited"; bad=1
  fi
  return $bad
}

WORKFLOWS=(extend-video image-to-video music-video music text-to-video video-to-video character-replacement)

if [ -n "$CHECK_ONLY" ]; then
  echo "checking $DEFS/ against profile '$PROFILE' ..."
  failed=0
  for w in "${WORKFLOWS[@]}"; do check_one "$w" || failed=1; done
  if [ "$failed" -eq 0 ]; then echo "all seven workflows carry the '$PROFILE' runtime block"; fi
  exit "$failed"
fi

for w in "${WORKFLOWS[@]}"; do apply_one "$w"; done

echo "applied profile '$PROFILE'. verifying:"
failed=0
for w in "${WORKFLOWS[@]}"; do check_one "$w" || failed=1; done
if [ "$failed" -ne 0 ]; then
  echo "VERIFY FAILED — do not build" >&2
  exit 1
fi

echo
grep -n '^  runtime:' "$DEFS"/*.yaml
grep -n '^hidden:' "$DEFS"/character-replacement.yaml
echo
echo "mock-output lines remaining (must be none):"
grep -n 'output_content_type\|output_kind' "$DEFS"/*.yaml || echo "  none"
echo
echo "h3 lines remaining (must be none):"
grep -n 'h3_comfy' "$DEFS"/*.yaml || echo "  none"
