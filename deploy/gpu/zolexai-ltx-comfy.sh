#!/usr/bin/env bash
# Launches the LTX 2.5 ComfyUI instance (docs/internal/ltx-comfy-runtime.md).
#
# Own checkout, own venv, own port. Nothing here touches the H3 instance.
# Loopback only: the worker on the same node is the only client.
#
# SageAttention (8 Sep 2026, docs/internal/ltx25_speed_optimization_report.md)
# is the whole speed result: an accelerated attention kernel for Blackwell,
# measured 1.59x end to end on the client's FAST 1080 graph at 30 s with the
# graph, the schedule, the transformer and the resolution all untouched, and
# detail, brightness, motion and audio measured unchanged. It is a server
# flag, not a workflow edit — nothing about the client's graph changes.
#
# Rollback and overrides live in one file, so neither needs this script
# edited (runbook §47):
#   echo --no-sage          > /workspace/comfy_extra_args   # stock launch line
#   echo --use-ck-attention > /workspace/comfy_extra_args   # no-build fallback
#   : > /workspace/comfy_extra_args                         # back to the default
# then: supervisorctl restart zolexai-ltx-comfy
set -euo pipefail

COMFY_DIR="${LTX_COMFY_DIR:-/workspace/ComfyUI-ltx}"
PORT="${LTX_COMFY_PORT:-8189}"
EXTRA_FILE="${LTX_COMFY_EXTRA_ARGS_FILE:-/workspace/comfy_extra_args}"

# The default, overridden wholesale by a non-empty extra-args file.
EXTRA="--use-sage-attention"
if [ -r "$EXTRA_FILE" ]; then
  # `sed`, not `grep -v`: on an empty file grep exits 1, which under
  # `set -e -o pipefail` killed this script and left ComfyUI FATAL with no
  # useful log. sed returns 0 on empty input. Found the hard way, 8 Sep 2026.
  FROM_FILE="$(sed -e 's/#.*//' "$EXTRA_FILE" \
    | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/^ //' -e 's/ $//')"
  if [ -n "$FROM_FILE" ]; then
    EXTRA="$FROM_FILE"
  fi
  # This script's own word for "the stock launch line, no extra flags".
  if [ "$FROM_FILE" = "--no-sage" ]; then
    EXTRA=""
  fi
fi

cd "$COMFY_DIR"
# shellcheck disable=SC2086
exec "$COMFY_DIR/.venv/bin/python" main.py \
  --listen 127.0.0.1 \
  --port "$PORT" \
  --disable-auto-launch \
  --preview-method none $EXTRA
