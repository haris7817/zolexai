#!/usr/bin/env bash
# Launches the LTX 2.5 ComfyUI instance (docs/internal/ltx-comfy-runtime.md).
#
# Own checkout, own venv, own port. Nothing here touches the H3 instance.
# Loopback only: the worker on the same node is the only client.
#
# NO accelerated attention by default (client decision, 8 Sep 2026). The
# SageAttention work measured 1.59x at 30 s with every metric we could put a
# number on unchanged, but the client judged the resulting videos worse and
# asked to go back to the stock kernel and the ~17 min / 30 s render. Their
# call, and it is the one this script encodes. Measurements are kept in
# docs/internal/ltx25_speed_optimization_report.md so nobody re-runs them.
#
# The extra-args file stays: it is how a flag gets tried, or restored, without
# editing this script (runbook §47).
#   echo --use-sage-attention > /workspace/comfy_extra_args   # the 1.59x path
#   echo --use-ck-attention   > /workspace/comfy_extra_args   # no-build variant
#   : > /workspace/comfy_extra_args                           # stock (default)
# then: supervisorctl restart zolexai-ltx-comfy
set -euo pipefail

COMFY_DIR="${LTX_COMFY_DIR:-/workspace/ComfyUI-ltx}"
PORT="${LTX_COMFY_PORT:-8189}"
EXTRA_FILE="${LTX_COMFY_EXTRA_ARGS_FILE:-/workspace/comfy_extra_args}"

# Stock by default; a non-empty extra-args file replaces this wholesale.
EXTRA=""
if [ -r "$EXTRA_FILE" ]; then
  # `sed`, not `grep -v`: on an empty file grep exits 1, which under
  # `set -e -o pipefail` killed this script and left ComfyUI FATAL with no
  # useful log. sed returns 0 on empty input. Found the hard way, 8 Sep 2026.
  FROM_FILE="$(sed -e 's/#.*//' "$EXTRA_FILE" \
    | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/^ //' -e 's/ $//')"
  if [ -n "$FROM_FILE" ]; then
    EXTRA="$FROM_FILE"
  fi
  # This script's own word for "the stock launch line", kept so the file can
  # say so explicitly rather than by being empty.
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
