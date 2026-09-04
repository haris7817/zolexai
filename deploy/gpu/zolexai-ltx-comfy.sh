#!/usr/bin/env bash
# Launches the LTX 2.5 ComfyUI instance (docs/internal/ltx-comfy-runtime.md).
#
# Own checkout, own venv, own port. Nothing here touches the H3 instance.
# Loopback only: the worker on the same node is the only client.
set -euo pipefail

COMFY_DIR="${LTX_COMFY_DIR:-/workspace/ComfyUI-ltx}"
PORT="${LTX_COMFY_PORT:-8189}"

cd "$COMFY_DIR"
exec "$COMFY_DIR/.venv/bin/python" main.py \
  --listen 127.0.0.1 \
  --port "$PORT" \
  --disable-auto-launch \
  --preview-method none
