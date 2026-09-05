#!/usr/bin/env bash
set -euxo pipefail
export UV_LINK_MODE=copy
cd /workspace/zolexai/apps/worker
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
python -c "import worker.comfy.ltx_graphs as g; print(\"worker import ok\")"
echo WORKER_ENV_DONE
