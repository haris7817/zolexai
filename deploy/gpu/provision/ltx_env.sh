#!/usr/bin/env bash
# Official Lightricks LTX-2 stack at the audited pin, with natten + ltx-kernels (NVFP4).
set -euxo pipefail
cd /workspace/ltx2-benchmark
git log --oneline -1
export UV_LINK_MODE=copy
# 1. torch 2.13 cu132 + natten (repo-pinned), plus the dev group which carries nvcc/cccl 13.2 in-venv.
uv sync --extra natten --group dev
PY=$(ls -d .venv/lib/python3.*/site-packages | head -1)
CU13=$(ls -d "$PY"/nvidia/cu13 2>/dev/null || true)
echo "venv site: $PY  cu13: $CU13"
uv run python -c "import torch;print(\"torch\",torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_capability())"
# 2. ltx-kernels for sm_120 with the venv toolchain (system nvcc is 12.8; torch is cu132).
if [ -n "$CU13" ]; then export CUDA_HOME="$CU13" CUDA_PATH="$CU13" PATH="$CU13/bin:$PATH"; fi
export TORCH_CUDA_ARCH_LIST="12.0"
nvcc --version | tail -1 || true
uv sync --extra natten --group dev --group kernels
uv run python -c "import ltx_kernels;print(\"ltx_kernels OK\", ltx_kernels.__file__)"
uv run python -c "import natten;print(\"natten\", natten.__version__)" || echo "natten import failed"
echo LTX_ENV_DONE
