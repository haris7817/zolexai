#!/usr/bin/env bash
set -euxo pipefail
cd /workspace/ltx2-benchmark
export UV_LINK_MODE=copy
CU13=/workspace/ltx2-benchmark/.venv/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME="$CU13" CUDA_PATH="$CU13" PATH="$CU13/bin:$PATH" TORCH_CUDA_ARCH_LIST="12.0"
nvcc --version | tail -1
uv sync --extra natten --group dev --group kernels
uv run python -c "import ltx_kernels;print(\"ltx_kernels OK\", ltx_kernels.__file__)"
uv run python -c "import natten;print(\"natten\", natten.__version__)" || echo "natten import failed"
uv run python -c "import torch;print(\"torch\",torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())"
echo LTX_KERNELS_DONE
