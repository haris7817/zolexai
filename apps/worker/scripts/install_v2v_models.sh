#!/usr/bin/env bash
set -euo pipefail

LTX_REPO_DIR="${LTX_REPO_DIR:-/workspace/ltx2-benchmark}"
LTX_PYTHON="${LTX_PYTHON:-$LTX_REPO_DIR/.venv/bin/python}"
BIREFNET_DIR="${ZOLEXAI_BIREFNET_MODEL:-$LTX_REPO_DIR/models/BiRefNet}"

if [[ ! -x "$LTX_PYTHON" ]]; then
  echo "LTX Python was not found at $LTX_PYTHON" >&2
  exit 1
fi

# BiRefNet's published modelling file is remote code, and it imports `kornia`
# and `timm` at load time. Neither was in the list this script shipped with,
# so the first real run on the RTX PRO 6000 (10 Sep 2026) died with
# "requires the following packages that were not found in your environment".
# They are what makes the difference between this script working on a fresh
# node and failing at the first customer job.
PACKAGES=(
  "accelerate>=1.0,<2"
  "huggingface-hub>=0.27,<2"
  "kornia>=0.7,<1"
  "numpy>=1.26,<3"
  "pillow>=10,<13"
  "safetensors>=0.4,<1"
  "timm>=1.0,<2"
  "transformers>=4.47,<6"
)

# OpenCV only when the environment has none. The LTX checkout already carries
# `opencv-python`, and adding `opencv-python-headless` beside it puts two
# distributions behind the same `cv2` module — with a `<5` pin that contradicts
# the 5.0 already installed. Both provide everything these scripts use.
if ! "$LTX_PYTHON" -c "import cv2" >/dev/null 2>&1; then
  PACKAGES+=("opencv-python-headless>=4.10")
fi

# The CUDA runtime packages are the LTX pipelines' floor, and resolving these
# additions is allowed to move them: on 10 Sep 2026 this install silently
# downgraded `nvidia-cudnn-cu13` from 9.21.1.3 to 9.20.0.48 on a node that was
# serving jobs, and `uv pip install --dry-run` did not predict it. So they are
# snapshotted and put back. Adding a person matte must not re-floor the CUDA
# stack underneath every other tool on the box.
nvidia_versions() {
  uv pip list --python "$LTX_PYTHON" --format=freeze 2>/dev/null | grep -i '^nvidia-' | sort
}
BEFORE="$(nvidia_versions)"

uv pip install --python "$LTX_PYTHON" "${PACKAGES[@]}"

MOVED="$(comm -23 <(printf '%s\n' "$BEFORE") <(nvidia_versions) | tr '\n' ' ')"
if [[ -n "${MOVED// /}" ]]; then
  echo "restoring CUDA packages this install moved: $MOVED"
  # shellcheck disable=SC2086
  uv pip install --python "$LTX_PYTHON" $MOVED
fi

"$LTX_REPO_DIR/.venv/bin/hf" download ZhengPeng7/BiRefNet \
  --local-dir "$BIREFNET_DIR"

# Prove it loads before a customer job finds out it does not. This is the
# check that would have caught both the missing imports above and the fp16
# input mismatch, in seconds, instead of at the first render.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOLEXAI_BIREFNET_MODEL="$BIREFNET_DIR" SCRIPT_DIR="$SCRIPT_DIR" "$LTX_PYTHON" - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["SCRIPT_DIR"])

from PIL import Image

from _person_segmentation import PersonSegmenter

segmenter = PersonSegmenter()
mask = segmenter.mask(Image.new("RGB", (256, 144), "black"))
print(f"BiRefNet loaded at {segmenter.dtype}, returned a {mask.shape} mask")
PY

echo "V2V person segmentation installed at $BIREFNET_DIR"
