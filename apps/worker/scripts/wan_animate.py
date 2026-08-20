"""Replace the person in a video with the person from a photo (Wan2.2-Animate).

Runs in the Wan environment, not the worker's — the model needs torch and CUDA
and this worker deliberately has neither. Same seam as `person_matte.py` and
`person_anchor.py`: a small, stable CLI behind a configured command, so the
model underneath can be replaced without touching worker code.

Two stages, both from the official repo:

  1. preprocessing, in REPLACEMENT mode — extracts the driving materials from
     the source video: a pose skeleton, a face crop sequence, the background
     with the person removed, and a mask of where they were;
  2. generation — the reference person performs that pose and that facial
     motion, composited back into the source's own background.

Replacement mode rather than animation mode: animation keeps the REFERENCE
photo's background, which loses the customer's scene entirely. Measured
20 Aug 2026 — animation returned the speaker against the reference's grey
studio backdrop, 6.2 dB against the source's own background.

Why this exists at all, when the LTX path already does identity: a pose
skeleton carries joint coordinates and nothing else, so it cannot leak the
source person's hair silhouette or clothing the way a canny edge map does, and
a dedicated face encoder drives expression from the source's own face crops.
Those are mechanisms LTX has no equivalent for.

LICENCE GATE — READ BEFORE ENABLING FOR CUSTOMER TRAFFIC
--------------------------------------------------------
The preprocessing checkpoint ships `det/yolov10m.onnx`, and YOLOv10 is
**AGPL-3.0** upstream. Redistribution inside an Apache-2.0 repository does not
relicense it, and AGPL's network-service clause is precisely the one a SaaS
cannot ignore. It is also the least load-bearing part of the stack — a person
bounding box, which the MIT-licensed BiRefNet matte this repo already ships
could supply instead. Everything else is clean: Wan2.2-Animate code and weights
Apache 2.0, SAM2 Apache 2.0, ViTPose Apache 2.0. FLUX.1-Kontext-dev is
non-commercial but is animation-mode retargeting only and is never fetched.

So this is for evaluation behind an off-by-default flag. Swap the detector
before it serves anyone.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: Wan-Animate's own sampling rate. The worker re-times the result to the
#: source's rate on delivery, so this only decides what the model renders at.
WAN_FPS = 30

#: How many of the previous segment's frames condition the next one. This is
#: what carries the character across a segment boundary.
CARRY_FRAMES = 1

#: The materials replacement mode cannot run without. A "successful"
#: preprocessing run that produced none of these found no person, and
#: generating from that would hand back the source untouched under a claim of
#: replacement — the one outcome this mode must never produce.
REQUIRED_MATERIALS = (
    "src_pose.mp4",
    "src_face.mp4",
    "src_bg.mp4",
    "src_mask.mp4",
    "src_ref.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="driving video")
    parser.add_argument("--reference", type=Path, required=True, help="person photo")
    parser.add_argument("--dest", type=Path, required=True, help="output mp4")
    parser.add_argument("--repo", type=Path, required=True, help="Wan2.2 checkout")
    parser.add_argument("--ckpt", type=Path, required=True, help="Wan2.2-Animate-14B dir")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--prompt", default="")
    # Mask shape. Bigger and coarser lets the character be re-imagined freely
    # but redraws more of the background; smaller and finer keeps more of the
    # real scene but can leak the source person's outline. Repo defaults.
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--w-len", type=int, default=1)
    parser.add_argument("--h-len", type=int, default=1)
    return parser.parse_args(argv)


def run(command: list[str], cwd: Path) -> None:
    """Run a stage, streaming its output so a stall is visible in the log."""
    print("$ " + " ".join(str(part) for part in command), flush=True)
    result = subprocess.run(command, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"stage failed with {result.returncode}")


def check_patched(repo: Path) -> None:
    """Refuse early if the checkout cannot run without FlashAttention.

    Two call sites bypass the module's own SDPA fallback — the animate CLIP
    encoder, and the face adapter's flash branch, whose mode="torch" sibling is
    unusable because `pre_attn_layout` is fetched and never applied to q/k/v.
    Both are patched on the node. Failing here beats failing forty minutes into
    a render.
    """
    checks = {
        "wan/modules/attention.py": "_sdpa_fallback",
        "wan/modules/animate/face_blocks.py": "No FlashAttention build",
    }
    missing = [
        name
        for name, marker in checks.items()
        if marker not in (repo / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit(
            "Wan checkout is missing the SDPA fallback patches: "
            + ", ".join(missing)
            + " — see docs/internal/gpu-worker-runbook.md"
        )


def preprocess(args: argparse.Namespace, materials: Path) -> None:
    run(
        [
            sys.executable,
            str(args.repo / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"),
            "--ckpt_path", str(args.ckpt / "process_checkpoint"),
            "--video_path", str(args.source),
            "--refer_path", str(args.reference),
            "--save_path", str(materials),
            "--resolution_area", str(args.width), str(args.height),
            "--iterations", str(args.iterations),
            "--k", str(args.k),
            "--w_len", str(args.w_len),
            "--h_len", str(args.h_len),
            "--replace_flag",
        ],
        cwd=args.repo,
    )
    absent = [name for name in REQUIRED_MATERIALS if not (materials / name).is_file()]
    if absent:
        raise SystemExit(
            "preprocessing produced no " + ", ".join(absent) + " — no usable person found"
        )


def generate(args: argparse.Namespace, materials: Path) -> None:
    command = [
        sys.executable, str(args.repo / "generate.py"),
        "--task", "animate-14B",
        "--ckpt_dir", str(args.ckpt),
        "--src_root_path", str(materials),
        "--refert_num", str(CARRY_FRAMES),
        "--replace_flag",
        # Relights the reference character into the source scene's lighting;
        # without it they stay lit for the photo they came from.
        "--use_relighting_lora",
        # The GPU is shared with a resident LTX worker holding ~24 GB, so keep
        # the text encoder on the CPU and offload between stages.
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--save_file", str(args.dest),
    ]
    if args.prompt.strip():
        command += ["--prompt", args.prompt.strip()]
    run(command, cwd=args.repo)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.source, args.reference):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    if not (args.repo / "generate.py").is_file():
        raise SystemExit(f"not a Wan2.2 checkout: {args.repo}")
    check_patched(args.repo)

    workspace = Path(tempfile.mkdtemp(prefix="wan-animate-"))
    try:
        materials = workspace / "materials"
        print("preprocessing the driving video", flush=True)
        preprocess(args, materials)
        print("generating", flush=True)
        generate(args, materials)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if not args.dest.is_file() or args.dest.stat().st_size == 0:
        raise SystemExit("no video was written")
    print(f"wrote {args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
