"""Which exact frame counts can this card decode, and which cannot?

The VAE fails on particular `(grid, conditioned, frame-count)` triples with
`CUBLAS_STATUS_INTERNAL_ERROR` — dimensions cast to int32 inside a batched
GEMM, upstream. There is no arithmetic rule: larger counts pass where smaller
ones fail, in both directions. `adapters/ltx._BAD_FRAME_BANDS` is therefore a
table of MEASUREMENTS, and this is what measures it.

The counts that matter are not round numbers. Music video cuts on musical
onsets, so its passes are lengths like 59.88s and 57.54s — frame counts nobody
would think to test. A 3-minute track produced 1437 / 1440 / 1381 / 63 and one
of them took the job down twice.

Runs the adapter's OWN command builder, so what is probed is exactly what
production sends — a probe with its own hand-written argv proves nothing about
the code path that actually fails.

    python scripts/frame_probe.py 1437 1528 1381
    CONDITIONED=1 python scripts/frame_probe.py 1440 1464
    GRID=768x768 python scripts/frame_probe.py 720 736

Each count is one render. Budget a couple of minutes each and run it on an
idle card — a probe that OOMs against a customer's job has measured nothing.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from worker.adapters.base import AdapterJob
from worker.adapters.ltx import ConditioningFrame, LtxAdapter
from worker.core.config import settings


def _conditioning_still(workspace: Path, width: int, height: int) -> Path:
    """A real frame to condition on, generated here rather than uploaded.

    Conditioned and unconditioned runs fail at DIFFERENT counts — that is the
    whole reason `_BAD_FRAME_BANDS` is keyed on it — so a probe that skips
    conditioning is measuring the other half of the table.
    """
    still = workspace / "conditioning.png"
    subprocess.run(
        [
            settings.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=1",
            "-frames:v", "1", str(still),
        ],
        check=True,
    )
    return still


async def main() -> int:
    counts = [int(value) for value in sys.argv[1:]]
    if not counts:
        raise SystemExit("usage: python scripts/frame_probe.py FRAMES [FRAMES...]")

    grid = os.getenv("GRID", "1024x576")
    width, height = (int(part) for part in grid.lower().split("x"))
    conditioned = os.getenv("CONDITIONED", "1") not in ("0", "", "false")
    prompt = os.getenv("PROMPT", "a koi pond at dawn, slow cinematic push in")

    workspace = Path(tempfile.mkdtemp(prefix="frame-probe-"))
    adapter = LtxAdapter()
    still = _conditioning_still(workspace, width, height) if conditioned else None

    print(f"grid:        {width}x{height}")
    print(f"conditioned: {conditioned}")
    print(f"counts:      {counts}")
    print(f"workspace:   {workspace}")
    print("-" * 62)

    results: list[tuple[int, bool, float, str]] = []
    for frames in counts:
        job = AdapterJob(
            job_id=f"probe-{frames}",
            workflow_id="text-to-video",
            workflow_version="1",
            prompt=prompt,
            parameters={"aspect_ratio": "16:9", "duration": "10s"},
            inputs=[],
            execution={"runtime": "ltx"},
            output_content_type="video/mp4",
            workspace=workspace,
        )
        output = workspace / f"probe-{frames}.mp4"
        cmd = adapter._command(
            job,
            frames / float(settings.ltx_frame_rate),
            output,
            conditioning=(
                [ConditioningFrame(still, 0, 1.0)] if still is not None else []
            ),
            dimensions=(width, height),
            num_frames=frames,
        )

        printable = f"{frames:>5} frames ({frames / float(settings.ltx_frame_rate):5.1f}s)"
        print(f"{printable}  … ", end="", flush=True)
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(settings.ltx_repo_dir),
            start_new_session=True,
        )
        stdout, _ = await process.communicate()
        elapsed = time.monotonic() - started
        text = stdout.decode("utf-8", "replace")

        if process.returncode == 0 and output.is_file():
            print(f"PASS  {elapsed:6.1f}s")
            results.append((frames, True, elapsed, ""))
            output.unlink(missing_ok=True)
        else:
            reason = "unknown"
            for needle in (
                "CUBLAS_STATUS_INTERNAL_ERROR",
                "CUBLAS_STATUS_NOT_SUPPORTED",
                "illegal memory access",
                "out of memory",
                "invalid argument",
            ):
                if needle in text:
                    reason = needle
                    break
            print(f"FAIL  {elapsed:6.1f}s  {reason}")
            results.append((frames, False, elapsed, reason))

    print("-" * 62)
    bad = [row for row in results if not row[1]]
    for frames, ok, elapsed, reason in results:
        print(f"  {frames:>5}  {'PASS' if ok else 'FAIL':<4}  {elapsed:6.1f}s  {reason}")
    print()
    if bad:
        print("Add the failing counts to _BAD_FRAME_BANDS with a landing that")
        print("PASSES here — a nudge onto an unmeasured count is a guess.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
