"""GPU benchmark for the client's LTX 2.5 graphs — prepared, not yet run.

    cd apps/worker
    P=.venv/bin/python; S=../../benchmarks/client-pack/ltx25/samples
    $P scripts/ltx_comfy_bench.py t2v --seconds 5 10 15 30 --aspect 16:9 9:16 1:1
    $P scripts/ltx_comfy_bench.py flf --first $S/first_last_frame_input.png --seconds 5 10 15 30
    $P scripts/ltx_comfy_bench.py flf --first first.png --last last.png --seconds 10
    $P scripts/ltx_comfy_bench.py cr --video $S/character_replacement_source.mp4 --image photo.png

Every cell runs the SAME compiled prompt the worker would submit — the same
adapter code path, against the live service — and records, per run:

    resolution · fps · output duration · wall-clock · VRAM peak · RAM peak

VRAM is sampled from `nvidia-smi --query-gpu=memory.used` at 1 Hz for the
whole run (peak and mean); RAM from psutil on the ComfyUI process when its
pid is given, otherwise system-wide. Results land in
`benchmarks/results/ltx25/<stamp>.json` plus a markdown table, and the
readiness report's benchmark section is filled from that file — never from
this docstring.

Quality is not measured here: the outputs are kept beside the JSON for a
human viewing against the ZIP samples (docs/internal/ltx-client-workflow-audit.md §2.1).

STATUS: WAITING FOR GPU VALIDATION.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from worker.adapters.base import AdapterInput, AdapterJob
from worker.adapters.character_replacement import CharacterReplacementAdapter
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.core.config import settings
from worker.media import probe_media

RESULTS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "results" / "ltx25"


@dataclass
class RunRecord:
    cell: str
    workflow: str
    seconds: float | None
    aspect: str | None
    inputs: dict[str, str]
    wall_seconds: float | None = None
    output: str | None = None
    output_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool | None = None
    vram_peak_mib: int | None = None
    vram_mean_mib: int | None = None
    ram_peak_mib: int | None = None
    error: str | None = None
    status: str = "WAITING FOR GPU VALIDATION"


class Sampler:
    """1 Hz nvidia-smi + psutil sampling for the duration of one run."""

    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.vram: list[int] = []
        self.ram: list[int] = []
        self._task: asyncio.Task[None] | None = None

    async def _loop(self) -> None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            psutil = None
        while True:
            if shutil.which("nvidia-smi"):
                completed = await asyncio.to_thread(
                    subprocess.run,
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                out = completed.stdout.strip().splitlines()
                if out:
                    self.vram.append(int(float(out[0])))
            if psutil is not None:
                if self.pid:
                    try:
                        self.ram.append(int(psutil.Process(self.pid).memory_info().rss / 2**20))
                    except psutil.Error:
                        pass
                else:
                    self.ram.append(int(psutil.virtual_memory().used / 2**20))
            await asyncio.sleep(1.0)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def _progress(status: str, progress: int, message: str, details=None) -> None:
    print(f"    [{status:>15}] {progress:3d}% {message}", flush=True)


def _job(
    workflow: str, workspace: Path, prompt: str, inputs: list[AdapterInput], **params
) -> AdapterJob:
    workspace.mkdir(parents=True, exist_ok=True)
    return AdapterJob(
        job_id=workspace.name,
        workflow_id=workflow,
        workflow_version="1",
        prompt=prompt,
        parameters=params,
        inputs=inputs,
        execution={"runtime": "ltx_comfy"},
        output_content_type="video/mp4",
        workspace=workspace,
    )


def _input(role: str, path: Path, kind: str) -> AdapterInput:
    return AdapterInput(
        role=role,
        kind=kind,
        content_type="video/mp4" if kind == "video" else "image/png",
        download_url="file://" + str(path),
        path=path,
    )


async def run_cell(
    record: RunRecord, job: AdapterJob, adapter, pid: int | None, keep_dir: Path
) -> RunRecord:
    sampler = Sampler(pid)
    sampler.start()
    started = time.monotonic()
    try:
        result = await adapter.run(job, _progress)
        record.wall_seconds = round(time.monotonic() - started, 1)
        kept = keep_dir / f"{record.cell}.mp4"
        shutil.copyfile(result.path, kept)
        info = await probe_media(kept)
        record.output = str(kept)
        record.output_seconds = info.duration_seconds
        record.width, record.height, record.fps = info.width, info.height, info.fps
        record.has_audio = info.has_audio
        record.status = "MEASURED"
    except Exception as exc:  # noqa: BLE001 - a benchmark records failures
        record.wall_seconds = round(time.monotonic() - started, 1)
        record.error = f"{type(exc).__name__}: {exc}"
        record.status = "FAILED"
    finally:
        await sampler.stop()
    if sampler.vram:
        record.vram_peak_mib = max(sampler.vram)
        record.vram_mean_mib = int(sum(sampler.vram) / len(sampler.vram))
    if sampler.ram:
        record.ram_peak_mib = max(sampler.ram)
    return record


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cell", choices=["t2v", "flf", "cr"])
    parser.add_argument("--seconds", nargs="*", type=int, default=[5])
    parser.add_argument("--aspect", nargs="*", default=["16:9"])
    parser.add_argument(
        "--prompt",
        default=(
            "A woman in a red coat walks along a rain-soaked pier at dusk, "
            "gulls overhead, waves against the pilings."
        ),
    )
    parser.add_argument("--first", type=Path)
    parser.add_argument("--last", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--comfy-pid", type=int, default=None, help="ComfyUI pid for RAM sampling")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--seed", type=int, default=None, help="fixed seed for every cell (Step 8 comparison)"
    )
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    out_dir = RESULTS_DIR / args.stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = out_dir / "workspaces"
    records: list[RunRecord] = []

    print(f"LTX ComfyUI at {settings.ltx_comfy_base_url}; results → {out_dir}")
    for repeat in range(args.repeat):
        if args.cell == "t2v":
            for seconds in args.seconds:
                for aspect in args.aspect:
                    cell = f"t2v-{seconds}s-{aspect.replace(':', 'x')}-r{repeat}"
                    print(f"== {cell}")
                    job = _job(
                        "text-to-video",
                        workspace_root / cell,
                        args.prompt,
                        [],
                        duration=f"{seconds}s",
                        aspect_ratio=aspect,
                        **({"seed": args.seed} if args.seed is not None else {}),
                    )
                    records.append(
                        await run_cell(
                            RunRecord(cell, "text-to-video", seconds, aspect, {}),
                            job,
                            LtxComfyAdapter(),
                            args.comfy_pid,
                            out_dir,
                        )
                    )
        elif args.cell == "flf":
            if not args.first:
                parser.error("flf needs --first")
            inputs = [_input("source_image", args.first, "image")]
            if args.last:
                inputs.append(_input("last_frame", args.last, "image"))
            for seconds in args.seconds:
                for aspect in args.aspect:
                    frames = "fl" if args.last else "f"
                    cell = f"flf-{frames}-{seconds}s-{aspect.replace(':', 'x')}-r{repeat}"
                    print(f"== {cell}")
                    job = _job(
                        "image-to-video",
                        workspace_root / cell,
                        args.prompt,
                        inputs,
                        duration=f"{seconds}s",
                        aspect_ratio=aspect,
                        **({"seed": args.seed} if args.seed is not None else {}),
                    )
                    records.append(
                        await run_cell(
                            RunRecord(
                                cell,
                                "image-to-video",
                                seconds,
                                aspect,
                                {i.role: str(i.path) for i in inputs},
                            ),
                            job,
                            LtxComfyAdapter(),
                            args.comfy_pid,
                            out_dir,
                        )
                    )
        else:
            if not (args.video and args.image):
                parser.error("cr needs --video and --image")
            inputs = [
                _input("source_video", args.video, "video"),
                _input("reference_image", args.image, "image"),
            ]
            cell = f"cr-r{repeat}"
            print(f"== {cell}")
            job = _job(
                "character-replacement",
                workspace_root / cell,
                args.prompt,
                inputs,
                **({"seed": args.seed} if args.seed is not None else {}),
            )
            records.append(
                await run_cell(
                    RunRecord(
                        cell,
                        "character-replacement",
                        None,
                        None,
                        {i.role: str(i.path) for i in inputs},
                    ),
                    job,
                    CharacterReplacementAdapter(),
                    args.comfy_pid,
                    out_dir,
                )
            )

    (out_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
    )
    lines = [
        "| cell | resolution | fps | output s | wall s | VRAM peak MiB | RAM peak MiB | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r.cell} | {r.width}x{r.height} | {r.fps} | {r.output_seconds} | "
            f"{r.wall_seconds} | {r.vram_peak_mib} | {r.ram_peak_mib} | "
            f"{r.status}{' — ' + r.error if r.error else ''} |"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(r.status == "MEASURED" for r in records) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
