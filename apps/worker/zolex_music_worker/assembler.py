from __future__ import annotations

from pathlib import Path

from .config import WorkerConfig
from .models import AudioInfo, EditorialPlan
from .upscaler import upscale_master
from .utils import run_command


def _concat_path(path: Path) -> str:
    value = str(path.resolve())
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines are not allowed in clip paths")
    return value.replace("'", "'\\''")


def assemble_working_master(
    *,
    plan: EditorialPlan,
    accepted_clips: list[Path],
    audio: AudioInfo,
    output: Path,
    config: WorkerConfig,
    log_path: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{_concat_path(path)}'\n" for path in accepted_clips),
        encoding="utf-8",
    )
    run_command(
        [
            config.ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            audio.aligned_wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(plan.total_frames),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ],
        timeout=1800,
        log_path=log_path,
    )


def assemble(
    *,
    plan: EditorialPlan,
    accepted_clips: list[Path],
    audio: AudioInfo,
    output: Path,
    config: WorkerConfig,
    log_path: Path,
) -> Path:
    """Create a working master, then finish a 4K delivery with unchanged AAC audio."""
    working_master = output.with_name("working-master.mp4")
    assemble_working_master(
        plan=plan,
        accepted_clips=accepted_clips,
        audio=audio,
        output=working_master,
        config=config,
        log_path=log_path,
    )
    upscale_master(
        input_path=working_master,
        output_path=output,
        plan=plan,
        config=config,
        log_path=log_path.with_name("upscale-command.json"),
    )
    return working_master
