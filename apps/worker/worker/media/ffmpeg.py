"""Running ffmpeg and ffprobe without stalling the worker.

The worker is one process, one event loop, and up to `max_concurrency` jobs as
asyncio tasks. `subprocess.run` here would block that loop — freezing the lease
keepalive, the other in-flight job's progress reports, and the claim loop — so
every invocation goes through `asyncio.create_subprocess_exec`.

The same reasoning applies to whatever M2 ends up calling for inference: if it
blocks, it must go through a thread or a subprocess, never straight onto the
loop.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)

#: ffmpeg is verbose on stderr and we only ever want the tail for diagnostics.
_STDERR_KEEP_CHARS = 2000


class FfmpegError(RuntimeError):
    """A media tool exited non-zero, timed out, or was not installed."""

    def __init__(self, message: str, *, stderr: str = "") -> None:
        self.stderr = stderr
        super().__init__(message)


def tools_available() -> bool:
    """True when both binaries are on PATH.

    Used to fail a job with a clear internal reason rather than a `FileNotFound`
    traceback, and to skip media tests on a machine without them.
    """
    return bool(
        shutil.which(settings.ffmpeg_path) and shutil.which(settings.ffprobe_path)
    )


async def _run(executable: str, args: list[str], *, timeout: float) -> tuple[bytes, str]:
    """Runs a tool to completion. Returns (stdout, tail of stderr).

    On timeout or cancellation the child is killed rather than left behind — an
    orphaned ffmpeg holds file handles in the workspace we are about to delete,
    and on a GPU node it would hold VRAM too.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise FfmpegError(f"{executable} is not installed or not on PATH") from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        await _terminate(process)
        raise
    except BaseException:
        await _terminate(process)
        raise

    tail = stderr.decode("utf-8", "replace")[-_STDERR_KEEP_CHARS:]
    if process.returncode != 0:
        raise FfmpegError(
            f"{Path(executable).name} exited {process.returncode}", stderr=tail
        )
    return stdout, tail


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except (TimeoutError, asyncio.CancelledError):
        logger.warning("media_tool_kill_timeout", extra={"pid": process.pid})


async def ffmpeg(args: list[str], *, timeout: float = 600.0) -> str:
    """Runs ffmpeg with the given arguments. Returns the stderr tail.

    `-nostdin` matters: without it ffmpeg can consume the worker's stdin and
    hang waiting on a terminal that is not there.
    """
    _, stderr = await _run(
        settings.ffmpeg_path,
        ["-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args],
        timeout=timeout,
    )
    return stderr


async def ffmpeg_stdout(args: list[str], *, timeout: float = 600.0) -> bytes:
    """Runs ffmpeg and returns what it wrote to stdout.

    For the one case where the output is data rather than a file: decoding an
    audio track to raw PCM so the timing layer can measure it. Writing that to
    a temporary file first would double the I/O for bytes nobody keeps.
    """
    stdout, _ = await _run(
        settings.ffmpeg_path,
        ["-hide_banner", "-nostdin", "-loglevel", "error", *args],
        timeout=timeout,
    )
    return stdout


async def ffprobe_json(path: Path, *, timeout: float = 60.0) -> dict[str, Any]:
    """Returns ffprobe's format+stream description of a file."""
    stdout, _ = await _run(
        settings.ffprobe_path,
        [
            "-hide_banner",
            "-loglevel", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=timeout,
    )
    try:
        return json.loads(stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise FfmpegError(f"ffprobe returned unparseable output for {path.name}") from exc
