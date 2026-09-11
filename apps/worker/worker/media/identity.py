"""Strict multi-person identity anchors for Video to Video."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media.ffmpeg import FfmpegError

logger = get_logger(__name__)


async def build_multi_identity_anchor(
    source: Path,
    references: Sequence[Path],
    dest: Path,
    *,
    start_seconds: float,
    width: int,
    height: int,
    background: Path | None = None,
    timeout: float = 900.0,
) -> Path:
    """Build a source-composed anchor containing one-to-four reference people.

    References are mapped to visible source people from screen-left to
    screen-right. Unlike the historical single-person helper, every failure is
    fatal. A requested replacement must never degrade into unconditioned video
    generation and still be presented as a successful replacement.
    """
    if not 1 <= len(references) <= 4:
        raise ValueError(f"multi-person identity needs 1..4 references, got {len(references)}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *settings.multi_person_anchor_argv,
        "--source",
        str(source),
        "--dest",
        str(dest),
        "--start-seconds",
        f"{max(0.0, start_seconds):.3f}",
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    for reference in references:
        command.extend(("--reference", str(reference)))
    if background is not None:
        command.extend(("--background", str(background)))

    logger.info(
        "multi_person_anchor_started",
        extra={"people": len(references), "start_seconds": round(start_seconds, 3)},
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(settings.ltx_repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as error:
        raise FfmpegError(f"multi-person anchor command is unavailable: {error}") from error

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        dest.unlink(missing_ok=True)
        raise FfmpegError(f"multi-person anchor timed out after {timeout:.0f}s") from None
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        dest.unlink(missing_ok=True)
        raise

    output = (stdout or b"").decode("utf-8", "replace").strip()
    if process.returncode != 0:
        dest.unlink(missing_ok=True)
        tail = " | ".join(output.splitlines()[-12:])
        raise FfmpegError(
            f"multi-person anchor exited {process.returncode}: {tail or 'no diagnostics'}"
        )
    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FfmpegError("multi-person anchor produced no image")

    logger.info(
        "multi_person_anchor_built",
        extra={"people": len(references), "anchor": dest.name},
    )
    return dest
