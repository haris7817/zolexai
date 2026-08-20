"""The Wan2.2-Animate identity provider, reached over the same seam as matting.

`scripts/wan_animate.py` runs in the Wan environment and does the work; this is
the worker's half — build the argv, wait, and be strict about what counts as
success.

Unlike `build_identity_anchor`, a failure here RAISES. The anchor degrades to a
weaker but honest result, so returning None and carrying on is right there.
This provider is the whole render: if it fails there is nothing to fall back to
except an ordinary restyle, which would hand the customer their own source
person back under a claim that the reference replaced them. The brief for this
feature is explicit that never happens.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from worker.adapters.base import AdapterError
from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)


async def replace_person(
    source: Path,
    reference: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    prompt: str = "",
    timeout: float = 5400.0,
) -> Path:
    """The source's motion and scene, performed by the reference person.

    Raises `AdapterError` on anything short of a finished file.
    """
    if not settings.wan_animate_argv:
        raise AdapterError(
            "This tool is temporarily unavailable.",
            internal_detail=(
                "v2v_identity_provider is 'wan_animate' but WAN_ANIMATE_COMMAND "
                "is not configured on this node"
            ),
            retriable=False,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *settings.wan_animate_argv,
        "--source", str(source),
        "--reference", str(reference),
        "--dest", str(dest),
        "--width", str(width),
        "--height", str(height),
    ]
    if prompt.strip():
        command += ["--prompt", prompt.strip()]

    logger.info("wan_animate_started", extra={"source": source.name})
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as error:
        raise AdapterError(
            "This tool is temporarily unavailable.",
            internal_detail=f"wan_animate could not start: {error}",
            retriable=False,
        ) from error

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        dest.unlink(missing_ok=True)
        raise AdapterError(
            "That generation took too long.",
            internal_detail=f"wan_animate exceeded {timeout:.0f}s",
            # A retry gets the same footage and the same clock.
            retriable=False,
        ) from None

    if process.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        tail = (stdout or b"").decode("utf-8", "replace").strip().splitlines()[-8:]
        dest.unlink(missing_ok=True)
        raise AdapterError(
            "That video could not be processed. Please try another.",
            internal_detail=(
                f"wan_animate returned {process.returncode}: " + " | ".join(tail)
            ),
            retriable=False,
        )

    logger.info("wan_animate_finished", extra={"output": dest.name})
    return dest
