"""What the uploaded photograph shows — optional grounding for I2V Director.

Image-to-Video Director mode works from two sources of truth that must never
fight: the uploaded image (WHO and WHAT exists) and the user's idea (WHAT
HAPPENS). The planner is a text model and cannot see the image, so left alone
it would invent appearances — and an invented detail in the caption pulls the
render away from the conditioned frames, which is drift by construction.

The first line of defence is refusal: the anchored planning brief forbids
invented visual detail outright, and that path needs no model at all. This
module is the second, better line: a subprocess asks the local checkpoint to
state what the photograph visibly shows, and those measured facts let the
planner write concrete continuity constraints ("two people", "a red scarf")
instead of writing nothing.

OFF by default (`settings.director_vision_enabled`) because whether the on-box
checkpoint accepts image input is unmeasured — the exact posture the guided
tier ships with. Every failure mode here — disabled, no staged image, a
text-only checkpoint, a timeout, garbage output — degrades to "" and the
refusal posture above, never to a failed job. Facts are garnish; the render
must not depend on them.
"""

from __future__ import annotations

import asyncio
import json
import re

from worker.adapters.base import AdapterJob
from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)

_FACTS_BEGIN = "===IMAGE_FACTS_BEGIN==="
_FACTS_END = "===IMAGE_FACTS_END==="

#: The describer's brief. Facts only, stated positively, in the vocabulary the
#: planner's continuity rules already use — so a fact can flow into the plan's
#: `continuity` list without translation.
_SYSTEM_PROMPT = """You describe photographs for a film crew that must keep every visible detail
consistent while animating them. State only what is VISIBLE. Never guess at
names, emotions, backstory or anything outside the frame.

Answer with short plain lines, nothing else:
PEOPLE: how many people, and for each: age group, build, hair, clothing with exact colours
OBJECTS: prominent objects or creatures, each with colour, material and shape
SETTING: the location and lighting, concretely
COMPOSITION: where the subjects are in the frame and how they relate

If a line has nothing to report, write it with "none"."""

_USER_PROMPT = (
    "Describe this photograph's visible facts for the crew. Be concrete and "
    "brief; exact colours matter more than atmosphere."
)

#: The facts are injected into the planner's user prompt, which a runaway
#: description must not crowd out. Measured plans run ~2 KB; the facts block
#: stays well under that.
_MAX_FACTS_CHARS = 900


async def source_image_facts(job: AdapterJob) -> str:
    """Visible facts about the job's source image, or "" when unavailable.

    "" is a real answer, not an error: the anchored planning brief is written
    to work without facts, so every failure here is logged and absorbed.
    """
    if not settings.director_vision_enabled:
        return ""
    item = job.input_for("source_image")
    if item is None or item.path is None:
        return ""

    payload = json.dumps(
        {
            "gemma_root": str(settings.director_gemma_root),
            "image_path": str(item.path),
            "system_prompt": _SYSTEM_PROMPT,
            "user_prompt": _USER_PROMPT,
            "max_new_tokens": 400,
            "begin_marker": _FACTS_BEGIN,
            "end_marker": _FACTS_END,
        }
    ).encode()

    try:
        process = await asyncio.create_subprocess_exec(
            *settings.director_vision_argv,
            cwd=str(settings.ltx_repo_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as error:
        logger.warning("director_vision_unavailable", extra={"reason": str(error)})
        return ""
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(payload),
            timeout=settings.director_vision_timeout_seconds,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        logger.warning(
            "director_vision_timeout",
            extra={"timeout_seconds": settings.director_vision_timeout_seconds},
        )
        return ""

    text = (stdout or b"").decode("utf-8", "replace")
    if process.returncode != 0:
        # The expected honest failure: a text-only checkpoint. One log line,
        # no customer impact.
        logger.warning(
            "director_vision_failed",
            extra={
                "returncode": process.returncode,
                "tail": " | ".join(text.strip().splitlines()[-5:]),
            },
        )
        return ""

    facts = _extract_facts(text)
    if facts:
        logger.info("director_vision_facts", extra={"characters": len(facts)})
    return facts


def _extract_facts(text: str) -> str:
    """The completion between the markers, tidied and bounded."""
    match = re.search(
        re.escape(_FACTS_BEGIN) + r"(.*?)" + re.escape(_FACTS_END), text, re.DOTALL
    )
    if not match:
        return ""
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    facts = "\n".join(lines)
    return facts[:_MAX_FACTS_CHARS].strip()
