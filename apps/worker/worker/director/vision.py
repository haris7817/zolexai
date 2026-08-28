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


#: The identity describer's brief — one caption-voice sentence about the
#: PERSON, because its output is appended to a generation prompt rather than
#: fed to a planner. Identity replacement lives or dies on this description:
#: the measured failure it exists to prevent is a customer prompt full of
#: meta-instructions ("preserve the exact face… no flicker") that names no
#: visible attribute of the person at all, which the model reads as content
#: vocabulary and renders as neither the source nor the reference.
_IDENTITY_SYSTEM_PROMPT = """You describe the person in a photograph for a film crew that must cast
their exact double. State only what is VISIBLE. Never guess names, emotions,
backstory or anything outside the frame.

Answer with ONE sentence, nothing else, and BEGIN it with the person's
apparent age stated as a number of years — "a man of about 55" — followed by
whatever visibly carries that age (grey or greying hair, a receding hairline,
lines around the eyes and mouth, weathered or youthful skin, a child's or
teenager's proportions). Then give build, hair (colour, LENGTH, style),
facial hair (always name a beard or moustache when one is visible), headwear
if any, and clothing with exact colours.
Never write "adult", "young adult", "middle-aged" or any other age BAND —
those are the words that let a video model cast whoever it likes; commit to a
number even when you are only approximately right.
Ignore jewellery, chains, necklaces and other accessories entirely — a chain
described near a face comes back rendered as braided hair. Never mention the
photograph, the background or the setting."""

_IDENTITY_USER_PROMPT = (
    "Describe the person in this photograph in one concrete sentence, "
    "starting with their age in years."
)

#: Roughly where each vague age band sits, in years, longest phrase first so
#: "young adult" is read before "young". Deliberately coarse: the point is to
#: put SOME number in front of the model, because the measured failure is a
#: prompt with no age in it at all, which the model answers with its own prior
#: — a twenty-something — whatever the photograph actually showed.
_BAND_YEARS: tuple[tuple[str, int], ...] = (
    ("young adult", 25),
    ("middle-aged", 50),
    ("middle aged", 50),
    ("teenaged", 16),
    ("teenage", 16),
    ("elderly", 72),
    ("young", 25),
    ("adult", 35),
)
#: "old" is deliberately absent. It is the one band that describes objects as
#: often as people — "an old leather jacket" — and rewriting a jacket's age
#: into the person's is a worse caption than the one it replaced. "elderly"
#: covers the case that matters.

_BAND_RES = tuple(
    (re.compile(rf"\b{re.escape(band)}\b", re.IGNORECASE), years)
    for band, years in _BAND_YEARS
)


def _pin_age(caption: str) -> str:
    """A caption that names no age in years gets the best one available.

    The describer is instructed to lead with a number and usually does. When
    it falls back to a band ("an adult man in a grey coat"), the band is
    translated once, here, rather than reaching a video model that reads
    "adult" as "whoever I usually draw" — the reported failure where an older
    reference photograph came back as a performer in their twenties.

    A caption already carrying any digit is left exactly as written: the
    describer's own number is better than this table's.
    """
    if not caption or any(char.isdigit() for char in caption):
        return caption
    for pattern, years in _BAND_RES:
        if pattern.search(caption):
            # Substituted as an ADJECTIVE, in the slot the band already
            # occupied, because that is the one form that survives every
            # sentence shape the describer produces: "a middle-aged man" reads
            # back as "a 50-year-old man", and a bare "Woman: adult, dark
            # hair" as "Woman: 35-year-old, dark hair".
            return pattern.sub(f"{years}-year-old", caption, count=1)
    return caption


#: A prompt suffix, not a planning document. One sentence fits well inside
#: this; a runaway description would dilute the user's own text.
_MAX_IDENTITY_CHARS = 350


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
    return await _describe(
        str(item.path),
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_USER_PROMPT,
        max_chars=_MAX_FACTS_CHARS,
    )


async def reference_person_facts(image_path) -> str:
    """One caption-voice sentence describing the person in `image_path`.

    Serves `v2v_reference_identity`, and is deliberately NOT gated on
    `settings.director_vision_enabled` — that switch governs whether Director
    planning may look at a source image, a separate product decision. This
    path's safety is its degradation: every failure returns "", the prompt
    goes to the model exactly as the user typed it, and the render proceeds.

    Measured on the production box 19 Aug 2026: the on-box gemma-4-e2b-it
    checkpoint loads as an image-text model in ~4s and answered in ~1s —
    "Woman: adult, dark hair, black leather jacket" for exactly that photo.

    That measured answer is also this function's known weakness and the reason
    for `_pin_age`: "adult" is not an age, and a prompt that states no age
    lets the model cast from its own prior. Client report, 28 Aug 2026 — "it
    doesn't follow the age that I put as reference".
    """
    caption = _pin_age(
        await _describe(
            str(image_path),
            system_prompt=_IDENTITY_SYSTEM_PROMPT,
            user_prompt=_IDENTITY_USER_PROMPT,
            max_chars=_MAX_IDENTITY_CHARS,
        )
    )
    if caption:
        # The caption text itself, not just its length: when a render grows
        # braids nobody asked for, the first question is what the describer
        # actually said about the photo, and this line is the only answer.
        logger.info("reference_person_described", extra={"caption": caption})
    return caption


async def _describe(
    image_path: str, *, system_prompt: str, user_prompt: str, max_chars: int
) -> str:
    payload = json.dumps(
        {
            "gemma_root": str(settings.director_gemma_root),
            "image_path": image_path,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
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

    facts = _extract_facts(text, max_chars)
    if facts:
        logger.info("director_vision_facts", extra={"characters": len(facts)})
    return facts


def _extract_facts(text: str, max_chars: int = _MAX_FACTS_CHARS) -> str:
    """The completion between the markers, tidied and bounded."""
    match = re.search(
        re.escape(_FACTS_BEGIN) + r"(.*?)" + re.escape(_FACTS_END), text, re.DOTALL
    )
    if not match:
        return ""
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    facts = "\n".join(lines)
    return facts[:max_chars].strip()
