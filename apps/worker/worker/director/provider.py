"""Where DirectorPlans come from.

The planning model is `gemma-4-e2b-it` — deliberately the same checkpoint the
LTX 2.5 runtime names as its official prompt enhancer, because it is already
licensed for this exact deployment (Apache 2.0), already sized for a spare
corner of the card (~11 GB bf16), and one download serves both purposes. It
runs OUTSIDE this process, in the LTX environment, for the same reason person
matting does: it is model work, this worker has no torch, and the subprocess
seam (`scripts/director_plan.py`) lets the model be swapped without touching
worker code.

`DirectorProvider` is the seam the product depends on, not Gemma: if measured
plan quality ever proves the local model insufficient, an external-LLM
provider implements the same one-method protocol and everything downstream —
validation, compilation, rendering — is unchanged.

Failure posture: one greedy attempt, one sampled retry, then a clean job
failure. Director mode never falls back to rendering the bare idea as a
standard prompt — the user asked for a planned dialogue scene, and silently
delivering a dialogue-less video would be answering a different request.
"""

from __future__ import annotations

import asyncio
import json
import re
import zlib
from dataclasses import dataclass
from typing import Any, Protocol

from worker.adapters.base import AdapterJob
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.director.plan import (
    MAX_CHARACTERS,
    DirectorPlan,
    DirectorPlanError,
    parse_plan,
    required_quotes,
    speech_budget,
    spoken_line_budget,
)

logger = get_logger(__name__)

_PLAN_BEGIN = "===DIRECTOR_PLAN_BEGIN==="
_PLAN_END = "===DIRECTOR_PLAN_END==="

#: Languages the product offers for generated dialogue. The five named ones are
#: the set Lightricks documents as validated for speech (Dub-It); "auto" means
#: "the language the idea itself is written in". Kept in sync with the API's
#: `DIALOGUE_LANGUAGES` and the frontend's selector by the contract tests.
DIALOGUE_LANGUAGES = ("auto", "english", "spanish", "french", "german", "russian")


class DirectorFailure(Exception):
    """Planning failed after its retry. Carries customer-safe copy."""

    def __init__(self, user_message: str, *, internal_detail: str = "") -> None:
        self.user_message = user_message
        self.internal_detail = internal_detail or user_message
        super().__init__(self.internal_detail)


@dataclass(frozen=True)
class DirectorRequest:
    idea: str
    duration_seconds: float
    language: str
    seed: int
    sample: bool
    """False for the deterministic first attempt; True adds temperature on the
    retry so a refused plan is not regenerated token for token."""


class DirectorProvider(Protocol):
    async def generate_plan(self, request: DirectorRequest) -> dict[str, Any]:
        """One planning attempt. Returns the raw (unvalidated) plan object."""
        ...


# ── The planning brief ───────────────────────────────────────────────────
#
# Written against what the runtime provably understands (official prompting
# guides, read 2026-08-18): dialogue as exact quoted words, delivery as audible
# description, camera from the dialogue-friendly vocabulary, acting as physical
# cues. The JSON shape mirrors `worker/director/plan.py`, which enforces every
# rule below deterministically after parsing — the prompt asks, the validator
# checks.

_SYSTEM_PROMPT = f"""You are a video director planning a short generated video with spoken dialogue.
Given an IDEA, a DURATION in seconds, and a DIALOGUE LANGUAGE, produce a JSON plan.

Output ONLY a JSON object, no markdown fences, no commentary, with this exact shape:
{{
  "scene": "<one sentence: the single location and lighting, concrete and visual>",
  "tone": "<two or three words>",
  "ambience": "<the quiet background sounds of this place, a short phrase>",
  "characters": [
    {{"id": "<short_snake_case>", "role": "<short noun phrase, e.g. detective>",
     "appearance": "<concrete visible description: age group, build, hair, clothing with colours>",
     "voice": "<audible voice description: pitch, pace, texture>"}}
  ],
  "timeline": [
    {{"start": <seconds>, "end": <seconds>, "action": "<what is visibly happening>",
     "camera": "<one of: medium shot | medium close-up | close-up | two-shot |
       over-the-shoulder shot | wide shot; plus 'static' or a subtle move>",
     "speaker": "<character id or null>", "dialogue": "<the exact spoken words, or null>",
     "delivery": "<audible manner, e.g. 'low and accusing', or null>"}}
  ]
}}

Hard rules:
- Use ONLY the characters the idea implies, at most {MAX_CHARACTERS}. Do not invent extra
  people. Keep every fact the idea states (counts, colours, named things) exactly as stated.
- "role" is the plain noun phrase prose would call them: "detective", "police chief",
  "woman", "robot". In "action" and "camera" text refer to characters by those role words
  only — NEVER by their id.
- If the idea already contains quoted dialogue or lines like 'Name: "..."', copy those
  spoken words VERBATIM into dialogue events. Never rewrite or drop them.
- All dialogue must be written in the DIALOGUE LANGUAGE.
- If the idea implies nobody would speak, leave every "dialogue" null rather than
  forcing a line into the scene.
- LINE COUNT: aim for about one spoken line per 5 seconds of video (a SPOKEN_LINES
  figure is given below). Fewer, better-placed lines beat more lines. A short clip
  should carry a single exchange, not a whole argument.
- The first spoken line starts AFTER the scene has established itself, not on the
  very first frame: open with a short action event that nobody speaks over.
- Never put two spoken lines back to back without a reaction, action or pause
  between them — that is what makes two lines run together as one.
- Speech pacing: stay within the TOTAL_WORDS figure given below, spread over the
  timeline. Short lines are better. Include events with no dialogue for reactions,
  movement and silence.
- The timeline covers 0 to DURATION seconds in 2-6 second events, in order, no overlaps.
- The conversation must progress: no line repeats an earlier line, and the last event
  resolves or lands the exchange.
- Dialogue needs readable faces: prefer medium shots, close-ups, two-shots and reaction
  shots, with a static camera or a subtle push-in. No fast camera moves.
- The ambience stays quiet under the voices. No background music unless the idea asks.
- Characters keep exactly the same appearance for the whole video.
"""


def _user_prompt(request: DirectorRequest) -> str:
    language = (
        "the same language the idea is written in"
        if request.language == "auto"
        else request.language.capitalize()
    )
    lines = [
        f"IDEA: {request.idea}",
        f"DURATION: {request.duration_seconds:g} seconds",
        f"DIALOGUE LANGUAGE: {language}",
        # Computed rather than left as arithmetic in the brief: a small
        # instruct model reliably obeys a number and unreliably derives one.
        f"SPOKEN_LINES: about {spoken_line_budget(request.duration_seconds)}",
        f"TOTAL_WORDS: at most {speech_budget(request.duration_seconds)}",
    ]
    quotes = required_quotes(request.idea)
    if quotes:
        lines.append(
            "REMINDER — these exact words from the idea must appear verbatim as dialogue: "
            + "; ".join(f'"{quote}"' for quote in quotes)
        )
    return "\n".join(lines)


class GemmaDirectorProvider:
    """Runs the local Gemma instruct checkpoint through the LTX environment."""

    async def generate_plan(self, request: DirectorRequest) -> dict[str, Any]:
        payload = json.dumps(
            {
                "gemma_root": str(settings.director_gemma_root),
                "system_prompt": _SYSTEM_PROMPT,
                "user_prompt": _user_prompt(request),
                "sample": request.sample,
                "seed": request.seed,
                "max_new_tokens": 1600,
                "begin_marker": _PLAN_BEGIN,
                "end_marker": _PLAN_END,
            }
        ).encode()

        process = await asyncio.create_subprocess_exec(
            *settings.director_planner_argv,
            cwd=str(settings.ltx_repo_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(payload),
                timeout=settings.director_planner_timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise DirectorFailure(
                "Planning this scene took too long. Please try again.",
                internal_detail=(
                    "director planner timed out after "
                    f"{settings.director_planner_timeout_seconds:.0f}s"
                ),
            ) from None

        text = (stdout or b"").decode("utf-8", "replace")
        if process.returncode != 0:
            tail = " | ".join(text.strip().splitlines()[-8:])
            raise DirectorFailure(
                "We couldn't plan this scene. Please try again.",
                internal_detail=f"director planner exited {process.returncode}: {tail}",
            )
        return _extract_plan_json(text)


def _extract_plan_json(text: str) -> dict[str, Any]:
    match = re.search(re.escape(_PLAN_BEGIN) + r"(.*?)" + re.escape(_PLAN_END), text, re.DOTALL)
    body = match.group(1) if match else text
    brace = re.search(r"\{.*\}", body, re.DOTALL)
    if not brace:
        raise DirectorPlanError(["the planner returned no JSON at all"])
    try:
        parsed = json.loads(brace.group(0))
    except json.JSONDecodeError as error:
        raise DirectorPlanError([f"the planner returned invalid JSON: {error}"]) from None
    if not isinstance(parsed, dict):
        raise DirectorPlanError(["the planner returned JSON that is not an object"])
    return parsed


# ── Orchestration ────────────────────────────────────────────────────────


def wants_director(job: AdapterJob) -> bool:
    """Whether this job asked for Director mode.

    Scoped to text-to-video by construction: image-to-video shares the same
    generation handler, and a prompt mode leaking across workflows would be the
    exact class of surprise this feature is built not to cause. The API only
    admits `prompt_mode` on workflows whose YAML declares it, so this check is
    the worker-side belt to that server-side braces.
    """
    return (
        job.workflow_id == "text-to-video"
        and str(job.parameters.get("prompt_mode") or "").strip().lower() == "director"
    )


def requested_language(job: AdapterJob) -> str:
    language = str(job.parameters.get("dialogue_language") or "auto").strip().lower()
    return language if language in DIALOGUE_LANGUAGES else "auto"


async def create_director_plan(
    job: AdapterJob,
    duration_seconds: float,
    *,
    provider: DirectorProvider | None = None,
) -> DirectorPlan:
    """Idea → validated DirectorPlan, or a clean `DirectorFailure`.

    Two attempts: greedy (deterministic, cheap to reason about), then sampled
    (a refused plan regenerated greedily would be refused again, token for
    token). The stored job keeps the user's idea verbatim throughout — the
    plan lives only in this process and in the log.
    """
    provider = provider or GemmaDirectorProvider()
    language = requested_language(job)
    seed = zlib.crc32(f"{job.job_id}:director".encode())

    failures: list[str] = []
    for attempt, sample in enumerate((False, True), start=1):
        job.raise_if_cancelled()
        request = DirectorRequest(
            idea=job.prompt,
            duration_seconds=duration_seconds,
            language=language,
            seed=seed + attempt,
            sample=sample,
        )
        try:
            raw = await provider.generate_plan(request)
            plan = parse_plan(
                raw,
                idea=job.prompt,
                duration_seconds=duration_seconds,
                language=language,
            )
        except DirectorPlanError as error:
            failures.append(f"attempt {attempt}: {error}")
            logger.warning(
                "director_plan_rejected",
                extra={"attempt": attempt, "problems": error.problems},
            )
            continue

        logger.info(
            "director_plan_ready",
            extra={
                "attempt": attempt,
                "characters": [entry.id for entry in plan.characters],
                "events": len(plan.timeline),
                "spoken_words": plan.spoken_words,
                "language": plan.language,
            },
        )
        return plan

    raise DirectorFailure(
        "We couldn't turn this idea into a scene plan. Try adding a little more "
        "detail to the idea, or switch to the standard prompt mode.",
        internal_detail=" || ".join(failures),
    )
