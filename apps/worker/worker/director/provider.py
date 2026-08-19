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
from worker.director.cerebras import (
    CerebrasDirectorProvider,
    DirectorProviderUnavailable,
)
from worker.director.plan import (
    MAX_CHARACTERS,
    MAX_SILENT_GAP,
    DirectorPlan,
    DirectorPlanError,
    pacing_problems,
    parse_plan,
    required_quotes,
    speech_budget,
    spoken_line_budget,
    target_spoken_lines,
    vocabulary_problems,
)
from worker.director.vision import source_image_facts

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

    source_anchored: bool = False
    """True when the video starts from an uploaded image (Image to Video).

    Switches the brief into its anchored register: the photograph owns WHO and
    WHAT, the idea owns WHAT HAPPENS, and invented visual detail is forbidden.
    A text-to-video request is byte-identical to what shipped before this
    field existed."""

    image_facts: str = ""
    """What the photograph visibly shows, when the vision step produced it.

    Empty is normal (the step is off by default) and the anchored brief is
    written to survive it — the planner then simply may not describe what it
    cannot see."""

    notes: tuple[str, ...] = ()
    """What was wrong with the previous draft, in the planner's own terms.

    Carries both kinds of correction — a validation failure and a pacing
    complaint — because the retry is one attempt and it should fix everything
    known to be wrong, not just the thing that raised."""


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
  "continuity": [
    "<a fact that must look identical in every frame: a prop's exact colour and
      shape, what each person is wearing, how many people are in the scene>"
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
- LINE COUNT: write the number of spoken lines given as TOTAL_LINES below. That is
  a target to hit, not a maximum to stay under. Fewer lines than that leaves the
  video silent for long stretches, which is worse than too many.
- THE CONVERSATION RUNS THE WHOLE VIDEO. Spread the lines from the first seconds
  to the last so someone is speaking or reacting throughout. Never leave more than
  {MAX_SILENT_GAP:g} seconds of the timeline with nobody speaking. The last spoken
  line lands near the end, not in the middle.
- Open on a SHORT establishing beat — one or two seconds of action nobody speaks
  over — then start the dialogue. Do not open with a long silence.
- Never put two spoken lines back to back without a reaction, action or pause
  between them — that is what makes two lines run together as one.
- Keep individual lines SHORT (a handful of words) and stay within the TOTAL_WORDS
  figure below. Many short lines beat a few long ones: short lines are what let the
  conversation cover the whole video without anyone rushing.
- The timeline covers 0 to DURATION seconds in 2-6 second events, in order, no overlaps.
- The conversation must progress: no line repeats an earlier line, and the last event
  resolves or lands the exchange.
- Dialogue needs readable faces: prefer medium shots, close-ups, two-shots and reaction
  shots, with a static camera or a subtle push-in. No fast camera moves.
- The ambience stays quiet under the voices. No background music unless the idea asks.
- Characters keep exactly the same appearance for the whole video.
- VOCABULARY: every line uses different words. If one line says "excellent", no other
  line may say "excellent" — pick another word. Reusing a distinctive word across lines
  is the single thing that makes generated dialogue sound generated.
- CONTINUITY: list 2-5 facts that must look identical in every single frame. Always
  include what each person is wearing and how many people are present. If any prop is
  picked up, taken off, put down or handled during the scene, describe it there in
  concrete detail (exact colour, material, shape) — a thing that leaves the frame and
  comes back is where the picture drifts.
- Write continuity facts as things that STAY, never as things to avoid: "the red felt
  hat stays the same red felt hat every time it appears", not "the hat does not change".
"""

#: Appended to the brief for Image to Video only. The register change is the
#: whole feature: on this path a text model is planning around a photograph it
#: cannot see, and the one catastrophic failure is CONFIDENT INVENTION — a
#: described red coat over a photographed blue one is drift pressure written
#: into the caption. Every rule below is a way of saying "describe nothing the
#: idea or the facts block does not state".
_ANCHORED_RULES = """
SOURCE IMAGE MODE — this video starts from a photograph the user uploaded:
- The photograph is the video's exact first frame and its visual truth. The
  photograph decides WHO and WHAT is in the scene; the IDEA decides what happens
  next. You cannot see the photograph unless a PHOTOGRAPH FACTS block is given.
- Cast exactly the people and things the idea (and the PHOTOGRAPH FACTS block,
  when given) says are in the photograph. Never add or remove anyone.
- NEVER invent visible details. "appearance" may carry ONLY visual facts stated
  by the idea or the PHOTOGRAPH FACTS block; when neither states any for a
  character, set "appearance" to "" — their identity then comes from the
  photograph itself. The same rule applies to "scene": name the setting as
  stated, and add no imagined visual detail.
- CONTINUITY on this path: always include how many people are present, plus
  every visual fact the idea or the PHOTOGRAPH FACTS block states. Do not
  describe clothing or props neither of them mentions.
- The action moves FORWARD from the photographed moment. Do not re-stage or
  restart what the photograph already shows; the first event begins exactly
  where the photograph leaves off.
- Camera: the video opens on the photograph's own framing. Keep every shot
  inside the space the photograph establishes — a static camera, a subtle
  push-in, or cuts between the people already in frame. Never call for a
  reveal of anything the photograph does not show.
"""


def _user_prompt(request: DirectorRequest) -> str:
    language = (
        "the same language the idea is written in"
        if request.language == "auto"
        else request.language.capitalize()
    )
    target = target_spoken_lines(request.duration_seconds)
    words = speech_budget(request.duration_seconds)
    lines = [
        f"IDEA: {request.idea}",
        f"DURATION: {request.duration_seconds:g} seconds",
        f"DIALOGUE LANGUAGE: {language}",
    ]
    if request.source_anchored and request.image_facts:
        lines.append(
            "PHOTOGRAPH FACTS — what the uploaded photograph visibly shows, "
            "measured by a vision model. Treat these as true and copy their "
            "visual details into appearance and continuity:\n"
            + request.image_facts
        )
    lines += [
        # Computed rather than left as arithmetic in the brief: a small
        # instruct model reliably obeys a number and unreliably derives one.
        #
        # A target with a FLOOR and the ceiling stated last, in that order. The
        # lyrics writer proved the failure of the other shape on 19 Aug: given
        # a maximum and a vague "about", the model multiplies the small numbers
        # and stops short, and the result is a mostly-silent render.
        f"TOTAL_LINES: write {target} spoken lines across the whole video. "
        f"Not fewer than {max(2, target - 1)}, and never more than "
        f"{spoken_line_budget(request.duration_seconds)}.",
        f"TOTAL_WORDS: keep the spoken words under {words} in total, "
        f"which is roughly {max(3, words // max(1, target))} words per line.",
    ]
    quotes = required_quotes(request.idea)
    if quotes:
        lines.append(
            "REMINDER — these exact words from the idea must appear verbatim as dialogue: "
            + "; ".join(f'"{quote}"' for quote in quotes)
        )
    if request.notes:
        lines.append(
            "\nYOUR PREVIOUS PLAN HAD THESE PROBLEMS. Fix every one of them:\n"
            + "\n".join(f"- {note}" for note in request.notes)
        )
    return "\n".join(lines)


def system_prompt(request: DirectorRequest) -> str:
    """The planning brief. One text, shared by every provider.

    Exposed so a hosted provider sends the same instructions as the local one:
    if the two drifted apart, a fallback would quietly produce a differently
    shaped plan than the primary and the difference would only ever show up in
    a customer's video.

    A text-to-video request receives the original brief byte for byte; only a
    source-anchored (Image to Video) request appends the anchored register.
    """
    if request.source_anchored:
        return _SYSTEM_PROMPT + _ANCHORED_RULES
    return _SYSTEM_PROMPT


def user_prompt(request: DirectorRequest) -> str:
    """The per-job half of the request. Shared for the same reason."""
    return _user_prompt(request)


class GemmaDirectorProvider:
    """Runs the local Gemma instruct checkpoint through the LTX environment."""

    async def generate_plan(self, request: DirectorRequest) -> dict[str, Any]:
        payload = json.dumps(
            {
                "gemma_root": str(settings.director_gemma_root),
                "system_prompt": system_prompt(request),
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


#: The workflows whose YAML declares `settings.prompt_modes`, mirrored here so
#: a mis-routed parameter cannot switch modes on a workflow that never offered
#: the choice. Extend, video-to-video and music video stay OUT deliberately:
#: their prompts describe continuations and restyles, not scenes to invent.
_DIRECTOR_WORKFLOWS = frozenset({"text-to-video", "image-to-video"})


def wants_director(job: AdapterJob) -> bool:
    """Whether this job asked for Director mode.

    Scoped to the workflows that declare the control by construction: every
    video workflow shares the same generation handler, and a prompt mode
    leaking across workflows would be the exact class of surprise this feature
    is built not to cause. The API only admits `prompt_mode` on workflows
    whose YAML declares it, so this check is the worker-side belt to that
    server-side braces.
    """
    return (
        job.workflow_id in _DIRECTOR_WORKFLOWS
        and str(job.parameters.get("prompt_mode") or "").strip().lower() == "director"
    )


def source_anchored(job: AdapterJob) -> bool:
    """Whether this Director job starts from an uploaded image.

    Keyed on the WORKFLOW, not on which inputs happen to be present — the same
    dispatch rule the adapter itself follows.
    """
    return job.workflow_id == "image-to-video"


def requested_language(job: AdapterJob) -> str:
    language = str(job.parameters.get("dialogue_language") or "auto").strip().lower()
    return language if language in DIALOGUE_LANGUAGES else "auto"


def default_providers() -> list[DirectorProvider]:
    """The providers to try, best first.

    Hosted before local, when it is configured: it plans in about two seconds
    against a far larger model, where the local checkpoint costs 18-26 seconds
    of the GPU the render is waiting for. The local one stays underneath as the
    floor, so a missing key, an outage or a revoked credential slows the
    feature down instead of taking it out.
    """
    hosted = CerebrasDirectorProvider()
    return [hosted, GemmaDirectorProvider()] if hosted.available else [GemmaDirectorProvider()]


async def create_director_plan(
    job: AdapterJob,
    duration_seconds: float,
    *,
    provider: DirectorProvider | None = None,
    providers: list[DirectorProvider] | None = None,
) -> DirectorPlan:
    """Idea → validated DirectorPlan, or a clean `DirectorFailure`.

    Two attempts per provider, and the second is never a blind repeat: it
    carries the previous draft's problems as corrections, whether those were
    validation failures (a rewritten user line, an unknown speaker) or pacing
    complaints (long silences, too few lines). A plan that only fails pacing is
    ACCEPTED on the final attempt — sparse dialogue is a worse video, not a
    broken one, and failing the job over it would serve the customer nothing.

    The stored job keeps the user's idea verbatim throughout; the plan lives
    only in this process and in the log.
    """
    chain = providers or ([provider] if provider is not None else default_providers())
    language = requested_language(job)
    seed = zlib.crc32(f"{job.job_id}:director".encode())
    anchored = source_anchored(job)
    # Optional, off by default, and absorbed on failure: the plan must be as
    # valid without the facts as with them (see worker/director/vision.py).
    facts = await source_image_facts(job) if anchored else ""

    failures: list[str] = []
    best: DirectorPlan | None = None
    attempt = 0

    for source in chain:
        notes: tuple[str, ...] = ()
        for sample in (False, True):
            attempt += 1
            job.raise_if_cancelled()
            request = DirectorRequest(
                idea=job.prompt,
                duration_seconds=duration_seconds,
                language=language,
                seed=seed + attempt,
                sample=sample,
                source_anchored=anchored,
                image_facts=facts,
                notes=notes,
            )
            try:
                raw = await source.generate_plan(request)
                plan = parse_plan(
                    raw,
                    idea=job.prompt,
                    duration_seconds=duration_seconds,
                    language=language,
                    source_anchored=anchored,
                )
            except DirectorProviderUnavailable as unavailable:
                # Not a failed attempt — this provider was never usable. Move
                # to the next one without spending its retry.
                logger.info(
                    "director_provider_skipped",
                    extra={"provider": type(source).__name__, "reason": str(unavailable)},
                )
                failures.append(f"{type(source).__name__}: {unavailable}")
                break
            except DirectorPlanError as error:
                failures.append(f"attempt {attempt}: {error}")
                logger.warning(
                    "director_plan_rejected",
                    extra={"attempt": attempt, "problems": error.problems},
                )
                notes = tuple(error.problems)
                continue

            # Quality complaints, gathered together: both are "this makes a
            # weaker video", neither is "this video is broken", and the retry
            # should fix everything known to be wrong rather than one thing.
            pacing = pacing_problems(plan) + vocabulary_problems(plan)
            if pacing:
                failures.append(f"attempt {attempt} quality: {'; '.join(pacing)}")
                logger.warning(
                    "director_plan_quality",
                    extra={
                        "attempt": attempt,
                        "provider": type(source).__name__,
                        "problems": pacing,
                    },
                )
                # Valid, just sparse. Hold it as the floor and spend the
                # remaining attempt asking for better; if nothing better
                # arrives, this still ships a video.
                best = best or plan
                if not sample:
                    notes = tuple(pacing)
                    continue
                break

            _log_ready(plan, attempt=attempt, provider=type(source).__name__, paced=True)
            return plan

    if best is not None:
        _log_ready(best, attempt=attempt, provider="fallback", paced=False)
        return best

    raise DirectorFailure(
        "We couldn't turn this idea into a scene plan. Try adding a little more "
        "detail to the idea, or switch to the standard prompt mode.",
        internal_detail=" || ".join(failures),
    )


def _log_ready(plan: DirectorPlan, *, attempt: int, provider: str, paced: bool) -> None:
    logger.info(
        "director_plan_ready",
        extra={
            "attempt": attempt,
            "provider": provider,
            "characters": [character.id for character in plan.characters],
            "events": len(plan.timeline),
            "spoken_lines": sum(1 for e in plan.timeline if (e.dialogue or "").strip()),
            "spoken_words": plan.spoken_words,
            "seconds_per_line": plan.seconds_per_spoken_line,
            "language": plan.language,
            # False means the plan shipped despite a pacing complaint, which is
            # the state worth finding in a log when a customer reports silence
            # or a repeated line.
            "well_paced": paced,
        },
    )
