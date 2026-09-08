"""Automatic dialogue — a video that talks without the customer writing lines.

The client's brief (`LTX25_Dialogue-2.zip`, 7 Sep 2026) asks for one thing at
its centre: when Auto Dialogue is on and the scene has somebody who could
speak, the video should speak, and it should say something worth saying.

## What this is, and what it deliberately is not

The client's pack pairs that idea with a second one — sectioned video, one
external master voice track, post-stitch lip synchronisation — which needs a
text-to-speech service holding a persistent voice identity and a lip-sync
stage. This platform has neither, and the pack itself refuses a multi-section
spoken job when no `voice_id` is configured rather than ship a video whose
voice changes halfway. That guard is right and it is not worked around here.

So this delivers the half that runs today: **LTX generates the speech itself**,
in its own voice, with its own lip movement, in a single pass. That is a real
constraint, not a stopgap dressed up — one pass means one voice, so it is
honest for the 8- and 15-second lengths the product offers on this path and
would not be honest across a seam. Nothing here sections anything.

## How little had to change

`worker/longform/language.py` already turns the soundtrack on the presence of
quoted words in the prompt: with none, the model is told plainly that nobody
speaks; with them, it speaks exactly those words, once, in a stated language.
Every measurement behind that — the narration leak, the repeated line, the
described-silence tail — still applies unchanged. This module's whole job is
to put quoted words into a prompt that had none, and then get out of the way.

Which is why the adapters take one line each and no downstream stage knows
this exists.

## Posture

Off unless switched on (`AUTO_DIALOGUE_ENABLED`), skipped for any prompt that
already has dialogue or asks for silence, and open on failure: a video whose
dialogue could not be written is rendered from the prompt the customer wrote,
exactly as it would have been before this module existed.
"""

from __future__ import annotations

from dataclasses import replace

from worker.adapters.base import AdapterError, AdapterJob
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.dialogue.decide import (
    AUTO_DIALOGUE_WORKFLOWS,
    Dialogue,
    Line,
    Speaker,
    compose,
    skip_reason,
)
from worker.dialogue.provider import (
    DialogueProvider,
    DialogueRejected,
    DialogueRequest,
    DialogueUnavailable,
    default_providers,
    parse,
)
from worker.dialogue import native
from worker.longform.language import spoken_language_name

logger = get_logger(__name__)


#: Skip reasons that are an ANSWER, not a failure. Each one describes a video
#: that is correct without generated lines — the customer asked for silence,
#: wrote their own dialogue, turned the sound off, or picked a workflow whose
#: prompt continues something rather than inventing a scene. A job that asked
#: for Auto Dialogue explicitly still renders on these, and the log says which
#: one closed the gate.
HONOURED_REFUSALS = frozenset(
    {
        "workflow_not_eligible",
        "director_mode_owns_the_dialogue",
        "sound_off",
        "empty_prompt",
        "forbidden_by_prompt",
        "dialogue_already_present",
        "too_short_to_speak",
    }
)


def requested_explicitly(job: AdapterJob) -> bool:
    """Whether this job's own parameters ASKED for automatic dialogue.

    The distinction that `enabled_for` deliberately flattens. A deployment
    default that quietly falls open is the right posture — the feature is an
    improvement on a prompt, and a writer being rate-limited should not lose a
    render. An explicit request is a different contract: the customer ticked
    the box, the panel promised speech, and handing back a silent video that
    looks like a successful generation is the failure the client reported on
    8 Sep 2026. So an explicit request that could not be written stops the job
    with a message, and the deployment default still falls open.
    """
    raw = job.parameters.get("auto_dialogue")
    if raw is None or str(raw).strip() == "":
        return False
    return str(raw).strip().lower() not in ("false", "no", "off", "0")


def max_speakers_for(job: AdapterJob) -> int:
    """How many visible people the writer may give words to.

    The client's request carries `maximum_speakers`; their own validator's
    ceiling (`native.MAX_SPEAKERS`) is both the default and the cap, because
    a screenplay with five voices in one pass is a voice the model has to
    invent and then hold, and it does not.
    """
    raw = job.parameters.get("maximum_speakers")
    if raw is None or str(raw).strip() == "":
        raw = job.execution.get("maximum_speakers")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return native.MAX_SPEAKERS
    return max(1, min(value, native.MAX_SPEAKERS))


def _unwritten(job: AdapterJob, detail: str) -> AdapterError:
    return AdapterError(
        "Auto Dialogue is on, but the dialogue writer could not be reached, "
        "so this video would have come back with nobody speaking. Please try "
        "again, or turn Auto Dialogue off to generate from your prompt as "
        "written.",
        internal_detail=f"auto_dialogue requested explicitly but no plan was produced: {detail}",
        retriable=True,
    )


async def _write_native(
    job: AdapterJob,
    seconds: float,
    chain: list[DialogueProvider],
    language: str,
    *,
    strict: bool = False,
    max_speakers: int = native.MAX_SPEAKERS,
) -> AdapterJob:
    """The client's native-dialogue format (their package's second revision,
    8 Sep 2026): one screenplay prompt, validated by their rules, for one
    native LTX audio-video pass. See `worker/dialogue/native.py`.

    Same posture as the older layouts: every failure falls open to the
    customer's own prompt, and a scene with nobody to speak is an answer.
    """
    language = language or "English"
    request = DialogueRequest(
        prompt=job.prompt.strip(),
        seconds=seconds,
        language=language,
        system=native.system_prompt(seconds, max_speakers=max_speakers),
        user=native.user_prompt(job.prompt, seconds, language),
    )
    for provider in chain:
        name = getattr(provider, "name", type(provider).__name__)
        plan = None
        attempt_request = request
        # The client's validator is strict on purpose, and a writer's first
        # answer misses it more often than not on exactly one thing — three
        # words over the range, a turn written as a string. Measured 8 Sep
        # 2026: the hosted writer returned 37 words for a 24–34 range and the
        # job fell open to a silent video. One corrective retry, naming the
        # rule that failed, costs two seconds and turns that into a pass.
        low, high = native.word_range(seconds)
        for attempt in (1, 2, 3):
            try:
                raw = await provider.write(attempt_request)
                if raw.get("has_speaker") is False:
                    logger.info(
                        "auto_dialogue_no_speaker", extra={"job_id": job.job_id, "provider": name}
                    )
                    return job
                plan = native.validate_script(raw, seconds, max_speakers=max_speakers)
                break
            except DialogueUnavailable as exc:
                logger.info(
                    "auto_dialogue_provider_unavailable",
                    extra={"job_id": job.job_id, "provider": name, "detail": str(exc)},
                )
                break
            except (DialogueRejected, native.NativeDialogueRejected) as exc:
                logger.warning(
                    "auto_dialogue_provider_rejected",
                    extra={"job_id": job.job_id, "provider": name, "attempt": attempt,
                           "detail": str(exc)},
                )
                attempt_request = DialogueRequest(
                    prompt=request.prompt, seconds=seconds, language=language,
                    system=request.system,
                    user=native.user_prompt(job.prompt, seconds, language)
                    + f"\n\nYour previous script was rejected: {exc}. "
                    f"Fix exactly that — aim for {(low + high) // 2} spoken words in total, "
                    f"between {low} and {high} — and return the complete JSON again.",
                )
                continue
            except Exception as exc:  # noqa: BLE001 — fail open, always
                logger.warning(
                    "auto_dialogue_provider_failed",
                    extra={"job_id": job.job_id, "provider": name, "error": type(exc).__name__},
                )
                break
        if plan is None:
            continue

        enriched = native.compose_native_prompt(plan, language)
        logger.info(
            "auto_dialogue_written",
            extra={
                "job_id": job.job_id,
                "provider": name,
                "seconds": seconds,
                "lines": len(plan.turns),
                "words": plan.total_words,
                "speakers": len(plan.speakers),
                "language": language,
                "layout": "native",
                "original_prompt_chars": len(job.prompt),
            },
        )
        return replace(job, prompt=enriched)

    if strict:
        raise _unwritten(job, "every provider failed on the native layout")
    logger.info(
        "auto_dialogue_skipped",
        extra={"job_id": job.job_id, "reason": "every_provider_failed"},
    )
    return job


def enabled_for(job: AdapterJob) -> bool:
    """Whether this job asked for automatic dialogue.

    An explicit `auto_dialogue` parameter wins; without one the deployment's
    default applies. The parameter is read but not yet offered in any
    workflow's YAML, so today this is the environment switch — which is what
    lets the feature reach a test deployment without a schema change on a
    surface the client is mid-test on.
    """
    raw = job.parameters.get("auto_dialogue")
    if raw is None or str(raw).strip() == "":
        return bool(settings.auto_dialogue_enabled)
    return str(raw).strip().lower() not in ("false", "no", "off", "0")


async def add_auto_dialogue(
    job: AdapterJob,
    seconds: float,
    *,
    providers: list[DialogueProvider] | None = None,
    carries_soundscape_clause: bool = True,
) -> AdapterJob:
    """The job, with spoken lines written into its prompt — or unchanged.

    Raises only for a job that asked for Auto Dialogue **in its own
    parameters** and could not be given any (`requested_explicitly`). Where
    the feature comes from the deployment default it never raises: the only
    thing a caller can do with that failure is render the video anyway, and
    that decision is made once, here, rather than at three call sites.

    `carries_soundscape_clause` says whether the caller's prompt already picks
    up `soundscape_clause` downstream. The ComfyUI text-to-video path does, so
    its anti-repeat rule arrives on its own; the HD path builds its prompt
    from the job's text alone, so this composes the rule in. Neither path
    changes at all when no line is written.
    """
    strict = requested_explicitly(job)
    reason = skip_reason(job, seconds, enabled=enabled_for(job))
    if reason:
        logger.info(
            "auto_dialogue_skipped",
            extra={"job_id": job.job_id, "reason": reason, "requested": strict},
        )
        if strict and reason not in HONOURED_REFUSALS:
            raise _unwritten(job, reason)
        return job

    language = spoken_language_name(job.parameters, job.execution)
    chain = providers if providers is not None else default_providers()
    if not chain:
        logger.info(
            "auto_dialogue_skipped",
            extra={"job_id": job.job_id, "reason": "no_provider_available"},
        )
        if strict:
            raise _unwritten(
                job, "no dialogue writer is configured on this deployment"
            )
        return job

    layout = str(job.execution.get("auto_dialogue_layout") or settings.auto_dialogue_layout)
    layout = layout.strip().lower()
    if layout == "native":
        return await _write_native(
            job,
            seconds,
            chain,
            language,
            strict=strict,
            max_speakers=max_speakers_for(job),
        )

    request = DialogueRequest(prompt=job.prompt.strip(), seconds=seconds, language=language)
    for provider in chain:
        name = getattr(provider, "name", type(provider).__name__)
        try:
            dialogue = parse(await provider.write(request), request)
        except DialogueUnavailable as exc:
            logger.info(
                "auto_dialogue_provider_unavailable",
                extra={"job_id": job.job_id, "provider": name, "detail": str(exc)},
            )
            continue
        except DialogueRejected as exc:
            logger.warning(
                "auto_dialogue_provider_rejected",
                extra={"job_id": job.job_id, "provider": name, "detail": str(exc)},
            )
            continue
        except Exception as exc:  # noqa: BLE001 — fail open, always
            logger.warning(
                "auto_dialogue_provider_failed",
                extra={"job_id": job.job_id, "provider": name, "error": type(exc).__name__},
            )
            continue

        if dialogue is None:
            # Not a failure: the scene has nobody to speak. A landscape stays
            # a landscape, and asking the next provider would only be shopping
            # for a different answer to a question already answered.
            logger.info(
                "auto_dialogue_no_speaker",
                extra={"job_id": job.job_id, "provider": name},
            )
            return job

        enriched = compose(
            job.prompt.strip(),
            dialogue,
            add_speech_rule=not carries_soundscape_clause,
            layout=layout,
        )
        if enriched == job.prompt.strip():
            return job
        logger.info(
            "auto_dialogue_written",
            extra={
                "job_id": job.job_id,
                "provider": name,
                "seconds": seconds,
                "lines": len(dialogue.lines),
                "words": dialogue.words,
                "speakers": len(dialogue.speakers),
                "language": request.language or "unstated",
                "layout": layout,
            },
        )
        return replace(job, prompt=enriched)

    if strict:
        raise _unwritten(job, "every provider failed")
    logger.info(
        "auto_dialogue_skipped",
        extra={"job_id": job.job_id, "reason": "every_provider_failed"},
    )
    return job


__all__ = [
    "AUTO_DIALOGUE_WORKFLOWS",
    "HONOURED_REFUSALS",
    "Dialogue",
    "Line",
    "Speaker",
    "add_auto_dialogue",
    "compose",
    "enabled_for",
    "max_speakers_for",
    "requested_explicitly",
    "skip_reason",
]
