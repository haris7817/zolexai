"""The client's LTX 2.5 guideline pack, applied to a customer's prompt.

*Pack received 9 Sep 2026 (`ltx-2.5-guideline-pack.zip`, eight files). Its own
header calls it "an implementation-oriented paraphrase of the official LTX-2.5
prompting guidance" and cites `ltx.io/blog/ltx-2-5-prompt-guide` and
`docs.ltx.io/models/ltx-2-5`. The files under `guidelines/` are theirs,
unedited; everything in this package is the routing, the assembly and the
validation around them.*

## What this does

One language-model call, before the render, that rewrites the customer's
description into a prompt written to the vendor's rules: a form is chosen
(`forms.select`), the guidelines that govern that form are assembled into a
system prompt (`compose`), and what comes back is checked against the
mechanically decidable half of `validation.txt` (`validate`) with one
corrective round if it fails.

## Where it sits, and why there

Two orderings were possible and only one is safe.

**It runs BEFORE `add_auto_dialogue`.** The pack's screenplay guide says a
rewriter that sees injected lines must treat them as authoritative and "not
add speakers or rewrite exact lines". Running first is a stronger guarantee
than instructing for it: the lines are composed onto an already-finished
prompt, so no model is ever in a position to paraphrase them. This matters
because paraphrase is the measured failure — the first real Auto Dialogue
render (7 Sep 2026) delivered four lines as "one verbatim, three paraphrased,
one phrase twice", and that was the model reading a prompt, not a rewriter
editing one. `validate.check` still enforces the rule for the case where a
customer wrote their own quoted lines, which this call DOES see.

**It stands `prompt_structuring` down for the job it rewrites.** `core.txt`
§7 forbids duplicating the negative prompt in the positive, and
`validation.txt` §11 warns about restating it as prohibitions;
`worker/longform/enhance.py` appends exactly that kind of continuity block.
Running both would say the same thing twice in the box where repetition
costs the most. So a rewritten job carries `prompt_structuring: False`, and
the deterministic structuring returns the moment the rewrite fails.

**`soundscape_clause` still runs.** It is not general prompt craft: it is
three GPU-measured fixes (the model narrating its own caption, a five-word
line looping for twenty seconds, the described-silence tail), and the pack has
no equivalent. Where the two overlap they agree.

## Posture: off, and open

Off unless switched on (`LTX25_GUIDELINES_ENABLED`, or
`execution.ltx25_guidelines`), because this rewrites every prompt on a
surface the client is mid-test on and the prompt text is the product. Open on
failure: a writer that is unavailable, slow, malformed or invalid leaves the
customer's prompt exactly as they wrote it and the job renders as it would
have. Unlike Auto Dialogue there is no "explicitly requested" variant — the
customer never asks for this, so there is never a promise to break.
"""

from __future__ import annotations

from dataclasses import replace

from worker.adapters.base import AdapterJob
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.dialogue.decide import sound_is_on
from worker.dialogue.provider import (
    DialogueProvider,
    DialogueRejected,
    DialogueRequest,
    DialogueUnavailable,
    default_providers,
)
from worker.longform.language import spoken_language_name, supplied_dialogue
from worker.prompt.ltx25 import compose
from worker.prompt.ltx25.forms import Form, quoted_lines, select
from worker.prompt.ltx25.validate import Report, check

logger = get_logger(__name__)

#: The workflows whose prompt describes a scene to invent. The same line
#: `AUTO_DIALOGUE_WORKFLOWS` draws, for the same reason: Extend Video and
#: Video to Video prompts describe a continuation or a restyle, Character
#: Replacement's is a character description wrapped in the pack's own lead
#: sentence, and Music Video has a song. Rewriting any of those to the rules
#: of a standalone scene would break the thing it is actually part of.
LTX25_WORKFLOWS = frozenset({"text-to-video", "image-to-video", "text-to-video-hd"})

#: Workflows whose first frame is a supplied picture.
_ANCHORED = frozenset({"image-to-video"})


def enabled_for(job: AdapterJob) -> bool:
    raw = job.execution.get("ltx25_guidelines")
    if raw is None or str(raw).strip() == "":
        return bool(settings.ltx25_guidelines_enabled)
    return str(raw).strip().lower() not in ("false", "no", "off", "0")


def skip_reason(job: AdapterJob, *, enabled: bool) -> str:
    if not enabled:
        return "disabled"
    if job.workflow_id not in LTX25_WORKFLOWS:
        return "workflow_not_eligible"
    if str(job.parameters.get("prompt_mode") or "").strip().lower() == "director":
        # Director mode writes its own prompt from a plan, with locks and
        # exits this has no idea about. Two writers for one prompt is the
        # failure `worker/dialogue/decide.py` refuses for the same reason.
        return "director_mode_owns_the_prompt"
    if not job.prompt.strip():
        return "empty_prompt"
    return ""


def _answer(raw: dict) -> str:
    text = raw.get("prompt")
    if not isinstance(text, str) or not text.strip():
        raise DialogueRejected("the prompt writer returned no prompt")
    return text.strip()


async def rewrite(
    job: AdapterJob,
    *,
    providers: list[DialogueProvider] | None = None,
) -> tuple[str, Form, Report] | None:
    """The rewritten prompt, or None when it could not be written.

    Public and side-effect free so a test — and a GPU day — can see the form,
    the text and the validator's verdict without running a render.
    """
    text = job.prompt.strip()
    sound_on = sound_is_on(job)
    anchored = job.workflow_id in _ANCHORED
    form = select(
        text,
        workflow_id=job.workflow_id,
        requested=str(job.execution.get("ltx25_form") or ""),
        anchored=anchored,
    )

    # Lines the customer wrote themselves. Auto Dialogue has not run yet — by
    # design, see the module docstring — so these are the only spoken words
    # that legitimately exist, and the validator treats every other quoted
    # string as an invention.
    supplied = tuple(quoted_lines(text)) if supplied_dialogue(text) else ()
    dialogue_allowed = sound_on and bool(supplied)
    language = spoken_language_name(job.parameters, job.execution)

    system = compose.system_prompt(form, sound_on=sound_on)
    user = compose.user_prompt(
        text,
        form,
        language=language if supplied else "",
        supplied_lines=supplied,
        dialogue_allowed=dialogue_allowed,
        anchored=anchored,
    )

    chain = providers if providers is not None else default_providers()
    if not chain:
        logger.info(
            "ltx25_skipped",
            extra={"job_id": job.job_id, "reason": "no_provider_available"},
        )
        return None

    for provider in chain:
        name = getattr(provider, "name", type(provider).__name__)
        ask = user
        for attempt in (1, 2):
            request = DialogueRequest(
                prompt=text, seconds=0.0, language=language, system=system, user=ask
            )
            try:
                written = _answer(await provider.write(request))
            except DialogueUnavailable as exc:
                logger.info(
                    "ltx25_provider_unavailable",
                    extra={"job_id": job.job_id, "provider": name, "detail": str(exc)},
                )
                break
            except DialogueRejected as exc:
                logger.warning(
                    "ltx25_provider_rejected",
                    extra={"job_id": job.job_id, "provider": name, "attempt": attempt,
                           "detail": str(exc)},
                )
                break
            except Exception as exc:  # noqa: BLE001 — fail open, always
                logger.warning(
                    "ltx25_provider_failed",
                    extra={"job_id": job.job_id, "provider": name,
                           "error": type(exc).__name__},
                )
                break

            report = check(
                written,
                form,
                sound_on=sound_on,
                dialogue_allowed=dialogue_allowed,
                supplied_lines=supplied,
            )
            if not report.errors:
                return written, form, report
            logger.warning(
                "ltx25_invalid",
                extra={
                    "job_id": job.job_id,
                    "provider": name,
                    "attempt": attempt,
                    "rules": [f.rule for f in report.errors],
                    "detail": report.instruction()[:400],
                },
            )
            if attempt == 2:
                break
            ask = compose.retry_prompt(user, report.instruction())

    return None


async def apply_guidelines(
    job: AdapterJob,
    *,
    providers: list[DialogueProvider] | None = None,
) -> AdapterJob:
    """The job, with its prompt rewritten to the pack — or unchanged.

    Never raises. Every failure is the customer's own prompt, rendered the way
    it would have been before this package existed.
    """
    reason = skip_reason(job, enabled=enabled_for(job))
    if reason:
        logger.info("ltx25_skipped", extra={"job_id": job.job_id, "reason": reason})
        return job

    try:
        written = await rewrite(job, providers=providers)
    except Exception as exc:  # noqa: BLE001 — fail open, always
        logger.warning(
            "ltx25_failed",
            extra={"job_id": job.job_id, "error": type(exc).__name__},
        )
        return job

    if written is None:
        logger.info(
            "ltx25_skipped",
            extra={"job_id": job.job_id, "reason": "no_valid_prompt_written"},
        )
        return job

    prompt, form, report = written
    logger.info(
        "ltx25_rewritten",
        extra={
            "job_id": job.job_id,
            "form": form.value,
            "result": report.result,
            "warnings": [f.rule for f in report.warnings],
            "chars_before": len(job.prompt.strip()),
            "chars_after": len(prompt),
        },
    )
    # The deterministic structuring stands down for this job only: the
    # rewritten prompt already carries its own continuity, and saying it twice
    # is `validation.txt` §11. A job that was NOT rewritten keeps it.
    return replace(
        job, prompt=prompt, execution={**job.execution, "prompt_structuring": False}
    )


__all__ = [
    "LTX25_WORKFLOWS",
    "Form",
    "apply_guidelines",
    "enabled_for",
    "rewrite",
    "skip_reason",
]
