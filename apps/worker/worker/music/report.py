"""The job report: what was planned, what was measured, what the customer gets.

Two audiences, one source of truth:

  * `job_report` is everything — written to the workspace as
    `lyrics-report.json` and logged as one record per job, with the fields
    the client's workflow lists under "Logging".
  * `customer_result` is the bounded subset that travels back to the API on
    completion and is shown on the result page: the lyrics in three forms,
    the coverage and rhyme numbers, and any warnings. It carries no model
    names, no file paths and no reference transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worker.music.verify import VerificationReport
from worker.music.workflow import WORKFLOW_VERSION, LyricsOptions, PreparedSong

#: Ceiling on the customer result as serialised JSON. The API refuses more.
RESULT_MAX_BYTES = 64 * 1024


def job_report(
    *,
    job_id: str,
    prompt: str,
    prepared: PreparedSong,
    options: LyricsOptions,
    verification: VerificationReport | None,
    retries: int,
    retry_reasons: list[str],
    status: str,
    failure_code: str | None,
    requested_language: str | None,
    requested_seconds: float,
    actual_seconds: float | None,
) -> dict[str, Any]:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "job_id": job_id,
        "request_sha256": prepared.request_sha256,
        "original_user_prompt": prompt,
        "requested_language": requested_language,
        "language": prepared.brief.language,
        "detected_language": verification.language_detected if verification else None,
        "requested_duration_seconds": round(requested_seconds, 2),
        "actual_duration_seconds": None if actual_seconds is None else round(actual_seconds, 2),
        "reference": {
            "used": prepared.reference is not None,
            "source": prepared.reference.source if prepared.reference else None,
            "analysis": prepared.reference.analysis if prepared.reference else {},
        },
        "blueprint_version": prepared.blueprint.version,
        "lyrics_source": "customer" if prepared.supplied else prepared.writer_name,
        "writer_rounds": prepared.rounds,
        "rhyme_repairs": prepared.repairs,
        "planned_vocal_coverage": round(prepared.preflight.planned_vocal_coverage, 4),
        "filler_ratio": round(prepared.preflight.filler_ratio, 4),
        "measured_vocal_coverage": (
            None if verification is None else verification.measured_vocal_coverage
        ),
        "coverage_method": verification.coverage_method if verification else None,
        "lyric_recall": None if verification is None else verification.lyric_recall,
        "rhyme": prepared.rhyme.to_dict(),
        "originality": {
            "shared_phrases_with_reference": list(prepared.preflight.shared_phrases),
            "result": "failed" if prepared.preflight.shared_phrases else "passed",
        },
        "missing_lines": [line.to_dict() for line in verification.missing_lines] if verification else [],
        "substituted_lines": [line.to_dict() for line in verification.substituted_lines] if verification else [],
        "retry_count": retries,
        "retry_reasons": retry_reasons,
        "status": status,
        "failure_code": failure_code,
        "options": options.to_dict(),
        "preflight": prepared.preflight.to_dict(),
        "verification": verification.to_dict() if verification else None,
    }


def customer_result(
    prepared: PreparedSong,
    verification: VerificationReport | None,
    *,
    retries: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    warnings = [p.detail for p in prepared.preflight.warnings]
    if verification is not None:
        warnings += [detail for _, detail in verification.warnings]
        if not verification.passed:
            warnings += [detail for _, detail in verification.errors]

    result: dict[str, Any] = {
        "workflow": WORKFLOW_VERSION,
        "status": "planned" if dry_run else "completed",
        "language": prepared.brief.language,
        "duration_seconds": round(prepared.blueprint.duration_seconds, 2),
        "lyrics_source": "customer" if prepared.supplied else "generated",
        "lyrics": prepared.written,
        "lyrics_lrc": prepared.timed.to_lrc(),
        "lyrics_srt": prepared.timed.to_srt(),
        "lyrics_json": {
            "sections": [
                {
                    "type": section.blueprint.kind,
                    "start": round(section.blueprint.start, 2),
                    "end": round(section.blueprint.end, 2),
                    "lines": [
                        {
                            "text": line.text,
                            "start": round(line.start, 2),
                            "end": round(line.end, 2),
                            "rhyme_group": line.rhyme_group,
                            "syllables": line.syllables,
                        }
                        for line in section.lines
                    ],
                }
                for section in prepared.timed.sections
                if section.lines
            ]
        },
        "planned_vocal_coverage": round(prepared.preflight.planned_vocal_coverage, 3),
        "measured_vocal_coverage": (
            None
            if verification is None or verification.measured_vocal_coverage is None
            else round(verification.measured_vocal_coverage, 3)
        ),
        "coverage_method": verification.coverage_method if verification else "unmeasured",
        "lyric_recall": (
            None if verification is None or verification.lyric_recall is None else round(verification.lyric_recall, 3)
        ),
        "rhyme_pass_rate": round(prepared.rhyme.pass_rate, 3),
        "rhyme_scheme": prepared.rhyme.scheme,
        "rhyme_validator_confidence": prepared.rhyme.confidence,
        "reference_used": prepared.reference is not None,
        "originality_check": "failed" if prepared.preflight.shared_phrases else "passed",
        "retries": retries,
        "warnings": warnings,
    }
    return _bounded(result)


def _bounded(result: dict[str, Any]) -> dict[str, Any]:
    """Drops the largest optional renderings until the result fits."""
    for key in ("lyrics_srt", "lyrics_json", "lyrics_lrc"):
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= RESULT_MAX_BYTES:
            return result
        result.pop(key, None)
    return result


def write_report(workspace: Path, report: dict[str, Any]) -> Path:
    path = workspace / "lyrics-report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = ["RESULT_MAX_BYTES", "customer_result", "job_report", "write_report"]
