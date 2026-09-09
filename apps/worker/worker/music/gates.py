"""The gates a lyric sheet passes before any audio is made.

The client's rule is that an unvalidated draft never reaches the music
model. This module is the checklist, computed from the timed sheet and
returned as one report — every number the workflow logs, every code it can
fail with — so the decision is made in one place and visible afterwards.

Gates (their names are the failure codes):

  VOCAL_COVERAGE_BELOW_90   planned sung time under the target
  FILLER_ABOVE_LIMIT        filler and unsung time over the allowance
  RHYME_VALIDATION_FAILED   a required rhyme group failed, strict mode,
                            and the validator's confidence is high enough
                            for that to mean something
  LANGUAGE_MISMATCH         the sheet reads as another language
  REFERENCE_SIMILARITY_TOO_HIGH  a run of words copied from the reference
  BLUEPRINT_VALIDATION_FAILED   a vocal section has no lines at all

The detail check ("Lahore" must appear) and repetition are reported as
warnings: they were already part of the older review, and a missing name is
what the writer is *asked to fix*, not what a finished song is refused for.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from worker.music.detect import written_in
from worker.music.lyrics import LyricBrief
from worker.music.rhyme import RhymeReport
from worker.music.timing import TimedLyrics
from worker.music.transcribe import normalise

#: Consecutive words shared with the reference that count as copying. Five
#: is long enough that a common phrase ("I love you so much") is not caught
#: and short enough that a copied line is.
SIMILARITY_NGRAM = 5


class LyricsValidationFailed(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class Problem:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class PreflightReport:
    planned_vocal_coverage: float
    filler_ratio: float
    words: int
    syllables: int
    per_section: list[dict[str, Any]]
    language_ok: bool | None
    rhyme_pass_rate: float
    rhyme_confidence: str
    repetition_score: float
    """Fraction of non-refrain lines that appear more than once."""
    must_keep_missing: tuple[str, ...]
    shared_phrases: tuple[str, ...]
    errors: tuple[Problem, ...]
    warnings: tuple[Problem, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "planned_vocal_coverage": round(self.planned_vocal_coverage, 4),
            "filler_ratio": round(self.filler_ratio, 4),
            "words": self.words,
            "syllables": self.syllables,
            "per_section": self.per_section,
            "language_ok": self.language_ok,
            "rhyme_pass_rate": round(self.rhyme_pass_rate, 3),
            "rhyme_confidence": self.rhyme_confidence,
            "repetition_score": round(self.repetition_score, 3),
            "must_keep_missing": list(self.must_keep_missing),
            "shared_phrases_with_reference": list(self.shared_phrases),
            "errors": [p.to_dict() for p in self.errors],
            "warnings": [p.to_dict() for p in self.warnings],
        }

    def notes(self) -> list[str]:
        """The problems as instructions a writer can act on."""
        out: list[str] = []
        for problem in self.errors + self.warnings:
            out.append(f"{problem.code.lower()}: {problem.detail}")
        return out


def shared_phrases(lines: list[str], reference_lines: tuple[str, ...] | list[str], *, n: int = SIMILARITY_NGRAM) -> list[str]:
    """Word runs of length `n` that appear in both texts, normalised."""
    if not reference_lines:
        return []
    reference_grams: set[tuple[str, ...]] = set()
    for line in reference_lines:
        words = normalise(line).split()
        for i in range(len(words) - n + 1):
            reference_grams.add(tuple(words[i : i + n]))
    if not reference_grams:
        return []
    found: list[str] = []
    for line in lines:
        words = normalise(line).split()
        hits = [i for i in range(len(words) - n + 1) if tuple(words[i : i + n]) in reference_grams]
        # Consecutive hits are one copied run, reported once at full length.
        run_start: int | None = None
        previous = -2
        for position in [*hits, None]:
            if position is not None and position == previous + 1 and run_start is not None:
                previous = position
                continue
            if run_start is not None:
                phrase = " ".join(words[run_start : previous + n])
                if phrase not in found:
                    found.append(phrase)
            run_start, previous = (position, position) if position is not None else (None, -2)
    return found


_REFRAIN = frozenset({"chorus", "hook", "drop", "refrain"})


def preflight(
    timed: TimedLyrics,
    rhyme: RhymeReport,
    brief: LyricBrief,
    *,
    coverage_target: float,
    max_filler_ratio: float,
    rhyme_mode: str,
    reference_lines: tuple[str, ...] = (),
    strict_rhyme_min_confidence: str = "high",
    supplied: bool = False,
) -> PreflightReport:
    """Every gate, computed once.

    `supplied` marks a customer's own sheet: it is measured and reported in
    full, but the only things that *fail* it are an empty sheet and a copied
    reference — the platform never rewrites a customer's words, so it does
    not refuse them over rhyme or density either. The warnings say what a
    generated sheet would have been sent back for.
    """
    errors: list[Problem] = []
    warnings: list[Problem] = []
    lines = [line.text for line in timed.lines]
    sheet_text = "\n".join(lines)

    if not lines:
        errors.append(Problem("BLUEPRINT_VALIDATION_FAILED", "no lyric lines were produced"))

    unwritten = timed.unwritten_sections
    if unwritten:
        # Leanness, not a defect (the older review's rule): the section's
        # time is sung by its neighbour, and the coverage gate below decides
        # whether that neighbour has the syllables to cover it.
        names = ", ".join(f"{s.kind}#{s.occurrence + 1}" for s in unwritten)
        warnings.append(Problem("SECTION_UNWRITTEN", f"planned sections with no lines: {names}"))

    coverage = timed.planned_coverage
    # A hair of tolerance: the blueprint squeezes the wordless parts to
    # exactly one tenth, so a fully written sheet lands on 0.8999….
    if lines and coverage < coverage_target - 0.005:
        deficit = timed.blueprint.duration_seconds * (coverage_target - coverage)
        more = max(1, round(deficit / max(1.0, timed.blueprint.seconds_per_line)))
        problem = Problem(
            "VOCAL_COVERAGE_BELOW_90",
            f"planned sung coverage is {coverage:.0%}; the target is {coverage_target:.0%}. "
            f"Write about {more} more full lines, spread across the sections with the fewest.",
        )
        (warnings if supplied else errors).append(problem)

    filler = timed.filler_ratio
    if lines and filler > max_filler_ratio:
        filler_lines = [line.text for line in timed.lines if line.filler]
        problem = Problem(
            "FILLER_ABOVE_LIMIT",
            f"filler and unsung time is {filler:.0%} of the song; at most {max_filler_ratio:.0%} is allowed"
            + (f". Replace these with real words: {filler_lines[:4]}" if filler_lines else ""),
        )
        (warnings if supplied else errors).append(problem)

    language_ok = written_in(sheet_text, brief.language) if lines else None
    if language_ok is False:
        # A customer's own sheet may be in any language — the selection
        # only says which language WE write in — so for a supplied sheet
        # this is information, not a refusal.
        problem = Problem("LANGUAGE_MISMATCH", f"the lyrics do not read as {brief.language!r}")
        (warnings if supplied else errors).append(problem)

    if rhyme.required and not rhyme.passed:
        failing = ", ".join(f"[{g.section}] group {g.label}: {g.reason}" for g in rhyme.failing[:6])
        problem = Problem("RHYME_VALIDATION_FAILED", f"{len(rhyme.failing)} rhyme group(s) failed — {failing}")
        enforce = (
            rhyme_mode == "strict"
            and not supplied
            and _confidence_rank(rhyme.confidence) >= _confidence_rank(strict_rhyme_min_confidence)
        )
        (errors if enforce else warnings).append(problem)

    shared = shared_phrases(lines, reference_lines)
    if shared:
        errors.append(
            Problem(
                "REFERENCE_SIMILARITY_TOO_HIGH",
                "these word runs are copied from the reference: " + "; ".join(repr(p) for p in shared[:5]),
            )
        )

    counts = Counter(
        normalise(line.text) for line in timed.lines if line.section not in _REFRAIN
    )
    repeated = sum(count for count in counts.values() if count > 1)
    non_refrain = sum(counts.values())
    repetition = repeated / non_refrain if non_refrain else 0.0
    if repetition > 0.3:
        warnings.append(Problem("REPETITION", f"{repetition:.0%} of verse lines are reused verbatim"))

    lowered = sheet_text.lower()
    missing = tuple(detail for detail in brief.must_keep if detail.lower() not in lowered)
    if missing and not supplied:
        warnings.append(Problem("DETAIL_MISSING", f"these must appear and do not: {list(missing)}"))

    return PreflightReport(
        planned_vocal_coverage=coverage,
        filler_ratio=filler,
        words=timed.words,
        syllables=timed.syllables,
        per_section=timed.per_section_rates(),
        language_ok=language_ok,
        rhyme_pass_rate=rhyme.pass_rate,
        rhyme_confidence=rhyme.confidence,
        repetition_score=repetition,
        must_keep_missing=missing,
        shared_phrases=tuple(shared),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _confidence_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


__all__ = ["LyricsValidationFailed", "PreflightReport", "Problem", "preflight", "shared_phrases"]
