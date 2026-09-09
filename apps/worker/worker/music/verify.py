"""Did the model actually sing the sheet? Measured, after the audio exists.

Planning a ninety-percent sung song and delivering one are different
claims, and the August measurements are the reason this module exists: the
same sheet produced 52.8% and 87.7% sung time on different runs, and what
the model pads with is not silence but vamping — "City summer night" six
times. A stem detector hears a singer; the customer hears the lyrics stop.
So two things are measured, on two different instruments:

  * **Vocal coverage** — how much of the song is sung at all — from the
    Demucs vocal stem, language-proof and lexically blind. Without a stem
    separator it falls back to the transcript's word spans.
  * **Lyric recall** — how many of the approved lines were sung — from a
    word-level transcript aligned to the sheet line by line. This is the
    lexical measure the stem cannot give.

Both degrade to "unmeasured" when the tool is not on the node, and say so.
The verdict — retry, deliver, fail — is the adapter's policy, not this
file's; this file reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from worker.media.vocals import spans_from_envelope, vocal_activity, vocal_fraction
from worker.music.timing import TimedLyrics
from worker.music.transcribe import Transcript, normalise, transcribe

_HOP = 0.05

#: Alignment scores. A line scoring at least `MATCH` was sung as written; one
#: between `SUBSTITUTED` and `MATCH` was sung with some words changed; below
#: that it was not heard.
MATCH_SCORE = 0.62
SUBSTITUTED_SCORE = 0.35


@dataclass(frozen=True)
class LineRecall:
    index: int
    section: str
    text: str
    score: float
    status: str
    """"matched", "substituted" or "missing"."""
    heard: str
    start: float | None
    end: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "section": self.section,
            "text": self.text,
            "score": round(self.score, 3),
            "status": self.status,
            "heard": self.heard,
            "start": None if self.start is None else round(self.start, 2),
            "end": None if self.end is None else round(self.end, 2),
        }


@dataclass(frozen=True)
class VerificationReport:
    duration_seconds: float
    measured_vocal_coverage: float | None
    coverage_method: str
    """"stem", "transcript" or "unmeasured"."""
    longest_gap_seconds: float | None
    lyric_recall: float | None
    recall_method: str
    lines: tuple[LineRecall, ...]
    language_detected: str | None
    language_probability: float | None
    language_ok: bool | None
    duration_ok: bool
    coverage_target: float
    recall_threshold: float
    errors: tuple[tuple[str, str], ...]
    warnings: tuple[tuple[str, str], ...]

    @property
    def measured(self) -> bool:
        return self.coverage_method != "unmeasured" or self.recall_method != "unmeasured"

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def missing_lines(self) -> tuple[LineRecall, ...]:
        return tuple(line for line in self.lines if line.status == "missing")

    @property
    def substituted_lines(self) -> tuple[LineRecall, ...]:
        return tuple(line for line in self.lines if line.status == "substituted")

    def score(self) -> float:
        """For choosing the best of several takes: coverage plus recall."""
        return (self.measured_vocal_coverage or 0.0) + (self.lyric_recall or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "measured": self.measured,
            "duration_seconds": round(self.duration_seconds, 2),
            "measured_vocal_coverage": (
                None if self.measured_vocal_coverage is None else round(self.measured_vocal_coverage, 4)
            ),
            "coverage_method": self.coverage_method,
            "coverage_target": self.coverage_target,
            "longest_gap_seconds": None if self.longest_gap_seconds is None else round(self.longest_gap_seconds, 2),
            "lyric_recall": None if self.lyric_recall is None else round(self.lyric_recall, 4),
            "recall_method": self.recall_method,
            "recall_threshold": self.recall_threshold,
            "language_detected": self.language_detected,
            "language_probability": (
                None if self.language_probability is None else round(self.language_probability, 3)
            ),
            "language_ok": self.language_ok,
            "duration_ok": self.duration_ok,
            "missing_lines": [line.to_dict() for line in self.missing_lines],
            "substituted_lines": [line.to_dict() for line in self.substituted_lines],
            "lines": [line.to_dict() for line in self.lines],
            "errors": [{"code": code, "detail": detail} for code, detail in self.errors],
            "warnings": [{"code": code, "detail": detail} for code, detail in self.warnings],
        }


# ── Alignment ────────────────────────────────────────────────────────────

_CJK_CODES = frozenset({"zh", "ja"})


def _tokens(text: str, code: str) -> list[str]:
    body = normalise(text)
    if code in _CJK_CODES:
        return [ch for ch in body if not ch.isspace()]
    return body.split()


def align_lines(timed: TimedLyrics, transcript: Transcript, *, code: str) -> list[LineRecall]:
    """Each sheet line against the best-matching run of transcript words.

    Greedy and in order: each line is searched from where the previous
    line landed, with a short back-off, so a chorus that the model sang
    early or late is still found near its neighbours rather than matched to
    an identical chorus at the other end of the song.
    """
    words = list(transcript.words)
    heard = [_tokens(w.text, code) for w in words]
    # A transcript word may split into several tokens (CJK); keep a map from
    # token position back to the word for timing.
    flat: list[str] = []
    owner: list[int] = []
    for position, tokens in enumerate(heard):
        for token in tokens:
            flat.append(token)
            owner.append(position)

    results: list[LineRecall] = []
    cursor = 0
    for line in timed.lines:
        wanted = _tokens(line.text, code)
        if not wanted or not flat:
            results.append(LineRecall(line.index, line.section, line.text, 0.0, "missing", "", None, None))
            continue
        width = len(wanted)
        best_score, best_at, best_width = 0.0, None, width
        # Words already matched to an earlier line are not available again
        # — beyond a little slack for a boundary the aligner drew a word or
        # two late. Without this an unsung line borrows its neighbour's.
        lowest = max(0, cursor - max(1, width // 3))
        for size in {max(1, width - 2), width, width + 2}:
            for start in range(lowest, max(lowest, len(flat) - size) + 1):
                window = flat[start : start + size]
                score = SequenceMatcher(None, wanted, window, autojunk=False).ratio()
                if score > best_score:
                    best_score, best_at, best_width = score, start, size
        # A weak match far ahead is not this line sung with changed words;
        # it is a later line's words, and claiming them would lose every
        # line in between. A strong match ahead IS credible — a skipped
        # section — and is kept.
        far_ahead = best_at is not None and best_at > cursor + 2 * width
        if best_at is None or best_score < SUBSTITUTED_SCORE or (best_score < MATCH_SCORE and far_ahead):
            results.append(LineRecall(line.index, line.section, line.text, best_score, "missing", "", None, None))
            continue
        status = "matched" if best_score >= MATCH_SCORE else "substituted"
        first, last = owner[best_at], owner[min(len(owner) - 1, best_at + best_width - 1)]
        results.append(
            LineRecall(
                line.index,
                line.section,
                line.text,
                best_score,
                status,
                " ".join(w.text for w in words[first : last + 1]),
                words[first].start,
                words[last].end,
            )
        )
        # Only a proper match consumes its words. A weak ("substituted")
        # match is often the NEXT line heard through a missing one; leaving
        # its words available lets that next line claim them.
        cursor = best_at + best_width if status == "matched" else best_at
    return results


def _word_spans(transcript: Transcript, duration: float) -> list[tuple[float, float]]:
    env = [0.0] * (int(duration / _HOP) + 1)
    for w in transcript.words:
        for i in range(int(w.start / _HOP), min(len(env), int(w.end / _HOP) + 1)):
            env[i] = 1.0
    return spans_from_envelope(env, abs_floor=0.5, rel_fraction=0.5)


def _longest_gap(spans: list[tuple[float, float]], duration: float) -> float:
    if not spans:
        return duration
    longest = spans[0][0]
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        longest = max(longest, start - end)
    return max(longest, duration - spans[-1][1])


async def verify_song(
    path: Path,
    timed: TimedLyrics,
    *,
    language: str,
    expected_seconds: float,
    duration_seconds: float,
    coverage_target: float,
    recall_threshold: float,
    duration_tolerance_seconds: float = 2.0,
    transcribe_fn=None,
    vocal_activity_fn=None,
) -> VerificationReport:
    # Resolved at call time so a test can swap the module's tools.
    transcribe_fn = transcribe_fn or transcribe
    vocal_activity_fn = vocal_activity_fn or vocal_activity
    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    duration = float(duration_seconds)

    duration_ok = abs(duration - expected_seconds) <= duration_tolerance_seconds
    if not duration_ok:
        errors.append(("OUTPUT_DURATION_MISMATCH", f"delivered {duration:.1f}s for a {expected_seconds:.0f}s request"))

    # ── Coverage, from the stem ──────────────────────────────────────
    spans = await vocal_activity_fn(path)
    coverage: float | None = None
    method = "unmeasured"
    gap: float | None = None
    if spans is not None:
        coverage = vocal_fraction(spans, 0.0, duration)
        gap = _longest_gap(spans, duration)
        method = "stem"

    # ── Recall, from the transcript ──────────────────────────────────
    transcript = await transcribe_fn(path, language=language)
    lines: tuple[LineRecall, ...] = ()
    recall: float | None = None
    recall_method = "unmeasured"
    detected: str | None = None
    probability: float | None = None
    language_ok: bool | None = None
    if transcript is not None:
        detected = transcript.language
        probability = transcript.language_probability
        if timed.lines:
            aligned = align_lines(timed, transcript, code=language)
            lines = tuple(aligned)
            # Recall is "the line was sung", so a line heard with some words
            # changed counts. Whisper on sung vocals is reliable for whether a
            # verse appeared and weak on the exact words (runbook §36.6;
            # measured 9 Sep 2026 on Spanish takes: 40% exact, every line
            # audibly present). The exact-match share is still reported.
            heard = sum(1 for line in aligned if line.status in {"matched", "substituted"})
            recall = heard / len(aligned)
            recall_method = "transcript"
        if coverage is None and transcript.words:
            word_spans = _word_spans(transcript, duration)
            coverage = vocal_fraction(word_spans, 0.0, duration)
            gap = _longest_gap(word_spans, duration)
            method = "transcript"
        if detected and probability is not None and probability >= 0.8:
            language_ok = detected == language
            if not language_ok:
                warnings.append(("LANGUAGE_MISMATCH", f"the transcriber heard {detected!r}, not {language!r}"))

    # The same hair of tolerance the preflight gives: the stem envelope is
    # measured in 50 ms hops, and a take at 89.7% is the target reached.
    if coverage is not None and coverage < coverage_target - 0.005:
        errors.append(
            (
                "VOCAL_COVERAGE_BELOW_90",
                f"measured sung coverage is {coverage:.0%} ({method}); the target is {coverage_target:.0%}"
                + (f"; longest unsung gap {gap:.0f}s" if gap else ""),
            )
        )
    if recall is not None and recall < recall_threshold:
        missing = [line.text for line in lines if line.status == "missing"][:5]
        errors.append(
            (
                "LYRIC_RECALL_TOO_LOW",
                f"{recall:.0%} of the lines were heard as written; the threshold is {recall_threshold:.0%}. "
                f"Not heard: {missing}",
            )
        )
    if coverage is None and recall is None:
        warnings.append(("UNMEASURED", "neither a stem separator nor a transcriber is available on this node"))

    return VerificationReport(
        duration_seconds=duration,
        measured_vocal_coverage=coverage,
        coverage_method=method,
        longest_gap_seconds=gap,
        lyric_recall=recall,
        recall_method=recall_method,
        lines=lines,
        language_detected=detected,
        language_probability=probability,
        language_ok=language_ok,
        duration_ok=duration_ok,
        coverage_target=coverage_target,
        recall_threshold=recall_threshold,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


__all__ = ["MATCH_SCORE", "SUBSTITUTED_SCORE", "LineRecall", "VerificationReport", "align_lines", "verify_song"]
