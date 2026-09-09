"""The Music Lyrics Workflow v2.0 — lyrics first, validated, then sung.

## What this orchestrates

The client's specification (ZolexAI Music Lyrics Workflow v2.0, 9 Sep 2026)
in the order it lists: normalise the request, analyse an optional reference,
build a timed blueprint, write the lyrics section by section, validate rhyme
strictly and repair only the failing lines, check density / language /
originality / timing, and only then hand a sheet and a production brief to
the music model. After the audio exists the adapter calls `verify_song`
(`worker/music/verify.py`) and decides on a retry.

Everything below the adapter is reused, not replaced: `plan_song` still
shapes the song, the writer chain (Cerebras → template) still writes it,
`polish_lyrics` still makes every chorus the same chorus. What is new is the
accounting — every second of the song assigned, every line timed, every
rhyme group judged — and the refusal to send an unvalidated sheet onward.

## The one rule that is older than this workflow

A customer's own lyrics are never rewritten. A supplied sheet goes through
every measurement here and the report says what it found, but the only
things that fail it are an empty sheet, a copied reference and the wrong
language. Refusing a customer's words over a rhyme heuristic would be worse
than the problem the workflow solves.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worker.adapters.base import AdapterJob
from worker.core.config import settings
from worker.core.logging import get_logger
from worker.music.blueprint import (
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_SECONDS_PER_LINE,
    SongBlueprint,
    build_blueprint,
)
from worker.music.gates import LyricsValidationFailed, PreflightReport, preflight
from worker.music.quality import Outline, Verdict, judge, outline as write_outline
from worker.music.lyrics import (
    LyricBrief,
    LyricsWriter,
    NoLyricsWriterAvailable,
    SongPlan,
    UnsupportedLyricLanguage,
    parse_sections,
    polish_lyrics,
)
from worker.music.reference import (
    ReferenceInvalid,
    ReferenceProfile,
    ReferenceUnavailable,
    analyse_reference,
    fetch_reference_url,
)
from worker.music.rhyme import SCHEMES, RhymeReport, repair_instructions, validate_rhymes
from worker.music.timing import TimedLyrics, lay_out

logger = get_logger(__name__)

WORKFLOW_VERSION = "music-lyrics/2.0"

POINTS_OF_VIEW: tuple[str, ...] = ("auto", "first_person", "second_person", "third_person")
RHYME_MODES: tuple[str, ...] = ("strict", "relaxed")

_POV_TEXT = {
    "first_person": "first person — the singer is 'I', living it",
    "second_person": "second person — the song speaks to 'you'",
    "third_person": "third person — the song tells someone else's story",
}


@dataclass(frozen=True)
class LyricsOptions:
    """Everything the workflow lets a request or a deployment decide."""

    coverage_target: float = DEFAULT_COVERAGE_TARGET
    max_filler_ratio: float = 0.10
    seconds_per_line: float = DEFAULT_SECONDS_PER_LINE
    rhyme_mode: str = "strict"
    rhyme_scheme: str = "auto"
    point_of_view: str = "auto"
    clean: bool = True
    dry_run: bool = False
    recall_threshold: float = 0.5
    verify_policy: str = "fail"
    max_retries: int = 2
    reference_url: str | None = None
    reference_influence: float = 0.65
    max_write_rounds: int = 3
    max_repair_rounds: int = 5
    quality_judge: bool = True
    coherence_floor: float = 0.90
    grammar_floor: float = 0.95
    max_break_seconds: float = 3.0
    trim: bool = True
    overshoot: float = 1.05
    overshoot_seconds: float = 22.0
    reference_strength: float = 0.3
    reference_similarity_threshold: float = 0.85
    reference_bpm_tolerance: float = 0.03

    @classmethod
    def from_job(cls, job: AdapterJob) -> LyricsOptions:
        params = job.parameters or {}
        execution = job.execution or {}

        def flag(name: str, default: bool) -> bool:
            value = params.get(name, execution.get(name, default))
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value) if value is not None else default

        scheme = str(params.get("rhyme_scheme") or "auto").strip()
        if scheme.upper() not in {s.upper() for s in SCHEMES}:
            scheme = "auto"
        mode = str(params.get("rhyme_mode") or getattr(settings, "music_rhyme_mode", "strict")).strip().lower()
        if mode not in RHYME_MODES:
            mode = "strict"
        pov = str(params.get("point_of_view") or "auto").strip().lower()
        if pov not in POINTS_OF_VIEW:
            pov = "auto"
        url = params.get("reference_audio_url")
        influence = _clamp(_float(params.get("reference_influence"), 0.65), 0.0, 1.0)

        # The client's floor: standard mode never plans for less than 90%.
        coverage = max(
            DEFAULT_COVERAGE_TARGET,
            _clamp(_float(execution.get("vocal_coverage_target"), float(getattr(settings, "music_vocal_coverage_target", 0.9))), 0.0, 1.0),
        )
        return cls(
            coverage_target=coverage,
            max_filler_ratio=_clamp(_float(execution.get("max_filler_ratio"), float(getattr(settings, "music_max_filler_ratio", 0.10))), 0.0, 1.0),
            seconds_per_line=max(1.0, _float(execution.get("seconds_per_line"), float(getattr(settings, "music_v2_seconds_per_line", DEFAULT_SECONDS_PER_LINE)))),
            rhyme_mode=mode,
            rhyme_scheme=scheme,
            point_of_view=pov,
            clean=flag("clean_mode", True),
            dry_run=flag("dry_run", False) or flag("music_dry_run", False),
            recall_threshold=_clamp(_float(execution.get("lyric_recall_threshold"), float(getattr(settings, "music_lyric_recall_threshold", 0.5))), 0.0, 1.0),
            verify_policy=str(execution.get("music_verify_policy") or getattr(settings, "music_verify_policy", "fail")),
            max_retries=max(0, int(execution.get("music_verify_max_retries", getattr(settings, "music_verify_max_retries", 2)) or 0)),
            reference_url=str(url).strip() if isinstance(url, str) and url.strip() else None,
            reference_influence=influence,
            quality_judge=flag("music_quality_judge", bool(getattr(settings, "music_quality_judge", True))),
            coherence_floor=_clamp(_float(execution.get("coherence_floor"), float(getattr(settings, "music_coherence_floor", 0.9))), 0.0, 1.0),
            grammar_floor=_clamp(_float(execution.get("grammar_floor"), float(getattr(settings, "music_grammar_floor", 0.95))), 0.0, 1.0),
            max_break_seconds=max(0.5, _float(execution.get("max_break_seconds"), float(getattr(settings, "music_max_break_seconds", 3.0)))),
            trim=flag("music_trim", bool(getattr(settings, "music_v2_trim", True))),
            overshoot=_clamp(_float(execution.get("music_overshoot"), float(getattr(settings, "music_v2_overshoot", 1.05))), 1.0, 2.0),
            overshoot_seconds=_clamp(_float(execution.get("music_overshoot_seconds"), float(getattr(settings, "music_v2_overshoot_seconds", 22.0))), 0.0, 90.0),
            reference_strength=_clamp(_float(execution.get("reference_strength"), float(getattr(settings, "music_reference_strength", 0.3))), 0.05, 0.6),
            reference_similarity_threshold=_clamp(_float(execution.get("reference_similarity_threshold"), float(getattr(settings, "music_reference_similarity_threshold", 0.85))), 0.0, 1.0),
            reference_bpm_tolerance=_clamp(_float(execution.get("reference_bpm_tolerance"), float(getattr(settings, "music_reference_bpm_tolerance", 0.03))), 0.0, 0.5),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items()}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class PreparedSong:
    written: str
    """The sheet as written: each section once, tagged."""
    sheet: str
    """The sheet the music model receives: every occurrence, in order."""
    blueprint: SongBlueprint
    timed: TimedLyrics
    rhyme: RhymeReport
    preflight: PreflightReport
    brief: LyricBrief
    caption: str
    bpm: int | None
    supplied: bool
    reference: ReferenceProfile | None = None
    key: str | None = None
    outline: Outline | None = None
    verdict: Verdict | None = None
    quality_repairs: int = 0
    writer_name: str = ""
    rounds: int = 0
    repairs: int = 0
    request_sha256: str = ""
    files: dict[str, Path] = field(default_factory=dict)


def request_digest(job: AdapterJob) -> str:
    body = json.dumps(
        {"prompt": job.prompt, "parameters": job.parameters, "workflow": job.workflow_id},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ── The workflow ─────────────────────────────────────────────────────────


async def prepare_song(
    *,
    job: AdapterJob,
    plan: SongPlan,
    brief: LyricBrief,
    writer: LyricsWriter | None,
    options: LyricsOptions,
    workspace: Path,
    reference_path: Path | None = None,
) -> PreparedSong:
    """Steps 1–6 of the specification. Raises `LyricsValidationFailed`."""
    workspace.mkdir(parents=True, exist_ok=True)
    digest = request_digest(job)

    reference = await _reference(options, workspace, reference_path, brief.language)
    # A reference's tempo and key are mandatory targets, not a suggestion
    # weighed by "influence" (client rule, 10 Sep 2026).
    bpm = reference.bpm if reference and reference.bpm else None
    blueprint = build_blueprint(
        plan,
        language=brief.language,
        coverage_target=options.coverage_target,
        seconds_per_line=options.seconds_per_line,
        rhyme_scheme=options.rhyme_scheme,
        bpm=bpm,
    )
    reference_lines = reference.transcript_lines if reference else ()

    supplied = str(job.parameters.get("lyrics") or "").strip()
    try:
        if supplied:
            prepared = _measure_supplied(supplied, blueprint, brief, options, reference_lines)
        else:
            prepared = await _write_and_validate(plan, blueprint, brief, writer, options, reference, reference_lines)
    except LyricsValidationFailed as failure:
        if failure.prepared is not None:
            failure.prepared.reference = reference
            failure.prepared.request_sha256 = digest
            failure.prepared.files = _write_files(workspace, failure.prepared, options)
        raise

    prepared.reference = reference
    prepared.bpm = bpm
    prepared.key = reference.key if reference else None
    prepared.request_sha256 = digest
    prepared.caption = production_caption(job.prompt, blueprint, brief, reference, prepared.timed)
    prepared.files = _write_files(workspace, prepared, options)

    logger.info(
        "music_lyrics_prepared",
        extra={
            "workflow_version": WORKFLOW_VERSION,
            "request_sha256": digest,
            "language": brief.language,
            "supplied": prepared.supplied,
            "writer": prepared.writer_name,
            "rounds": prepared.rounds,
            "repairs": prepared.repairs,
            "lines": len(prepared.timed.lines),
            "planned_vocal_coverage": round(prepared.preflight.planned_vocal_coverage, 3),
            "filler_ratio": round(prepared.preflight.filler_ratio, 3),
            "rhyme_pass_rate": round(prepared.rhyme.pass_rate, 3),
            "rhyme_confidence": prepared.rhyme.confidence,
            "reference": reference.source if reference else None,
            "bpm": bpm,
            "warnings": [p.code for p in prepared.preflight.warnings],
        },
    )
    return prepared


async def _reference(
    options: LyricsOptions, workspace: Path, upload: Path | None, language: str
) -> ReferenceProfile | None:
    """An upload wins over a link when both are given (the spec's rule)."""
    if upload is not None:
        source, path = "upload", upload
    elif options.reference_url:
        try:
            path = await fetch_reference_url(options.reference_url, workspace / "reference")
        except ReferenceUnavailable as exc:
            raise LyricsValidationFailed("REFERENCE_UNAVAILABLE", str(exc)) from exc
        source = "url"
    else:
        return None
    try:
        return await analyse_reference(path, source=source, language_hint=language)
    except ReferenceInvalid as exc:
        raise LyricsValidationFailed("REFERENCE_AUDIO_INVALID", str(exc)) from exc


def _measure_supplied(
    text: str,
    blueprint: SongBlueprint,
    brief: LyricBrief,
    options: LyricsOptions,
    reference_lines: tuple[str, ...],
) -> PreparedSong:
    sections = parse_sections(text)
    timed = lay_out(sections, blueprint)
    rhyme = validate_rhymes(sections, code=brief.language, scheme=blueprint.rhyme_scheme, mode=options.rhyme_mode)
    report = preflight(
        timed, rhyme, brief,
        coverage_target=options.coverage_target,
        max_filler_ratio=options.max_filler_ratio,
        rhyme_mode=options.rhyme_mode,
        reference_lines=reference_lines,
        supplied=True,
    )
    if not report.passed:
        first = report.errors[0]
        raise LyricsValidationFailed(first.code, first.detail)
    # The customer's words go to the model exactly as written — the timed
    # layout is for the report and the LRC, not for the sheet.
    return PreparedSong(
        written=text, sheet=text, blueprint=blueprint, timed=timed, rhyme=rhyme,
        preflight=report, brief=brief, caption="", bpm=blueprint.bpm, supplied=True,
        writer_name="customer",
    )


async def _write_and_validate(
    plan: SongPlan,
    blueprint: SongBlueprint,
    brief: LyricBrief,
    writer: LyricsWriter | None,
    options: LyricsOptions,
    reference: ReferenceProfile | None,
    reference_lines: tuple[str, ...],
) -> PreparedSong:
    if writer is None:
        raise NoLyricsWriterAvailable("no lyrics writer is configured")

    directed = dataclasses.replace(
        brief,
        rhyme_scheme=blueprint.rhyme_scheme,
        perspective=_POV_TEXT.get(options.point_of_view, brief.perspective),
        clean=options.clean,
        section_targets=tuple(blueprint.writer_targets()),
        reference_direction=reference.describe() if reference else "",
        # The floor rate credits coverage; a writer is asked for an ordinary
        # delivery, which is about forty percent denser than the floor.
        syllables_per_line=max(5, round(blueprint.seconds_per_line * blueprint.syllables_per_second * 1.4)),
    )

    # Song concept → narrative outline → lyrics (the client's order).
    story: Outline = Outline()
    if options.quality_judge:
        story = await write_outline(writer, brief)
        if not story.empty:
            directed = dataclasses.replace(directed, outline_text=story.describe())

    best: tuple[str, TimedLyrics, RhymeReport, PreflightReport, Verdict] | None = None
    notes: list[str] | None = None
    rounds = 0
    repairs = 0
    quality_repairs = 0

    for round_index in range(max(1, options.max_write_rounds)):
        rounds = round_index + 1
        try:
            draft = strip_labels(polish_lyrics(await writer.write(directed, plan, notes), plan))
        except (NoLyricsWriterAvailable, UnsupportedLyricLanguage):
            raise
        sections = parse_sections(draft)
        timed = lay_out(sections, blueprint)
        rhyme = validate_rhymes(sections, code=brief.language, scheme=blueprint.rhyme_scheme, mode=options.rhyme_mode)

        # Targeted rhyme repair before the full-sheet verdict: only the
        # failing endings change, everything else stays as written.
        for _ in range(max(0, options.max_repair_rounds)):
            if (rhyme.passed and not rhyme.meter_failing) or not hasattr(writer, "rewrite_lines"):
                break
            repaired = await writer.rewrite_lines(  # type: ignore[attr-defined]
                directed, plan, draft, repair_instructions(rhyme, sections)
            )
            if not repaired or repaired.strip() == draft.strip():
                break
            repairs += 1
            repaired = strip_labels(repaired)
            candidate_sections = parse_sections(polish_lyrics(repaired, plan))
            candidate_rhyme = validate_rhymes(
                candidate_sections, code=brief.language, scheme=blueprint.rhyme_scheme, mode=options.rhyme_mode
            )
            better = candidate_rhyme.pass_rate > rhyme.pass_rate or (
                candidate_rhyme.pass_rate == rhyme.pass_rate
                and candidate_rhyme.meter_pass_rate >= rhyme.meter_pass_rate
            )
            if better:
                draft, sections, rhyme = polish_lyrics(repaired, plan), candidate_sections, candidate_rhyme
                timed = lay_out(sections, blueprint)
            else:
                break

        # The quality judge: coherence, grammar, meaning. Objected lines go
        # back as targeted repairs, then the sheet is judged again.
        verdict = Verdict(1.0, 1.0, (), "judge disabled", measured=False)
        if options.quality_judge:
            verdict = await judge(writer, [l for _, ls in sections for l in ls], story, brief.language)
            for _ in range(2):
                if verdict.passes(coherence_floor=options.coherence_floor, grammar_floor=options.grammar_floor):
                    break
                if not verdict.problems or not hasattr(writer, "rewrite_lines"):
                    break
                repaired = await writer.rewrite_lines(directed, plan, draft, verdict.instructions())  # type: ignore[attr-defined]
                if not repaired or repaired.strip() == draft.strip():
                    break
                quality_repairs += 1
                candidate = strip_labels(polish_lyrics(repaired, plan))
                candidate_sections = parse_sections(candidate)
                candidate_rhyme = validate_rhymes(
                    candidate_sections, code=brief.language, scheme=blueprint.rhyme_scheme, mode=options.rhyme_mode
                )
                if candidate_rhyme.pass_rate < rhyme.pass_rate:
                    break
                draft, sections, rhyme = candidate, candidate_sections, candidate_rhyme
                timed = lay_out(sections, blueprint)
                verdict = await judge(writer, [l for _, ls in sections for l in ls], story, brief.language)

        report = preflight(
            timed, rhyme, brief,
            coverage_target=options.coverage_target,
            max_filler_ratio=options.max_filler_ratio,
            rhyme_mode=options.rhyme_mode,
            reference_lines=reference_lines,
        )
        quality_ok = verdict.passes(coherence_floor=options.coherence_floor, grammar_floor=options.grammar_floor)
        logger.info(
            "music_lyrics_round",
            extra={
                "round": rounds,
                "lines": len(timed.lines),
                "planned_vocal_coverage": round(report.planned_vocal_coverage, 3),
                "filler_ratio": round(report.filler_ratio, 3),
                "rhyme_pass_rate": round(rhyme.pass_rate, 3),
                "coherence": round(verdict.coherence, 2),
                "grammar": round(verdict.grammar, 2),
                "quality_problems": len(verdict.problems),
                "errors": [p.code for p in report.errors],
                "warnings": [p.code for p in report.warnings],
            },
        )
        if best is None or _rank(report, verdict) > _rank(best[3], best[4]):
            best = (draft, timed, rhyme, report, verdict)
        if report.passed and quality_ok:
            break
        notes = report.notes() + repair_instructions(rhyme, sections) + verdict.instructions()

    assert best is not None
    draft, timed, rhyme, report, verdict = best
    prepared = PreparedSong(
        written=draft,
        sheet=timed.sheet(),
        blueprint=blueprint,
        timed=timed,
        rhyme=rhyme,
        preflight=report,
        brief=directed,
        caption="",
        bpm=blueprint.bpm,
        supplied=False,
        writer_name=getattr(writer, "last_writer", "") or getattr(writer, "name", type(writer).__name__),
        rounds=rounds,
        repairs=repairs,
        outline=story,
        verdict=verdict,
        quality_repairs=quality_repairs,
    )
    # The client's floors (0.90 / 0.95) are what the writer is sent back to
    # reach; a REFUSAL needs a clearly bad sheet. The judge is a language
    # model's opinion — on the first live Spanish jobs it scored grammar
    # 0.40 on lyrics a native reader found merely plain — so a job is only
    # refused below the hard floor, and the scores travel with the result.
    hard_floor = float(getattr(settings, "music_quality_hard_floor", 0.5))
    clearly_bad = verdict.measured and (verdict.coherence < hard_floor or verdict.grammar < hard_floor)
    if report.passed and clearly_bad:
        first = verdict.problems[0] if verdict.problems else None
        detail = (
            f"coherence {verdict.coherence:.2f} (floor {options.coherence_floor:.2f}), grammar {verdict.grammar:.2f} "
            f"(floor {options.grammar_floor:.2f}); {len(verdict.problems)} line(s) objected to"
            + (f", first: line {first[0]}: {first[1]}" if first else "")
        )
        raise LyricsValidationFailed("LYRIC_QUALITY_FAILED", detail, prepared=prepared)
    if not report.passed:
        # The best draft and its measurements are kept for the operator even
        # though the job stops here: a refusal with nothing to read is how
        # the first 3-minute failure on the node went unexplained.
        first = report.errors[0]
        raise LyricsValidationFailed(first.code, first.detail, prepared=prepared)
    return prepared


_LABEL = re.compile(r"\s*[\(\[]\s*(?:rhyme\s*)?[A-Za-z]\d?\s*[\)\]]\s*$")


def strip_labels(sheet: str) -> str:
    """Removes a rhyme-group label a writer wrote at the end of a line.

    Asked for an AABB scheme, the hosted model sometimes annotates its own
    work — "Morning breeze dances through the flow (A)" — and the music
    model would sing the "A" (seen on the first live 5-minute song, 9 Sep
    2026). Tags and the words themselves are untouched.
    """
    out: list[str] = []
    for line in sheet.splitlines():
        if line.strip().startswith("["):
            out.append(line)
        else:
            out.append(_LABEL.sub("", line))
    return "\n".join(out)


def _rank(report: PreflightReport, verdict: Verdict | None = None) -> float:
    quality = 0.0 if verdict is None else (verdict.coherence + verdict.grammar - len(verdict.problems) * 0.2)
    return -len(report.errors) * 10 + report.planned_vocal_coverage - report.filler_ratio + report.rhyme_pass_rate + quality


# ── The production brief ─────────────────────────────────────────────────


def production_caption(
    prompt: str,
    blueprint: SongBlueprint,
    brief: LyricBrief,
    reference: ReferenceProfile | None,
    timed: TimedLyrics,
) -> str:
    """The customer's prompt plus the arrangement the sheet was written for.

    The prompt itself is never rewritten; the direction is appended in the
    style-caption vocabulary the music model reads. It states what the
    workflow needs from the performance — every section sung, vocals from
    the start, no long instrumental passages — because a model given a
    dense sheet and no such direction will still sometimes open with forty
    seconds of beat.
    """
    parts = [prompt.strip()]
    direction: list[str] = []
    if blueprint.bpm:
        direction.append(f"{blueprint.bpm} BPM")
    if reference is not None and reference.describe():
        direction.append(reference.describe())
    if prepared_key := getattr(blueprint, "key", None):
        direction.append(f"in {prepared_key}")
    direction.append(f"structure: {blueprint.outline}")
    if not blueprint.sections or blueprint.vocal_sections:
        direction.append(
            "lead vocal sings every lyric section in full, vocals begin within the first seconds, "
            "continuous singing with only brief instrumental gaps, no humming or ad-libs in place of words, "
            "no long instrumental intro or outro, an original melody and an original voice"
        )
        if brief.language and brief.language != "en":
            direction.append(f"sung entirely in {brief.language} with native pronunciation")
    parts.append("; ".join(direction))
    return "\n\n".join(part for part in parts if part)


# ── Files ────────────────────────────────────────────────────────────────


def _write_files(workspace: Path, prepared: PreparedSong, options: LyricsOptions) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def put(name: str, content: str) -> None:
        path = workspace / name
        path.write_text(content, encoding="utf-8")
        files[name] = path

    put("lyrics.txt", prepared.written)
    put("lyrics-sheet.txt", prepared.sheet)
    put("lyrics.json", prepared.timed.to_json())
    put("lyrics.lrc", prepared.timed.to_lrc())
    put("lyrics.srt", prepared.timed.to_srt())
    put("blueprint.json", json.dumps(prepared.blueprint.to_dict(), indent=2))
    put("rhyme-validation.json", json.dumps(prepared.rhyme.to_dict(), indent=2, ensure_ascii=False))
    put("preflight.json", json.dumps(prepared.preflight.to_dict(), indent=2, ensure_ascii=False))
    if prepared.reference is not None:
        put("reference-profile.json", json.dumps(prepared.reference.to_dict(), indent=2))
    put("options.json", json.dumps(options.to_dict(), indent=2))
    if prepared.outline is not None:
        put("outline.json", json.dumps(prepared.outline.to_dict(), indent=2, ensure_ascii=False))
    if prepared.verdict is not None:
        put("quality.json", json.dumps(prepared.verdict.to_dict(), indent=2, ensure_ascii=False))
    return files


__all__ = [
    "POINTS_OF_VIEW",
    "RHYME_MODES",
    "WORKFLOW_VERSION",
    "LyricsOptions",
    "PreparedSong",
    "prepare_song",
    "production_caption",
    "request_digest",
]
