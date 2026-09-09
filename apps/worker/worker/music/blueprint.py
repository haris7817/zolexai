"""The song blueprint: where every second of the song goes, before a word.

`plan_song` (`worker/music/lyrics.py`) decides a song's *shape* — intro,
verse, chorus, bridge, outro — for a genre and a length. The blueprint takes
that shape and makes it accountable to the client's lyrics workflow:

  * at least 90% of the song must carry sung words, so the wordless parts
    (intro, outro, breaks) are squeezed to fit the remaining tenth;
  * every vocal section gets a start, an end, a line target and a rhyme
    scheme, so the writer is asked for a specific amount of a specific thing
    rather than "some lyrics";
  * a chorus repeats — the writer writes it once, the sheet carries it at
    every occurrence, and the timeline counts every occurrence.

The line target is time divided by seconds-per-line. That number is the most
important one in the module and it is NOT the 8 s/line the older path aims
for: the August matrix (see `_SECONDS_PER_LINE` in `lyrics.py`) shows the
model only reaches ninety-percent coverage at roughly 3.5–4 s/line. Denser
sheets take away the intros and breaks a song normally has — which is
exactly what a ninety-percent rule asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from worker.music.lyrics import Section, SongPlan

#: Seconds of song per sung line the blueprint targets. Measured, see above.
DEFAULT_SECONDS_PER_LINE = 4.0

#: The client's floor for sung coverage, and the ceiling on everything else.
DEFAULT_COVERAGE_TARGET = 0.90

#: The SLOWEST plausible sung delivery, in syllables per second, by genre.
#:
#: Used to turn a line into a *planned* vocal-seconds figure: a line is
#: credited with `syllables / rate` seconds of singing, capped at its slot.
#: Slow on purpose — a singer stretches a short line over a bar, so crediting
#: a fast rate would call a sheet of three-word lines "half instrumental"
#: when it is not. What the number still catches is the real defect: a
#: section of two short lines over thirty seconds cannot be sung for thirty
#: seconds at any tempo, and the gate says so before the audio exists.
SYLLABLES_PER_SECOND: dict[str, float] = {
    "rap": 3.5,
    "hip-hop": 3.5,
    "drill": 3.2,
    "trap": 3.0,
    "ballad": 1.4,
    "acoustic": 1.6,
    "folk": 1.6,
    "r&b": 1.8,
    "soul": 1.8,
    "lo-fi": 1.6,
}
DEFAULT_SYLLABLES_PER_SECOND = 1.8

#: Section kinds that carry no words. Mirrors `Section.carries_words`.
_WORDLESS_KINDS = frozenset({"intro", "outro", "solo", "movement", "breakdown", "build"})

_REFRAIN_KINDS = frozenset({"chorus", "hook", "drop", "refrain"})


@dataclass(frozen=True)
class BlueprintSection:
    index: int
    kind: str
    start: float
    end: float
    vocal_required: bool
    target_lines: int
    rhyme_scheme: str
    occurrence: int
    """Which occurrence of this kind this is (0 for the first chorus…)."""

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def is_refrain(self) -> bool:
        return self.kind in _REFRAIN_KINDS

    @property
    def tag(self) -> str:
        """The `[tag]` a sheet uses for this section."""
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "type": self.kind,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "vocal_required": self.vocal_required,
            "target_lines": self.target_lines,
            "rhyme_scheme": self.rhyme_scheme,
            "occurrence": self.occurrence,
        }


@dataclass(frozen=True)
class SongBlueprint:
    language: str
    duration_seconds: float
    genre: str
    bpm: int | None
    time_signature: str
    sections: tuple[BlueprintSection, ...]
    seconds_per_line: float
    coverage_target: float
    syllables_per_second: float
    rhyme_scheme: str
    version: str = "2.0"

    @property
    def target_vocal_seconds(self) -> float:
        return self.duration_seconds * self.coverage_target

    @property
    def maximum_nonvocal_seconds(self) -> float:
        return self.duration_seconds - self.target_vocal_seconds

    @property
    def vocal_sections(self) -> tuple[BlueprintSection, ...]:
        return tuple(section for section in self.sections if section.vocal_required)

    @property
    def planned_vocal_seconds(self) -> float:
        return sum(section.seconds for section in self.vocal_sections)

    def writer_targets(self) -> list[tuple[str, int]]:
        """(tag, lines) for each section the writer must actually write.

        A refrain is written once — its later occurrences reuse the text — so
        the writer sees each refrain kind once, at its first occurrence, with
        the largest line target any of its occurrences carries.
        """
        targets: list[tuple[str, int]] = []
        seen: dict[str, int] = {}
        for section in self.vocal_sections:
            if section.is_refrain:
                if section.kind in seen:
                    position = seen[section.kind]
                    tag, lines = targets[position]
                    targets[position] = (tag, max(lines, section.target_lines))
                    continue
                seen[section.kind] = len(targets)
            targets.append((section.kind, section.target_lines))
        return targets

    @property
    def total_target_lines(self) -> int:
        return sum(lines for _, lines in self.writer_targets())

    @property
    def outline(self) -> str:
        return " → ".join(section.kind for section in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 2),
            "genre": self.genre,
            "bpm": self.bpm,
            "time_signature": self.time_signature,
            "seconds_per_line": self.seconds_per_line,
            "syllables_per_second": self.syllables_per_second,
            "rhyme_scheme": self.rhyme_scheme,
            "coverage_target": self.coverage_target,
            "target_vocal_seconds": round(self.target_vocal_seconds, 2),
            "maximum_nonvocal_seconds": round(self.maximum_nonvocal_seconds, 2),
            "planned_vocal_seconds": round(self.planned_vocal_seconds, 2),
            "sections": [section.to_dict() for section in self.sections],
            "writer_targets": [
                {"section": tag, "lines": lines} for tag, lines in self.writer_targets()
            ],
        }


def resolve_scheme(scheme: str | None, genre: str) -> str:
    """"auto" → the scheme the genre usually uses; anything else as given."""
    chosen = (scheme or "auto").strip().upper()
    if chosen != "AUTO":
        return chosen
    if genre in {"rap", "hip-hop", "drill", "trap", "grime"}:
        return "AABB"
    if genre in {"country", "folk", "pop", "rock", "indie"}:
        return "ABAB"
    return "AABB"


def build_blueprint(
    plan: SongPlan,
    *,
    language: str,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    seconds_per_line: float = DEFAULT_SECONDS_PER_LINE,
    rhyme_scheme: str | None = "auto",
    bpm: int | None = None,
    time_signature: str = "4/4",
    minimum_lines: int = 2,
) -> SongBlueprint:
    """A timed, accountable blueprint from a song plan.

    Wordless sections are scaled so that together they take at most
    `1 - coverage_target` of the song; the time they give up goes to the
    vocal sections in proportion. A plan whose every section is wordless (a
    true instrumental) is returned as it is — coverage is then not the
    question being asked.
    """
    duration = float(plan.total_seconds)
    if duration <= 0:
        raise ValueError("duration must be positive")
    coverage_target = min(1.0, max(0.0, coverage_target))

    wordless = [s for s in plan.sections if s.kind in _WORDLESS_KINDS or not s.carries_words]
    vocal = [s for s in plan.sections if s not in wordless]
    if plan.wordless or not vocal:
        sections = tuple(
            _section(index, s.kind, start, start + s.seconds, False, 0, "", 0)
            for index, (s, start) in enumerate(_starts(plan.sections))
        )
        return SongBlueprint(
            language=language,
            duration_seconds=duration,
            genre=plan.genre,
            bpm=bpm,
            time_signature=time_signature,
            sections=sections,
            seconds_per_line=seconds_per_line,
            coverage_target=coverage_target,
            syllables_per_second=_rate(plan.genre),
            rhyme_scheme=resolve_scheme(rhyme_scheme, plan.genre),
        )

    wordless_seconds = sum(s.seconds for s in wordless)
    allowed = duration * (1.0 - coverage_target)
    wordless_scale = min(1.0, allowed / wordless_seconds) if wordless_seconds else 1.0
    vocal_seconds = sum(s.seconds for s in vocal)
    vocal_scale = (duration - wordless_seconds * wordless_scale) / vocal_seconds

    scheme = resolve_scheme(rhyme_scheme, plan.genre)
    rate = _rate(plan.genre)

    occurrences: dict[str, int] = {}
    sections: list[BlueprintSection] = []
    cursor = 0.0
    for index, section in enumerate(plan.sections):
        is_vocal = section in vocal
        seconds = section.seconds * (vocal_scale if is_vocal else wordless_scale)
        end = cursor + seconds
        occurrence = occurrences.get(section.kind, 0)
        occurrences[section.kind] = occurrence + 1
        target = max(minimum_lines, round(seconds / max(0.5, seconds_per_line))) if is_vocal else 0
        sections.append(
            _section(index, section.kind, cursor, end, is_vocal, target, scheme if is_vocal else "", occurrence)
        )
        cursor = end

    # Rounding drift: the last section absorbs it so the timeline ends
    # exactly on the requested length.
    if sections:
        last = sections[-1]
        sections[-1] = _section(
            last.index, last.kind, last.start, duration, last.vocal_required,
            last.target_lines, last.rhyme_scheme, last.occurrence,
        )

    return SongBlueprint(
        language=language,
        duration_seconds=duration,
        genre=plan.genre,
        bpm=bpm,
        time_signature=time_signature,
        sections=tuple(sections),
        seconds_per_line=seconds_per_line,
        coverage_target=coverage_target,
        syllables_per_second=rate,
        rhyme_scheme=scheme,
    )


def _rate(genre: str) -> float:
    return SYLLABLES_PER_SECOND.get(genre, DEFAULT_SYLLABLES_PER_SECOND)


def _starts(sections: list[Section]) -> list[tuple[Section, float]]:
    out: list[tuple[Section, float]] = []
    cursor = 0.0
    for section in sections:
        out.append((section, cursor))
        cursor += section.seconds
    return out


def _section(
    index: int, kind: str, start: float, end: float, vocal: bool,
    target: int, scheme: str, occurrence: int,
) -> BlueprintSection:
    return BlueprintSection(
        index=index,
        kind=kind,
        start=start,
        end=end,
        vocal_required=vocal,
        target_lines=target,
        rhyme_scheme=scheme,
        occurrence=occurrence,
    )


__all__ = [
    "DEFAULT_COVERAGE_TARGET",
    "DEFAULT_SECONDS_PER_LINE",
    "SYLLABLES_PER_SECOND",
    "BlueprintSection",
    "SongBlueprint",
    "build_blueprint",
    "resolve_scheme",
]
