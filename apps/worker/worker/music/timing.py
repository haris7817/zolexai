"""Laying a written sheet onto the blueprint's timeline.

The writer produces sections of lines; the blueprint has sections of
seconds. This module marries them: each written section is placed on its
blueprint slot, a refrain's text is reused at every occurrence, and every
line gets a start and an end — each section's time divided among its lines
in proportion to their syllables.

Two things come out of that, and both are *planned* rather than measured:

  * a timeline the music model is asked to follow, delivered as the
    `.lrc`/`.srt`/JSON outputs the client's workflow specifies;
  * the planned vocal coverage — how much of the song those lines can
    actually fill at the genre's delivery speed. A section with two short
    lines over thirty seconds does not count as thirty seconds of singing.

The measured version of the same number comes after the audio exists
(`worker/music/verify.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from worker.music.blueprint import BlueprintSection, SongBlueprint
from worker.music.rhyme import scheme_labels
from worker.music.syllables import is_filler_line, syllables

#: Seconds between one line's end and the next's start in the timeline.
#: Zero: a breath between lines is part of continuous singing, not one of
#: the "pauses" the client's ten-percent allowance is for, and a timeline
#: that deducted it would make ninety percent unreachable by construction
#: (fourteen lines in a minute would cost eight percent on their own).
LINE_GAP_SECONDS = 0.0

_REFRAIN_KINDS = frozenset({"chorus", "hook", "drop", "refrain"})


@dataclass(frozen=True)
class TimedLine:
    index: int
    section_index: int
    section: str
    occurrence: int
    text: str
    start: float
    end: float
    syllables: int
    rhyme_group: str
    filler: bool
    sung_seconds: float
    """How long the line takes to sing at the genre's delivery speed —
    capped at its slot. The planned coverage is the sum of these."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "section_index": self.section_index,
            "section": self.section,
            "occurrence": self.occurrence,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "syllables": self.syllables,
            "rhyme_group": self.rhyme_group,
            "filler": self.filler,
            "sung_seconds": round(self.sung_seconds, 3),
        }


@dataclass(frozen=True)
class TimedSection:
    blueprint: BlueprintSection
    lines: tuple[TimedLine, ...]
    written: bool
    """False when the blueprint wanted words here and the sheet had none."""

    @property
    def sung_seconds(self) -> float:
        return sum(line.sung_seconds for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.blueprint.to_dict(),
            "written": self.written,
            "lines": [line.to_dict() for line in self.lines],
            "sung_seconds": round(self.sung_seconds, 3),
        }


@dataclass(frozen=True)
class TimedLyrics:
    blueprint: SongBlueprint
    sections: tuple[TimedSection, ...]
    unplaced: tuple[tuple[str, list[str]], ...]
    """Written sections the blueprint had no slot for (an extra verse, an
    unknown tag). Reported; never silently dropped."""

    @property
    def lines(self) -> tuple[TimedLine, ...]:
        return tuple(line for section in self.sections for line in section.lines)

    @property
    def planned_vocal_seconds(self) -> float:
        return sum(line.sung_seconds for line in self.lines if not line.filler)

    @property
    def filler_seconds(self) -> float:
        return sum(line.sung_seconds for line in self.lines if line.filler)

    @property
    def planned_coverage(self) -> float:
        return self.planned_vocal_seconds / self.blueprint.duration_seconds

    @property
    def filler_ratio(self) -> float:
        """Filler lines plus every second nobody sings, over the song."""
        nonvocal = self.blueprint.duration_seconds - self.planned_vocal_seconds
        return max(0.0, nonvocal) / self.blueprint.duration_seconds

    @property
    def words(self) -> int:
        return sum(len(line.text.split()) for line in self.lines)

    @property
    def syllables(self) -> int:
        return sum(line.syllables for line in self.lines)

    @property
    def unwritten_sections(self) -> tuple[BlueprintSection, ...]:
        return tuple(s.blueprint for s in self.sections if s.blueprint.vocal_required and not s.written)

    def per_section_rates(self) -> list[dict[str, Any]]:
        out = []
        for section in self.sections:
            seconds = section.blueprint.seconds
            words = sum(len(line.text.split()) for line in section.lines)
            syl = sum(line.syllables for line in section.lines)
            out.append(
                {
                    "section": section.blueprint.kind,
                    "index": section.blueprint.index,
                    "lines": len(section.lines),
                    "target_lines": section.blueprint.target_lines,
                    "words_per_second": round(words / seconds, 2) if seconds else 0.0,
                    "syllables_per_second": round(syl / seconds, 2) if seconds else 0.0,
                    "sung_fraction": round(section.sung_seconds / seconds, 3) if seconds else 0.0,
                }
            )
        return out

    # ── Renderings ───────────────────────────────────────────────────

    def sheet(self) -> str:
        """The expanded sheet for the music model: every occurrence, in
        order, tagged. A refrain appears at each of its slots."""
        parts: list[str] = []
        for section in self.sections:
            if not section.lines:
                continue
            parts.append(f"[{section.blueprint.tag}]")
            parts.extend(line.text for line in section.lines)
            parts.append("")
        return "\n".join(parts).strip()

    def to_lrc(self) -> str:
        rows = []
        for line in self.lines:
            rows.append(f"[{_lrc_stamp(line.start)}]{line.text}")
        return "\n".join(rows) + ("\n" if rows else "")

    def to_srt(self) -> str:
        rows = []
        for number, line in enumerate(self.lines, start=1):
            rows.append(f"{number}\n{_srt_stamp(line.start)} --> {_srt_stamp(line.end)}\n{line.text}\n")
        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": round(self.blueprint.duration_seconds, 2),
            "language": self.blueprint.language,
            "planned_vocal_seconds": round(self.planned_vocal_seconds, 2),
            "planned_vocal_coverage": round(self.planned_coverage, 4),
            "filler_ratio": round(self.filler_ratio, 4),
            "words": self.words,
            "syllables": self.syllables,
            "sections": [section.to_dict() for section in self.sections],
            "unplaced": [{"section": tag, "lines": lines} for tag, lines in self.unplaced],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def lay_out(
    sections: list[tuple[str, list[str]]],
    blueprint: SongBlueprint,
    *,
    gap_seconds: float = LINE_GAP_SECONDS,
) -> TimedLyrics:
    """Places written sections on the blueprint.

    Matching is by tag, in order: the first written `verse` fills the first
    blueprint verse, the second the second. A refrain written once is reused
    for every later occurrence of its kind. Written sections that find no
    slot are returned in `unplaced`.
    """
    written = [(tag.strip().lower(), [line for line in lines if line.strip()]) for tag, lines in sections]
    used = [False] * len(written)
    refrain_text: dict[str, list[str]] = {}
    timed_sections: list[TimedSection] = []
    global_index = 0

    for section in blueprint.sections:
        if not section.vocal_required:
            timed_sections.append(TimedSection(section, (), written=True))
            continue

        lines: list[str] | None = None
        for position, (tag, text) in enumerate(written):
            if used[position] or tag != section.kind or not text:
                continue
            used[position] = True
            lines = text
            break
        if lines is None and section.kind in _REFRAIN_KINDS:
            lines = refrain_text.get(section.kind)
        if lines is None and section.kind in _REFRAIN_KINDS:
            # An untagged second chorus written under another refrain tag.
            for position, (tag, text) in enumerate(written):
                if not used[position] and tag in _REFRAIN_KINDS and text:
                    used[position] = True
                    lines = text
                    break
        if lines is None:
            # Any leftover written section of any word-carrying kind fills a
            # slot rather than being dropped: a writer who tagged a verse
            # "pre-chorus" still wrote a verse's worth of words.
            for position, (tag, text) in enumerate(written):
                if not used[position] and text and tag not in _REFRAIN_KINDS:
                    used[position] = True
                    lines = text
                    break
        if lines is None:
            # An unwritten slot (a pre-chorus the writer skipped) hands its
            # time to the previous written section: the sheet the model gets
            # has no such section, so the neighbour's lines are what fill
            # that time — and they are credited only for what their
            # syllables can cover. Reported as unwritten either way.
            timed_sections.append(TimedSection(section, (), written=False))
            continue
        if section.kind in _REFRAIN_KINDS:
            refrain_text.setdefault(section.kind, lines)

        timed_lines = _time_lines(lines, section, blueprint, start_index=global_index, gap=gap_seconds)
        global_index += len(timed_lines)
        timed_sections.append(TimedSection(section, tuple(timed_lines), written=True))

    timed_sections = _absorb_unwritten(timed_sections, blueprint, gap_seconds)
    unplaced = tuple((tag, text) for (tag, text), taken in zip(written, used, strict=True) if not taken and text)
    return TimedLyrics(blueprint=blueprint, sections=tuple(timed_sections), unplaced=unplaced)


def _absorb_unwritten(
    sections: list[TimedSection], blueprint: SongBlueprint, gap: float
) -> list[TimedSection]:
    """Re-times written sections so an unwritten vocal slot's time is sung
    by its previous written neighbour (or its next, at the start)."""
    if not any(s.blueprint.vocal_required and not s.written for s in sections):
        return sections
    out = list(sections)
    for index, section in enumerate(out):
        if section.written or not section.blueprint.vocal_required:
            continue
        donor = next((i for i in range(index - 1, -1, -1) if out[i].written and out[i].lines), None)
        if donor is None:
            donor = next((i for i in range(index + 1, len(out)) if out[i].written and out[i].lines), None)
        if donor is None:
            continue
        host = out[donor]
        start = min(host.blueprint.start, section.blueprint.start)
        end = max(host.blueprint.end, section.blueprint.end)
        widened = BlueprintSection(
            index=host.blueprint.index, kind=host.blueprint.kind, start=start, end=end,
            vocal_required=True, target_lines=host.blueprint.target_lines,
            rhyme_scheme=host.blueprint.rhyme_scheme, occurrence=host.blueprint.occurrence,
        )
        first_index = host.lines[0].index
        retimed = _time_lines([line.text for line in host.lines], widened, blueprint, start_index=first_index, gap=gap)
        out[donor] = TimedSection(widened, tuple(retimed), written=True)
    return out


def _time_lines(
    lines: list[str],
    section: BlueprintSection,
    blueprint: SongBlueprint,
    *,
    start_index: int,
    gap: float,
) -> list[TimedLine]:
    code = blueprint.language
    counts = [max(1, syllables(line, code)) for line in lines]
    total = sum(counts)
    labels = scheme_labels(len(lines), section.rhyme_scheme or blueprint.rhyme_scheme)
    slot_total = section.seconds
    gaps = gap * max(0, len(lines) - 1)
    usable = max(0.0, slot_total - gaps)
    rate = blueprint.syllables_per_second

    timed: list[TimedLine] = []
    cursor = section.start
    for offset, (text, count) in enumerate(zip(lines, counts, strict=True)):
        share = usable * count / total if total else 0.0
        end = cursor + share
        sung = min(share, count / rate) if rate > 0 else share
        timed.append(
            TimedLine(
                index=start_index + offset,
                section_index=section.index,
                section=section.kind,
                occurrence=section.occurrence,
                text=text,
                start=cursor,
                end=end,
                syllables=count,
                rhyme_group=labels[offset],
                filler=is_filler_line(text),
                sung_seconds=sung,
            )
        )
        cursor = end + gap
    return timed


def _lrc_stamp(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def _srt_stamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


__all__ = ["LINE_GAP_SECONDS", "TimedLine", "TimedLyrics", "TimedSection", "lay_out"]
