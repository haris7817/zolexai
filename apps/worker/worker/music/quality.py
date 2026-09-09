"""The lyric quality and flow engine: outline first, judged before singing.

The client's finding (10 Sep 2026): endings rhymed but the lines did not
connect, the story changed direction, and words were chosen to force a
rhyme. Their order is now the workflow's order — song concept → narrative
outline → lyrics → rhyme/meter validation → music — and the lyrics are
judged for coherence, grammar and meaning by a reader before the music
model ever sees them.

Two hosted-model calls, both through the lyrics writer's own endpoint:

* **`outline`** turns the brief into a narrative outline: theme, narrator,
  who is addressed, the central emotion, the conflict, how it develops,
  the final message and the hook idea. The writer is then asked to write
  *that* song, so every verse continues one story.
* **`judge`** reads the finished sheet against the outline and returns a
  coherence score, a grammar/naturalness score and the lines it objects
  to with a reason each. Below the client's floors (coherence 0.90,
  grammar 0.95) the objected lines go back to the writer as targeted
  repairs, and the sheet is judged again.

A judge is an opinion, and this one is a language model's. It is recorded
with its scores in the report, its floors are configurable, and when no
model is available the stage is reported as unmeasured rather than failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from worker.core.logging import get_logger
from worker.music.lyrics import LyricBrief

logger = get_logger(__name__)

COHERENCE_FLOOR = 0.90
GRAMMAR_FLOOR = 0.95


@dataclass(frozen=True)
class Outline:
    theme: str = ""
    narrator: str = ""
    addressee: str = ""
    emotion: str = ""
    conflict: str = ""
    development: str = ""
    final_message: str = ""
    hook: str = ""
    source: str = "none"

    @property
    def empty(self) -> bool:
        return not any((self.theme, self.narrator, self.emotion, self.hook))

    def describe(self) -> str:
        parts = [
            ("THEME", self.theme), ("NARRATOR", self.narrator), ("ADDRESSED TO", self.addressee),
            ("CENTRAL EMOTION", self.emotion), ("CONFLICT", self.conflict),
            ("HOW IT DEVELOPS", self.development), ("FINAL MESSAGE", self.final_message),
            ("HOOK IDEA", self.hook),
        ]
        return "\n".join(f"{label}: {value}" for label, value in parts if value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme, "narrator": self.narrator, "addressee": self.addressee,
            "emotion": self.emotion, "conflict": self.conflict, "development": self.development,
            "final_message": self.final_message, "hook": self.hook, "source": self.source,
        }


@dataclass(frozen=True)
class Verdict:
    coherence: float
    grammar: float
    problems: tuple[tuple[int, str], ...]
    """(1-based line number over the sung lines, reason)."""
    summary: str = ""
    measured: bool = True

    def passes(self, *, coherence_floor: float = COHERENCE_FLOOR, grammar_floor: float = GRAMMAR_FLOOR) -> bool:
        if not self.measured:
            return True
        return self.coherence >= coherence_floor and self.grammar >= grammar_floor and not self.problems

    def instructions(self) -> list[str]:
        return [
            f"quality: line {number}: {reason}. Rewrite this line so it continues the story naturally in "
            f"correct, idiomatic language; keep its rhyme partner's ending sound and its syllable count."
            for number, reason in self.problems
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured,
            "coherence": round(self.coherence, 3),
            "grammar": round(self.grammar, 3),
            "problems": [{"line": n, "reason": r} for n, r in self.problems],
            "summary": self.summary,
        }


_UNMEASURED = Verdict(coherence=1.0, grammar=1.0, problems=(), summary="no judge available", measured=False)


def _language_name(code: str) -> str:
    from worker.music.language import resolve_language

    try:
        language = resolve_language(code)
    except Exception:
        language = None
    return language.name if language else code


async def outline(writer: Any, brief: LyricBrief) -> Outline:
    """A narrative outline for the song, or an empty one when unavailable."""
    ask = getattr(writer, "ask_json", None)
    if ask is None:
        return Outline()
    language = _language_name(brief.language)
    system = (
        "You are a songwriter planning a song before writing a word of it. Answer with one JSON "
        "object and nothing else, with these string keys: theme, narrator, addressee, emotion, "
        "conflict, development, final_message, hook. Each value is one or two plain sentences in "
        f"English describing the plan for a song whose lyrics will be in {language}. The story must "
        "be specific and stay the same from first verse to last."
    )
    user = (
        f"REQUEST: {brief.topic}\n"
        + (f"WRITE ABOUT: {brief.perspective}\n" if brief.perspective else "")
        + (f"MUST INCLUDE: {', '.join(brief.must_keep)}\n" if brief.must_keep else "")
        + f"GENRE: {brief.genre}\n"
    )
    data = await ask(system, user, max_tokens=900)
    if not isinstance(data, dict):
        return Outline()
    text = lambda key: str(data.get(key) or "").strip()[:400]  # noqa: E731
    result = Outline(
        theme=text("theme"), narrator=text("narrator"), addressee=text("addressee"),
        emotion=text("emotion"), conflict=text("conflict"), development=text("development"),
        final_message=text("final_message"), hook=text("hook"), source=getattr(writer, "name", "writer"),
    )
    logger.info("music_outline_written", extra={"empty": result.empty, "theme": result.theme[:80]})
    return result


async def judge(writer: Any, sheet_lines: list[str], plan: Outline, language_code: str) -> Verdict:
    """Coherence, grammar and the lines a careful reader would object to."""
    ask = getattr(writer, "ask_json", None)
    if ask is None or not sheet_lines:
        return _UNMEASURED
    language = _language_name(language_code)
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(sheet_lines, start=1))
    system = (
        f"You are a native-speaking copy editor of {language} song lyrics. Answer with one JSON "
        "object and nothing else: {\"coherence\": 0-1, \"grammar\": 0-1, "
        "\"problems\": [{\"line\": n, \"reason\": \"...\"}], \"summary\": \"...\"}. "
        "Judge ERRORS, not taste: a line is a problem only if it is ungrammatical, uses an invented "
        "or wrong word, is meaningless, contradicts the story, or is a filler sound put there only "
        "to rhyme. Plain, simple or unpoetic lines are NOT problems. coherence = the fraction of "
        "lines that belong to one consistent story matching the outline; grammar = the fraction of "
        f"lines that are correct, natural {language}. Song lyrics allow poetic licence, inversion "
        "and imagery. List only lines with a real error, with a short reason each."
    )
    user = ("OUTLINE:\n" + plan.describe() + "\n\n" if not plan.empty else "") + "LYRICS:\n" + numbered
    data = await ask(system, user, max_tokens=1600, effort="medium")
    if not isinstance(data, dict):
        return _UNMEASURED
    try:
        coherence = max(0.0, min(1.0, float(data.get("coherence", 0))))
        grammar = max(0.0, min(1.0, float(data.get("grammar", 0))))
    except (TypeError, ValueError):
        return _UNMEASURED
    problems: list[tuple[int, str]] = []
    for item in data.get("problems") or []:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        if 1 <= number <= len(sheet_lines):
            problems.append((number, str(item.get("reason") or "")[:200]))
    verdict = Verdict(coherence, grammar, tuple(problems), str(data.get("summary") or "")[:300])
    logger.info(
        "music_lyrics_judged",
        extra={"coherence": round(coherence, 2), "grammar": round(grammar, 2), "problems": len(problems)},
    )
    return verdict


__all__ = ["COHERENCE_FLOOR", "GRAMMAR_FLOOR", "Outline", "Verdict", "judge", "outline"]
