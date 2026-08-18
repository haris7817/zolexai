"""Song structure and a lyric quality pass.

The client's complaint about generated lyrics was specific: repetitive, badly
structured, inconsistently rhymed. Those are three different problems and only
one of them is really a writing problem, so this module treats them separately.

**Structure** is arithmetic, and it is done here. A four-minute pop song is not
a four-minute ambient piece with different words — it has a different number of
sections, in a different order, at different lengths. `plan_song` builds that
skeleton per genre and per requested length, and it is the plan the writer is
asked to fill and the plan the result is checked against.

**Repetition and rhyme** are measurable. `review_lyrics` measures them: how
much of the non-chorus material is duplicated, how many lines find a rhyme
partner inside their own section, whether the sections the plan asked for are
the sections that arrived, whether the names and details the user gave survived.
It produces a list of concrete, quotable problems rather than a score nobody
can act on.

**Writing** is not done here, and cannot be faked. `polish_lyrics` performs only
the repairs that are unambiguously improvements — making every chorus the same
chorus, removing lines that repeat immediately, dropping empty sections — and
reports what it could not fix. A configured `LyricsWriter` gets those reports
back and rewrites; with no writer configured, the honest outcome is a structure
plan and no lyrics, which is what the adapter then works with.

What this deliberately does not do is promise perfect rhyme or professional
songwriting. It is a floor, not a ceiling: it makes obvious failures visible
and repairs the mechanical ones.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)


# ── Structure ────────────────────────────────────────────────────────────

#: Section kinds, and roughly how many seconds one of them runs. Real songs
#: vary wildly; these are only used to decide how many sections fit a
#: requested length, so being in the right neighbourhood is enough.
_SECTION_SECONDS: dict[str, float] = {
    "intro": 12.0,
    "verse": 30.0,
    "pre-chorus": 12.0,
    "chorus": 24.0,
    "hook": 20.0,
    "bridge": 20.0,
    "build": 16.0,
    "drop": 28.0,
    "breakdown": 20.0,
    "movement": 60.0,
    "solo": 20.0,
    "outro": 14.0,
}

#: Per-genre skeletons: (opening, one repeatable cycle, late-song section,
#: closing). The cycle repeats to fill the requested length, which is what
#: makes a five-minute song more song rather than a slower one.
#:
#: These differ on purpose. Forcing verse/pre-chorus/chorus onto an ambient
#: piece or a drum-and-bass track produces a structure the genre does not have,
#: and the client explicitly asked not to do that.
_STRUCTURES: dict[str, tuple[list[str], list[str], str | None, list[str]]] = {
    "pop": (["intro"], ["verse", "pre-chorus", "chorus"], "bridge", ["chorus", "outro"]),
    "rock": (["intro"], ["verse", "chorus"], "solo", ["chorus", "outro"]),
    "hip-hop": (["intro"], ["verse", "hook"], None, ["hook", "outro"]),
    "rnb": (["intro"], ["verse", "chorus"], "bridge", ["chorus", "outro"]),
    "ballad": (["intro"], ["verse", "chorus"], "bridge", ["chorus", "outro"]),
    "country": (["intro"], ["verse", "chorus"], "bridge", ["chorus", "outro"]),
    "folk": (["intro"], ["verse", "chorus"], None, ["verse", "outro"]),
    "electronic": (["intro"], ["build", "drop", "breakdown"], None, ["drop", "outro"]),
    "ambient": (["intro"], ["movement"], None, ["outro"]),
    "instrumental": (["intro"], ["movement"], None, ["outro"]),
}
_DEFAULT_GENRE = "pop"

#: Genres whose sections carry no sung words. Asking a writer for lyrics here
#: and then checking their rhyme scheme would be measuring nothing.
_WORDLESS = frozenset({"ambient", "instrumental"})

#: Words that identify a genre in free-text prompt. Longest match wins, so
#: "hip hop ballad" resolves on the more specific phrase it contains first.
_GENRE_WORDS: dict[str, str] = {
    "hip hop": "hip-hop", "hip-hop": "hip-hop", "rap": "hip-hop", "trap": "hip-hop",
    "r&b": "rnb", "rnb": "rnb", "soul": "rnb",
    "ballad": "ballad", "slow song": "ballad",
    "rock": "rock", "metal": "rock", "punk": "rock", "grunge": "rock",
    "country": "country", "bluegrass": "country",
    "folk": "folk", "acoustic": "folk", "singer-songwriter": "folk",
    "edm": "electronic", "electronic": "electronic", "house": "electronic",
    "techno": "electronic", "dubstep": "electronic", "synthwave": "electronic",
    "ambient": "ambient", "lo-fi": "ambient", "lofi": "ambient",
    "instrumental": "instrumental", "no vocals": "instrumental",
    "pop": "pop",
}


@dataclass(frozen=True)
class Section:
    index: int
    kind: str
    seconds: float

    @property
    def is_chorus(self) -> bool:
        """Sections that are SUPPOSED to repeat.

        The distinction matters everywhere below: repeating a chorus is the
        song working, and repeating a verse is the song failing.
        """
        return self.kind in ("chorus", "hook", "drop")

    @property
    def carries_words(self) -> bool:
        return self.kind not in ("intro", "outro", "solo", "movement", "breakdown", "build")


#: Seconds of song one sung line needs, at ordinary tempo and allowing for
#: intros, instrumental breaks and repeats.
#:
#: This is a MEASUREMENT, not a style preference, and it is the most important
#: number in this module — and it is a BAND, bounded on both sides:
#:
#:   * 8 lines at  60s (7.5s/line) → the model sang only the chorus and
#:     silently dropped both verses (RTX 5090, 2026-08-13);
#:   * 5 lines at 120s (24s/line)  → an 82-second instrumental intro plus
#:     wordless padding (same session);
#:   * 9 lines at 120s (13.3s/line) → vocals at 30s, every line sung;
#:   * 12 lines at 240s (20s/line)  → every line sung, real arrangement.
#:
#: A model given more words than the clock can hold does not compress them; it
#: discards them without saying which. Given too few, it pads with instrumental
#: — which reads to a customer as "lyrics not present". 13s/line is the
#: densest point proven safe, so the budget it produces is a true ceiling.
_SECONDS_PER_LINE = 13.0


def line_budget(total_seconds: float, seconds_per_line: float | None = None) -> int:
    """How many sung lines a song of this length can actually carry.

    Two is the floor: a song with one line is a loop, not a song.
    """
    per_line = seconds_per_line or settings.music_seconds_per_line or _SECONDS_PER_LINE
    return max(2, int(total_seconds / max(1.0, per_line)))


#: The density a writer should AIM for, as seconds of song per sung line.
#:
#: Distinct from `_SECONDS_PER_LINE` above, and both are needed. That one is the
#: ceiling — the densest sheet the model will sing without silently dropping
#: lines. This one is the target, and it exists because the band is bounded on
#: BOTH sides: at 120s, 9 lines sang everything, and 5 lines produced an
#: 82-second instrumental intro. A writer told only "at most 9" writes 6 and
#: the song is half wordless, which reads to a customer as "lyrics not present"
#: just as surely as no lyrics at all.
_TARGET_SECONDS_PER_LINE = 16.0

#: Fewer lines than this is a loop, not a song, whatever the duration.
_MINIMUM_LINES = 4


def target_lines(plan: SongPlan) -> int:
    """How many sung lines a writer should actually aim to produce.

    Sits inside the measured band from both directions: dense enough that
    vocals arrive early and the song does not pad, sparse enough that nothing
    gets dropped. Never exceeds the plan's own ceiling.
    """
    by_density = round(plan.total_seconds / _TARGET_SECONDS_PER_LINE)
    return min(plan.line_budget, max(_MINIMUM_LINES, by_density))


@dataclass(frozen=True)
class LyricFit:
    """Whether a lyric sheet fits the time available."""

    lines: int
    budget: int

    @property
    def fits(self) -> bool:
        return self.lines <= self.budget

    @property
    def overflow(self) -> int:
        return max(0, self.lines - self.budget)


def check_lyric_fit(
    text: str, total_seconds: float, seconds_per_line: float | None = None
) -> LyricFit:
    """Measures a lyric sheet against the duration it has to fit into.

    The caller decides what to do about a bad fit, and for user-supplied
    lyrics the answer must be *tell them* — silently handing the model more
    words than it can sing means the customer's own lines vanish with no
    explanation, which is the lyrical form of generalising away "two cars".
    """
    lines = [line for _, section in parse_sections(text) for line in section]
    return LyricFit(lines=len(lines), budget=line_budget(total_seconds, seconds_per_line))


@dataclass(frozen=True)
class SongPlan:
    genre: str
    total_seconds: float
    sections: list[Section]

    @property
    def has_lyrics(self) -> bool:
        return self.genre not in _WORDLESS and any(s.carries_words for s in self.sections)

    @property
    def line_budget(self) -> int:
        """Total sung lines this plan's duration can carry."""
        return line_budget(self.total_seconds)

    @property
    def lines_per_section(self) -> int:
        """How many lines each word-carrying section should aim for.

        Two is the floor for the same reason as `line_budget` — a section with
        a single line reads as an unfinished thought.
        """
        singing = sum(1 for section in self.sections if section.carries_words)
        return max(2, self.line_budget // max(1, singing))

    @property
    def outline(self) -> str:
        """A one-line structure summary, for prompts and for logs."""
        return " → ".join(section.kind for section in self.sections)


def detect_genre(prompt: str) -> str:
    """The genre named in a free-text prompt, or pop.

    Pop is the default because it is the structure most listeners expect from
    an unqualified "write me a song", not because it is neutral.
    """
    text = " " + re.sub(r"[^a-z&\- ]+", " ", prompt.lower()) + " "
    for phrase in sorted(_GENRE_WORDS, key=len, reverse=True):
        if f" {phrase} " in text:
            return _GENRE_WORDS[phrase]
    return _DEFAULT_GENRE


def plan_song(total_seconds: float, *, genre: str | None = None, prompt: str = "") -> SongPlan:
    """A section-by-section skeleton for a song of this length and genre.

    The cycle repeats until the length is covered, then every section is
    scaled by one factor so the plan sums to exactly the requested duration —
    the user picked five minutes and five minutes is what the plan describes.
    """
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")

    resolved = (genre or detect_genre(prompt) or _DEFAULT_GENRE).lower()
    opening, cycle, late, closing = _STRUCTURES.get(resolved, _STRUCTURES[_DEFAULT_GENRE])

    def length(kinds: list[str]) -> float:
        return sum(_SECTION_SECONDS.get(kind, 20.0) for kind in kinds)

    fixed = length(opening) + length(closing)
    cycle_length = length(cycle)

    # At least one cycle even for a very short request: a song with an intro
    # and an outro and nothing between them is not a song.
    cycles = max(1, round((total_seconds - fixed) / cycle_length)) if cycle_length else 1

    kinds = [*opening]
    for _ in range(cycles):
        kinds += cycle
    # The late-song section earns its place only when there is enough song for
    # a departure to mean anything.
    if late and cycles >= 2 and total_seconds >= 120:
        kinds.append(late)
    kinds += closing

    nominal = length(kinds)
    scale = total_seconds / nominal if nominal else 1.0
    sections = [
        Section(index=index, kind=kind, seconds=_SECTION_SECONDS.get(kind, 20.0) * scale)
        for index, kind in enumerate(kinds)
    ]

    logger.info(
        "song_planned",
        extra={
            "genre": resolved,
            "total_seconds": round(total_seconds, 1),
            "sections": len(sections),
        },
    )
    return SongPlan(genre=resolved, total_seconds=total_seconds, sections=sections)


# ── The brief: what must survive whatever happens to the words ───────────


@dataclass(frozen=True)
class LyricBrief:
    """The parts of the user's request that a rewrite may not quietly discard.

    The client's video complaint was that explicit details get generalised
    away; the same failure in lyrics is a song about "a city" when the user
    asked for a song about Lahore, or one addressed to "you" when they asked
    for a mother singing to a daughter. `must_keep` is extracted once and
    checked after every pass.
    """

    topic: str
    genre: str
    mood: str = ""
    language: str = "en"
    """Canonical ISO 639-1 code — see `worker/music/language.py`. Never a
    display name: the writer compares it against what it can write, and
    "Spanish" vs "es" is exactly the mismatch that comparison must not have."""
    perspective: str = ""
    must_keep: list[str] = field(default_factory=list)

    @classmethod
    def from_prompt(cls, prompt: str, *, genre: str | None = None) -> LyricBrief:
        return cls(
            topic=prompt.strip(),
            genre=genre or detect_genre(prompt),
            must_keep=salient_details(prompt),
        )


#: Words that start a sentence and are capitalised for that reason alone. They
#: are not the names the user cares about, and treating them as such would make
#: every review complain about a "missing detail" that was never a detail.
_SENTENCE_WORDS = frozenset(
    {
        "a", "an", "and", "the", "i", "in", "on", "at", "it", "is", "of", "for", "to",
        "with", "write", "make", "song", "about", "sing", "her", "his", "their", "my",
        "we", "he", "she", "they", "you", "this", "that", "but", "so", "then", "when",
    }
)


def salient_details(prompt: str) -> list[str]:
    """Proper nouns and explicit numbers from a prompt, in order of appearance.

    Deliberately shallow — no model, no dictionary. It catches the things whose
    loss is most obvious to the person who typed them: names, places, and
    counts. Duplicates collapse; ordinary sentence-leading words do not count.
    """
    found: list[str] = []
    for match in re.finditer(r"\b\d+\b|\b[A-Z][a-zA-Z'’-]+\b", prompt):
        token = match.group(0)
        if token.lower() in _SENTENCE_WORDS:
            continue
        if token not in found:
            found.append(token)
    return found


# ── Rhyme ────────────────────────────────────────────────────────────────

#: English spelling → a rough sound for the purpose of comparing endings. Not
#: a pronunciation dictionary; enough that "night"/"light" and "days"/"phase"
#: are seen as rhymes while "night"/"cat" is not.
_SOUND_RULES: tuple[tuple[str, str], ...] = (
    ("ight", "ite"), ("igh", "ie"), ("ough", "uf"), ("augh", "af"),
    ("tion", "shun"), ("sion", "shun"), ("ceive", "eve"), ("ie", "ee"),
    ("ea", "ee"), ("ee", "ee"), ("ey", "ee"), ("y", "ee"),
    ("ai", "ay"), ("ay", "ay"), ("ei", "ay"),
    ("oa", "oh"), ("ow", "oh"), ("oe", "oh"),
    ("oo", "oo"), ("ou", "ow"), ("ue", "oo"), ("ew", "oo"),
    ("ph", "f"), ("ck", "k"), ("qu", "kw"),
)
_VOWELS = "aeiou"


def rhyme_key(word: str) -> str:
    """A comparable ending for a word: last vowel sound onwards.

    Two words rhyme, for our purposes, when their keys match. That is a
    heuristic and it is wrong sometimes in both directions — but it is applied
    identically to every line, so the *rate* it measures is meaningful even
    where an individual verdict is not.
    """
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return ""

    # A silent final 'e' changes the vowel before it rather than sounding
    # itself: "fire" must not key on "e".
    if len(cleaned) > 3 and cleaned.endswith("e") and cleaned[-2] not in _VOWELS:
        cleaned = cleaned[:-1]

    for spelling, sound in _SOUND_RULES:
        cleaned = cleaned.replace(spelling, sound)

    last_vowel = max((cleaned.rfind(vowel) for vowel in _VOWELS), default=-1)
    return cleaned[last_vowel:] if last_vowel >= 0 else cleaned[-2:]


def lines_rhyme(first: str, second: str) -> bool:
    """Whether two lines end on the same sound."""
    def final_word(line: str) -> str:
        words = re.findall(r"[A-Za-z'’]+", line)
        return words[-1] if words else ""

    left, right = rhyme_key(final_word(first)), rhyme_key(final_word(second))
    if not left or not right:
        return False
    # An identical final word is a repeat, not a rhyme — counting it would let
    # the laziest possible lyric score perfectly.
    if final_word(first).lower() == final_word(second).lower():
        return False
    return left == right


# ── Review ───────────────────────────────────────────────────────────────

_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*$")


@dataclass(frozen=True)
class LyricIssue:
    kind: str
    """One of: structure, repetition, rhyme, detail, empty."""
    detail: str


@dataclass(frozen=True)
class LyricsReview:
    issues: list[LyricIssue]
    rhyme_rate: float
    """Fraction of lines that find a rhyme partner within their own section."""
    unique_rate: float
    """Fraction of non-chorus lines that appear exactly once in the song."""

    @property
    def acceptable(self) -> bool:
        """Whether this draft is worth using without another pass.

        The thresholds are deliberately modest. The client asked for
        "noticeably better than an unreviewed one-pass generation", not for
        perfection, and a bar set where nothing clears it would just mean
        every song burns the maximum number of rewrites and ships anyway.
        """
        blocking = [issue for issue in self.issues if issue.kind != "rhyme"]
        return not blocking and self.rhyme_rate >= 0.5 and self.unique_rate >= 0.7

    @property
    def notes(self) -> list[str]:
        """The issues as instructions a writer can act on."""
        return [f"{issue.kind}: {issue.detail}" for issue in self.issues]


def parse_sections(text: str) -> list[tuple[str, list[str]]]:
    """Splits `[verse]`-tagged lyrics into (tag, lines) pairs.

    Untagged text before the first tag becomes a `verse`, because that is what
    a model that ignored the tagging instruction has almost certainly written.
    """
    sections: list[tuple[str, list[str]]] = []
    current_tag: str = "verse"
    current: list[str] = []

    for raw in text.splitlines():
        tag = _TAG.match(raw)
        if tag:
            if current:
                sections.append((current_tag, current))
            current_tag, current = tag.group(1).strip().lower(), []
            continue
        line = raw.strip()
        if line:
            current.append(line)

    if current:
        sections.append((current_tag, current))
    return sections


def review_lyrics(text: str, plan: SongPlan, brief: LyricBrief | None = None) -> LyricsReview:
    """Measures a draft against its plan and its brief."""
    sections = parse_sections(text)
    issues: list[LyricIssue] = []

    if not sections or not any(lines for _, lines in sections):
        return LyricsReview(
            issues=[LyricIssue("empty", "no lyric lines were produced")],
            rhyme_rate=0.0,
            unique_rate=0.0,
        )

    # ── Structure ────────────────────────────────────────────────────
    # Only the kinds a song cannot be a song without count as missing. A
    # pre-chorus or bridge that never arrived is leanness, not a defect — and
    # at short durations the line budget arithmetically cannot hold every
    # planned kind at two lines each, so demanding them all would make every
    # one-minute song unacceptable by construction.
    essential = {"verse", "chorus", "hook", "drop", "movement"}
    wanted = [s.kind for s in plan.sections if s.carries_words]
    arrived = [tag for tag, lines in sections if lines]
    missing = [
        kind
        for kind in dict.fromkeys(wanted)
        if kind in essential and kind not in arrived
    ]
    if missing:
        issues.append(
            LyricIssue("structure", f"the plan asks for {missing} and the draft has none")
        )
    thin = [tag for tag, lines in sections if 0 < len(lines) < 2]
    if thin:
        issues.append(LyricIssue("structure", f"sections with a single line: {thin}"))

    # ── Density: will the model actually sing all of this? ───────────
    # Blocking on purpose. A sheet longer than the clock allows is not a
    # stylistic quibble — the model drops the excess without saying so, and
    # the customer gets a song missing its verses.
    total_lines = sum(len(lines) for _, lines in sections)
    if total_lines > plan.line_budget:
        issues.append(
            LyricIssue(
                "density",
                f"{total_lines} lines will not fit a {plan.total_seconds:.0f}s song; "
                f"write at most {plan.line_budget} lines "
                f"(~{plan.lines_per_section} per section) or the model will drop some",
            )
        )

    # ── Repetition ───────────────────────────────────────────────────
    chorus_tags = {"chorus", "hook", "drop", "refrain"}
    verse_lines = [
        line for tag, lines in sections if tag not in chorus_tags for line in lines
    ]
    counts = Counter(line.lower() for line in verse_lines)
    repeated = [line for line, count in counts.items() if count > 1]
    unique_rate = (
        sum(1 for line in verse_lines if counts[line.lower()] == 1) / len(verse_lines)
        if verse_lines
        else 1.0
    )
    if repeated:
        issues.append(
            LyricIssue(
                "repetition",
                f"{len(repeated)} line(s) outside the chorus are reused verbatim, "
                f"starting with {repeated[0]!r}",
            )
        )

    # Choruses that disagree with each other read as a mistake rather than as
    # variation, which is why polish_lyrics fixes exactly this.
    choruses = [tuple(lines) for tag, lines in sections if tag in chorus_tags]
    if len(set(choruses)) > 1:
        issues.append(
            LyricIssue("repetition", "the chorus is not the same twice — it should be")
        )

    # ── Rhyme ────────────────────────────────────────────────────────
    partnered = 0
    total = 0
    for _, lines in sections:
        if len(lines) < 2:
            continue
        for index, line in enumerate(lines):
            total += 1
            if any(
                lines_rhyme(line, other)
                for offset, other in enumerate(lines)
                if offset != index
            ):
                partnered += 1
    rhyme_rate = partnered / total if total else 0.0
    if rhyme_rate < 0.5:
        issues.append(
            LyricIssue(
                "rhyme",
                f"only {rhyme_rate:.0%} of lines rhyme with another line in their "
                "section; aim for a consistent scheme within each section",
            )
        )

    # ── Details the user gave ────────────────────────────────────────
    if brief and brief.must_keep:
        lowered = text.lower()
        lost = [detail for detail in brief.must_keep if detail.lower() not in lowered]
        if lost:
            issues.append(
                LyricIssue("detail", f"these must appear and do not: {lost}")
            )

    return LyricsReview(issues=issues, rhyme_rate=rhyme_rate, unique_rate=unique_rate)


# ── Polish: only the repairs that cannot be wrong ────────────────────────


def polish_lyrics(text: str, plan: SongPlan) -> str:
    """Mechanical repairs to a draft. Never invents a line.

    Three fixes, all of them things a listener notices and none of them a
    judgement call:

      1. every chorus becomes the same chorus (the longest one written);
      2. a line immediately repeating itself inside a verse is dropped;
      3. empty sections and stray blank tags disappear.

    Anything requiring new words — a weak rhyme, a thin verse — is left alone
    and reported by `review_lyrics` instead, because guessing at replacement
    words produces exactly the robotic forced rhyme the client complained
    about.
    """
    if not plan.has_lyrics:
        # An instrumental plan that came back with words is a writer ignoring
        # its brief. Sung words over an ambient piece is not a small stylistic
        # difference, it is a different product.
        return ""

    sections = parse_sections(text)
    if not sections:
        return text.strip()

    chorus_tags = {"chorus", "hook", "drop", "refrain"}
    choruses = [lines for tag, lines in sections if tag in chorus_tags and lines]
    canonical = max(choruses, key=len) if choruses else None

    rebuilt: list[str] = []
    for tag, lines in sections:
        if tag in chorus_tags and canonical:
            kept = list(canonical)
        else:
            kept = []
            for line in lines:
                if kept and line.strip().lower() == kept[-1].strip().lower():
                    continue
                kept.append(line)
        if not kept:
            continue
        rebuilt.append(f"[{tag}]")
        rebuilt += kept
        rebuilt.append("")

    return "\n".join(rebuilt).strip()


# ── The writer seam ──────────────────────────────────────────────────────


class UnsupportedLyricLanguage(RuntimeError):
    """A writer was asked for a language it cannot write.

    Deliberately not a soft failure. The caller's options are to pick a writer
    that can, or to tell the customer — and both need to know it happened,
    which a returned sheet of English words does not communicate.
    """


class LyricsWriteFailed(RuntimeError):
    """A writer ran and could not produce a usable sheet.

    Distinct from `UnsupportedLyricLanguage`, which is a writer declining
    before it starts. This one means it tried: the service was unreachable,
    rate-limited, badly configured, or it answered in the wrong language.

    `retriable` records which kind, and it is for the *writer's own* retry
    budget rather than the job's — a chain of writers moves on to the next one
    either way. It exists so that a revoked API key is not retried three times
    before the fallback is reached.
    """

    def __init__(self, detail: str, *, retriable: bool = True) -> None:
        self.retriable = retriable
        super().__init__(detail)


class NoLyricsWriterAvailable(RuntimeError):
    """Every configured writer refused or failed.

    The end of the chain, and deliberately not a quiet return of `None`: an
    empty sheet is how the music model is told to make an instrumental, so
    silently returning nothing here would deliver a wordless track to someone
    who asked for a song with words.
    """


class LyricsWriter(Protocol):
    """Whatever actually writes words.

    Kept a protocol because the choice is not made: a hosted language model, a
    local one, or the music model's own lyric mode are all reasonable and none
    of them should require changing anything above this line. `notes` carries
    the previous review's complaints, so a second pass is a targeted revision
    rather than a fresh roll of the dice.
    """

    supported_languages: frozenset[str]
    """
    Canonical language codes this writer can actually write in. Empty means
    "any" — a language model does not need to enumerate them.

    This exists because the honest answer to "write me a chorus in Urdu" from a
    writer with an English phrasebook is *no*, and the caller needs to be able
    to ask before it has a sheet of English words in its hand. A writer that
    answered by writing English anyway is what made the language selector look
    connected while changing nothing about the song.
    """

    async def write(
        self, brief: LyricBrief, plan: SongPlan, notes: list[str] | None = None
    ) -> str: ...


async def write_lyrics(
    brief: LyricBrief,
    plan: SongPlan,
    writer: LyricsWriter | None,
    *,
    max_rounds: int = 2,
) -> tuple[str, LyricsReview] | None:
    """Draft, review, revise once, polish. None when no writer is configured.

    Bounded at `max_rounds` on purpose: each round is a full generation, and
    the improvement from a third is not worth the latency a user waits through.
    The best draft seen wins even if it never became "acceptable", because
    shipping the better of two flawed drafts beats failing the job over rhyme.
    """
    if writer is None or not plan.has_lyrics:
        return None

    best: tuple[str, LyricsReview] | None = None
    notes: list[str] | None = None

    for round_index in range(max(1, max_rounds)):
        draft = polish_lyrics(await writer.write(brief, plan, notes), plan)
        review = review_lyrics(draft, plan, brief)
        logger.info(
            "lyrics_reviewed",
            extra={
                "round": round_index + 1,
                "issues": len(review.issues),
                "rhyme_rate": round(review.rhyme_rate, 2),
                "unique_rate": round(review.unique_rate, 2),
            },
        )
        if best is None or _score(review) > _score(best[1]):
            best = (draft, review)
        if review.acceptable:
            break
        notes = review.notes

    return best


def _score(review: LyricsReview) -> float:
    """Ranks two drafts. Structural problems outweigh imperfect rhyme."""
    blocking = sum(1 for issue in review.issues if issue.kind != "rhyme")
    return review.rhyme_rate + review.unique_rate - blocking
