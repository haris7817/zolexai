"""The built-in lyrics writer: templates, no model, no network.

## Why this exists

The music model treats an empty lyric sheet as "make an instrumental" —
verified on the GPU twice, 2026-08-13 and again 2026-08-16. Until this module,
nothing in the pipeline ever wrote a sheet, so every production music job was
silently asking for an instrumental. That was the client's "lyrics not present"
complaint, in its entirety.

## What it is, and is not

It is a floor: a deterministic writer that turns a prompt into a structured,
rhymed, topic-grounded lyric sheet that the model will actually sing. Every
couplet in the bank rhymes under `lines_rhyme` — pinned by a test, because the
reviewer that judges these drafts uses exactly that function.

It is not a poet. The lines are curated stock imagery with the user's subject
and named details woven in. A language-model writer will replace it behind the
same `LyricsWriter` protocol without touching anything else; this one exists so
that songs have words *today* and so the platform never again depends on a
writer that was never built.

## The density band (measured, load-bearing)

Line count is a band, not just a ceiling. On the GPU:

  * 8 lines at  60s → the model sang only the chorus, dropped both verses;
  * 5 lines at 120s → an 82-second instrumental intro, wordless padding;
  * 9 lines at 120s → vocals at 30s, every line sung;
  * 12 lines at 240s → every line sung, real arrangement.

So the writer aims for ~one line per 16 seconds — inside the band from both
sides — and never exceeds the plan's budget. The chorus is written ONCE: the
model repeats a chorus by itself (observed in the 240s benchmark), so writing
it twice would just double-charge the budget.
"""

from __future__ import annotations

import random
import re
import zlib

from worker.core.logging import get_logger
from worker.music.lyrics import (
    _GENRE_WORDS,
    LyricBrief,
    SongPlan,
    UnsupportedLyricLanguage,
)

logger = get_logger(__name__)

#: The density the writer aims for. Sits inside the measured band: dense enough
#: that vocals arrive early, sparse enough that nothing gets dropped.
_TARGET_SECONDS_PER_LINE = 16.0

#: Fewer lines than this is a loop, not a song, regardless of duration.
_MINIMUM_LINES = 4

# ── Moods ─────────────────────────────────────────────────────────────────

_BRIGHT_WORDS = frozenset(
    "upbeat happy joyful joy party dance dancing summer celebration celebrate "
    "fun energetic bright sunny wedding victory anthem uplifting hopeful".split()
)
_DARK_WORDS = frozenset(
    "sad heartbreak heartbroken lonely loneliness melancholy loss losing lost "
    "breakup grief missing goodbye farewell cry crying tears rainy sorrow "
    "leaving broken".split()
)


def detect_mood(prompt: str) -> str:
    """"bright", "dark", or "neutral" — decides which couplets qualify."""
    words = set(re.findall(r"[a-z'-]+", prompt.lower()))
    bright = len(words & _BRIGHT_WORDS)
    dark = len(words & _DARK_WORDS)
    if bright > dark:
        return "bright"
    if dark > bright:
        return "dark"
    return "neutral"


# ── The subject: what the song is about, in the user's words ─────────────

#: Words that describe the *recording*, not the song's subject. A prompt like
#: "pop song about summer in Lahore, female vocals, clear singing" is about
#: summer in Lahore; the rest is production direction and must not be sung.
#:
#: Mood adjectives belong here too. They are real instructions — `detect_mood`
#: reads them — but they are instructions ABOUT the song, and a chorus that
#: opens "Summer nights in Karachi, catchy and energetic" is the writer
#: singing the brief back at the customer.
_PRODUCTION_TERMS = re.compile(
    r"\b(vocals?|vocalist|voice|singing|singer|sung|male|female|man|woman|"
    r"bpm|tempo|key|beats?|drums?|guitars?|pianos?|synths?|bass|strings|"
    r"acoustic|electric|clear|clean|catchy|studio|quality|production|mix|"
    r"instrumental|audio|sound|style|genre|"
    r"upbeat|energetic|uplifting|joyful|happy|sad|slow|fast|emotional|"
    r"heartfelt|melancholy|dreamy|chill|relaxing|dark|bright|moody|"
    r"cinematic|epic|smooth|soft|loud|powerful|gentle|romantic|nostalgic|"
    r"and|with|plus)\b",
    re.IGNORECASE,
)

#: Filler that precedes the subject when there is no "about" to anchor on.
_SONG_WORDS = re.compile(
    r"\b(write|make|create|generate|compose|an?|the|song|track|tune|music|"
    r"about|with|please|me|for)\b",
    re.IGNORECASE,
)


#: Genre and style names `_PRODUCTION_TERMS` does not cover, plus the English
#: number words. Both are things `salient_details` collects and neither is ever
#: a thing to sing about.
_STYLE_WORDS = frozenset(
    """latin latino latina afrobeat afrobeats reggaeton salsa bachata merengue
    cumbia flamenco bossa samba tango kpop jpop synthpop dreampop britpop
    jazz blues disco funk soul motown gospel reggae ska punk grunge indie
    techno trance dubstep drill grime afro amapiano bollywood qawwali ghazal
    orchestral symphonic classical baroque opera choir anthem ballad
    christmas holiday lofi chillhop trap hardstyle
    one two three four five six seven eight nine ten"""
    .split()
)


def singable_details(details: list[str]) -> list[str]:
    """The must-keep details that are actually things to sing ABOUT.

    `salient_details` finds capitalised words and bare numbers, which is the
    right shallow rule for names, places and counts — but it cannot tell a name
    from an ordinary word that happens to open the sentence. "Romantic Latin
    pop about two people…" yields "Romantic", "Latin" and "Two", and a brief
    that then DEMANDS those words appear gets exactly what it asked for.
    Measured on the GPU, 2026-08-19:

        "La brisa trae un aire romantic / bajo el cielo Latin del mar"
        "estamos Two en un baile fiel"
        "É o nosso verão, Two souls in love"

    Two failures at once: the writer singing the brief back at the customer,
    and an English word wedged into a Spanish song — which is the language
    guarantee leaking through the one door that bypasses it, since a detail is
    demanded verbatim and is therefore never translated.

    Real names, places and digits survive, because losing those is the failure
    the must-keep mechanism exists to prevent in the first place.

    Filtered here rather than inside `salient_details` so that the function
    itself — and the tests pinning it — keep describing what a prompt contains;
    this describes what is worth demanding of a writer, which is a narrower
    question asked in only one place.
    """
    return [
        detail
        for detail in details
        if not _PRODUCTION_TERMS.fullmatch(detail.strip())
        and detail.strip().lower() not in _STYLE_WORDS
    ]


def extract_subject(prompt: str) -> str:
    """The thing the song is about, lifted from the prompt.

    Prefers the clause after "about". Comma-separated segments that are pure
    production direction ("female vocals", "clear singing") are dropped; the
    rest survive verbatim, because the subject is the one part of the prompt
    the user will listen for.
    """
    match = re.search(r"\babout\b(.+)", prompt, re.IGNORECASE)
    text = match.group(1) if match else _SONG_WORDS.sub(" ", _strip_genre(prompt))

    segments = []
    for segment in re.split(r"[,;.]", text):
        cleaned = segment.strip(" \t-—")
        if not cleaned:
            continue
        # A segment is production direction when removing those terms leaves
        # nothing meaningful behind ("catchy and energetic" → ""), and subject
        # matter when something survives ("summer in Lahore" → itself). What
        # survives keeps its original wording: the subject is the one part of
        # the prompt the customer listens for, so it is never paraphrased.
        if not _PRODUCTION_TERMS.sub("", cleaned).strip(" \t-—"):
            continue
        segments.append(cleaned)

    subject = ", ".join(segments).strip()
    subject = re.sub(r"\s+", " ", subject)
    if len(subject) > 60:
        subject = subject[:60].rsplit(" ", 1)[0]
    return subject or "this moment"


def _strip_genre(prompt: str) -> str:
    """The prompt with genre words removed, so "an energetic synthwave track"
    does not become a song about synthwave.

    Strips every trigger word the genre detector knows, not just the one that
    resolved — the resolved name ("electronic") is often not the word the
    user typed ("synthwave")."""
    pattern = "|".join(
        re.escape(phrase) for phrase in sorted(_GENRE_WORDS, key=len, reverse=True)
    )
    return re.sub(rf"\b(?:{pattern})\b", " ", prompt, flags=re.IGNORECASE)


# ── The bank ──────────────────────────────────────────────────────────────
#
# Every couplet's two lines rhyme under `lines_rhyme` — the same function the
# reviewer scores drafts with — and `tests/test_lyrics_writer.py` pins that for
# every entry, because a bank couplet that stops rhyming under a heuristic
# change would silently drag every song below the review threshold.
#
# Moods: a "neutral" couplet serves any song; "bright"/"dark" serve their own.

_VERSE: tuple[tuple[str, tuple[str, str]], ...] = (
    ("bright", ("There's a rhythm in the streets tonight", "every window spilling golden light")),
    ("bright", ("Dust on the mirror, sun in our eyes", "we found a doorway under open skies")),
    ("bright", ("Feel the pavement humming at our feet", "every corner drumming out the beat")),
    ("bright", ("Hold the moment, hold it tight", "we could live forever if we time it right")),
    ("bright", ("If the whole sky opens into rain", "we will sing it back to sun again")),
    ("neutral", ("Every heartbeat keeps us moving on", "we'll be dancing till the dark is gone")),
    ("neutral", ("We took the long way down this road", "shook off the weight of every load")),
    ("neutral", ("The evening pulls us close and near", "there is nothing left for us to fear")),
    ("dark", ("The echo of your voice is fading slow", "I hold it like an afterglow")),
    ("dark", ("Empty rooms remember where you'd stand", "I still reach out for a missing hand")),
    ("dark", ("I keep the memories behind the glass", "waiting for this heaviness to pass")),
    ("dark", ("The night is long, the silence runs deep", "full of promises I couldn't keep")),
    ("dark", ("Cold coffee by an unmade bed", "your name still ringing in my head")),
    ("dark", ("I wear the winter like a second skin", "still looking for a way to let the light back in")),
)

_PRE_CHORUS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("bright", ("One step closer to the edge of night", "we are reaching for the light")),
    ("bright", ("Turn it up until the rooftops shake", "no mistakes tonight, we're wide awake")),
    ("neutral", ("Something's rising we can't name", "nothing's ever gonna feel the same")),
    ("dark", ("If it ever falls apart", "we still know the way back to the start")),
)

#: Chorus blocks open with the subject alone — a title line, the thing the
#: user typed, sung back to them — then a rhymed couplet under it.
_CHORUS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("bright", ("tonight we come alive", "open hearts in overdrive")),
    ("bright", ("we're never gonna fade away", "this feeling's here to stay")),
    ("neutral", ("singing till the break of day", "nothing's gonna steal us away")),
    ("dark", ("I carry you in every song", "right here is where you belong")),
    ("dark", ("even when the lights go down", "you're the quiet in this town")),
)

_HOOK: tuple[tuple[str, tuple[str, str]], ...] = (
    ("neutral", ("hold it down", "all my people run this town")),
    ("neutral", ("say it loud", "standing tall, standing proud")),
)

_DROP: tuple[tuple[str, tuple[str, str]], ...] = (
    ("neutral", ("feel the bass, feel the glow", "moving fast and moving slow")),
    ("neutral", ("hands up to the sky", "we were made to fly")),
)

_BRIDGE: tuple[tuple[str, tuple[str, str]], ...] = (
    ("bright", ("Strip it back to just a spark", "we can glow inside the dark")),
    ("neutral", ("Take it back to something true", "a different point of view")),
    ("dark", ("Maybe time will turn the tide", "till then I keep you by my side")),
)

#: When a named detail from the prompt (a place, a name, a number) has not
#: landed anywhere else, this couplet carries it — because the user listening
#: for "Lahore" and never hearing it is the lyrical form of the generalisation
#: complaint this project keeps re-learning.
_DETAIL_COUPLET = ("{detail} keeps calling out my name", "after this we won't be the same")

_BANKS: dict[str, tuple[tuple[str, tuple[str, str]], ...]] = {
    "verse": _VERSE,
    "pre-chorus": _PRE_CHORUS,
    "chorus": _CHORUS,
    "hook": _HOOK,
    "drop": _DROP,
    "bridge": _BRIDGE,
}

#: Section kinds that behave like a chorus: written once, the model repeats
#: them, and their block opens with the subject line.
_ANCHOR_KINDS = ("chorus", "hook", "drop")


class TemplateLyricsWriter:
    """A `LyricsWriter` with no dependencies beyond this file.

    Deterministic per (topic, genre, length): a retried job sings the same
    words, for the same reason the provider gets a fixed seed. A revision round
    (non-empty `notes`) reshuffles the bank so the second draft is a different
    draft rather than the same one resubmitted.
    """

    name = "template"
    """Internal label, matching the value that selects it in
    MUSIC_LYRICS_WRITER. The chain logs which writer answered, and a log line
    naming the config value is one somebody can act on."""

    supported_languages = frozenset({"en"})
    """
    English, and only English — every couplet in the bank above is English.

    This is the writer admitting its own limit so the adapter can refuse a
    request it cannot honour. It replaces a warning that was logged and then
    ignored: the old code noticed it had been asked for Urdu, said so in the
    log, and returned English anyway. Nobody reads a log line that the customer
    never sees, and the resulting song was in the wrong language with nothing
    in the product to indicate it.
    """

    async def write(
        self, brief: LyricBrief, plan: SongPlan, notes: list[str] | None = None
    ) -> str:
        seed = zlib.crc32(
            f"{brief.topic}|{plan.genre}|{plan.total_seconds:.0f}|{len(notes or [])}".encode()
        )
        rng = random.Random(seed)

        # The adapter checks `supported_languages` before it ever gets here, so
        # this is a backstop against a future caller that forgets. It raises
        # rather than warns: returning English from a call that asked for
        # Spanish is the bug, not the recovery.
        requested = brief.language.strip().lower()
        if requested and requested not in self.supported_languages:
            raise UnsupportedLyricLanguage(
                f"the template writer's bank is English; it cannot write {requested!r}"
            )

        mood = detect_mood(brief.topic)
        subject = extract_subject(brief.topic)
        sheet = self._compose(rng, plan, mood, subject)
        sheet = self._ensure_details(sheet, brief, plan)

        text = _render(sheet)
        logger.info(
            "lyrics_written",
            extra={
                "writer": "template",
                "mood": mood,
                "lines": sum(len(lines) for _, lines in sheet),
                "budget": plan.line_budget,
                "sections": len(sheet),
            },
        )
        return text

    # ── Composition ──────────────────────────────────────────────────────

    def _compose(
        self, rng: random.Random, plan: SongPlan, mood: str, subject: str
    ) -> list[tuple[str, list[str]]]:
        wanted = list(dict.fromkeys(s.kind for s in plan.sections if s.carries_words))
        target = self._target_lines(plan)

        pools = {kind: _pool(rng, _BANKS[kind], mood) for kind in _BANKS}
        anchor = next((kind for kind in _ANCHOR_KINDS if kind in wanted), None)

        sheet: list[tuple[str, list[str]]] = []
        budget = target

        def emit(kind: str, lines: list[str]) -> None:
            nonlocal budget
            sheet.append((kind, lines))
            budget -= len(lines)

        # The core of any song comes first and is never traded away: one
        # verse and the anchor. The anchor opens with the subject line only
        # when the budget can afford a third line — at 1 minute it cannot,
        # and the subject then reaches the sheet through `_ensure_details`.
        if "verse" in wanted and pools["verse"]:
            emit("verse", list(pools["verse"].pop()))
        if anchor and pools[anchor]:
            couplet = pools[anchor].pop()
            lines = [_title(subject), *couplet] if budget >= 3 else list(couplet)
            emit(anchor, lines)

        if "pre-chorus" in wanted and budget >= 2 and pools["pre-chorus"]:
            # Before the anchor, where a pre-chorus lives.
            position = next(
                (i for i, (kind, _) in enumerate(sheet) if kind == anchor), len(sheet)
            )
            sheet.insert(position, ("pre-chorus", list(pools["pre-chorus"].pop())))
            budget -= 2

        # Fill the remaining budget with distinct verses, a bridge earning its
        # place only once the song is long enough to want a departure.
        bridge_due = "bridge" in wanted
        while budget >= 2:
            if bridge_due and len(sheet) >= 3 and pools["bridge"]:
                emit("bridge", list(pools["bridge"].pop()))
                bridge_due = False
                continue
            if "verse" in wanted and pools["verse"]:
                emit("verse", list(pools["verse"].pop()))
                continue
            if anchor and "verse" not in wanted and pools[anchor]:
                # Verse-less genres (electronic: only the drop carries words).
                # The single anchor block GROWS rather than gaining a sibling:
                # two differing chorus-family blocks read as a mistake, and
                # the reviewer rightly blocks on exactly that.
                index = next(i for i, (kind, _) in enumerate(sheet) if kind == anchor)
                kind, lines = sheet[index]
                sheet[index] = (kind, [*lines, *pools[anchor].pop()])
                budget -= 2
                continue
            break

        return sheet

    def _target_lines(self, plan: SongPlan) -> int:
        by_density = round(plan.total_seconds / _TARGET_SECONDS_PER_LINE)
        return min(plan.line_budget, max(_MINIMUM_LINES, by_density))

    # ── Details ──────────────────────────────────────────────────────────

    def _ensure_details(
        self,
        sheet: list[tuple[str, list[str]]],
        brief: LyricBrief,
        plan: SongPlan,
    ) -> list[tuple[str, list[str]]]:
        """Every named detail from the prompt appears somewhere in the sheet.

        Most arrive inside the subject line already. For any that did not, the
        last verse is rewritten around the detail rather than appended — the
        line budget is measured, and exceeding it is how words get dropped.
        """
        text = _render(sheet).lower()
        missing = [d for d in brief.must_keep if d.lower() not in text]
        if not missing:
            return sheet

        detail = ", ".join(missing[:2])
        couplet = [_DETAIL_COUPLET[0].format(detail=_title(detail)), _DETAIL_COUPLET[1]]
        for index in range(len(sheet) - 1, -1, -1):
            kind, _ = sheet[index]
            if kind == "verse":
                sheet[index] = ("verse", couplet)
                return sheet
        # No verse anywhere (electronic, hip-hop edge): the detail joins the
        # first anchor block's subject line instead of displacing a rhyme.
        if sheet:
            kind, lines = sheet[0]
            sheet[0] = (kind, [f"{lines[0]} — {_title(detail)}", *lines[1:]])
        return sheet


# ── Helpers ───────────────────────────────────────────────────────────────


def _pool(
    rng: random.Random,
    bank: tuple[tuple[str, tuple[str, str]], ...],
    mood: str,
) -> list[tuple[str, str]]:
    """The couplets this song may use, shuffled, most-suitable last (pop()).

    Neutral couplets serve every mood; bright and dark serve their own. A
    neutral song draws from bright and neutral, because an unqualified request
    for a song is a request for a pleasant one, not a heartbroken one.
    """
    allowed = {"bright": ("bright", "neutral"), "dark": ("dark", "neutral")}.get(
        mood, ("neutral", "bright")
    )
    matching = [couplet for tag, couplet in bank if tag in allowed]
    rng.shuffle(matching)
    return matching


def _title(subject: str) -> str:
    return subject[:1].upper() + subject[1:] if subject else subject


def _render(sheet: list[tuple[str, list[str]]]) -> str:
    parts: list[str] = []
    for kind, lines in sheet:
        parts.append(f"[{kind}]")
        parts.extend(lines)
        parts.append("")
    return "\n".join(parts).strip()
