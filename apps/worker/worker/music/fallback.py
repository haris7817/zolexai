"""Ordering the lyric writers, and the one rule that orders them.

## The rule

**A fallback must be able to write the language that was asked for.** That is
the whole content of this file. Everything else — trying Cerebras first,
dropping to the template bank, giving up cleanly — falls out of it.

The failure this prevents is specific and was already live in the product
before any of this existed: a customer picks Spanish, the writer that can
actually be reached only writes English, and the platform ships English lyrics
sung with Spanish phonetics. Nobody downstream can tell. The customer can, the
moment they press play.

So the chain does not fall back to *a writer*, it falls back to *a writer that
claims this language*. `TemplateLyricsWriter` declares `{"en"}`, so:

    English  + Cerebras down  → the template bank writes the song.
    Spanish  + Cerebras down  → the template bank is not asked, and the job
                                fails saying so.

The second line looks like a worse outcome than falling back. It is a much
better one: the alternative is a song in the wrong language delivered as though
nothing happened.

## Why not one writer with a fallback inside it

Because then every writer would need to know about every other writer, and the
Cerebras client would have to import the template bank in order to give up.
Composition keeps each writer ignorant of the others and keeps the ordering
policy — which is a *product* decision — in one readable place.
"""

from __future__ import annotations

from worker.core.logging import get_logger
from worker.music.lyrics import (
    LyricBrief,
    LyricsWriteFailed,
    LyricsWriter,
    NoLyricsWriterAvailable,
    SongPlan,
    UnsupportedLyricLanguage,
)

logger = get_logger(__name__)


def writer_name(writer: object) -> str:
    """A stable internal label for logs. Never customer-facing."""
    return str(getattr(writer, "name", type(writer).__name__))


def is_available(writer: object) -> bool:
    """Whether a writer can be attempted at all right now.

    Writers that need configuration (an API key, a service) expose
    `available`; ones that need nothing simply do not have the attribute and
    are always available. Defaulting to True rather than requiring every writer
    to declare it keeps the protocol as small as it already was.
    """
    return bool(getattr(writer, "available", True))


def can_write(writer: object, language: str) -> bool:
    """Whether this writer claims it can write `language`.

    An empty `supported_languages` means "any", which is how a language model
    answers a question it has no list for — see the protocol in `lyrics.py`.
    """
    supported: frozenset[str] = getattr(writer, "supported_languages", frozenset())
    return not supported or language in supported


class FallbackLyricsWriter:
    """Tries each writer in order; the first usable sheet wins.

    Satisfies `LyricsWriter` itself, so the adapter, the review loop and the
    tests above it cannot tell a chain from a single writer — which is what
    lets the chain be configured rather than coded in.
    """

    name = "chain"

    last_writer: str = ""
    """Which member produced the most recent sheet, for the adapter's log.

    An attribute rather than a return value because the `LyricsWriter` protocol
    returns a string, and widening it would touch every writer and every test
    for the sake of one diagnostic.
    """

    def __init__(self, writers: list[LyricsWriter]) -> None:
        self._writers = list(writers)

    @property
    def writers(self) -> list[LyricsWriter]:
        return list(self._writers)

    @property
    def available(self) -> bool:
        """Whether ANY member of the chain can be attempted.

        Answered separately from `supported_languages` because the protocol's
        "empty set means any language" convention cannot also express "no
        languages at all" — an unconfigured chain returning an empty set would
        claim it can write everything. So the two questions stay two questions,
        and the adapter asks this one first.
        """
        return any(is_available(writer) for writer in self._writers)

    def unavailable_reason(self) -> str:
        reasons = [
            f"{writer_name(writer)}: "
            + getattr(writer, "unavailable_reason", lambda: "not configured")()
            for writer in self._writers
            if not is_available(writer)
        ]
        return "; ".join(reasons) or "no lyrics writer is configured"

    @property
    def supported_languages(self) -> frozenset[str]:
        """Every language SOME currently-available member can write.

        Availability is part of the answer on purpose. A chain holding a
        Cerebras writer with no API key must not claim it can write Japanese,
        because the adapter asks this question *before* writing in order to
        refuse an impossible request early, with a message telling the customer
        what to do instead. Claiming the language and then failing mid-job
        would turn a clear refusal into an opaque one.

        Empty means "any", per the protocol — reachable either because a member
        is a language model with no list to give, or because nothing is
        available at all. `available` above is what distinguishes those.
        """
        languages: set[str] = set()
        for writer in self._writers:
            if not is_available(writer):
                continue
            supported: frozenset[str] = getattr(
                writer, "supported_languages", frozenset()
            )
            if not supported:
                return frozenset()
            languages |= supported
        return frozenset(languages)

    async def write(
        self, brief: LyricBrief, plan: SongPlan, notes: list[str] | None = None
    ) -> str:
        language = brief.language.strip().lower() or "en"
        problems: list[str] = []

        for position, writer in enumerate(self._writers, start=1):
            label = writer_name(writer)

            if not is_available(writer):
                reason = getattr(writer, "unavailable_reason", lambda: "not configured")()
                problems.append(f"{label}: unavailable ({reason})")
                logger.info(
                    "lyrics_writer_skipped",
                    extra={"writer": label, "reason": reason, "language": language},
                )
                continue

            if not can_write(writer, language):
                # The rule this module exists for. Skipping is not a
                # degradation to be logged quietly and forgotten — it is the
                # platform declining to substitute a language nobody asked for.
                problems.append(f"{label}: does not write {language!r}")
                logger.info(
                    "lyrics_writer_skipped",
                    extra={
                        "writer": label,
                        "reason": "language not supported",
                        "language": language,
                    },
                )
                continue

            try:
                text = await writer.write(brief, plan, notes)
            except (LyricsWriteFailed, UnsupportedLyricLanguage) as failure:
                problems.append(f"{label}: {failure}")
                logger.warning(
                    "lyrics_writer_failed",
                    extra={
                        "writer": label,
                        "language": language,
                        "retriable": getattr(failure, "retriable", None),
                        "detail": str(failure)[:400],
                    },
                )
                continue

            if not text.strip():
                problems.append(f"{label}: returned an empty sheet")
                logger.warning(
                    "lyrics_writer_failed",
                    extra={
                        "writer": label,
                        "language": language,
                        "detail": "empty sheet",
                    },
                )
                continue

            logger.info(
                "lyrics_writer_used",
                extra={
                    "writer": label,
                    "language": language,
                    "characters": len(text),
                    # Which position in the chain answered. 1 is the primary;
                    # anything higher means the primary did not, which is the
                    # number worth alerting on.
                    "position": position,
                },
            )
            self.last_writer = label
            return text

        raise NoLyricsWriterAvailable(
            f"no configured lyrics writer could write {language!r}: "
            + " || ".join(problems)
        )
