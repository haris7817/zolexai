"""The template lyrics writer — the thing that makes production tracks sing.

Why this suite is strict: the music model treats an empty lyric sheet as
"make an instrumental" (verified on the GPU, 2026-08-16), so this writer is
the difference between the product having vocals and not having them. And the
reviewer that scores its drafts uses `lines_rhyme`, so every couplet in the
bank is pinned against exactly that function — a bank entry that stops
rhyming under a heuristic change would silently drag every song below the
review threshold with no test failing anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.music import (
    LyricBrief,
    TemplateLyricsWriter,
    detect_mood,
    extract_subject,
    lines_rhyme,
    parse_sections,
    plan_song,
    review_lyrics,
    write_lyrics,
)
from worker.music.writer import _BANKS, _DETAIL_COUPLET

# ── The bank itself ──────────────────────────────────────────────────────


def test_every_bank_couplet_rhymes_under_the_reviewers_own_function() -> None:
    """The reviewer scores rhyme with `lines_rhyme`; a bank couplet that fails
    it is dead weight that lowers every song it appears in."""
    for kind, bank in _BANKS.items():
        for mood, (first, second) in bank:
            assert lines_rhyme(first, second), (
                f"bank couplet does not rhyme ({kind}, {mood}): "
                f"{first!r} / {second!r}"
            )


def test_the_detail_couplet_rhymes_with_a_detail_inserted() -> None:
    first = _DETAIL_COUPLET[0].format(detail="Lahore")
    assert lines_rhyme(first, _DETAIL_COUPLET[1])


# ── Subject and mood ─────────────────────────────────────────────────────


def test_the_subject_is_the_about_clause_without_production_direction() -> None:
    subject = extract_subject(
        "an upbeat pop song about summer in Lahore, female vocals, clear singing"
    )
    assert subject == "summer in Lahore"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "an upbeat pop song about summer nights in Karachi, female vocals, "
            "catchy and energetic",
            "summer nights in Karachi",
        ),
        (
            "a slow ballad about missing my hometown, male vocals, piano and "
            "strings, heartfelt",
            "missing my hometown",
        ),
        (
            "an upbeat pop anthem about winning together as a team, joyful and "
            "uplifting, female vocals",
            "winning together as a team",
        ),
        (
            "a song about my daughter Ayesha and her first day at school",
            "my daughter Ayesha and her first day at school",
        ),
    ],
)
def test_mood_direction_never_becomes_the_subject(prompt: str, expected: str) -> None:
    """Mood words are instructions ABOUT the song, not its subject.

    They are real — `detect_mood` reads them — but the subject line opens the
    chorus, and "Summer nights in Karachi, catchy and energetic" is the writer
    singing the customer's brief back at them.
    """
    assert extract_subject(prompt) == expected


def test_multiple_subject_segments_survive_and_production_terms_do_not() -> None:
    subject = extract_subject("a ballad about love, loss and hope, male vocals")
    assert "love" in subject
    assert "loss and hope" in subject
    assert "vocals" not in subject.lower()


def test_a_prompt_without_about_still_yields_a_subject() -> None:
    subject = extract_subject("energetic synthwave track")
    assert subject
    assert "synthwave" not in subject.lower()
    assert "track" not in subject.lower()


def test_mood_detection() -> None:
    assert detect_mood("an upbeat pop song about summer") == "bright"
    assert detect_mood("a sad piano song about heartbreak") == "dark"
    assert detect_mood("a song about mountains") == "neutral"


# ── The sheets it writes ─────────────────────────────────────────────────

#: The product's range: every offered length, several genres, both moods.
_MATRIX = [
    ("an upbeat pop song about summer in Lahore, female vocals", 60),
    ("an upbeat pop song about summer in Lahore, female vocals", 120),
    ("a sad ballad about missing home", 180),
    ("a country song about the open road", 240),
    ("an upbeat pop anthem about winning together", 300),
    ("a hip-hop track about hustle in the city", 180),
    ("a rock song about breaking free", 120),
]


@pytest.mark.parametrize(("prompt", "seconds"), _MATRIX)
async def test_written_sheets_pass_their_own_review(prompt: str, seconds: int) -> None:
    """The full production path: write → polish → review, via `write_lyrics`.

    `acceptable` is the reviewer's shipping bar — no blocking issues, rhyme
    and uniqueness above threshold. The writer failing its own reviewer would
    mean every job burns the revision round and ships flagged anyway.
    """
    brief = LyricBrief.from_prompt(prompt)
    plan = plan_song(seconds, genre=brief.genre)

    written = await write_lyrics(brief, plan, TemplateLyricsWriter())
    assert written is not None
    text, review = written

    assert review.acceptable, f"{review.issues} for {prompt!r} at {seconds}s\n{text}"


@pytest.mark.parametrize(("prompt", "seconds"), _MATRIX)
async def test_written_sheets_respect_the_measured_density_band(
    prompt: str, seconds: int
) -> None:
    """More lines than the budget and the model silently drops some; too few
    and the track opens with a minute of instrumental. Both were measured."""
    brief = LyricBrief.from_prompt(prompt)
    plan = plan_song(seconds, genre=brief.genre)
    text = await TemplateLyricsWriter().write(brief, plan)

    lines = [line for _, section in parse_sections(text) for line in section]
    assert len(lines) <= plan.line_budget
    # The sparse side of the band: hip-hop/electronic run leaner by design,
    # so the floor is one line per 24 seconds — the measured-bad sparse case
    # was one per 24s on a pop song; staying strictly denser than that.
    assert len(lines) >= min(plan.line_budget, max(4, int(seconds / 22)))


async def test_named_details_from_the_prompt_reach_the_sheet() -> None:
    """The user listening for "Lahore" and never hearing it is the lyrical
    form of the generalisation complaint this project keeps re-learning."""
    for seconds in (60, 120, 300):
        brief = LyricBrief.from_prompt(
            "an upbeat pop song about summer in Lahore, female vocals"
        )
        plan = plan_song(seconds, genre=brief.genre)
        text = await TemplateLyricsWriter().write(brief, plan)
        assert "lahore" in text.lower(), f"Lahore missing at {seconds}s:\n{text}"


async def test_an_electronic_track_grows_one_drop_rather_than_two_that_differ() -> None:
    """Two differing chorus-family blocks read as a mistake and the reviewer
    blocks on exactly that — verse-less genres extend the single block."""
    brief = LyricBrief.from_prompt("an edm track about neon nights")
    plan = plan_song(180, genre=brief.genre)
    text = await TemplateLyricsWriter().write(brief, plan)

    sections = parse_sections(text)
    assert [tag for tag, _ in sections].count("drop") == 1
    review = review_lyrics(text, plan, brief)
    assert not [i for i in review.issues if i.kind == "repetition"], review.issues


async def test_the_writer_is_deterministic_and_revisions_differ() -> None:
    """Deterministic per request for the same reason the provider gets a fixed
    seed: a retried job must reproduce its own song. A revision round must be
    a different draft, or the second pass is the first pass resubmitted."""
    brief = LyricBrief.from_prompt("a pop song about summer in Lahore")
    plan = plan_song(120, genre=brief.genre)
    writer = TemplateLyricsWriter()

    first = await writer.write(brief, plan)
    again = await writer.write(brief, plan)
    revised = await writer.write(brief, plan, notes=["rhyme: too low"])

    assert first == again
    assert revised != first


async def test_dark_prompts_do_not_get_party_couplets() -> None:
    brief = LyricBrief.from_prompt("a sad ballad about losing my father")
    plan = plan_song(120, genre=brief.genre)
    text = (await TemplateLyricsWriter().write(brief, plan)).lower()

    # Spot-check: the brightest bank lines stay out of a grief song.
    assert "wide awake" not in text
    assert "overdrive" not in text


# ── Adapter integration: the writer resolves like the provider does ─────


async def test_the_adapter_resolves_the_configured_chain_by_default(
    tmp_path: Path,
) -> None:
    """The default is now the hosted writer with this bank behind it."""
    from worker.adapters.music import MusicAdapter
    from worker.music.fallback import FallbackLyricsWriter

    writer = MusicAdapter()._resolve_writer()
    assert isinstance(writer, FallbackLyricsWriter)
    assert [w.name for w in writer.writers] == ["cerebras", "template"]


async def test_the_default_chain_changes_nothing_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The load-bearing safety property of changing that default.

    A deployment that has not configured Cerebras must behave EXACTLY as it did
    when this bank was the only writer: English lyrics get written, and a
    language the bank cannot write is refused rather than answered in English.
    Anything else would mean a config default silently altered live behaviour.
    """
    from worker.adapters.music import MusicAdapter
    from worker.core.config import settings

    monkeypatch.setattr(settings, "cerebras_api_key", "")
    writer = MusicAdapter()._resolve_writer()

    assert writer.supported_languages == frozenset({"en"})
    assert writer.available is True


async def test_a_single_configured_writer_is_not_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming exactly one writer keeps the pre-chain shape, so a deployment
    pinned to the bank gets the bank rather than a chain of one."""
    from worker.adapters.music import MusicAdapter
    from worker.core.config import settings

    monkeypatch.setattr(settings, "music_lyrics_writer", "template")
    assert isinstance(MusicAdapter()._resolve_writer(), TemplateLyricsWriter)


async def test_an_unknown_name_inside_a_chain_is_dropped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worker.adapters.music import MusicAdapter
    from worker.core.config import settings

    monkeypatch.setattr(settings, "music_lyrics_writer", "not-a-writer,template")
    assert isinstance(MusicAdapter()._resolve_writer(), TemplateLyricsWriter)


async def test_an_unknown_writer_name_degrades_to_no_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike an unknown provider, an unknown writer must not fail the job —
    a misconfiguration here should cost lyric quality, not the track."""
    from worker.adapters.music import MusicAdapter
    from worker.core.config import settings

    monkeypatch.setattr(settings, "music_lyrics_writer", "not-a-writer")
    assert MusicAdapter()._resolve_writer() is None
