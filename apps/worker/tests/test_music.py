"""Music generation: minutes, sections, lyric quality, and the provider seam.

The model lives behind `MusicGenerationProvider`, so almost everything here is
provable without one. These tests drive a fake provider that writes genuine
MP3s, which means the assembly, loudness, validation, cancellation and
duplicate-detection paths run for real — the only thing a chosen model adds is
the music itself.

That seam is the point. The same suite passes whichever model is configured,
and `tests/test_acestep_provider.py` covers the one file that knows which model
that currently is.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from tests.conftest import collect, make_job, needs_ffmpeg
from worker.adapters.base import AdapterError, JobCancelled
from worker.adapters.music import MusicAdapter
from worker.core.config import settings
from worker.media import ffmpeg, probe_media
from worker.music import (
    LyricBrief,
    MusicGenerationProvider,
    MusicRequest,
    MusicTake,
    ProviderGenerationError,
    ProviderUnavailable,
    check_lyric_fit,
    detect_genre,
    line_budget,
    lines_rhyme,
    parse_sections,
    plan_song,
    polish_lyrics,
    review_lyrics,
    rhyme_key,
    salient_details,
    vocal_intent,
    write_lyrics,
)


def music_job(workspace: Path, duration: str = "1m", **overrides):
    defaults = dict(
        workflow_id="music",
        prompt="an upbeat pop song about summer in Lahore, hopeful, female vocals",
        parameters={"duration": duration},
        inputs=[],
        execution={"runtime": "music"},
        output_content_type="audio/mpeg",
    )
    return make_job(workspace, **{**defaults, **overrides})


class FakeProvider:
    """A provider that produces real audio without a model.

    Records every `MusicRequest` it receives, which is how the tests assert
    what the adapter actually asked for — duration, lyrics, bpm, key, seed —
    rather than only what came back.

    `vary=False` makes every section identical, proving the "a repeated section
    is a loop, not a longer song" guard.
    """

    name = "fake"

    def __init__(
        self,
        *,
        max_seconds: float = 600.0,
        vary: bool = True,
        unavailable: bool = False,
        fail: bool = False,
        empty: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.max_seconds = max_seconds
        self.requests: list[MusicRequest] = []
        self._vary = vary
        self._unavailable = unavailable
        self._fail = fail
        self._empty = empty
        self._delay = delay

    async def generate(self, request, workspace, on_progress=None):
        self.requests.append(request)
        if self._unavailable:
            raise ProviderUnavailable("no music service configured")
        if self._fail:
            raise ProviderGenerationError("the model fell over")
        if self._delay:
            await asyncio.sleep(self._delay)
        if on_progress is not None:
            await on_progress(0.5)
        if self._empty:
            return []

        destination = workspace / "provider"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"take-{len(self.requests):02d}.mp3"
        frequency = 200 + ((request.seed or 0) % 400) if self._vary else 440
        await ffmpeg(
            [
                "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=44100",
                "-t", f"{request.duration_seconds:.3f}",
                "-c:a", "libmp3lame", "-b:a", "128k",
                str(path),
            ]
        )
        return [MusicTake(path=path, seed=request.seed)]


def test_the_fake_provider_satisfies_the_protocol() -> None:
    """If this fails the seam has changed shape and a model swap is no longer
    a drop-in — which is the entire reason the seam exists."""
    assert isinstance(FakeProvider(), MusicGenerationProvider)


# ── The provider seam ────────────────────────────────────────────────────


async def test_an_unavailable_provider_fails_once_rather_than_retrying(
    workspace: Path,
) -> None:
    """A service that is not there will not be there on attempt three either.

    The customer must see generic copy while the log names the real cause —
    producing silence or a tone instead would be the worst outcome, a job that
    looks successful and delivers something that is not a song.
    """
    with pytest.raises(AdapterError) as raised:
        await collect(music_job(workspace), MusicAdapter(FakeProvider(unavailable=True)))

    assert raised.value.retriable is False
    assert "no music service configured" in raised.value.internal_detail
    assert raised.value.user_message == "This tool is temporarily unavailable."


async def test_a_generation_failure_is_retriable(workspace: Path) -> None:
    """Distinct from unavailable: the model ran and fell over, and another
    attempt may genuinely succeed."""
    with pytest.raises(AdapterError) as raised:
        await collect(music_job(workspace), MusicAdapter(FakeProvider(fail=True)))

    assert raised.value.retriable is True
    assert "fell over" in raised.value.internal_detail


async def test_a_provider_returning_no_audio_fails_the_job(workspace: Path) -> None:
    with pytest.raises(AdapterError) as raised:
        await collect(music_job(workspace), MusicAdapter(FakeProvider(empty=True)))

    assert "returned no audio" in raised.value.internal_detail


async def test_an_unknown_provider_name_is_a_configuration_error(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing music to a model is configuration, not a code change. A typo in
    that configuration must fail loudly rather than silently doing nothing."""
    monkeypatch.setattr(settings, "music_provider", "not-a-provider")

    with pytest.raises(AdapterError) as raised:
        await collect(music_job(workspace), MusicAdapter())

    assert raised.value.retriable is False
    assert "not-a-provider" in raised.value.internal_detail
    assert "not-a-provider" not in raised.value.user_message


def test_the_configured_provider_is_resolved_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the adapter must not touch the network or import a model's
    dependencies — worker startup happens on nodes that never run music."""
    monkeypatch.setattr(settings, "music_provider", "acestep")
    provider = MusicAdapter()._resolve_provider()
    assert provider.name == "acestep"
    assert provider.max_seconds >= 300, "must cover the product's longest song"


def test_the_adapter_claims_only_music() -> None:
    adapter = MusicAdapter()
    assert adapter.supports("music")
    assert not adapter.supports("music-video")
    assert not adapter.supports("text-to-video")


# ── Length, in minutes ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("chosen", "seconds"),
    [("1m", 60), ("2m", 120), ("3m", 180), ("4m", 240), ("5m", 300)],
)
def test_the_whole_offered_range_reads_as_minutes(
    chosen: str, seconds: int, workspace: Path
) -> None:
    """Music is chosen in minutes, not in video-style second presets. The list
    itself belongs to the workflow YAML, so extending it past five is an edit
    there and nothing here has to change."""
    assert MusicAdapter()._requested_seconds(music_job(workspace, chosen)) == seconds


@pytest.mark.parametrize("bad", [None, "", "soon", "0m", "-2m"])
def test_an_unusable_duration_fails_before_any_generation(
    bad, workspace: Path
) -> None:
    job = music_job(workspace, parameters={"duration": bad} if bad is not None else {})
    with pytest.raises(AdapterError) as raised:
        MusicAdapter()._requested_seconds(job)

    assert raised.value.retriable is False


@pytest.mark.parametrize("minutes", [1, 2, 3, 4, 5])
def test_a_provider_that_covers_the_range_needs_no_sectioning(
    minutes: int, workspace: Path
) -> None:
    """The current provider does 600s in one pass, so the whole 1–5 minute
    product range is a single generation and no seam is ever created."""
    adapter = MusicAdapter()
    sections = adapter._plan_sections(
        music_job(workspace), FakeProvider(max_seconds=600.0), minutes * 60.0, 1.5
    )

    assert len(sections) == 1
    assert sections[0].duration_seconds == pytest.approx(minutes * 60.0)


@pytest.mark.parametrize("minutes", [2, 3, 4, 5])
def test_a_limited_provider_still_delivers_the_chosen_length(
    minutes: int, workspace: Path
) -> None:
    """The fallback path, against a deliberately small ceiling.

    Crossfading N sections consumes N-1 fades of material, so generating
    exactly the requested length and then fading would deliver a song audibly
    short of what the user picked. The overlap is paid for in the plan.
    """
    adapter = MusicAdapter()
    total, fade = minutes * 60.0, 1.5
    provider = FakeProvider(max_seconds=60.0)

    sections = adapter._plan_sections(music_job(workspace), provider, total, fade)
    generated = sum(section.duration_seconds for section in sections)

    assert len(sections) > 1, "a 60s ceiling cannot cover a multi-minute song"
    assert generated - (len(sections) - 1) * fade == pytest.approx(total, abs=0.01)
    assert all(s.duration_seconds <= provider.max_seconds + 1e-6 for s in sections)


def test_sections_are_even_rather_than_ceiling_plus_a_fragment(
    workspace: Path,
) -> None:
    """Splitting five minutes as 60/60/60/60/60/7 ends the song on a
    seven-second fragment that sounds like exactly what it is."""
    sections = MusicAdapter()._plan_sections(
        music_job(workspace), FakeProvider(max_seconds=60.0), 300.0, 1.5
    )
    lengths = [section.duration_seconds for section in sections]

    assert max(lengths) - min(lengths) < 0.01


def test_the_ceiling_comes_from_the_provider_not_from_a_constant(
    workspace: Path,
) -> None:
    """A length limit is a property of a model, so replacing the model must
    change the plan without any edit to the adapter."""
    adapter = MusicAdapter()
    job = music_job(workspace)

    big = adapter._plan_sections(job, FakeProvider(max_seconds=600.0), 300.0, 1.5)
    small = adapter._plan_sections(job, FakeProvider(max_seconds=45.0), 300.0, 1.5)

    assert len(big) == 1
    assert len(small) > len(big)


# ── Song structure ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a hard rock anthem with distorted guitars", "rock"),
        ("a slow ballad for a wedding", "ballad"),
        ("a trap beat with heavy 808s", "hip-hop"),
        ("ambient lo-fi study music", "ambient"),
        ("something cheerful", "pop"),
    ],
)
def test_the_genre_is_read_from_what_the_user_asked_for(
    prompt: str, expected: str
) -> None:
    assert detect_genre(prompt) == expected


def test_structure_is_not_the_same_shape_for_every_genre() -> None:
    """The client asked specifically that structure not be forced. A pop song
    has a pre-chorus and a hip-hop track has a hook; an ambient piece has
    neither, and pretending otherwise produces a form the genre does not have.
    """
    pop = plan_song(180, genre="pop")
    hiphop = plan_song(180, genre="hip-hop")
    ambient = plan_song(180, genre="ambient")
    electronic = plan_song(180, genre="electronic")

    assert "pre-chorus" in pop.outline
    assert "hook" in hiphop.outline and "pre-chorus" not in hiphop.outline
    assert "drop" in electronic.outline
    assert "movement" in ambient.outline
    assert len({pop.outline, hiphop.outline, ambient.outline, electronic.outline}) == 4


def test_an_instrumental_plan_asks_for_no_words() -> None:
    """And a writer that returns some anyway is overruled — sung words over an
    ambient piece is a different product, not a stylistic difference."""
    plan = plan_song(180, genre="ambient")
    assert plan.has_lyrics is False
    assert polish_lyrics("[verse]\nwords that should not be here", plan) == ""


# ── Who decides whether anyone sings ─────────────────────────────────────


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        # The reported failure: a genre word chose silence over a request for
        # a voice that is sitting in the same sentence.
        ("a lo-fi pop song with soft female vocals about rain", True),
        ("an ambient pop ballad with a female singer", True),
        ("a cinematic song with instrumental verses and a big sung chorus", True),
        # The negation is spent on the intro; the request survives it.
        ("a dreamy song, no vocals in the intro, then she sings", True),
        # Asked for plainly, and the only way the product offers.
        ("an instrumental piano piece", False),
        ("ambient music for sleeping, no vocals", False),
        ("a beat with no lyrics", False),
        # Said nothing either way: the genre still decides.
        ("a calm lo-fi study beat", None),
        ("an upbeat pop song about summer in the city", None),
    ],
)
def test_a_stated_vocal_request_is_read_from_the_prompt(
    prompt: str, expected: bool | None
) -> None:
    assert vocal_intent(prompt) is expected


def test_a_wordless_genre_asked_for_vocals_gets_somewhere_to_put_them() -> None:
    """The trap, closed at both ends.

    `ambient` is wordless AND its skeleton is intro/movement/outro — not one
    of those sections carries words. Flipping only the flag would produce a
    plan that claims to have lyrics and has nowhere to sing them, so the plan
    borrows a worded shape while keeping its own genre name.
    """
    silent = plan_song(180, genre="ambient")
    singing = plan_song(180, genre="ambient", vocals=True)

    assert silent.has_lyrics is False
    assert singing.has_lyrics is True
    assert singing.genre == "ambient", "the music is still ambient"
    assert "verse" in singing.outline and "chorus" in singing.outline
    assert singing.lines_per_section >= 2


def test_a_stated_instrumental_silences_a_worded_genre() -> None:
    """It has to work in both directions, or the prompt is not the channel."""
    plan = plan_song(180, genre="pop", vocals=False)
    assert plan.has_lyrics is False
    assert polish_lyrics("[verse]\nwords that should not be here", plan) == ""


@pytest.mark.parametrize("seconds", [60, 120, 180, 240, 300])
def test_a_plan_describes_exactly_the_length_that_was_asked_for(seconds: int) -> None:
    plan = plan_song(seconds, genre="pop")
    assert sum(section.seconds for section in plan.sections) == pytest.approx(seconds)


def test_a_longer_song_gets_more_song_rather_than_slower_sections() -> None:
    """Five minutes should be more verses and choruses, not a two-minute
    structure stretched until every section drags."""
    short = plan_song(90, genre="pop")
    long = plan_song(300, genre="pop")

    assert len(long.sections) > len(short.sections)
    assert long.outline.count("chorus") > short.outline.count("chorus")


# ── Lyric quality: the measurable half ───────────────────────────────────


def test_rhyme_matching_survives_english_spelling() -> None:
    assert rhyme_key("night") == rhyme_key("light")
    assert rhyme_key("fire") == rhyme_key("desire")
    assert rhyme_key("night") != rhyme_key("cat")


def test_repeating_a_word_is_not_rhyming_with_it() -> None:
    """Otherwise the laziest possible lyric — every line ending "away" —
    scores as perfectly rhymed, which is the failure being measured."""
    assert lines_rhyme("we drove away", "we walked away") is False
    assert lines_rhyme("we drove away", "there is nothing left to say") is True


def test_sections_are_read_from_their_tags() -> None:
    sections = parse_sections("[verse]\nline one\nline two\n\n[chorus]\nhook line\n")
    assert sections == [("verse", ["line one", "line two"]), ("chorus", ["hook line"])]


def test_the_review_names_repetition_outside_the_chorus() -> None:
    """A chorus repeating is the song working. A verse repeating is the song
    failing, and the two must not be measured together."""
    plan = plan_song(120, genre="pop")
    lyrics = (
        "[verse]\nthe same tired line\nthe same tired line\n"
        "[chorus]\nsing it out loud\nlift it off the ground\n"
        "[verse]\nthe same tired line\nsomething else entirely\n"
        "[chorus]\nsing it out loud\nlift it off the ground\n"
    )
    review = review_lyrics(lyrics, plan)

    assert any(issue.kind == "repetition" for issue in review.issues)
    assert review.unique_rate < 0.7
    assert review.acceptable is False


def test_the_review_notices_a_chorus_that_changed_its_mind() -> None:
    plan = plan_song(120, genre="pop")
    lyrics = (
        "[verse]\nwalking through the rain\nlooking for a train\n"
        "[chorus]\nhold me in the light\n"
        "[chorus]\nhold me through the night\n"
    )
    review = review_lyrics(lyrics, plan)
    assert any("same twice" in issue.detail for issue in review.issues)


def test_the_review_reports_details_the_user_gave_and_the_draft_lost() -> None:
    """The lyric version of the client's video complaint: explicit details
    quietly generalised away. A song about Lahore that never says Lahore."""
    brief = LyricBrief.from_prompt("a hopeful pop song about summer in Lahore")
    plan = plan_song(120, genre="pop")

    lyrics = "[verse]\nsummer in the city\nwarm and bright and pretty\n"
    review = review_lyrics(lyrics, plan, brief)

    assert any(issue.kind == "detail" and "Lahore" in issue.detail for issue in review.issues)


def test_salient_details_picks_names_and_counts_not_sentence_openers() -> None:
    assert salient_details("A song about Maria in Lahore with 3 brothers") == [
        "Maria",
        "Lahore",
        "3",
    ]


def test_polish_makes_every_chorus_the_same_chorus() -> None:
    """The one repair that is always an improvement and never a judgement
    call. Everything needing new words is reported instead, because inventing
    replacements is how the forced robotic rhyme gets in."""
    plan = plan_song(120, genre="pop")
    polished = polish_lyrics(
        "[verse]\nwalking through the rain\nwalking through the rain\n"
        "[chorus]\nhold me in the light\nkeep me here tonight\n"
        "[chorus]\nhold me in the dark\n",
        plan,
    )
    sections = parse_sections(polished)

    choruses = [lines for tag, lines in sections if tag == "chorus"]
    assert len(choruses) == 2 and choruses[0] == choruses[1]
    # …and the verse's immediate self-repeat is gone.
    assert [lines for tag, lines in sections if tag == "verse"] == [
        ["walking through the rain"]
    ]


class _Writer:
    """A stand-in lyricist: hands back prepared drafts and records its notes."""

    def __init__(self, *drafts: str) -> None:
        self.drafts = list(drafts)
        self.notes: list[list[str] | None] = []

    async def write(self, brief, plan, notes=None) -> str:
        self.notes.append(notes)
        return self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]


async def test_a_weak_draft_is_sent_back_with_the_reasons_it_was_weak() -> None:
    """The quality pass that makes this better than one-shot generation: the
    second attempt is a targeted revision, not another roll of the dice."""
    plan = plan_song(120, genre="pop")
    brief = LyricBrief.from_prompt("a pop song about the sea")
    writer = _Writer(
        # Repetition across sections, which polish cannot silently repair —
        # only a rewrite can, so it must reach the writer as a note.
        "[verse]\nover and over\nwaiting for the rain\n"
        "[chorus]\nnothing here at all\n"
        "[verse]\nover and over\nwaiting for the rain\n",
        "[verse]\nsalt on the window\nlight where the waves go\n"
        "[chorus]\nout past the harbour wall\nlisten for the gulls that call\n"
        "[verse]\nropes on the harbour side\nwaiting for the turning tide\n"
        "[chorus]\nout past the harbour wall\nlisten for the gulls that call\n",
    )

    written = await write_lyrics(brief, plan, writer)
    assert written is not None
    text, review = written

    assert writer.notes[0] is None, "the first attempt gets no notes"
    assert writer.notes[1], "the second attempt must be told what was wrong"
    assert any("repetition" in note for note in writer.notes[1])
    assert "harbour" in text
    assert review.unique_rate > 0.7


async def test_no_writer_means_no_lyrics_rather_than_invented_ones() -> None:
    """With nothing configured to write words, the structure plan still reaches
    the provider. Filling the gap with generated filler would be worse than an
    instrumental."""
    plan = plan_song(120, genre="pop")
    assert await write_lyrics(LyricBrief.from_prompt("a pop song"), plan, None) is None


async def test_revision_stops_after_a_bounded_number_of_rounds() -> None:
    """Each round is a full generation. A third rarely helps and the user waits
    through all of them, so the best draft seen ships even if it never became
    flawless."""
    plan = plan_song(120, genre="pop")
    writer = _Writer("[verse]\nsame\nsame\n")

    written = await write_lyrics(LyricBrief.from_prompt("a pop song"), plan, writer)
    assert written is not None
    assert len(writer.notes) == 2


# ── Lyric density: the constraint measured on the GPU ────────────────────


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(60, 10), (120, 20), (180, 30), (240, 40), (300, 50)],
)
def test_the_line_budget_scales_with_the_songs_length(
    seconds: int, expected: int
) -> None:
    """Measured on the GPU, and re-measured when the model changed under it.

    The previous budgets (13s/line: 4 lines at 60s, 23 at 300s) came from an
    RTX 5090 and an older ACE-Step checkpoint. Against the build in production
    that density is where coverage falls apart — a three-minute song sang for
    52.8% of its length with a 43-second hole in the middle and nothing for the
    first thirty seconds, measured 2026-08-21 from a separated vocal stem across
    twelve cells. See `_SECONDS_PER_LINE` for the whole matrix.
    """
    assert line_budget(seconds) == expected


def test_a_short_song_is_planned_for_fewer_words_than_a_long_one() -> None:
    assert plan_song(60, genre="pop").line_budget < plan_song(300, genre="pop").line_budget
    # A one-line section reads as an unfinished thought.
    assert plan_song(60, genre="pop").lines_per_section >= 2


def test_an_oversized_sheet_is_measured_rather_than_guessed_at() -> None:
    sixteen_lines = (
        "[Verse 1]\na\nb\nc\nd\n[Chorus]\ne\nf\ng\nh\n"
        "[Verse 2]\ni\nj\nk\nl\n[Chorus]\nm\nn\no\np\n"
    )
    tight = check_lyric_fit(sixteen_lines, 60)
    roomy = check_lyric_fit(sixteen_lines, 300)

    assert tight.lines == 16
    assert tight.fits is False and tight.overflow > 0
    assert roomy.fits is True


def test_the_quality_pass_blocks_a_draft_that_will_not_fit() -> None:
    """Density is a blocking issue, not a stylistic one — the alternative is a
    song delivered without its verses and nobody told."""
    plan = plan_song(60, genre="pop")
    review = review_lyrics(
        "[Verse 1]\nline one here\nline two here\nline three here\nline four here\n"
        "[Chorus]\nline five here\nline six here\nline seven here\nline eight here\n"
        "[Verse 2]\nline nine here\nline ten here\nline eleven here\nline twelve here\n"
        "[Chorus]\nline a here\nline b here\nline c here\nline d here\n",
        plan,
    )

    density = [issue for issue in review.issues if issue.kind == "density"]
    assert density, "a 16-line sheet must not pass for a 60-second song"
    assert "drop" in density[0].detail
    assert review.acceptable is False


# ── End to end against a fake provider ───────────────────────────────────


@needs_ffmpeg
async def test_a_one_minute_request_produces_a_verified_one_minute_track(
    workspace: Path,
) -> None:
    provider = FakeProvider()
    result, reported = await collect(music_job(workspace, "1m"), MusicAdapter(provider))

    assert result.content_type == "audio/mpeg"
    assert result.kind == "audio"
    assert result.duration_seconds == pytest.approx(60.0, abs=2.0)

    info = await probe_media(result.path)
    assert info.has_audio is True
    assert info.has_video is False
    assert result.path.stat().st_size > 1024

    # One generation, and the user's words reached it untouched.
    assert len(provider.requests) == 1
    assert provider.requests[0].prompt == (
        "an upbeat pop song about summer in Lahore, hopeful, female vocals"
    )
    assert provider.requests[0].duration_seconds == pytest.approx(60.0)

    progress = [value for _, value, _ in reported]
    assert progress == sorted(progress)
    assert progress[-1] < 100


@needs_ffmpeg
async def test_the_customers_own_settings_reach_the_provider(workspace: Path) -> None:
    """bpm, key and supplied lyrics are product fields; they must survive the
    trip to the model rather than being dropped on the floor."""
    provider = FakeProvider()
    job = music_job(
        workspace,
        "1m",
        parameters={
            "duration": "1m",
            "bpm": 128,
            "key": "A Minor",
            "lyrics": "[Verse 1]\nsomething the user wrote\nand meant to keep\n",
        },
    )
    await collect(job, MusicAdapter(provider))

    request = provider.requests[0]
    assert request.bpm == 128
    assert request.key == "A Minor"
    assert "something the user wrote" in (request.lyrics or "")
    assert request.instrumental is False


@needs_ffmpeg
async def test_an_instrumental_request_carries_no_lyrics(workspace: Path) -> None:
    """`instrumental` is one concept, so it is one field: no words means no
    vocals, and the provider maps that to whatever its model expects."""
    provider = FakeProvider()
    job = music_job(workspace, "1m", parameters={"duration": "1m", "instrumental": True})
    await collect(job, MusicAdapter(provider))

    assert provider.requests[0].instrumental is True
    assert not (provider.requests[0].lyrics or "").strip()


@needs_ffmpeg
async def test_a_users_own_lyrics_are_never_rewritten(workspace: Path) -> None:
    """Even when they will not fit. Truncating here would mean choosing which
    of the customer's lines to lose, which is worse than the model doing it —
    the overflow is logged instead."""
    supplied = "[Verse 1]\n" + "\n".join(f"line number {n}" for n in range(1, 15))
    provider = FakeProvider()
    job = music_job(workspace, "1m", parameters={"duration": "1m", "lyrics": supplied})
    await collect(job, MusicAdapter(provider))

    assert provider.requests[0].lyrics == supplied


@needs_ffmpeg
async def test_a_retried_job_reproduces_its_own_song(workspace: Path) -> None:
    """Seeds are derived from the job id, so attempt two is the same song
    rather than a surprising new one."""
    first, second = FakeProvider(), FakeProvider()
    await collect(music_job(workspace, "1m"), MusicAdapter(first))
    await collect(music_job(workspace, "1m"), MusicAdapter(second))

    assert first.requests[0].seed == second.requests[0].seed
    assert first.requests[0].seed is not None


@needs_ffmpeg
async def test_a_long_song_is_assembled_from_sections_that_still_add_up(
    workspace: Path,
) -> None:
    """The fallback path: several generations, crossfaded, loudness-matched —
    and the delivered length is still the minute count the user picked."""
    provider = FakeProvider(max_seconds=20.0)
    job = music_job(
        workspace, "1m", execution={"runtime": "music", "max_segment_seconds": 20}
    )
    result, reported = await collect(job, MusicAdapter(provider))

    assert len(provider.requests) >= 3, "60s against a 20s ceiling is several sections"
    assert result.duration_seconds == pytest.approx(60.0, abs=2.0)

    # Each section knows which part of the song it is, so the provider is not
    # asked for the same average twenty seconds three times over.
    captions = {request.prompt for request in provider.requests}
    assert len(captions) > 1

    messages = [message for _, _, message in reported]
    assert any("section 1 of" in message.lower() for message in messages)


@needs_ffmpeg
async def test_repeating_one_section_is_refused_rather_than_shipped(
    workspace: Path,
) -> None:
    """The obvious cheat for long songs, and the one the client called out. A
    provider returning identical audio for every section produces a file of the
    right length that is unmistakably a loop."""
    job = music_job(
        workspace, "1m", execution={"runtime": "music", "max_segment_seconds": 20}
    )
    with pytest.raises(AdapterError) as raised:
        await collect(job, MusicAdapter(FakeProvider(max_seconds=20.0, vary=False)))

    assert "byte-identical" in raised.value.internal_detail
    assert "loop" in raised.value.internal_detail


@needs_ffmpeg
async def test_cancelling_a_long_song_stops_between_sections(workspace: Path) -> None:
    """A five-minute song is several generations. Cancelling one must stop the
    next from starting rather than paying for the whole plan."""
    provider = FakeProvider(max_seconds=20.0)
    cancelled = asyncio.Event()
    seen = 0

    async def cancel_after_the_first_section(
        status: str, progress: int, message: str, _details=None
    ) -> None:
        nonlocal seen
        if status == "generating":
            seen += 1
            # Reports bracket each section — start, provider progress, finish.
            # This cancels a job whose first section genuinely completed.
            if seen >= 3:
                cancelled.set()

    job = music_job(
        workspace,
        "1m",
        execution={"runtime": "music", "max_segment_seconds": 20},
        _cancelled=cancelled,
    )

    began = time.monotonic()
    with pytest.raises(JobCancelled):
        await MusicAdapter(provider).run(job, cancel_after_the_first_section)

    assert time.monotonic() - began < 30
    assert len(provider.requests) == 1, "a later section started on a cancelled job"
