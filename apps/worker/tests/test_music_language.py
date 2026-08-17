"""Multilingual lyrics: the selection has to survive all the way to the model.

The bug this suite exists for is not a crash. Every stage looked correct — the
dropdown changed, the API stored the value, the worker logged it — and the song
still came back in English, because two things were true at once:

  * the ACE-Step payload carried no language field at all, so the service
    applied its own `en` default to a Spanish lyric sheet; and
  * the template lyric writer, asked for Spanish, logged a warning and returned
    English words anyway.

Neither failure raised anything, and neither was visible in a finished job. So
these tests assert on the *payload the provider builds* and on *refusals*,
rather than on a job merely completing — a job completing is what it did all
along.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import collect, needs_ffmpeg
from tests.test_music import FakeProvider, music_job
from worker.adapters.base import AdapterError
from worker.adapters.music import MusicAdapter
from worker.music import (
    LyricBrief,
    MusicRequest,
    ProviderGenerationError,
    TemplateLyricsWriter,
    UnknownLanguage,
    UnsupportedLyricLanguage,
    offered,
    plan_song,
    resolve_language,
)
from worker.music.acestep import _VOCAL_LANGUAGES, AceStepProvider

# ── The catalogue ────────────────────────────────────────────────────────


def test_a_display_name_resolves_to_its_canonical_code() -> None:
    """What the UI sends today is a display name, and it must land as a code.

    "Spanish" reaching the model unchanged is not a milder version of this bug,
    it is the same one: the service does not validate the field, so an
    unrecognised value is interpolated into the lyric prompt and quietly
    ignored rather than rejected.
    """
    resolved = resolve_language("Spanish")
    assert resolved is not None
    assert resolved.code == "es"


def test_resolution_accepts_the_code_as_well_as_the_name() -> None:
    """Job history stores whatever representation was current when it ran, so
    "Reuse settings" on an old job must not become a different song."""
    assert resolve_language("es") == resolve_language("Spanish")


@pytest.mark.parametrize("value", ["", "   ", None])
def test_no_selection_is_not_a_selection_of_english(value: str | None) -> None:
    """None means "nobody said", which the provider treats differently from a
    deliberate choice of English."""
    assert resolve_language(value) is None


def test_an_unrecognised_language_is_refused_rather_than_defaulted() -> None:
    """A closed dropdown cannot produce this, so it is a client bug — and a
    client bug that silently sang English is what this change is about."""
    with pytest.raises(UnknownLanguage):
        resolve_language("Klingon")


def test_every_offered_language_is_one_the_model_can_actually_sing() -> None:
    """The product's offer and the model's repertoire are allowed to differ,
    but a language in the dropdown that the model cannot sing is a guaranteed
    failure at the last step. Read from ACE-Step 1.5 (commit 6d467e4)
    `acestep/constants.py:VALID_LANGUAGES`."""
    for language in offered():
        assert language.code in _VOCAL_LANGUAGES, language.name


def test_the_catalogue_matches_what_the_web_app_offers() -> None:
    """Pinned because the two lists live in different apps and drift silently.

    A language in the dropdown but missing here is a job that fails; one here
    but missing there is merely unreachable. The first is customer-visible, so
    the lists are asserted equal rather than one-way.
    """
    web = Path(__file__).resolve().parents[2] / "web/src/features/generation/form.ts"
    listed = web.read_text(encoding="utf-8").split("export const LYRIC_LANGUAGES = [", 1)
    names = [
        line.strip().strip(",").strip('"')
        for line in listed[1].split("]", 1)[0].splitlines()
        if '"' in line
    ]
    assert names == [language.name for language in offered()]


# ── The provider payload ─────────────────────────────────────────────────


def payload(**overrides) -> dict:
    defaults = dict(prompt="a pop song", duration_seconds=60.0)
    request = MusicRequest(**{**defaults, **overrides})
    return AceStepProvider(base_url="http://music.test").build_payload(request)


def test_the_chosen_language_reaches_the_service_as_its_own_field() -> None:
    """The regression. `vocal_language` was absent from every payload this
    provider had ever built, so the service applied `en` to everything."""
    assert payload(lyrics="[verse]\nCorazón", language="es")["vocal_language"] == "es"


def test_the_lyrics_and_the_language_stay_separate_fields() -> None:
    """The words are not a hint about the language and the language is not a
    prefix on the words — the service takes two inputs, and conflating them
    would put a language tag into the sung text."""
    built = payload(lyrics="[verse]\nCorazón de verano", language="es")
    assert built["lyrics"] == "[verse]\nCorazón de verano"


def test_no_chosen_language_sends_no_field_at_all() -> None:
    """Absence stays absent rather than becoming an explicit "en": the
    provider's job is to carry a decision, not to invent one."""
    assert "vocal_language" not in payload(lyrics="[verse]\nsomething")


def test_an_instrumental_carries_no_vocal_language() -> None:
    """There are no vocals for it to describe. Sending one would be harmless
    and meaningless, and meaningless fields are how the next reader learns the
    wrong thing about the contract."""
    built = payload(lyrics=None, language="es")
    assert built["lyrics"] == ""
    assert "vocal_language" not in built


def test_a_language_the_model_cannot_sing_fails_instead_of_singing_english() -> None:
    """The service would accept an unknown code without complaint — it does not
    validate the field — and sing English. Refusing is the only way that
    failure is ever visible."""
    with pytest.raises(ProviderGenerationError):
        payload(lyrics="[verse]\nwords", language="xx")


# ── The lyric writer ─────────────────────────────────────────────────────


async def test_the_template_writer_refuses_a_language_it_cannot_write() -> None:
    """It used to log a warning and return English. Nobody sees a log line; the
    customer sees a song in the wrong language."""
    brief = LyricBrief(topic="a summer song", genre="pop", language="es")
    with pytest.raises(UnsupportedLyricLanguage):
        await TemplateLyricsWriter().write(brief, plan_song(120.0, genre="pop"))


async def test_the_template_writer_still_writes_english() -> None:
    """The guard on the refusal above: it must refuse Spanish, not everything."""
    brief = LyricBrief(topic="a summer song", genre="pop", language="en")
    assert (
        await TemplateLyricsWriter().write(brief, plan_song(120.0, genre="pop"))
    ).strip()


# ── The adapter, end to end through a fake provider ──────────────────────


@needs_ffmpeg
async def test_the_selection_reaches_the_provider_with_the_users_own_lyrics(
    workspace: Path,
) -> None:
    """Custom lyrics is the path that fully works in every offered language:
    the words are the customer's, untouched, and the selection tells the model
    how to pronounce them."""
    supplied = "[Verse 1]\nCorazón de verano\nbajo el sol temprano"
    provider = FakeProvider()
    job = music_job(
        workspace,
        "1m",
        parameters={"duration": "1m", "lyrics": supplied, "lyrics_language": "Spanish"},
    )
    await collect(job, MusicAdapter(provider))

    assert provider.requests[0].language == "es"
    assert provider.requests[0].lyrics == supplied


@needs_ffmpeg
async def test_english_still_reaches_the_provider_as_english(workspace: Path) -> None:
    """The regression that matters most: nothing about English changes."""
    provider = FakeProvider()
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "English"}
    )
    await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))

    assert provider.requests[0].language == "en"
    assert (provider.requests[0].lyrics or "").strip()


@needs_ffmpeg
async def test_automatic_lyrics_in_a_language_we_cannot_write_is_refused(
    workspace: Path,
) -> None:
    """No silent English. The customer asked for Spanish words and the writer
    has an English phrasebook, so the job stops and says so rather than
    shipping English lyrics labelled Spanish."""
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "Spanish"}
    )
    with pytest.raises(AdapterError) as caught:
        await collect(job, MusicAdapter(FakeProvider(), writer=TemplateLyricsWriter()))

    assert caught.value.retriable is False


@needs_ffmpeg
async def test_an_instrumental_ignores_the_chosen_language(workspace: Path) -> None:
    """A wordless genre has nothing to sing, so a language selection must
    neither reach the model nor fail the job."""
    provider = FakeProvider()
    job = music_job(
        workspace,
        "1m",
        prompt="an ambient instrumental piece",
        parameters={"duration": "1m", "lyrics_language": "Spanish"},
    )
    await collect(job, MusicAdapter(provider, writer=TemplateLyricsWriter()))

    assert provider.requests[0].instrumental is True


@needs_ffmpeg
async def test_a_language_the_platform_does_not_offer_stops_the_job(
    workspace: Path,
) -> None:
    """Arrives only from a client bug, and fails naming the value rather than
    quietly generating the wrong song."""
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "Elvish"}
    )
    with pytest.raises(AdapterError) as caught:
        await collect(job, MusicAdapter(FakeProvider()))

    assert "Elvish" in (caught.value.internal_detail or "")
