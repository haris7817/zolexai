"""Automatic lyrics: the hosted writer, the fallback chain, and the language rule.

Everything here runs against a mocked HTTP transport. That is not a compromise
— the parts most likely to break are not the model's poetry but the *policy*
around it: which failures retry and which do not, whether a wrong-language
answer can reach the music model, whether a customer's own lyrics can trigger
an API call they never asked for, and whether "Spanish is unavailable" ever
quietly becomes "here is an English song".

Those are all decidable without a language model, and a live smoke test is
precisely the thing that would paper over them — a real model usually answers
in the right language, so the guard that catches it when it does not would
never be exercised.

The lettered tests map to the acceptance list the work was specified against.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.conftest import collect, make_job, needs_ffmpeg
from tests.test_music import FakeProvider
from worker.adapters.base import AdapterError
from worker.music import (
    LyricBrief,
    LyricsWriteFailed,
    NoLyricsWriterAvailable,
    plan_song,
)
from worker.music.cerebras import CerebrasLyricsWriter
from worker.music.fallback import FallbackLyricsWriter
from worker.music.writer import TemplateLyricsWriter

# ── Sample sheets, in the languages the tests ask for ────────────────────

SPANISH = """[verse]
Bajo las luces de la noche de verano
Tu mano en la mia, el mar nos llama
[chorus]
Y bailamos hasta que salga el sol
Con el corazon en la arena dorada
Todo lo que quiero es esta cancion"""

ENGLISH = """[verse]
Under the lights of a warm summer night
Your hand in mine and the sea is calling
[chorus]
And we dance until the sun comes up
With our hearts down in the golden sand
All that I want is this song tonight"""


def cerebras(
    *,
    status: int = 200,
    content: str = SPANISH,
    body: object | None = None,
    fail: type[Exception] | None = None,
    fail_times: int = 999,
    then_content: str | None = None,
    seen: dict | None = None,
) -> httpx.MockTransport:
    """A stand-in for the Cerebras endpoint, recording what it was sent.

    `fail_times` plus `then_content` is how the "one bad attempt, then a good
    one" cases are expressed — which is the shape of every retry test here.
    """
    state = {"calls": 0}
    recorded = seen if seen is not None else {}

    def handle(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        recorded["calls"] = state["calls"]
        recorded.setdefault("requests", []).append(json.loads(request.content))
        recorded.setdefault("headers", []).append(dict(request.headers))

        if fail is not None and state["calls"] <= fail_times:
            raise fail("injected")
        if status != 200 and state["calls"] <= fail_times:
            return httpx.Response(status, text="service said no")

        text = content
        if then_content is not None and state["calls"] > 1:
            text = then_content
        if body is not None:
            return httpx.Response(200, json=body)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 120},
            },
        )

    return httpx.MockTransport(handle)


def writer(transport: httpx.MockTransport, **overrides) -> CerebrasLyricsWriter:
    defaults = dict(
        api_key="sk-test-not-a-real-key",
        model="gemma-4-31b",
        base_url="https://api.cerebras.test",
        timeout_seconds=5.0,
        max_retries=1,
        enabled=True,
    )
    return CerebrasLyricsWriter(transport=transport, **{**defaults, **overrides})


def brief_and_plan(language: str = "es", seconds: float = 120.0):
    brief = LyricBrief.from_prompt(
        "romantic latin pop about falling in love on a summer night by the ocean"
    )
    brief = type(brief)(**{**brief.__dict__, "language": language})
    return brief, plan_song(seconds, genre=brief.genre)


def music_job(workspace: Path, duration: str = "1m", **overrides):
    defaults = dict(
        workflow_id="music",
        prompt="a romantic latin pop song about a summer night by the ocean",
        parameters={"duration": duration},
        inputs=[],
        execution={"runtime": "music"},
        output_content_type="audio/mpeg",
    )
    return make_job(workspace, **{**defaults, **overrides})


# ── TEST A — Cerebras succeeds ───────────────────────────────────────────


async def test_a_spanish_request_gets_spanish_lyrics_from_cerebras() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    text = await writer(cerebras(seen=seen)).write(brief, plan)

    assert "bailamos" in text
    assert seen["calls"] == 1
    # The language reaches the model as an instruction, by name, not as a code
    # it has no reason to understand.
    system = seen["requests"][0]["messages"][0]["content"]
    assert "Spanish" in system
    assert seen["requests"][0]["model"] == "gemma-4-31b"


@needs_ffmpeg
async def test_a_generated_sheet_reaches_the_music_model_with_its_language(
    workspace: Path,
) -> None:
    """The end of the acceptance flow: blank lyrics, Spanish selected, and what
    the music model is handed is a Spanish sheet plus `es` as the sung
    language — two separate fields, neither derived from the other."""
    provider = FakeProvider()
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "Spanish"}
    )
    await collect(
        job,
        _adapter(provider, FallbackLyricsWriter([writer(cerebras()), TemplateLyricsWriter()])),
    )

    request = provider.requests[0]
    assert request.language == "es"
    assert "bailamos" in (request.lyrics or "")
    assert request.instrumental is False


# ── TEST B — rate limit ──────────────────────────────────────────────────


async def test_a_rate_limit_is_retried_once_and_then_succeeds() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    text = await writer(
        cerebras(status=429, fail_times=1, seen=seen)
    ).write(brief, plan)

    assert "bailamos" in text
    assert seen["calls"] == 2


async def test_a_persistent_rate_limit_gives_up_rather_than_hammering() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed) as caught:
        await writer(cerebras(status=429, seen=seen)).write(brief, plan)

    assert caught.value.retriable is True
    assert seen["calls"] == 2, "one retry, not an unbounded loop"


@needs_ffmpeg
async def test_a_rate_limited_english_song_still_gets_written(workspace: Path) -> None:
    """B, end to end: Cerebras is unreachable, English was asked for, and the
    local bank can write English — so the song is delivered rather than
    failed."""
    provider = FakeProvider()
    chain = FallbackLyricsWriter(
        [writer(cerebras(status=429)), TemplateLyricsWriter()]
    )
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "English"}
    )
    await collect(job, _adapter(provider, chain))

    assert (provider.requests[0].lyrics or "").strip()
    assert chain.last_writer == "template"


# ── TEST C — invalid key ─────────────────────────────────────────────────


@pytest.mark.parametrize("status", [401, 403])
async def test_an_authorization_failure_is_not_retried(status: int) -> None:
    """A revoked key will be just as revoked half a second later. Spending the
    retry on it only delays the fallback."""
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed) as caught:
        await writer(cerebras(status=status, seen=seen)).write(brief, plan)

    assert caught.value.retriable is False
    assert seen["calls"] == 1


async def test_a_missing_key_skips_cerebras_without_calling_it() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("en")
    chain = FallbackLyricsWriter(
        [writer(cerebras(seen=seen), api_key=""), TemplateLyricsWriter()]
    )
    text = await chain.write(brief, plan)

    assert text.strip()
    assert chain.last_writer == "template"
    assert seen.get("calls") is None, "an unconfigured writer must not make a request"


async def test_the_feature_switch_turns_the_hosted_writer_off() -> None:
    seen: dict = {}
    disabled = writer(cerebras(seen=seen), enabled=False)

    assert disabled.available is False
    assert "ENABLED" in disabled.unavailable_reason()
    assert seen.get("calls") is None


# ── TEST D — timeout ─────────────────────────────────────────────────────


async def test_a_timeout_is_transient_and_bounded() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed) as caught:
        await writer(
            cerebras(fail=httpx.ReadTimeout, seen=seen)
        ).write(brief, plan)

    assert caught.value.retriable is True
    assert seen["calls"] == 2


async def test_a_network_error_does_not_escape_as_an_httpx_exception() -> None:
    """The job runner classifies `LyricsWriteFailed`. A raw `httpx.ConnectError`
    reaching it would be an unhandled crash rather than a fallback."""
    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed):
        await writer(cerebras(fail=httpx.ConnectError)).write(brief, plan)


# ── TEST E — empty and malformed responses ───────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"message": {"role": "assistant", "content": ""}}]},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"error": {"message": "something went wrong"}},
        {},
    ],
    ids=["empty-string", "no-choices", "no-content", "error-object", "empty-object"],
)
async def test_an_unusable_response_shape_fails_cleanly(body: dict) -> None:
    """Every one of these used to be a `KeyError` waiting to happen. None of
    them may reach the job runner as anything but a handled failure."""
    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed):
        await writer(cerebras(body=body)).write(brief, plan)


async def test_a_body_that_is_not_json_fails_cleanly() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed):
        await writer(httpx.MockTransport(handle)).write(brief, plan)


# ── TEST F — wrong language ──────────────────────────────────────────────


async def test_english_returned_for_a_spanish_request_is_retried_not_accepted() -> None:
    """The single most important test in this file.

    An English sheet is well-formed, rhymes, fits the budget and passes every
    other check in the pipeline. Nothing downstream can tell it is wrong, so if
    it is not caught here it is sung.
    """
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    text = await writer(
        cerebras(content=ENGLISH, then_content=SPANISH, seen=seen)
    ).write(brief, plan)

    assert "bailamos" in text
    assert seen["calls"] == 2
    # The second attempt says so in terms the first one cannot have missed.
    assert "CRITICAL CORRECTION" in seen["requests"][1]["messages"][0]["content"]


async def test_lyrics_that_stay_in_the_wrong_language_are_refused() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed) as caught:
        await writer(cerebras(content=ENGLISH, seen=seen)).write(brief, plan)

    assert "not 'es'" in str(caught.value)
    assert seen["calls"] == 2


async def test_a_wrong_language_answer_never_becomes_an_english_song() -> None:
    """F composed with the fallback rule: Cerebras answers Spanish requests in
    English, the local bank only writes English, and the correct outcome is a
    clean failure — NOT the English sheet either of them could have produced."""
    brief, plan = brief_and_plan("es")
    chain = FallbackLyricsWriter(
        [writer(cerebras(content=ENGLISH)), TemplateLyricsWriter()]
    )
    with pytest.raises(NoLyricsWriterAvailable) as caught:
        await chain.write(brief, plan)

    assert "'es'" in str(caught.value)


# ── TEST G — custom lyrics ───────────────────────────────────────────────


@needs_ffmpeg
async def test_the_customers_own_lyrics_never_reach_the_lyrics_service(
    workspace: Path,
) -> None:
    seen: dict = {}
    supplied = "[verse]\nBajo las luces, tu y yo\n[chorus]\nEsta es mi cancion"
    provider = FakeProvider()
    job = music_job(
        workspace,
        "1m",
        parameters={
            "duration": "1m",
            "lyrics": supplied,
            "lyrics_language": "Spanish",
        },
    )
    await collect(
        job,
        _adapter(
            provider,
            FallbackLyricsWriter([writer(cerebras(seen=seen)), TemplateLyricsWriter()]),
        ),
    )

    assert provider.requests[0].lyrics == supplied, "not rewritten, not translated"
    assert seen.get("calls") is None, "custom lyrics must cost zero API calls"


# ── TEST H — instrumental ────────────────────────────────────────────────


@needs_ffmpeg
async def test_an_instrumental_calls_no_lyrics_provider_at_all(
    workspace: Path,
) -> None:
    seen: dict = {}
    provider = FakeProvider()
    job = music_job(
        workspace,
        "1m",
        prompt="an ambient instrumental piece, no vocals",
        parameters={"duration": "1m", "lyrics_language": "Spanish"},
    )
    await collect(
        job,
        _adapter(
            provider,
            FallbackLyricsWriter([writer(cerebras(seen=seen)), TemplateLyricsWriter()]),
        ),
    )

    assert provider.requests[0].instrumental is True
    assert seen.get("calls") is None


@needs_ffmpeg
async def test_the_explicit_instrumental_flag_also_writes_nothing(
    workspace: Path,
) -> None:
    seen: dict = {}
    provider = FakeProvider()
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "instrumental": True}
    )
    await collect(
        job,
        _adapter(
            provider,
            FallbackLyricsWriter([writer(cerebras(seen=seen)), TemplateLyricsWriter()]),
        ),
    )

    assert provider.requests[0].instrumental is True
    assert seen.get("calls") is None


# ── TEST I — both providers fail ─────────────────────────────────────────


@needs_ffmpeg
async def test_when_no_writer_can_serve_the_job_fails_rather_than_going_silent(
    workspace: Path,
) -> None:
    """An empty sheet is how the music model is told to make an INSTRUMENTAL.
    So "we could not write lyrics" must never be expressed as "no lyrics" —
    that delivers a wordless track to someone who asked for a song."""
    provider = FakeProvider()
    chain = FallbackLyricsWriter(
        [writer(cerebras(status=500)), TemplateLyricsWriter()]
    )
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "Spanish"}
    )
    with pytest.raises(AdapterError) as caught:
        await collect(job, _adapter(provider, chain))

    assert not provider.requests, "no GPU time is spent on a song with no words"
    assert "lyrics" in caught.value.user_message.lower()


@needs_ffmpeg
async def test_spanish_with_nothing_configured_is_refused_before_any_work(
    workspace: Path,
) -> None:
    """The no-API-key deployment. The chain can write English and nothing else,
    so a Spanish request stops with copy that tells the customer what to do."""
    provider = FakeProvider()
    chain = FallbackLyricsWriter(
        [writer(cerebras(), api_key=""), TemplateLyricsWriter()]
    )
    job = music_job(
        workspace, "1m", parameters={"duration": "1m", "lyrics_language": "Spanish"}
    )
    with pytest.raises(AdapterError) as caught:
        await collect(job, _adapter(provider, chain))

    assert caught.value.retriable is False
    assert "English" in caught.value.user_message
    assert not provider.requests


# ── Secret handling ──────────────────────────────────────────────────────


async def test_the_api_key_travels_in_a_header_and_nowhere_else() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    await writer(cerebras(seen=seen)).write(brief, plan)

    assert seen["headers"][0]["authorization"] == "Bearer sk-test-not-a-real-key"
    # Not in the body, which is the thing that gets logged, stored and echoed.
    assert "sk-test-not-a-real-key" not in json.dumps(seen["requests"][0])


async def test_a_service_error_body_is_bounded_before_it_reaches_a_log() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="x" * 10_000)

    brief, plan = brief_and_plan("es")
    with pytest.raises(LyricsWriteFailed) as caught:
        await writer(httpx.MockTransport(handle), max_retries=0).write(brief, plan)

    assert len(str(caught.value)) < 500


# ── Duration awareness ───────────────────────────────────────────────────


@pytest.mark.parametrize("seconds", [60.0, 120.0, 180.0, 300.0])
async def test_the_line_budget_the_model_is_given_tracks_the_songs_length(
    seconds: float,
) -> None:
    """A 30-second song and a 5-minute song must not be asked for the same
    sheet. The number comes from the plan, which is measured, rather than from
    the model being asked to work it out."""
    seen: dict = {}
    brief, plan = brief_and_plan("es", seconds)
    await writer(cerebras(seen=seen)).write(brief, plan)

    user = seen["requests"][0]["messages"][1]["content"]
    assert f"at most {plan.line_budget}" in user
    assert f"{seconds:.0f} seconds" in user


async def test_a_longer_song_is_asked_for_more_words_than_a_short_one() -> None:
    short, long = {}, {}
    brief, plan_short = brief_and_plan("es", 60.0)
    _, plan_long = brief_and_plan("es", 300.0)
    await writer(cerebras(seen=short)).write(brief, plan_short)
    await writer(cerebras(seen=long)).write(brief, plan_long)

    assert (
        long["requests"][0]["max_completion_tokens"]
        > short["requests"][0]["max_completion_tokens"]
    )
    assert plan_long.line_budget > plan_short.line_budget


# ── The brief the model actually receives ────────────────────────────────


async def test_the_structure_asked_for_is_the_structure_the_pipeline_checks() -> None:
    """The reviewer downstream blocks a sheet missing the sections the plan
    wanted. Asking for different ones would guarantee a rewrite round."""
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    await writer(cerebras(seen=seen)).write(brief, plan)

    user = seen["requests"][0]["messages"][1]["content"]
    for section in plan.sections:
        if section.carries_words:
            assert f"[{section.kind}]" in user


async def test_named_details_from_the_prompt_are_demanded_not_hoped_for() -> None:
    seen: dict = {}
    brief = LyricBrief.from_prompt("a pop song about Sara in Lahore")
    brief = type(brief)(**{**brief.__dict__, "language": "es"})
    await writer(cerebras(seen=seen)).write(brief, plan_song(120.0, genre=brief.genre))

    user = seen["requests"][0]["messages"][1]["content"]
    assert "Sara" in user and "Lahore" in user


async def test_a_revision_round_carries_the_previous_complaints() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("es")
    await writer(cerebras(seen=seen)).write(
        brief, plan, notes=["density: 20 lines will not fit a 120s song"]
    )

    user = seen["requests"][0]["messages"][1]["content"]
    assert "will not fit" in user


async def test_production_direction_is_marked_as_something_never_to_sing() -> None:
    """"Warm female vocalist, acoustic guitar" describes the recording. A
    chorus that opens with those words is the writer singing the brief back."""
    seen: dict = {}
    brief = LyricBrief.from_prompt(
        "latin pop about a summer night, warm female vocalist, acoustic guitar"
    )
    brief = type(brief)(**{**brief.__dict__, "language": "es"})
    await writer(cerebras(seen=seen)).write(brief, plan_song(120.0, genre=brief.genre))

    user = seen["requests"][0]["messages"][1]["content"]
    assert "never sing these words" in user
    assert "WRITE ABOUT:" in user


async def test_a_genre_word_is_never_demanded_as_a_lyric() -> None:
    """Regression from the first real Spanish song (GPU, 2026-08-19).

    `salient_details` finds capitalised words, so "Romantic Latin pop…" yields
    "Romantic" and "Latin" — and a brief demanding they appear produced
    "La brisa trae un aire romantic / bajo el cielo Latin del mar". Names and
    places still must survive: dropping those is the failure this whole
    must-keep mechanism exists to prevent.
    """
    seen: dict = {}
    brief = LyricBrief.from_prompt(
        "Romantic Latin pop about Maria in Lahore, acoustic guitar"
    )
    brief = type(brief)(**{**brief.__dict__, "language": "es"})
    await writer(cerebras(seen=seen)).write(brief, plan_song(120.0, genre=brief.genre))

    user = seen["requests"][0]["messages"][1]["content"]
    demanded = user.split("MUST APPEAR IN THE LYRICS:")[1].splitlines()[0]
    assert "Maria" in demanded and "Lahore" in demanded
    assert "Romantic" not in demanded and "Latin" not in demanded


async def test_a_markdown_fence_the_model_was_told_not_to_add_is_removed() -> None:
    """It is told not to and mostly does not. When it does, the fence would
    reach the music model as a lyric line."""
    fenced = "```\n" + SPANISH + "\n```"
    brief, plan = brief_and_plan("es")
    text = await writer(cerebras(content=fenced)).write(brief, plan)

    assert not text.startswith("`")
    assert "bailamos" in text


# ── The chain's ordering rule ────────────────────────────────────────────


async def test_the_hosted_writer_is_tried_before_the_local_bank() -> None:
    seen: dict = {}
    brief, plan = brief_and_plan("en")
    chain = FallbackLyricsWriter(
        [writer(cerebras(content=ENGLISH, seen=seen)), TemplateLyricsWriter()]
    )
    text = await chain.write(brief, plan)

    assert chain.last_writer == "cerebras"
    assert "golden sand" in text
    assert seen["calls"] == 1


async def test_a_chain_with_no_available_member_says_so_rather_than_claiming_any() -> None:
    """An empty `supported_languages` means "any language". A chain that can
    run nothing must not answer that question with "any"."""
    chain = FallbackLyricsWriter([writer(cerebras(), api_key="")])

    assert chain.available is False
    assert chain.supported_languages == frozenset()
    assert "CEREBRAS_API_KEY" in chain.unavailable_reason()


async def test_an_available_hosted_writer_unlocks_every_offered_language() -> None:
    chain = FallbackLyricsWriter([writer(cerebras()), TemplateLyricsWriter()])

    assert chain.available is True
    assert chain.supported_languages == frozenset(), "empty means any"


# ── Helper ───────────────────────────────────────────────────────────────


def _adapter(provider, lyrics_writer):
    from worker.adapters.music import MusicAdapter

    return MusicAdapter(provider, writer=lyrics_writer)
