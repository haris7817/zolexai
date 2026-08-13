"""The ACE-Step provider: the one file that knows which music model we run.

Everything here runs against a mocked transport rather than the real service,
which matters for two reasons. The service holds ~24 GB of weights and cannot
live in CI; and the parts most likely to break are not the model but the
*protocol* — a parameter name, a doubly-encoded field, which exception a
connection failure becomes. Those are exactly the things a mock can pin and a
live smoke test tends to paper over.

The regression this suite exists for: `/query_result` takes **`task_id_list`**,
not `task_ids`. The wrong name returns `200 OK` with an empty list, which is
indistinguishable from "still generating" — so the bug presents as every job
hanging until timeout, with no error anywhere. It cost real time to find once.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from worker.music import MusicRequest, ProviderGenerationError, ProviderUnavailable
from worker.music.acestep import AceStepProvider

AUDIO = b"ID3\x04\x00" + b"\x00" * 4096


def service(
    *,
    polls_before_ready: int = 0,
    status: int = 1,
    submit_status: int = 200,
    audio: bytes = AUDIO,
    takes: int = 1,
    seen: dict | None = None,
) -> httpx.MockTransport:
    """A stand-in for the running service, recording what it was sent."""
    state = {"polls": 0}
    recorded = seen if seen is not None else {}

    def handle(request: httpx.Request) -> httpx.Response:
        recorded.setdefault("requests", []).append(request)

        if request.url.path == "/release_task":
            recorded["submit"] = json.loads(request.content)
            if submit_status != 200:
                return httpx.Response(submit_status, text="rejected")
            return httpx.Response(200, json={"data": {"task_id": "task-1"}, "code": 200})

        if request.url.path == "/query_result":
            recorded["poll"] = json.loads(request.content)
            state["polls"] += 1
            if state["polls"] <= polls_before_ready:
                return httpx.Response(200, json={"data": [], "code": 200})
            entries = [
                {
                    "file": f"/v1/audio?path=take{n}.mp3",
                    "seed_value": "12345,67890",
                    "metas": {"bpm": 120, "duration": 60, "keyscale": "C Major"},
                }
                for n in range(takes)
            ]
            return httpx.Response(
                200,
                json={
                    "data": [
                        # `result` is a JSON document encoded as a STRING inside
                        # the JSON response — decoded twice, deliberately.
                        {"task_id": "task-1", "status": status, "result": json.dumps(entries)}
                    ],
                    "code": 200,
                },
            )

        if request.url.path == "/v1/audio":
            return httpx.Response(200, content=audio)

        return httpx.Response(404)

    return httpx.MockTransport(handle)


def provider(transport: httpx.MockTransport, **kwargs) -> AceStepProvider:
    defaults = dict(
        base_url="http://music.test",
        max_seconds=600.0,
        generation_timeout=5.0,
        poll_seconds=0.0,
        transport=transport,
    )
    return AceStepProvider(**{**defaults, **kwargs})


def request(**overrides) -> MusicRequest:
    defaults = dict(prompt="an upbeat pop song", duration_seconds=60.0)
    return MusicRequest(**{**defaults, **overrides})


# ── Payload construction (pure) ──────────────────────────────────────────


def test_the_prompt_reaches_the_service_verbatim() -> None:
    """Same rule as the video runtime: whatever the user typed is what the
    model is given — no prefixing, rewriting or 'improving'."""
    typed = 'a "sad" song about 2 brothers in Lahore — café, 50% tempo'
    payload = AceStepProvider(base_url="http://x").build_payload(request(prompt=typed))
    assert payload["caption"] == typed


def test_an_instrumental_is_an_empty_lyrics_string() -> None:
    """Verified on the GPU: empty lyrics yields no vocals, rather than the
    model inventing its own words. That equivalence is the whole mapping for
    the product's `instrumental` field, so it is pinned here."""
    payload = AceStepProvider(base_url="http://x").build_payload(request(lyrics=None))
    assert payload["lyrics"] == ""

    blank = AceStepProvider(base_url="http://x").build_payload(request(lyrics="   \n "))
    assert blank["lyrics"] == ""


def test_lyrics_keep_their_structure_tags() -> None:
    """`[Verse 1]` / `[Chorus]` are native to the service and are exactly what
    worker/music/lyrics.py emits — the two halves must stay compatible."""
    sheet = "[Verse 1]\ncity lights\n\n[Chorus]\nedge of gold\n"
    payload = AceStepProvider(base_url="http://x").build_payload(request(lyrics=sheet))
    assert payload["lyrics"] == sheet


def test_optional_musical_controls_are_only_sent_when_chosen() -> None:
    """Sending null bpm/key would override the model's own judgement with
    nothing, so absent means absent."""
    bare = AceStepProvider(base_url="http://x").build_payload(request())
    assert "bpm" not in bare and "key_scale" not in bare

    full = AceStepProvider(base_url="http://x").build_payload(
        request(bpm=128, key="A Minor")
    )
    assert full["bpm"] == 128
    assert full["key_scale"] == "A Minor"


def test_a_seed_makes_a_retry_reproduce_rather_than_surprise() -> None:
    seeded = AceStepProvider(base_url="http://x").build_payload(request(seed=42))
    assert seeded["use_random_seed"] is False
    assert seeded["seed"] == 42

    unseeded = AceStepProvider(base_url="http://x").build_payload(request())
    assert unseeded["use_random_seed"] is True
    assert "seed" not in unseeded


def test_duration_is_sent_as_the_services_own_field() -> None:
    payload = AceStepProvider(base_url="http://x").build_payload(
        request(duration_seconds=240.0)
    )
    assert payload["audio_duration"] == pytest.approx(240.0)


# ── The protocol ─────────────────────────────────────────────────────────


async def test_the_poll_uses_task_id_list_not_task_ids(workspace: Path) -> None:
    """THE regression guard.

    `task_ids` returns 200 with an empty list — identical to "still working" —
    so the wrong name makes every job hang until it times out, with nothing in
    any log to say why.
    """
    seen: dict = {}
    await provider(service(seen=seen)).generate(request(), workspace)

    assert "task_id_list" in seen["poll"]
    assert seen["poll"]["task_id_list"] == ["task-1"]
    assert "task_ids" not in seen["poll"]


async def test_a_finished_task_yields_a_downloaded_take(workspace: Path) -> None:
    takes = await provider(service()).generate(request(), workspace)

    assert len(takes) == 1
    assert takes[0].path.exists()
    assert takes[0].path.read_bytes() == AUDIO
    assert takes[0].path.parent.parent == workspace, "must write inside the job workspace"
    # Metadata is carried for diagnostics but never surfaces publicly.
    assert takes[0].seed == 12345
    assert takes[0].metadata["keyscale"] == "C Major"


async def test_it_polls_until_the_result_appears(workspace: Path) -> None:
    """Generation is asynchronous; an empty `data` means not finished, not
    failed."""
    transport = service(polls_before_ready=3)
    takes = await provider(transport).generate(request(), workspace)
    assert len(takes) == 1


async def test_every_take_offered_is_returned(workspace: Path) -> None:
    """The service returns two takes per request by default. The adapter picks
    one, but discarding the rest here would remove a product choice."""
    takes = await provider(service(takes=2)).generate(request(), workspace)
    assert len(takes) == 2
    assert takes[0].path != takes[1].path


# ── Failure modes ────────────────────────────────────────────────────────


async def test_an_unreachable_service_is_unavailable_not_a_generation_failure(
    workspace: Path,
) -> None:
    """The distinction the adapter turns into retriable vs not: a service that
    is not running will not be running on attempt three either."""

    def refuse(request_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderUnavailable, match="unreachable"):
        await provider(httpx.MockTransport(refuse)).generate(request(), workspace)


async def test_an_empty_base_url_is_unavailable(workspace: Path) -> None:
    with pytest.raises(ProviderUnavailable, match="ACESTEP_BASE_URL"):
        await AceStepProvider(base_url="").generate(request(), workspace)


async def test_a_rejected_request_is_a_generation_failure(workspace: Path) -> None:
    with pytest.raises(ProviderGenerationError, match="rejected the request"):
        await provider(service(submit_status=422)).generate(request(), workspace)


async def test_a_reported_failure_stops_polling(workspace: Path) -> None:
    """Better to fail now than to keep asking until the timeout expires."""
    with pytest.raises(ProviderGenerationError, match="failed"):
        await provider(service(status=-1)).generate(request(), workspace)


async def test_a_task_that_never_finishes_times_out(workspace: Path) -> None:
    transport = service(polls_before_ready=10_000)
    with pytest.raises(ProviderGenerationError, match="within"):
        await provider(transport, generation_timeout=0.05).generate(request(), workspace)


async def test_an_empty_audio_file_is_refused(workspace: Path) -> None:
    """A zero-byte file would otherwise reach assembly and fail there, pointing
    at the wrong component."""
    with pytest.raises(ProviderGenerationError, match="empty audio"):
        await provider(service(audio=b"")).generate(request(), workspace)


async def test_a_length_beyond_the_providers_ceiling_is_refused_before_submitting(
    workspace: Path,
) -> None:
    """The adapter sections anything longer; a request that slipped through is
    a platform bug and must not silently produce a short song."""
    seen: dict = {}
    with pytest.raises(ProviderGenerationError, match="ceiling"):
        await provider(service(seen=seen), max_seconds=120.0).generate(
            request(duration_seconds=300.0), workspace
        )
    assert "submit" not in seen, "nothing should have been sent"


# ── Progress ─────────────────────────────────────────────────────────────


async def test_progress_is_reported_as_a_fraction_and_never_claims_completion(
    workspace: Path,
) -> None:
    """The provider reports 0..1 and knows nothing about the job lifecycle —
    mapping that onto the customer's bar is the adapter's job. It stays below
    1.0 so the bar never shows finished a moment before something fails."""
    seen: list[float] = []

    async def on_progress(fraction: float) -> None:
        seen.append(fraction)

    await provider(service(polls_before_ready=3)).generate(
        request(), workspace, on_progress
    )

    assert seen, "a long generation must report progress"
    assert all(0.0 <= value < 1.0 for value in seen)
    assert seen == sorted(seen)
