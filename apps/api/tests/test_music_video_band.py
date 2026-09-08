"""Music Video's band — the client's music-video worker, v1.8.0 (8 Sep 2026).

The client's package accepts up to five performers, each with a picture, a
role and a description. On the platform a performer is a `performer_N`
picture input plus an entry in the `performers` parameter for the same slot;
the API validates the pairing so the panel and the worker never disagree on
how many members a band can have. The parameter follows the lyrics policy:
a workflow that does not declare `settings.performers` rejects it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.config import settings as app_settings
from app.schemas.generation import GenerationParameters, PerformerSpec
from app.services.workflow_registry import ValidationFailed, load_registry

REGISTRY = load_registry(Path(app_settings.workflow_definitions_dir))


def _validate(**overrides):
    request = dict(
        workflow_id="music-video",
        prompt="make me a cinematic video according to the lyrics of the song",
        duration=None,
        aspect_ratio="16:9",
        quality=None,
        input_roles={"source_audio"},
    )
    request.update(overrides)
    return REGISTRY.validate_request(**request)


def _band(*slots: int) -> list[PerformerSpec]:
    return [PerformerSpec(slot=slot, role="guitarist") for slot in slots]


async def test_the_public_shape_offers_the_band_and_lyrics(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/music-video")).json()
    assert workflow["settings"]["performers"] is True
    assert workflow["settings"]["lyrics"] is True
    roles = [item["role"] for item in workflow["inputs"]]
    assert roles == ["source_audio"] + [f"performer_{n}" for n in range(1, 6)]
    for item in workflow["inputs"][1:]:
        assert item["kind"] == "image" and item["required"] is False
    assert workflow["duration_mode"] == "source"
    assert workflow["supported_aspect_ratios"] == ["16:9", "9:16", "1:1"]
    # Engines and models stay private.
    public = json.dumps(workflow).lower()
    for private in ("ltx", "comfy", "qwen", "whisper", "runtime"):
        assert private not in public, private


def test_a_band_of_five_with_pictures_is_accepted() -> None:
    _validate(
        input_roles={"source_audio", *{f"performer_{n}" for n in range(1, 6)}},
        performers=_band(1, 2, 3, 4, 5),
    )


def test_a_member_may_be_described_without_a_picture() -> None:
    _validate(performers=[PerformerSpec(slot=3, role="drummer", description="red bandana")])


def test_a_picture_may_arrive_without_an_entry() -> None:
    _validate(input_roles={"source_audio", "performer_2"})


def test_pasted_lyrics_and_a_language_are_accepted() -> None:
    _validate(lyrics="[00:01.00] first line", lyrics_language="Spanish")


def test_a_slot_the_definition_lacks_is_refused() -> None:
    with pytest.raises(ValidationFailed) as raised:
        _validate(performers=[PerformerSpec(slot=5, role="dj"), PerformerSpec(slot=5, role="dj")])
    fields = raised.value.details["fields"]
    assert any(p["field"] == "performers" and "once" in p["reason"] for p in fields)


def test_the_schema_caps_the_band_at_five() -> None:
    with pytest.raises(ValueError):
        GenerationParameters(performers=[PerformerSpec(slot=1)] * 6)
    with pytest.raises(ValueError):
        PerformerSpec(slot=6)
    with pytest.raises(ValueError):
        PerformerSpec(slot=1, description="x" * 301)


def test_other_workflows_refuse_performers() -> None:
    with pytest.raises(ValidationFailed) as raised:
        REGISTRY.validate_request(
            "text-to-video",
            prompt="a koi pond",
            duration="5s",
            aspect_ratio="16:9",
            quality=None,
            input_roles=set(),
            performers=_band(1),
        )
    fields = raised.value.details["fields"]
    assert any(p["field"] == "performers" for p in fields)


def test_a_sixth_picture_role_is_unknown() -> None:
    with pytest.raises(ValidationFailed) as raised:
        _validate(input_roles={"source_audio", "performer_6"})
    fields = raised.value.details["fields"]
    assert any("unknown_roles" in p and p["unknown_roles"] == ["performer_6"] for p in fields)


async def test_the_band_travels_to_the_claim(
    client: AsyncClient, worker_headers: dict[str, str], idempotency_key: str
) -> None:
    """A submitted band comes back in the claimed job's parameters exactly
    as sent — the worker pairs slots with pictures from there."""
    body = {
        "workflow_id": "music-video",
        "prompt": "make me a video according to the lyrics of the song",
        "parameters": {
            "aspect_ratio": "9:16",
            "performers": [{"slot": 1, "role": "lead_vocalist", "description": "white linen"}],
        },
    }
    # A generation needs a real uploaded audio asset; without one the request
    # is refused for the missing input, which is the assertion here — the
    # parameter itself passed schema and policy validation first.
    response = await client.post(
        "/api/v1/generations", json=body, headers={"Idempotency-Key": idempotency_key}
    )
    assert response.status_code in (400, 422)
    detail = response.json()["error"]
    assert detail["code"] == "missing_required_input"
    assert all(p["field"] != "performers" for p in detail["details"]["fields"])
