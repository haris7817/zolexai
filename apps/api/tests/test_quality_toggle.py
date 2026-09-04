"""The Fast/Best toggle — where it remains, and where it left.

Video to Video keeps the two levels (client-approved 27 Aug 2026): Fast
restyles from a prompt, Best replaces the person from a reference photo —
different work, so the customer chooses. Text to Video lost the toggle on
5 Sep 2026 (client decision, final milestone): one workflow, one engine,
nothing to select — and with it the only route to the engine "Best" named.
Engines are never named in anything public.
"""

from __future__ import annotations

from pathlib import Path

from httpx import AsyncClient

from app.core.config import settings as app_settings
from app.services.workflow_registry import load_registry

REGISTRY = load_registry(Path(app_settings.workflow_definitions_dir))


def _validate(**overrides):
    request = dict(
        workflow_id="text-to-video",
        prompt="a koi pond at dawn",
        duration="5s",
        aspect_ratio="16:9",
        quality=None,
        input_roles=set(),
    )
    request.update(overrides)
    return REGISTRY.validate_request(**request)


async def test_text_to_video_has_no_toggle_and_the_final_ladder(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/text-to-video")).json()
    assert workflow["supported_quality_levels"] == []
    assert workflow["supported_durations_by_quality"] == {}
    assert workflow["settings"]["quality"] is False
    assert workflow["settings"]["prompt_modes"] is False
    assert workflow["settings"]["sound"] is True
    assert workflow["supported_durations"] == ["5s", "10s", "15s", "30s"]
    assert workflow["supported_aspect_ratios"] == ["16:9", "9:16", "1:1"]
    # Engines stay private: no runtime name anywhere in the public shape.
    import json

    assert "ltx" not in json.dumps(workflow).lower()
    assert "h3" not in json.dumps(workflow).lower()
    assert "comfy" not in json.dumps(workflow).lower()


def test_text_to_video_refuses_a_quality_level_now() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    for level in ("best", "fast", "standard"):
        with pytest.raises(ValidationFailed) as raised:
            _validate(duration="5s", quality=level)
        assert any(p["field"] == "quality" for p in raised.value.details["fields"])


def test_text_to_video_sells_exactly_the_four_lengths() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    for length in ("5s", "10s", "15s", "30s"):
        _validate(duration=length)
    for gone in ("20s", "60s"):
        with pytest.raises(ValidationFailed) as raised:
            _validate(duration=gone)
        [problem] = [p for p in raised.value.details["fields"] if p["field"] == "duration"]
        assert problem["allowed"] == ["5s", "10s", "15s", "30s"]


def test_sound_needs_the_workflow_to_declare_it() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    # text-to-video declares it: accepted.
    _validate(sound=False)
    # music does not: rejected.
    with pytest.raises(ValidationFailed) as raised:
        REGISTRY.validate_request(
            workflow_id="music",
            prompt="an upbeat song",
            duration="2m",
            aspect_ratio=None,
            quality=None,
            input_roles=set(),
            sound=False,
        )
    assert any(p["field"] == "sound" for p in raised.value.details["fields"])


async def test_video_to_video_offers_the_toggle_too(client: AsyncClient) -> None:
    """Extended to Video to Video on 28 Aug 2026, after two production jobs
    failed with "needs a reference photo of the person".

    The workflow was routed to one engine that replaces a person from a
    reference photo and has no plain-restyle behaviour, so the tool's own
    headline promise — restyling footage from a prompt — was refused before
    it started, and no setting existed that would accept the job. The two
    levels are genuinely different work here, not two speeds of the same
    work, which is why the customer chooses rather than the product guessing.
    """
    workflow = (await client.get("/api/v1/workflows/video-to-video")).json()
    assert workflow["supported_quality_levels"] == ["fast", "best"]
    assert workflow["settings"]["quality"] is True
    # Duration stays source-derived: the toggle picks an engine, not a length.
    assert workflow["duration_mode"] == "source"
    assert workflow["supported_durations"] == []
    assert workflow["supported_durations_by_quality"] == {}
    # The copy says which level uses the photo (28 Aug wording: "On Best, the
    # person in your video is replaced…").
    roles = {item["role"]: item for item in workflow["inputs"]}
    help_text = roles["reference_image"]["help"].lower()
    assert "best" in help_text
    # Engines stay private here as everywhere else.
    import json

    assert "ltx" not in json.dumps(workflow).lower()


def test_video_to_video_accepts_both_levels_and_refuses_a_third() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    for level in ("fast", "best", None):
        REGISTRY.validate_request(
            workflow_id="video-to-video",
            prompt="a rain-soaked neon street",
            duration=None,
            aspect_ratio="16:9",
            quality=level,
            input_roles={"source_video"},
        )
    with pytest.raises(ValidationFailed):
        REGISTRY.validate_request(
            workflow_id="video-to-video",
            prompt="a rain-soaked neon street",
            duration=None,
            aspect_ratio="16:9",
            quality="ultra",
            input_roles={"source_video"},
        )
