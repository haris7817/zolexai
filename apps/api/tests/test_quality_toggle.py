"""The Fast/Best toggle — client-approved 27 Aug 2026, pinned.

Two engines behind one control: Fast is the speed engine with the full
duration ladder; Best is the quality engine, which sells 5-30s (its lattice
has a 60s the product does not offer) and generates native audio, hence the
sound on/off choice that only exists there. Engines are never named in
anything public — the runtime mapping is the deployment's execution block.
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


async def test_the_catalogue_serves_the_toggle(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/text-to-video")).json()
    assert workflow["supported_quality_levels"] == ["fast", "best"]
    assert workflow["supported_durations_by_quality"] == {
        "best": ["5s", "10s", "15s", "20s", "30s"]
    }
    assert workflow["settings"]["quality"] is True
    assert workflow["settings"]["sound"] is True
    # Engines stay private: no runtime name anywhere in the public shape.
    import json

    assert "ltx" not in json.dumps(workflow).lower()
    assert "h3" not in json.dumps(workflow).lower()


def test_absent_quality_is_the_default_and_offers_the_full_ladder() -> None:
    # The absence-is-default contract: a client from before the toggle
    # existed keeps its behaviour, 60s included.
    _validate(duration="60s", quality=None)


def test_best_does_not_sell_sixty_seconds() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    with pytest.raises(ValidationFailed) as raised:
        _validate(duration="60s", quality="best")
    [problem] = [
        p for p in raised.value.details["fields"] if p["field"] == "duration"
    ]
    assert "60s" not in problem["allowed"]
    assert "30s" in problem["allowed"]


def test_fast_offers_the_full_ladder() -> None:
    _validate(duration="60s", quality="fast")


def test_sound_needs_the_workflow_to_declare_it() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    # text-to-video declares it: accepted.
    _validate(quality="best", sound=False)
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
    # The copy has to say which level needs the photo, because one does and
    # the other does not.
    roles = {item["role"]: item for item in workflow["inputs"]}
    help_text = roles["reference_image"]["help"].lower()
    assert "fast" in help_text and "best" in help_text
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
