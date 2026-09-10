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


async def test_video_to_video_quality_now_picks_the_delivered_size(
    client: AsyncClient,
) -> None:
    """This control has meant two different things, and the second is why the
    wording matters.

    From 28 Aug 2026 it chose an ENGINE. Video to Video had been routed to one
    that replaces a person from a reference photo and has no plain-restyle
    behaviour, so the tool's headline promise — restyle footage from a prompt —
    was refused before it started, and Fast existed to accept that job again.

    From 10 Sep 2026 the client gated identity on whether a photo was
    attached, which is what Fast was really for, so the levels no longer
    differed by any work a customer could name. The control now picks the
    delivered SIZE. What it does not pick is how much detail is generated:
    every level renders the same proxy and resizes once, which the worker
    suite pins in `test_every_quality_level_renders_the_same_picture`.
    """
    workflow = (await client.get("/api/v1/workflows/video-to-video")).json()
    assert workflow["supported_quality_levels"] == ["1080p", "4k", "8k"]
    assert workflow["settings"]["quality"] is True
    # Duration stays source-derived: the control picks a frame, not a length.
    assert workflow["duration_mode"] == "source"
    assert workflow["supported_durations"] == []
    assert workflow["supported_durations_by_quality"] == {}
    # The copy leads with what an empty slot does, because that is now the
    # default path rather than a degraded one.
    roles = {item["role"]: item for item in workflow["inputs"]}
    assert roles["reference_image"]["required"] is False
    help_text = roles["reference_image"]["help"].lower()
    assert "empty" in help_text
    assert "on best" not in help_text, "the retired level must not survive in the copy"
    # Engines stay private here as everywhere else.
    import json

    assert "ltx" not in json.dumps(workflow).lower()


def test_video_to_video_accepts_every_offered_size_and_refuses_a_third() -> None:
    import pytest

    from app.services.workflow_registry import ValidationFailed

    for level in ("1080p", "4k", "8k", None):
        REGISTRY.validate_request(
            workflow_id="video-to-video",
            prompt="a rain-soaked neon street",
            duration=None,
            aspect_ratio="16:9",
            quality=level,
            input_roles={"source_video"},
        )
    # The retired levels are refused rather than silently accepted: a client
    # still sending "best" is out of date, and answering it with a guess is
    # how a stale build keeps working until it suddenly does not.
    for retired in ("fast", "best", "ultra"):
        with pytest.raises(ValidationFailed):
            REGISTRY.validate_request(
                workflow_id="video-to-video",
                prompt="a rain-soaked neon street",
                duration=None,
                aspect_ratio="16:9",
                quality=retired,
                input_roles={"source_video"},
            )


def test_the_photo_stays_optional_at_every_size() -> None:
    """The tool's two paths are chosen by the upload, not by the size. A
    prompt-only 8K job and a photo-led 1080p job are both valid requests."""
    for level in ("1080p", "8k"):
        REGISTRY.validate_request(
            workflow_id="video-to-video",
            prompt="a rain-soaked neon street",
            duration=None,
            aspect_ratio="16:9",
            quality=level,
            input_roles={"source_video"},
        )
        REGISTRY.validate_request(
            workflow_id="video-to-video",
            prompt="a rain-soaked neon street",
            duration=None,
            aspect_ratio="16:9",
            quality=level,
            input_roles={"source_video", "reference_image"},
        )
