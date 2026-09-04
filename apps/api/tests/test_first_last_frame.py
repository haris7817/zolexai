"""First/Last Frame Video — the public contract of the replaced Image to Video.

Client decision, 5 Sep 2026: the client's LTX 2.5 First/Last Frame workflow
replaces both Image to Video engines. The workflow id stays `image-to-video`
(history, Extend and saved links keep working); the product name, the second
optional input and the ladder are what changed. Engines stay private.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.config import settings as app_settings
from app.services.workflow_registry import ValidationFailed, load_registry

REGISTRY = load_registry(Path(app_settings.workflow_definitions_dir))


async def test_the_catalogue_names_it_and_offers_two_stills(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/image-to-video")).json()
    assert workflow["name"] == "First/Last Frame Video"
    roles = {item["role"]: item for item in workflow["inputs"]}
    assert list(roles) == ["source_image", "last_frame"]
    assert roles["source_image"]["required"] is True
    assert roles["source_image"]["label"] == "FIRST FRAME"
    assert roles["last_frame"]["required"] is False
    assert roles["last_frame"]["kind"] == "image"
    assert "optional" in roles["last_frame"]["help"].lower()
    assert workflow["supported_quality_levels"] == []
    assert workflow["settings"]["prompt_modes"] is False
    assert workflow["settings"]["sound"] is True
    assert workflow["supported_aspect_ratios"] == ["16:9", "9:16", "1:1"]
    body = json.dumps(workflow).lower()
    for private in ("ltx", "comfy", "h3", "gguf", "lora"):
        assert private not in body


def test_the_last_frame_is_optional_and_the_first_is_not() -> None:
    REGISTRY.validate_request(
        workflow_id="image-to-video",
        prompt="a slow push in",
        duration="10s",
        aspect_ratio="16:9",
        quality=None,
        input_roles={"source_image"},
    )
    REGISTRY.validate_request(
        workflow_id="image-to-video",
        prompt="a slow push in",
        duration="10s",
        aspect_ratio="16:9",
        quality=None,
        input_roles={"source_image", "last_frame"},
    )
    with pytest.raises(ValidationFailed) as raised:
        REGISTRY.validate_request(
            workflow_id="image-to-video",
            prompt="a slow push in",
            duration="10s",
            aspect_ratio="16:9",
            quality=None,
            input_roles={"last_frame"},
        )
    [problem] = [p for p in raised.value.details["fields"] if p["field"] == "inputs"]
    assert "source_image" in problem["missing_roles"]


def test_an_unknown_input_role_is_still_refused() -> None:
    with pytest.raises(ValidationFailed):
        REGISTRY.validate_request(
            workflow_id="image-to-video",
            prompt="a slow push in",
            duration="10s",
            aspect_ratio="16:9",
            quality=None,
            input_roles={"source_image", "reference_image"},
        )
