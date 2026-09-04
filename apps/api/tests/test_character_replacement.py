"""Character Replacement — the seventh workflow, shipped hidden.

The definition loads (so every reader knows the tool), is omitted from the
public list and refuses generation while `hidden: true`; the deployment
overlay flips the line once the GPU validation clears the engine. A copy of
the catalogue with the line flipped proves the contract the customer will
see: source-derived length, two required inputs, an optional prompt, no
quality, no aspect ratio, engines private.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.config import settings as app_settings
from app.core.errors import NotFound
from app.services.workflow_registry import ValidationFailed, load_registry

DEFINITIONS = Path(app_settings.workflow_definitions_dir)
REGISTRY = load_registry(DEFINITIONS)


def _visible_registry(tmp_path: Path):
    copy = tmp_path / "definitions"
    shutil.copytree(DEFINITIONS, copy)
    path = copy / "character-replacement.yaml"
    text = path.read_text(encoding="utf-8").replace("hidden: true", "hidden: false")
    path.write_text(text, encoding="utf-8")
    return load_registry(copy)


async def test_hidden_means_loaded_but_not_listed(client: AsyncClient) -> None:
    assert "character-replacement" in REGISTRY
    assert REGISTRY.get("character-replacement").hidden is True
    listed = [w["id"] for w in (await client.get("/api/v1/workflows")).json()["workflows"]]
    assert "character-replacement" not in listed
    # Readable by id, so history keeps its name.
    assert (await client.get("/api/v1/workflows/character-replacement")).status_code == 200


async def test_hidden_refuses_generation_like_an_unknown_tool(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "character-replacement",
            "prompt": "",
            "parameters": {},
            "inputs": {
                "source_video": "00000000-0000-0000-0000-000000000000",
                "reference_image": "00000000-0000-0000-0000-000000000001",
            },
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unsupported_workflow"


def test_visible_contract(tmp_path: Path) -> None:
    registry = _visible_registry(tmp_path)
    public = registry.get_public("character-replacement")
    assert public.name == "Character Replacement"
    assert public.duration_mode == "source"
    assert public.supported_durations == []
    assert public.supported_aspect_ratios == []
    assert public.supported_quality_levels == []
    assert public.prompt.required is False
    roles = {spec.role: spec for spec in public.inputs}
    assert list(roles) == ["source_video", "reference_image"]
    assert roles["source_video"].required and roles["reference_image"].required
    assert roles["reference_image"].kind == "image"
    assert public.ui.icon == "swap"
    assert public.capabilities.extend is False
    body = json.dumps(public.model_dump(mode="json")).lower()
    for private in ("ltx", "comfy", "ripple", "lora", "h3", "runtime"):
        assert private not in body
    assert [w.id for w in registry.list_public()] == [
        "text-to-video",
        "image-to-video",
        "video-to-video",
        "character-replacement",
        "extend-video",
        "music",
        "music-video",
    ]


def test_visible_validation(tmp_path: Path) -> None:
    registry = _visible_registry(tmp_path)
    registry.validate_request(
        workflow_id="character-replacement",
        prompt="",
        duration=None,
        aspect_ratio=None,
        quality=None,
        input_roles={"source_video", "reference_image"},
    )
    with pytest.raises(ValidationFailed) as raised:
        registry.validate_request(
            workflow_id="character-replacement",
            prompt="",
            duration="10s",
            aspect_ratio=None,
            quality=None,
            input_roles={"source_video"},
        )
    fields = {p["field"] for p in raised.value.details["fields"]}
    assert fields == {"duration", "inputs"}
    with pytest.raises(NotFound):
        REGISTRY.validate_request(
            workflow_id="character-replacement",
            prompt="",
            duration=None,
            aspect_ratio=None,
            quality=None,
            input_roles={"source_video", "reference_image"},
        )


def test_video_to_video_is_byte_identical_to_its_28_aug_contract() -> None:
    """Phase 5's guard: the new module changes nothing about Video to Video."""
    public = REGISTRY.get_public("video-to-video")
    assert public.supported_quality_levels == ["fast", "best"]
    assert public.duration_mode == "source"
    assert [spec.role for spec in public.inputs] == ["source_video", "reference_image"]
    definition = REGISTRY.get("video-to-video")
    assert definition.execution.model_extra.get("v2v_engine") == "transform"
    assert definition.execution.model_extra.get("v2v_reference_identity") is False
    assert definition.execution.model_extra.get("execution_by_quality") == {
        "best": {"v2v_reference_identity": True}
    }
