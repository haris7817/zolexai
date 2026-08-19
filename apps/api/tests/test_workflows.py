"""Workflow registry and catalogue tests.

The load-bearing one is `test_execution_block_never_reaches_a_client`: it is the
regression guard for the rule that no provider, model or runtime detail may be
visible to a browser (directive §11, §12). If someone later swaps the explicit
projection in `to_public()` for a `model_dump(exclude=...)`, this fails.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.services.workflow_registry import WorkflowRegistryError, load_registry

EXPECTED_IDS = [
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "extend-video",
    "music",
    "music-video",
]


async def test_lists_every_workflow_in_display_order(client: AsyncClient) -> None:
    response = await client.get("/api/v1/workflows")
    assert response.status_code == 200
    assert [w["id"] for w in response.json()["workflows"]] == EXPECTED_IDS


async def test_execution_block_never_reaches_a_client(client: AsyncClient) -> None:
    """No private execution detail may appear in any public response."""
    body = (await client.get("/api/v1/workflows")).text.lower()

    for forbidden in ("execution", "runtime", "output_content_type", "mock"):
        assert forbidden not in body, f"'{forbidden}' leaked into the public catalogue"

    # M2 added tuning to the private block — conditioning strengths, cut
    # alignment, per-workflow wall-clock budgets, pass ceilings. Those describe
    # how a model is driven, which makes them exactly as private as `runtime`,
    # and the allowlist projection is what has to keep them out.
    for tuning in ("timeout_seconds", "v2v_", "align_cuts", "max_segment", "keyframes"):
        assert tuning not in body, f"'{tuning}' leaked into the public catalogue"

    # And the same for a single-workflow response.
    detail = (await client.get("/api/v1/workflows/text-to-video")).text.lower()
    assert "execution" not in detail
    assert "runtime" not in detail


async def test_no_provider_or_infrastructure_names_anywhere(client: AsyncClient) -> None:
    """Guards the customer-facing vocabulary (directive §12)."""
    body = (await client.get("/api/v1/workflows")).text.lower()
    for name in ("ltx", "comfyui", "comfy", "vast.ai", "vastai", "pytorch", "cuda", "gpu"):
        assert name not in body, f"internal name '{name}' is visible to clients"


async def test_music_declares_no_frame_and_no_extend(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/music")).json()
    assert workflow["output_type"] == "audio"
    assert workflow["supported_aspect_ratios"] == []
    assert workflow["supported_quality_levels"] == []
    assert workflow["capabilities"]["extend"] is False


async def test_video_to_video_has_an_optional_reference_image(client: AsyncClient) -> None:
    """The reference image drives person identity (live since 19 Aug 2026).

    M1 pinned the OPPOSITE promise here — the help text was forbidden from
    mentioning identity while the input was only a look hint. Now that
    `v2v_reference_identity` ships, the copy must say who the person will be
    and set the single-person expectation, because a multi-person source has
    everyone re-imagined (measured on the GPU, 19 Aug 2026).
    """
    workflow = (await client.get("/api/v1/workflows/video-to-video")).json()
    roles = {item["role"]: item for item in workflow["inputs"]}

    assert roles["source_video"]["required"] is True
    assert roles["reference_image"]["required"] is False
    assert roles["reference_image"]["kind"] == "image"
    help_text = roles["reference_image"]["help"]
    assert "person" in help_text.lower()
    assert "one person" in help_text.lower()


async def test_unknown_workflow_is_a_clean_404(client: AsyncClient) -> None:
    response = await client.get("/api/v1/workflows/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unsupported_workflow"


# ── Startup validation ───────────────────────────────────────────────────


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")


VALID = """
    id: sample
    name: Sample
    category: video
    output_type: video
    description: A sample
    supported_durations: ["5s"]
    supported_aspect_ratios: ["16:9"]
    supported_quality_levels: [High]
    settings: {quality: true}
    ui: {icon: sparkles, thumb: "linear-gradient(140deg,#111,#222)"}
"""


def test_video_to_video_ships_the_transform_engine() -> None:
    """The SHIPPED definition selects the structure-conditioned engine.

    Everything else about the transform engine is proven in the worker suite,
    but the worker's tests build their own execution blocks — so none of them
    would notice if this key were dropped from the YAML. The customer-visible
    effect of dropping it is silent: video-to-video keeps working and quietly
    goes back to the weak restyle the client complained about, which is exactly
    the kind of regression nothing else here would catch.
    """
    registry = load_registry(Path(__file__).resolve().parents[3] / "workflow-definitions")
    execution = registry.get("video-to-video").execution

    assert getattr(execution, "v2v_engine", None) == "transform"


def test_valid_definition_loads(tmp_path: Path) -> None:
    _write(tmp_path, "sample.yaml", VALID)
    registry = load_registry(tmp_path)
    assert registry.ids() == ["sample"]


def test_filename_must_match_id(tmp_path: Path) -> None:
    _write(tmp_path, "wrong-name.yaml", VALID)
    with pytest.raises(WorkflowRegistryError, match="does not match the filename"):
        load_registry(tmp_path)


def test_audio_cannot_declare_aspect_ratios(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "sample.yaml",
        VALID.replace("output_type: video", "output_type: audio"),
    )
    with pytest.raises(WorkflowRegistryError, match="aspect ratios"):
        load_registry(tmp_path)


def test_quality_control_without_levels_is_rejected(tmp_path: Path) -> None:
    """A control with nothing to choose from would render an empty widget."""
    _write(
        tmp_path,
        "sample.yaml",
        VALID.replace("supported_quality_levels: [High]", "supported_quality_levels: []"),
    )
    with pytest.raises(WorkflowRegistryError, match="supported_quality_levels"):
        load_registry(tmp_path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    """A typo must fail the boot, not be silently ignored."""
    _write(tmp_path, "sample.yaml", VALID + "\nsuported_durations: ['5s']")
    with pytest.raises(WorkflowRegistryError):
        load_registry(tmp_path)


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(WorkflowRegistryError, match="No workflow definitions"):
        load_registry(tmp_path)
