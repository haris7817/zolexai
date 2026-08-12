"""Duration modes — the M2 client requirements, pinned.

Three behaviours were asked for by name (change log CR-006..CR-009):

  * Video to Video and Music Video take their duration from the uploaded file
    automatically — the user is never offered a choice.
  * Video Extension offers exactly 5 / 10 / 15 / 30 / 60 seconds.
  * Music is chosen in minutes, not video-style second presets.

These tests pin the public catalogue, the request validation and the startup
checks that keep a definition file from contradicting its own mode. If any of
them fails, a client requirement has silently regressed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.services.workflow_registry import WorkflowRegistryError, load_registry

# ── The catalogue serves the modes the client asked for ──────────────────


async def test_extension_offers_exactly_the_five_requested_durations(
    client: AsyncClient,
) -> None:
    """CR-008, verbatim: 5 / 10 / 15 / 30 / 60 seconds."""
    workflow = (await client.get("/api/v1/workflows/extend-video")).json()
    assert workflow["duration_mode"] == "fixed"
    assert workflow["supported_durations"] == ["5s", "10s", "15s", "30s", "60s"]


@pytest.mark.parametrize("workflow_id", ["video-to-video", "music-video"])
async def test_source_workflows_offer_no_duration_choice(
    client: AsyncClient, workflow_id: str
) -> None:
    """CR-006/CR-007: duration is automatic from the uploaded file."""
    workflow = (await client.get(f"/api/v1/workflows/{workflow_id}")).json()
    assert workflow["duration_mode"] == "source"
    assert workflow["supported_durations"] == []


async def test_music_is_chosen_in_minutes(client: AsyncClient) -> None:
    """CR-009. The 5-minute ceiling is provisional pending the model benchmark;
    what is pinned is the unit and that a real range exists."""
    workflow = (await client.get("/api/v1/workflows/music")).json()
    assert workflow["duration_mode"] == "minutes"
    assert workflow["supported_durations"][0] == "1m"
    assert all(d.endswith("m") for d in workflow["supported_durations"])
    assert len(workflow["supported_durations"]) >= 3


async def test_every_workflow_declares_its_mode(client: AsyncClient) -> None:
    """The frontend branches on this field; a workflow without one would render
    a broken duration control."""
    workflows = (await client.get("/api/v1/workflows")).json()["workflows"]
    assert all(w["duration_mode"] in ("fixed", "source", "minutes") for w in workflows)


# ── Request validation per mode ──────────────────────────────────────────


async def test_a_minutes_duration_is_accepted(client: AsyncClient) -> None:
    """Music needs no uploaded input, so this exercises the whole create path."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "music",
            "prompt": "A dreamy synthwave track with a driving beat",
            "parameters": {"duration": "3m", "prompt_adherence": 75},
        },
    )
    assert response.status_code == 202, response.text


async def test_the_old_second_presets_are_gone_from_music(client: AsyncClient) -> None:
    """"30s" was valid in M1. A stale client sending it must learn the new
    values from the error rather than silently failing."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "music",
            "prompt": "A dreamy synthwave track",
            "parameters": {"duration": "30s"},
        },
    )
    assert response.status_code == 422
    problems = response.json()["error"]["details"]["fields"]
    duration_problem = next(p for p in problems if p["field"] == "duration")
    assert duration_problem["allowed"] == ["1m", "2m", "3m", "4m", "5m"]


async def test_a_missing_duration_is_rejected_where_one_is_required(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "text-to-video",
            "prompt": "A cinematic drone shot",
            "parameters": {"aspect_ratio": "16:9", "quality": "High"},
        },
    )
    assert response.status_code == 422
    problems = response.json()["error"]["details"]["fields"]
    assert any(p["field"] == "duration" for p in problems)


async def test_a_source_workflow_rejects_a_supplied_duration(client: AsyncClient) -> None:
    """The UI promises "Same as source video"; a stray duration parameter would
    quietly contradict it, so the API refuses rather than ignores."""
    response = await client.post(
        "/api/v1/generations",
        json={
            "workflow_id": "video-to-video",
            "prompt": "Make it look like a watercolour painting",
            "parameters": {"duration": "10s", "aspect_ratio": "16:9", "quality": "High"},
        },
    )
    assert response.status_code == 422
    problems = response.json()["error"]["details"]["fields"]
    duration_problem = next(p for p in problems if p["field"] == "duration")
    assert "automatic" in duration_problem["reason"].lower()


# ── Startup validation — a file cannot contradict its own mode ───────────


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")


def _definition(extra: str) -> str:
    base = textwrap.dedent("""
        id: sample
        name: Sample
        category: video
        output_type: video
        description: A sample
        supported_aspect_ratios: ["16:9"]
        ui: {icon: sparkles, thumb: "linear-gradient(140deg,#111,#222)"}
    """).strip()
    return base + "\n" + textwrap.dedent(extra).strip()


def test_source_mode_with_a_duration_list_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "sample.yaml",
        _definition("""
            duration_mode: source
            supported_durations: ["5s"]
            inputs:
              - {role: source_video, kind: video, required: true, label: V, drop_hint: a video}
        """),
    )
    with pytest.raises(WorkflowRegistryError, match="must be empty"):
        load_registry(tmp_path)


def test_source_mode_needs_a_required_media_input(tmp_path: Path) -> None:
    """The duration has to come from somewhere."""
    _write(
        tmp_path,
        "sample.yaml",
        _definition("""
            duration_mode: source
            supported_durations: []
        """),
    )
    with pytest.raises(WorkflowRegistryError, match="video or audio input"):
        load_registry(tmp_path)


def test_fixed_mode_with_no_durations_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "sample.yaml",
        _definition("""
            duration_mode: fixed
            supported_durations: []
        """),
    )
    with pytest.raises(WorkflowRegistryError, match="needs supported_durations"):
        load_registry(tmp_path)


def test_minutes_mode_rejects_second_style_entries(tmp_path: Path) -> None:
    """Mixing units in one list is exactly the drift the modes exist to stop."""
    _write(
        tmp_path,
        "sample.yaml",
        _definition("""
            duration_mode: minutes
            supported_durations: ["30s", "1m"]
        """),
    )
    with pytest.raises(WorkflowRegistryError, match="must look like"):
        load_registry(tmp_path)
