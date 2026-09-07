"""Text to Video HD — the client's FAST 1080 graph as a product tool.

The tool is an addition: its own adapter, its own workflow definition, and it
shares only the ComfyUI service object with the tools that were already
there. These pin that isolation, because the whole reason for a fourth
adapter rather than a branch inside `ltx_comfy` is that nothing it does can
reach Text to Video, Image to Video or Extend Video.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from worker.adapters.base import AdapterError, AdapterJob
from worker.adapters.ltx_hd import WORKFLOW_ID, LtxHdAdapter, frames_for
from worker.adapters.registry import get_adapter
from worker.comfy.ltx_graphs import Fast1080Edits, compile_fast_1080, load_graph
from worker.core.config import settings
from worker.providers.ltx_comfy import GRAPH_FILES, GRAPH_SHA256

REPO = Path(__file__).resolve().parents[3]
DEFINITION = REPO / "workflow-definitions/text-to-video-hd.yaml"
CLIENT_GRAPH = REPO / "benchmarks/client-pack/ltx25" / GRAPH_FILES["fast_1080"]


def _job(workspace: Path, **params) -> AdapterJob:
    return AdapterJob(
        job_id="hd-job",
        workflow_id=WORKFLOW_ID,
        workflow_version="1",
        prompt="a photorealistic skydiving sequence",
        parameters={"duration": "8s", **params},
        execution={"runtime": "ltx_hd"},
        workspace=workspace,
    )


# ── The frame lattice ──────────────────────────────────────────────────────


def test_the_frame_count_is_the_graphs_own_arithmetic() -> None:
    """`1 + floor(fps*seconds/8)*8`, reproduced so a length can be checked
    before any GPU time is spent. Every offered length lands on 8k+1."""
    assert frames_for(8) == 193
    assert frames_for(15) == 361
    assert frames_for(30) == 721
    for seconds in (5, 8, 10, 15, 30):
        assert (frames_for(seconds) - 1) % 8 == 0, seconds


# ── Isolation from the tools that were already there ───────────────────────


def test_the_adapter_answers_for_its_own_workflow_and_no_other() -> None:
    adapter = get_adapter("ltx_hd")
    assert adapter.name == "ltx_hd"
    assert adapter.supports(WORKFLOW_ID)
    for other in ("text-to-video", "image-to-video", "extend-video",
                  "character-replacement", "video-to-video", "music-video"):
        assert not adapter.supports(other), other


def test_the_existing_runtimes_do_not_answer_for_this_workflow() -> None:
    for runtime in ("ltx_comfy", "character_replacement", "ltx"):
        assert not get_adapter(runtime).supports(WORKFLOW_ID), runtime


def test_the_client_graph_is_registered_by_its_delivered_hash() -> None:
    """A fourth graph in the pack's manifest, so drift is caught the same way
    the other three are."""
    assert GRAPH_FILES["fast_1080"].startswith("client_original/")
    import hashlib

    assert hashlib.sha256(CLIENT_GRAPH.read_bytes()).hexdigest() == GRAPH_SHA256["fast_1080"]


# ── The workflow definition ────────────────────────────────────────────────


def test_the_definition_offers_only_the_lengths_that_were_benchmarked() -> None:
    text = DEFINITION.read_text(encoding="utf-8")
    assert f"id: {WORKFLOW_ID}" in text
    assert 'supported_durations: ["8s", "15s"]' in text
    # 30 s renders but is deliberately not offered; the ceiling agrees.
    assert settings.ltx_hd_max_seconds == 15.0
    # The graph pins its canvas, so one ratio.
    assert 'supported_aspect_ratios: ["16:9"]' in text
    # Ships on the mock; the deployment overlay routes it.
    assert re.search(r"^  runtime: mock$", text, re.M)


def test_the_definition_does_not_promise_extend() -> None:
    """Extend continues a clip on the First/Last Frame graph, a different
    canvas. Claiming it here would offer a path nothing implements."""
    assert "extend: false" in DEFINITION.read_text(encoding="utf-8")


# ── What a job may set ─────────────────────────────────────────────────────


def test_a_job_sets_its_own_inputs_and_keeps_the_clients_negative() -> None:
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    api = compile_fast_1080(
        load_graph(CLIENT_GRAPH),
        Fast1080Edits(
            positive="a job's prompt", negative=None, seconds=15, seed=99,
            filename_prefix="zolexai/job/output", image="placeholder.png",
        ),
        catalogue,
    )
    texts = [
        e["inputs"]["value"]
        for e in api.values()
        if e["class_type"] == "PrimitiveStringMultiline"
    ]
    assert "a job's prompt" in texts
    # The client wrote a long negative into the graph; unset keeps it.
    assert any("low resolution" in t for t in texts if isinstance(t, str))
    [noise] = [e for e in api.values() if e["class_type"] == "RandomNoise"]
    assert noise["inputs"]["noise_seed"] == 99
    duration = [
        e["inputs"]["value"]
        for e in api.values()
        if e["class_type"] == "PrimitiveFloat"
        and "determines frames" in e.get("_meta", {}).get("title", "")
    ]
    assert duration == [15.0]


def test_a_length_above_the_ceiling_is_refused_before_any_gpu_time(tmp_path: Path) -> None:
    adapter = LtxHdAdapter()
    with pytest.raises(AdapterError) as caught:
        adapter._seconds(_job(tmp_path, duration="30s"))
    assert "up to 15 seconds" in caught.value.user_message
    assert caught.value.retriable is False


def test_a_missing_length_is_refused(tmp_path: Path) -> None:
    adapter = LtxHdAdapter()
    with pytest.raises(AdapterError):
        adapter._seconds(_job(tmp_path, duration=None))


def test_a_customer_seed_wins_and_is_otherwise_stable_per_job(tmp_path: Path) -> None:
    adapter = LtxHdAdapter()
    assert adapter._seed(_job(tmp_path, seed=1234)) == 1234
    first = adapter._seed(_job(tmp_path))
    assert first == adapter._seed(_job(tmp_path))
