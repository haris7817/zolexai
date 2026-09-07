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
from worker.adapters.ltx_hd import SUPPORTED, WORKFLOW_ID, LtxHdAdapter, frames_for
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


def test_the_adapter_answers_for_text_to_video_and_the_hd_id_only() -> None:
    """Text to Video itself (the client's ask, 7 Sep 2026) and the HD id kept
    for a stable name. Nothing else: Image to Video and Extend stay on
    ltx_comfy, and this graph's canvas would be wrong for them anyway."""
    adapter = get_adapter("ltx_hd")
    assert adapter.name == "ltx_hd"
    assert SUPPORTED == {"text-to-video", WORKFLOW_ID}
    for wanted in SUPPORTED:
        assert adapter.supports(wanted), wanted
    for other in ("image-to-video", "extend-video", "character-replacement",
                  "video-to-video", "music-video", "music"):
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
    # All three offered ratios are real renders at the model's stride.
    assert 'supported_aspect_ratios: ["16:9", "9:16", "1:1"]' in text
    # Ships on the mock; the deployment overlay routes it.
    assert re.search(r"^  runtime: mock$", text, re.M)
    # Hidden as a tool of its own: the graph is Text to Video now.
    assert re.search(r"^hidden: true$", text, re.M)


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


def test_a_smaller_canvas_reaches_the_latent_and_the_final_scale_stays_1080p() -> None:
    """The speed lever: generate small, let the graph's own lanczos+crop node
    deliver 1920x1080. The final ImageScale is deliberately NOT touched —
    it is what makes the output size independent of the canvas."""
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    api = compile_fast_1080(
        load_graph(CLIENT_GRAPH),
        Fast1080Edits(
            positive="p", negative=None, seconds=8, seed=1,
            filename_prefix="x", image="placeholder.png", canvas=(1280, 736),
        ),
        catalogue,
    )
    [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1280, 736)
    [scale] = [e for e in api.values() if e["class_type"] == "ImageScale"]
    assert (scale["inputs"]["width"], scale["inputs"]["height"]) == (1920, 1080)
    assert scale["inputs"]["crop"] == "center"


def test_a_canvas_off_the_32_grid_is_refused_at_compile_time() -> None:
    from worker.comfy.ltx_graphs import GraphError

    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    with pytest.raises(GraphError):
        compile_fast_1080(
            load_graph(CLIENT_GRAPH),
            Fast1080Edits(positive="p", negative=None, seconds=8, seed=1,
                          filename_prefix="x", image="placeholder.png", canvas=(1280, 720)),
            catalogue,
        )


def test_the_canvas_comes_from_the_job_then_the_deployment_then_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = LtxHdAdapter()
    assert adapter._canvas(_job(tmp_path)) is None                      # graph's own
    monkeypatch.setattr(settings, "ltx_hd_canvas", "1280x736")
    assert adapter._canvas(_job(tmp_path)) == (1280, 736)              # deployment
    job = _job(tmp_path)
    job.execution["canvas"] = "native"
    assert adapter._canvas(job) is None                                 # job wins


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


def test_the_clients_negative_is_kept_unless_a_deployment_overrides_it(tmp_path: Path) -> None:
    """None means the graph's own negative survives. Until 7 Sep the adapter
    silently swapped in the first/last-frame negative — `negative_for` had
    no entry for this workflow — which contradicted the parity claim for
    that one widget."""
    adapter = LtxHdAdapter()
    assert adapter._negative(_job(tmp_path)) is None
    overridden = _job(tmp_path)
    overridden.execution["negative_prompt"] = "  blurry, low quality  "
    assert adapter._negative(overridden) == "blurry, low quality"
    overridden.execution["negative_prompt"] = "   "
    assert adapter._negative(overridden) is None


def test_the_offered_ratios_are_accepted_and_anything_else_is_refused(tmp_path: Path) -> None:
    """Portrait and square became real renders on 8 Sep 2026 (client ask).
    A ratio with no size mapping still refuses before any GPU time, because
    returning a shape the customer did not choose is the thing to avoid."""
    adapter = LtxHdAdapter()
    assert adapter._aspect(_job(tmp_path)) == "16:9"               # absent → 16:9
    for ratio in ("16:9", "9:16", "1:1"):
        assert adapter._aspect(_job(tmp_path, aspect_ratio=ratio)) == ratio
    with pytest.raises(AdapterError) as caught:
        adapter._aspect(_job(tmp_path, aspect_ratio="21:9"))
    assert caught.value.retriable is False


def test_portrait_turns_the_graphs_own_two_size_widgets_on_their_side() -> None:
    """The whole of portrait support: generate 1088x1920, deliver 1080x1920.
    The 8 spare pixels come off the width, mirroring what the graph already
    does to the height at 16:9."""
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    for ratio, canvas, delivery in (
        ("9:16", (1088, 1920), (1080, 1920)),
        ("1:1", (1088, 1088), (1080, 1080)),
    ):
        api = compile_fast_1080(
            load_graph(CLIENT_GRAPH),
            Fast1080Edits(
                positive="p", negative=None, seconds=8, seed=1,
                filename_prefix="x", image="placeholder.png",
                canvas=canvas, delivery=delivery,
            ),
            catalogue,
        )
        [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
        assert (latent["inputs"]["width"], latent["inputs"]["height"]) == canvas, ratio
        [scale] = [e for e in api.values() if e["class_type"] == "ImageScale"]
        assert (scale["inputs"]["width"], scale["inputs"]["height"]) == delivery, ratio
        assert scale["inputs"]["crop"] == "center", ratio


def test_landscape_still_leaves_both_size_widgets_exactly_as_delivered() -> None:
    """16:9 maps to (None, None), so every speed and quality measurement
    taken before portrait existed still describes what landscape renders."""
    from worker.adapters.ltx_hd import ASPECTS

    assert ASPECTS["16:9"] == (None, None)
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    api = compile_fast_1080(
        load_graph(CLIENT_GRAPH),
        Fast1080Edits(positive="p", negative=None, seconds=8, seed=1,
                      filename_prefix="x", image="placeholder.png"),
        catalogue,
    )
    [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1920, 1088)
    [scale] = [e for e in api.values() if e["class_type"] == "ImageScale"]
    assert (scale["inputs"]["width"], scale["inputs"]["height"]) == (1920, 1080)


def test_the_720p_keyword_gives_each_ratio_its_own_canvas_and_a_1080p_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client's speed plan (8 Sep 2026): generate at the LTX 720p size,
    let the graph's own closing node upscale to 1080p. Square gets a square
    canvas rather than a transposed landscape one — otherwise a 1:1 job
    would generate 16:9 and have its sides cropped off."""
    from worker.adapters.ltx_hd import ASPECTS, DRAFT_CANVAS

    adapter = LtxHdAdapter()
    monkeypatch.setattr(settings, "ltx_hd_canvas", "720p")
    assert adapter._canvas(_job(tmp_path), "16:9") == (1280, 704)
    assert adapter._canvas(_job(tmp_path), "9:16") == (704, 1280)
    assert adapter._canvas(_job(tmp_path), "1:1") == (960, 960)
    # every generation side stays on the model's 32 grid
    for ratio, (width, height) in DRAFT_CANVAS.items():
        assert width % 32 == 0 and height % 32 == 0, ratio
    # and the delivered size is still 1080p, which is what the customer gets
    assert ASPECTS["16:9"][1] is None          # the graph's own 1920x1080
    assert ASPECTS["9:16"][1] == (1080, 1920)


def test_the_720p_canvas_reaches_the_latent_while_the_delivery_stays_1080p() -> None:
    """Compiled proof for landscape: generate 1280x704, deliver 1920x1080
    through the graph's own lanczos centre-crop. The soundtrack never passes
    through that node, so the upscale cannot touch it."""
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    api = compile_fast_1080(
        load_graph(CLIENT_GRAPH),
        Fast1080Edits(positive="p", negative=None, seconds=8, seed=1,
                      filename_prefix="x", image="placeholder.png",
                      canvas=(1280, 704), delivery=None),
        catalogue,
    )
    [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1280, 704)
    [scale] = [e for e in api.values() if e["class_type"] == "ImageScale"]
    assert (scale["inputs"]["width"], scale["inputs"]["height"]) == (1920, 1080)
    assert scale["inputs"]["upscale_method"] == "lanczos"
    # the audio branch reaches CreateVideo without meeting ImageScale
    [video] = [e for e in api.values() if e["class_type"] == "CreateVideo"]
    assert video["inputs"]["audio"][0] != scale["inputs"]["image"][0]


def test_a_landscape_speed_canvas_is_transposed_for_a_portrait_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment sets one canvas string for every job. It is a size lever,
    not an orientation choice, so it follows the ratio instead of handing the
    graph a landscape frame to crop down its sides."""
    adapter = LtxHdAdapter()
    monkeypatch.setattr(settings, "ltx_hd_canvas", "1280x736")
    assert adapter._canvas(_job(tmp_path), "16:9") == (1280, 736)
    assert adapter._canvas(_job(tmp_path), "9:16") == (736, 1280)


def test_a_customer_seed_wins_and_is_otherwise_stable_per_job(tmp_path: Path) -> None:
    adapter = LtxHdAdapter()
    assert adapter._seed(_job(tmp_path, seed=1234)) == 1234
    first = adapter._seed(_job(tmp_path))
    assert first == adapter._seed(_job(tmp_path))
