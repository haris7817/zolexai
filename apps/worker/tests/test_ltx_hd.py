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


def test_the_draft_keyword_gives_each_ratio_its_own_canvas_and_a_1080p_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client's 10 Sep 2026 instruction, which is the second revision of
    this one map: 1280x704 / 704x1280 became **864x480 / 480x864**. Square
    gets a square canvas rather than a transposed landscape one — otherwise a
    1:1 job would generate 16:9 and have its sides cropped off — and it
    tracks the landscape pixel budget rather than staying where it was.

    The selecting keyword is still "720p" even though nothing here is 720p
    any more: it names `LTX_HD_CANVAS` as the live GPU worker already has it
    set, and renaming it would have quietly returned that worker to native
    1920x1088 on its next restart."""
    from worker.adapters.ltx_hd import ASPECTS, DRAFT_CANVAS

    adapter = LtxHdAdapter()
    monkeypatch.setattr(settings, "ltx_hd_canvas", "720p")
    assert adapter._canvas(_job(tmp_path), "16:9") == (864, 480)
    assert adapter._canvas(_job(tmp_path), "9:16") == (480, 864)
    assert adapter._canvas(_job(tmp_path), "1:1") == (640, 640)
    monkeypatch.setattr(settings, "ltx_hd_canvas", "draft")
    assert adapter._canvas(_job(tmp_path), "16:9") == (864, 480)
    # every generation side stays on the model's 32 grid — the one constraint
    # in this map that is the model's and not the client's
    for ratio, (width, height) in DRAFT_CANVAS.items():
        assert width % 32 == 0 and height % 32 == 0, ratio
    # the square canvas carries the landscape one's pixel budget, within the
    # rounding the 32 grid forces
    assert 0.9 <= (640 * 640) / (864 * 480) <= 1.1
    # and the delivered size is still 1080p, which is what the customer gets
    assert ASPECTS["16:9"][1] is None          # the graph's own 1920x1080
    assert ASPECTS["9:16"][1] == (1080, 1920)


def test_the_draft_canvas_reaches_the_latent_while_the_delivery_stays_1080p() -> None:
    """Compiled proof for landscape: generate 864x480, deliver 1920x1080
    through the graph's own lanczos centre-crop. The soundtrack never passes
    through that node, so the upscale cannot touch it."""
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    api = compile_fast_1080(
        load_graph(CLIENT_GRAPH),
        Fast1080Edits(positive="p", negative=None, seconds=8, seed=1,
                      filename_prefix="x", image="placeholder.png",
                      canvas=(864, 480), delivery=None),
        catalogue,
    )
    [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (864, 480)
    [scale] = [e for e in api.values() if e["class_type"] == "ImageScale"]
    assert (scale["inputs"]["width"], scale["inputs"]["height"]) == (1920, 1080)
    assert scale["inputs"]["upscale_method"] == "lanczos"
    # the audio branch reaches CreateVideo without meeting ImageScale
    [video] = [e for e in api.values() if e["class_type"] == "CreateVideo"]
    assert video["inputs"]["audio"][0] != scale["inputs"]["image"][0]


def test_4k_is_a_delivery_tier_reached_by_one_resize_from_the_generated_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Client request (8 Sep 2026): 4K "in the same way we do 1920x1080".
    The graph's closing node is parked at the canvas (an identity resize)
    and ffmpeg does one lanczos scale-to-cover + centre-crop to the exact 4K
    frame per ratio, soundtrack copied through."""
    from worker.adapters.ltx_hd import DELIVERY_4K

    adapter = LtxHdAdapter()
    assert adapter._delivery_tier(_job(tmp_path)) == "1080p"
    monkeypatch.setattr(settings, "ltx_hd_delivery", "4k")
    assert adapter._delivery_tier(_job(tmp_path)) == "4k"
    job = _job(tmp_path)
    job.execution["delivery"] = "1080p"
    assert adapter._delivery_tier(job) == "1080p"            # the job wins
    job.execution["delivery"] = "definitely-not-a-tier"
    assert adapter._delivery_tier(job) == "1080p"            # a typo cannot 4K everything
    assert DELIVERY_4K == {"16:9": (3840, 2160), "9:16": (2160, 3840), "1:1": (2160, 2160)}
    for width, height in DELIVERY_4K.values():
        assert width % 2 == 0 and height % 2 == 0


def test_8k_is_the_same_one_resize_never_4k_and_then_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Client instruction, 10 Sep 2026: "do not upscale to 4K first. Replace
    the 4K destination with 8K so it scales directly from 480p to 8K."

    So 8K is a third value of the same tier setting, reached by the same
    single ffmpeg pass — not a stage added after the 4K one. What proves the
    "directly" is that the graph's own closing node is parked at the
    generation canvas for 8K exactly as it is for 4K, so the only resize in
    the whole pipeline is the ffmpeg one."""
    from worker.adapters.ltx_hd import ASPECTS, DELIVERY_4K, DELIVERY_8K, DRAFT_CANVAS

    adapter = LtxHdAdapter()
    monkeypatch.setattr(settings, "ltx_hd_delivery", "8k")
    assert adapter._delivery_tier(_job(tmp_path)) == "8k"
    job = _job(tmp_path)
    job.execution["delivery"] = "4k"
    assert adapter._delivery_tier(job) == "4k"               # the job still wins
    job.execution["delivery"] = "definitely-not-a-tier"
    assert adapter._delivery_tier(job) == "1080p"            # a typo cannot 8K everything

    assert DELIVERY_8K == {"16:9": (7680, 4320), "9:16": (4320, 7680), "1:1": (4320, 4320)}
    # exactly twice 4K on every side, which is what "replace the destination"
    # means rather than a differently-shaped frame
    for ratio, (width, height) in DELIVERY_4K.items():
        assert DELIVERY_8K[ratio] == (width * 2, height * 2)
    for width, height in DELIVERY_8K.values():
        assert width % 2 == 0 and height % 2 == 0
    # and the client's stated pipeline end to end: 864x480 in, 7680x4320 out,
    # at the aspect the customer picked
    assert DRAFT_CANVAS["16:9"] == (864, 480) and DELIVERY_8K["16:9"] == (7680, 4320)
    assert DRAFT_CANVAS["9:16"] == (480, 864) and DELIVERY_8K["9:16"] == (4320, 7680)
    assert ASPECTS["16:9"][1] is None


def test_480p_to_8k_is_one_resize_and_only_while_a_draft_canvas_is_set() -> None:
    """The client's pipeline in one test: "LTX generates 864x480 → existing
    GPU Lanczos runs once → final 7680x4320".

    The graph's own closing `ImageScale` is parked at the generation canvas —
    an identity resize — so the ONLY enlargement in the whole pipeline is the
    ffmpeg one, which is what "do not upscale to 4K first" asks for.

    **And the coupling that is easy to miss:** "directly from 480p" holds
    only while `LTX_HD_CANVAS` names a draft canvas. On `native` the graph
    delivers its own 1920x1080 and the finish enlarges that instead — still
    one resize, but from a different frame. Both settings have to be on the
    node together, which is why the job log prints both.
    """
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())

    def scale_node(canvas, delivery):
        api = compile_fast_1080(
            load_graph(CLIENT_GRAPH),
            Fast1080Edits(positive="p", negative=None, seconds=8, seed=1,
                          filename_prefix="x", image="placeholder.png",
                          canvas=canvas, delivery=delivery),
            catalogue,
        )
        [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
        [scale] = [e for e in api.values() if e["class_type"] == "ImageScale"]
        return (
            (latent["inputs"]["width"], latent["inputs"]["height"]),
            (scale["inputs"]["width"], scale["inputs"]["height"]),
        )

    # 8K with the draft canvas: generated 864x480, and the graph's own scaler
    # parked on the same numbers, so nothing is enlarged before ffmpeg.
    generated, parked = scale_node((864, 480), (864, 480))
    assert generated == (864, 480)
    assert parked == generated, "the graph must not enlarge before the ffmpeg finish"

    # 8K on a native canvas: the graph still delivers 1920x1080 and the
    # finish starts there. One resize, a different starting frame.
    generated, parked = scale_node(None, None)
    assert parked == (1920, 1080) and generated != (864, 480)


def test_the_distilled_schedule_is_never_touched_by_a_compile() -> None:
    """The client's 10 Sep 2026 review opened by asking us not to raise the
    workflow from 8 to 12 steps, and named the three nodes that carry the
    distilled model's fixed schedule.

    We never did and this is why we cannot: the compiler writes prompts, a
    duration, a seed, a filename, the image slot, the enhancer switch and the
    two size widgets. Nothing it can reach is a sampler. Reading the values
    back off a compiled prompt is the check, because a graph edit made
    somewhere else would show up here rather than on the GPU."""
    catalogue = json.loads((Path(__file__).parent / "data/ltx_object_info.json").read_text())
    api = compile_fast_1080(
        load_graph(CLIENT_GRAPH),
        Fast1080Edits(positive="p", negative=None, seconds=8, seed=1,
                      filename_prefix="x", image="placeholder.png",
                      canvas=(864, 480), delivery=None),
        catalogue,
    )
    sigmas = [e for e in api.values() if e["class_type"] == "ManualSigmas"]
    assert [e["inputs"]["sigmas"] for e in sigmas] == [
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    ]
    # nine sigmas is an eight-step schedule; twelve steps would be thirteen
    assert len(sigmas[0]["inputs"]["sigmas"].split(",")) == 9
    [guider] = [e for e in api.values() if e["class_type"] == "CFGGuider"]
    assert float(guider["inputs"]["cfg"]) == 1.0
    [sampler] = [e for e in api.values() if e["class_type"] == "KSamplerSelect"]
    assert sampler["inputs"]["sampler_name"] == "euler_ancestral"


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
