"""The client's LTX 2.5 graphs, compiled without a GPU.

The frozen files under `benchmarks/client-pack/ltx25/` are used directly —
the same bytes the service will receive — so a graph edit that breaks the
conversion breaks here first. What these tests pin: the flattening
invariants (subgraphs, nested subgraphs, Set/Get links, promoted widgets),
the sanctioned edit set and nothing beyond it, the optional-last-frame
bypass, and the `/object_info` verification that guards the deploy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worker.comfy.ltx_graphs import (
    ASPECT_LABELS,
    GenerationEdits,
    GraphError,
    ReplacementEdits,
    SeedPlan,
    aspect_label_for,
    character_frames_for_seconds,
    class_types_of,
    compile_character_replacement,
    compile_first_last_frame,
    compile_text_to_video,
    flatten,
    frames_for_seconds,
    graph_sha256,
    load_graph,
    model_files_referenced,
    oriented_canvas,
    structural_summary,
    verify_against_object_info,
)
from worker.core.config import settings
from worker.providers.ltx_comfy import GRAPH_FILES, GRAPH_SHA256, LtxComfyService

PACK = settings.ltx_comfy_workflows_dir


def _graph(name: str) -> dict:
    return load_graph(PACK / GRAPH_FILES[name])


def _t2v_edits(**overrides) -> GenerationEdits:
    base = dict(
        positive="a koi pond at dawn",
        negative="blurry",
        seconds=10,
        aspect_label=ASPECT_LABELS["16:9"],
        seed_base=4242,
        filename_prefix="zolexai/job/output",
    )
    base.update(overrides)
    return GenerationEdits(**base)


# ── The pack is the pack ─────────────────────────────────────────────────────


def test_frozen_graphs_are_byte_identical_to_the_delivered_zip() -> None:
    for name, expected in GRAPH_SHA256.items():
        assert graph_sha256(PACK / GRAPH_FILES[name]) == expected, name
    assert LtxComfyService().graphs_match_the_pack() == []


# ── Flattening ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "nodes", "links"),
    [
        ("text_to_video", 45, 62),
        ("first_last_frame", 54, 77),
        ("character_replacement", 72, 117),
    ],
)
def test_flattening_reaches_every_real_node_and_link(name: str, nodes: int, links: int) -> None:
    """Root nodes plus every subgraph's nodes, minus the virtual ones; every
    surviving link points at a surviving node. The totals are the graphs'
    own, counted by hand in the Phase 0 audit."""
    flat = flatten(_graph(name))
    summary = structural_summary(flat)
    assert summary["nodes"] == nodes
    assert summary["links"] == links
    for node in flat.nodes.values():
        for src in node.inputs.values():
            if isinstance(src, tuple):
                assert src[0] in flat.nodes


def test_no_virtual_or_subgraph_typed_node_survives() -> None:
    for name in GRAPH_FILES:
        graph = _graph(name)
        uuids = {sg["id"] for sg in graph["definitions"]["subgraphs"]}
        api = flatten(graph).to_api_prompt()
        types = class_types_of(api)
        assert not types & {"GetNode", "SetNode", "Reroute", "MarkdownNote", "Note"}
        assert not types & uuids


def test_subgraph_ids_follow_the_frontend_convention() -> None:
    api = flatten(_graph("text_to_video")).to_api_prompt()
    inner = [nid for nid in api if ":" in nid]
    assert inner and all(nid.startswith("5464:") for nid in inner)
    # The nested instance in the character graph gets a two-level id.
    api = flatten(_graph("character_replacement")).to_api_prompt()
    assert any(nid.count(":") == 2 for nid in api)


def test_promoted_sampler_widget_reaches_the_inner_node() -> None:
    """The instance node carries `sampler_name` as a widget value for a
    subgraph input that has no link; the inner KSamplerSelect must get it."""
    api = flatten(_graph("text_to_video")).to_api_prompt()
    [sampler] = [e for e in api.values() if e["class_type"] == "KSamplerSelect"]
    assert sampler["inputs"]["sampler_name"] == "euler_ancestral"
    api = flatten(_graph("first_last_frame")).to_api_prompt()
    [sampler] = [e for e in api.values() if e["class_type"] == "KSamplerSelect"]
    assert sampler["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"


def test_set_get_links_resolve_across_the_subgraph_boundary() -> None:
    """A GetNode inside the character graph's 'Replace first frame' subgraph
    reads the root SetNode 'ref_image'; the batch node must end up wired to
    the resize node that feeds it."""
    api = flatten(_graph("character_replacement")).to_api_prompt()
    [batch] = [e for e in api.values() if e["class_type"] == "BatchImagesNode"]
    ref_image = batch["inputs"]["images.image0"]
    ref_video = batch["inputs"]["images.image1"]
    assert api[ref_image[0]]["class_type"] == "ImageResizeKJv2"
    assert api[ref_video[0]]["class_type"] == "ImageResizeKJv2"
    assert ref_image[0] != ref_video[0]


def test_widget_values_that_are_links_are_shadowed_by_the_link() -> None:
    api = flatten(_graph("text_to_video")).to_api_prompt()
    [latent] = [e for e in api.values() if e["class_type"] == "EmptyLTXVLatentVideo"]
    # width/height/length are linked in the graph; the stale widget numbers
    # (1088/1920/121) must not leak through.
    assert isinstance(latent["inputs"]["width"], list)
    assert isinstance(latent["inputs"]["height"], list)
    assert isinstance(latent["inputs"]["length"], list)


def test_ui_only_widget_keys_are_dropped() -> None:
    for name in GRAPH_FILES:
        api = flatten(_graph(name)).to_api_prompt()
        for entry in api.values():
            assert "videopreview" not in entry["inputs"]
            assert "control_after_generate" not in entry["inputs"]
            assert "choose video to upload" not in entry["inputs"]


# ── The sanctioned edits, and nothing else ──────────────────────────────────


def test_text_to_video_edits_land_on_the_user_inputs_only() -> None:
    graph = _graph("text_to_video")
    untouched = flatten(graph).to_api_prompt()
    api = compile_text_to_video(graph, _t2v_edits(seconds=15, aspect_label=ASPECT_LABELS["9:16"]))

    positive = [
        e
        for e in api.values()
        if e["class_type"] == "CLIPTextEncode" and "positive" in e["_meta"]["title"]
    ]
    negative = [
        e
        for e in api.values()
        if e["class_type"] == "CLIPTextEncode" and "negative" in e["_meta"]["title"]
    ]
    assert positive[0]["inputs"]["text"] == "a koi pond at dawn"
    assert negative[0]["inputs"]["text"] == "blurry"

    [slider] = [e for e in api.values() if e["class_type"] == "mxSlider"]
    assert (
        slider["inputs"]["Xi"] == 15
        and slider["inputs"]["Xf"] == 15.0
        and slider["inputs"]["isfloatX"] == 0
    )

    [selector] = [e for e in api.values() if e["class_type"] == "ResolutionSelector"]
    assert selector["inputs"] == {
        "aspect_ratio": "9:16 (Portrait Widescreen)",
        "megapixels": 0.9,
        "multiple": 32,
    }

    [combine] = [e for e in api.values() if e["class_type"] == "VHS_VideoCombine"]
    assert combine["inputs"]["filename_prefix"] == "zolexai/job/output"
    assert combine["inputs"]["save_output"] is True
    assert combine["inputs"]["format"] == "video/h264-mp4" and combine["inputs"]["crf"] == 10

    for entry in api.values():
        if entry["class_type"] == "RandomNoise":
            assert entry["inputs"]["noise_seed"] == 4242

    # Everything the edits did not name is byte-identical to the flattened
    # graph: samplers, sigmas, LoRA strengths, VAEs, upscaler, memory nodes.
    edited_types = {
        "CLIPTextEncode",
        "mxSlider",
        "ResolutionSelector",
        "VHS_VideoCombine",
        "RandomNoise",
    }
    for nid, entry in untouched.items():
        if entry["class_type"] in edited_types:
            continue
        assert api[nid] == entry, nid


def test_text_to_video_keeps_the_packs_models_loras_and_schedules() -> None:
    api = compile_text_to_video(_graph("text_to_video"), _t2v_edits())
    models = model_files_referenced(api)
    assert models["UnetLoaderGGUF.unet_name"] == ["LTX-2.5-Distilled-Q8_0.gguf"]
    assert models["CLIPLoader.clip_name"] == [
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    ]
    assert set(models["VAELoaderKJ.vae_name"]) == {
        "taeltx2_3.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
    }
    assert models["Power Lora Loader (rgthree).lora"] == [
        "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
        "ltx2.3-transition.safetensors",
    ]
    assert models["LoraLoaderModelOnly.lora_name"] == ["ltx-2-19b-ic-lora-detailer.safetensors"]
    [power] = [e for e in api.values() if e["class_type"] == "Power Lora Loader (rgthree)"]
    assert power["inputs"]["lora_1"]["strength"] == 0.4
    assert power["inputs"]["lora_2"]["strength"] == 0.8
    [detailer] = [e for e in api.values() if e["class_type"] == "LoraLoaderModelOnly"]
    assert detailer["inputs"]["strength_model"] == 0.3
    sigmas = sorted(
        e["inputs"]["sigmas"] for e in api.values() if e["class_type"] == "ManualSigmas"
    )
    assert sigmas == [
        "0.85, 0.7250, 0.4219, 0.0",
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    ]


def test_each_random_noise_node_gets_its_own_seed_from_one_base() -> None:
    api = compile_first_last_frame(
        _graph("first_last_frame"),
        _t2v_edits(seed_base=100, first_image="a.png", last_image="b.png"),
    )
    seeds = [e["inputs"]["noise_seed"] for e in api.values() if e["class_type"] == "RandomNoise"]
    assert len(seeds) == 2 and len(set(seeds)) == 2 and 100 in seeds
    plan = SeedPlan(100)
    assert plan.for_index(0) == 100 and plan.for_index(1) == 100 + 7_919


def test_first_last_frame_with_both_stills_runs_the_graph_unmodified() -> None:
    graph = _graph("first_last_frame")
    untouched = flatten(graph).to_api_prompt()
    api = compile_first_last_frame(
        graph, _t2v_edits(first_image="first.png", last_image="last.png")
    )
    assert len(api) == len(untouched)
    loaders = {
        e["_meta"]["title"]: e["inputs"]["image"]
        for e in api.values()
        if e["class_type"] == "LoadImage"
    }
    assert loaders == {"Load Image1": "first.png", "Load Image2": "last.png"}
    [kj] = [e for e in api.values() if e["class_type"] == "LTXVImgToVideoInplaceKJ"]
    assert kj["inputs"]["num_images.index_1"] == 0 and kj["inputs"]["num_images.index_2"] == -1
    [inplace] = [e for e in api.values() if e["class_type"] == "LTXVImgToVideoInplace"]
    assert inplace["inputs"]["strength"] == 0.8


def test_first_frame_only_keeps_the_conditioning_node_with_one_image() -> None:
    """The KJ two-image node becomes a one-image node (its own counter set to
    1, the second image group gone); the second loader chain disappears; the
    stage-2 first-frame conditioning stays exactly as shipped. Bypassing the
    node instead lost the identity after frame 0 on the GPU (5 Sep 2026)."""
    graph = _graph("first_last_frame")
    api = compile_first_last_frame(graph, _t2v_edits(first_image="first.png", last_image=None))
    types = class_types_of(api)
    assert "LTXVImgToVideoInplaceKJ" in types
    assert "LTXVImgToVideoInplace" in types
    assert [e["inputs"]["image"] for e in api.values() if e["class_type"] == "LoadImage"] == [
        "first.png"
    ]
    [kj] = [e for e in api.values() if e["class_type"] == "LTXVImgToVideoInplaceKJ"]
    assert kj["inputs"]["num_images"] == "1"
    assert kj["inputs"]["num_images.index_1"] == 0 and kj["inputs"]["num_images.strength_1"] == 1
    assert not any(k.endswith("_2") for k in kj["inputs"])
    assert isinstance(kj["inputs"]["num_images.image_1"], list)
    assert len(api) == 51  # 54 minus the second loader, its resize and its preprocess
    [inplace] = [e for e in api.values() if e["class_type"] == "LTXVImgToVideoInplace"]
    assert inplace["inputs"]["strength"] == 0.8


def test_first_last_frame_requires_a_first_image() -> None:
    with pytest.raises(GraphError):
        compile_first_last_frame(_graph("first_last_frame"), _t2v_edits(first_image=None))


def test_character_replacement_edits_land_on_the_user_inputs_only() -> None:
    graph = _graph("character_replacement")
    untouched = flatten(graph)
    untouched.prune_unreachable()
    untouched_api = untouched.to_api_prompt()
    api = compile_character_replacement(
        graph,
        ReplacementEdits(
            positive="the same man in a grey suit",
            negative="source actor",
            video="zolex_job_source.mp4",
            image="zolex_job_ref.png",
            seconds=8,
            width=1280,
            height=736,
            seed_base=77,
            filename_prefix="zolexai/job/output",
        ),
    )
    [video] = [e for e in api.values() if e["class_type"] == "VHS_LoadVideoFFmpeg"]
    assert video["inputs"]["video"] == "zolex_job_source.mp4"
    assert video["inputs"]["format"] == "LTXV"
    assert isinstance(video["inputs"]["force_rate"], list) and isinstance(
        video["inputs"]["frame_load_cap"], list
    )
    [image] = [e for e in api.values() if e["class_type"] == "LoadImage"]
    assert image["inputs"]["image"] == "zolex_job_ref.png"
    constants = {
        e["_meta"]["title"]: e["inputs"]["value"]
        for e in api.values()
        if e["class_type"] == "INTConstant"
    }
    assert constants == {"Set Width": 1280, "Set Height": 736, "Set Length (seconds)": 8}
    [conditioning] = [e for e in api.values() if e["class_type"] == "LTXVConditioning"]
    assert (
        api[conditioning["inputs"]["positive"][0]]["inputs"]["text"]
        == "the same man in a grey suit"
    )
    assert api[conditioning["inputs"]["negative"][0]]["inputs"]["text"] == "source actor"
    [lora] = [e for e in api.values() if e["class_type"] == "LoraLoaderModelOnly"]
    assert lora["inputs"] == {
        "model": lora["inputs"]["model"],
        "lora_name": "LTX/LTX-2.5/LTX25_Ripple_v11.safetensors",
        "strength_model": 1.35,
    }
    edited = {
        "CLIPTextEncode",
        "VHS_LoadVideoFFmpeg",
        "LoadImage",
        "INTConstant",
        "RandomNoise",
        "VHS_VideoCombine",
    }
    for nid, entry in untouched_api.items():
        if entry["class_type"] in edited:
            continue
        assert api[nid] == entry, nid
    # The two dangling nodes the pack left disconnected are pruned; nothing else.
    assert set(untouched_api) == set(api)


def test_character_graph_keeps_single_pass_and_source_audio_switches() -> None:
    api = compile_character_replacement(
        _graph("character_replacement"),
        ReplacementEdits("p", "n", "v.mp4", "i.png", 8, 736, 1280, 1, "x"),
    )
    booleans = {
        e["_meta"]["title"]: e["inputs"]["value"]
        for e in api.values()
        if e["class_type"] == "PrimitiveBoolean"
    }
    assert booleans == {"Set Single Pass": True, "Use Audio from Video Input": True}
    samplers = [
        e["inputs"]["sampler_name"] for e in api.values() if e["class_type"] == "KSamplerSelect"
    ]
    assert samplers and set(samplers) == {"lcm"}
    [shift] = [e for e in api.values() if e["class_type"] == "ModelSamplingSD3"]
    assert shift["inputs"]["shift"] == 13


# ── Lattice and canvas ──────────────────────────────────────────────────────


def test_product_durations_land_on_the_lattice() -> None:
    assert [frames_for_seconds(s) for s in (5, 10, 15, 30)] == [121, 241, 361, 721]
    # Every whole second lands on 8k+1 at 24 fps; a fractional one does not.
    with pytest.raises(GraphError):
        frames_for_seconds(2.5)


def test_character_graph_frame_formula() -> None:
    assert character_frames_for_seconds(8) == 193  # the ZIP sample: 8 s → 193 frames
    assert character_frames_for_seconds(5) == 121
    assert character_frames_for_seconds(20) == 481


def test_aspect_labels_resolve_against_live_options_first() -> None:
    live = ["1:1 (Square)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)"]
    assert aspect_label_for("9:16", live) == "9:16 (Portrait Widescreen)"
    assert aspect_label_for("16:9") == "16:9 (Widescreen)"
    with pytest.raises(GraphError):
        aspect_label_for("4:5")
    with pytest.raises(GraphError):
        aspect_label_for("4:5", live)


def test_character_canvas_keeps_the_packs_pixels_and_follows_the_source() -> None:
    assert oriented_canvas((736, 1280), source_width=576, source_height=1024) == (736, 1280)
    assert oriented_canvas((736, 1280), source_width=1920, source_height=1080) == (1280, 736)
    assert oriented_canvas((736, 1280), source_width=None, source_height=None) == (736, 1280)


# ── Verification against a live server ─────────────────────────────────────


def _object_info_accepting(api: dict) -> dict:
    """A catalogue that declares every input the prompt uses, combos included."""
    info: dict = {}
    for entry in api.values():
        spec = info.setdefault(entry["class_type"], {"input": {"required": {}, "optional": {}}})
        for name, value in entry["inputs"].items():
            if isinstance(value, str) and not name.startswith("text"):
                slot = spec["input"]["required"].setdefault(name, [[]])
                if value not in slot[0]:
                    slot[0].append(value)
            else:
                spec["input"]["required"][name] = ["*"]
    return info


def test_verification_passes_against_a_catalogue_that_offers_everything() -> None:
    api = compile_text_to_video(_graph("text_to_video"), _t2v_edits())
    assert verify_against_object_info(api, _object_info_accepting(api)) == []


def test_verification_names_the_missing_node_pack_and_the_missing_model() -> None:
    api = compile_text_to_video(_graph("text_to_video"), _t2v_edits())
    info = _object_info_accepting(api)
    del info["UnetLoaderGGUF"]
    info["CLIPLoader"]["input"]["required"]["clip_name"] = [["some-other-encoder.safetensors"]]
    problems = verify_against_object_info(api, info)
    assert any("UnetLoaderGGUF" in p and "not installed" in p for p in problems)
    assert any("clip_name=" in p and "not among" in p for p in problems)


def test_verification_ignores_per_job_files_and_dynamic_inputs() -> None:
    api = compile_first_last_frame(_graph("first_last_frame"), _t2v_edits(first_image="job.png"))
    info = _object_info_accepting(api)
    info["LoadImage"]["input"]["required"]["image"] = [["something-else.png"]]
    del info["Power Lora Loader (rgthree)"]["input"]["required"]["lora_1"]
    assert verify_against_object_info(api, info) == []


def test_compiled_prompts_are_json_serialisable() -> None:
    for name in GRAPH_FILES:
        api = flatten(_graph(name)).to_api_prompt()
        json.dumps(api)


def test_pack_directory_is_where_the_worker_looks(tmp_path: Path) -> None:
    assert (PACK / "ltx25_text_to_video.json").is_file()
    assert PACK == settings.ltx_comfy_workflows_dir


def test_combo_options_reads_the_v3_schema_shape() -> None:
    """ComfyUI 0.34 core nodes report combos as ["COMBO", {"options": [...]}]
    (seen live on the GPU node, 5 Sep 2026); the legacy shapes still parse."""
    from worker.comfy.ltx_graphs import combo_options

    info = {
        "ResolutionSelector": {
            "input": {
                "required": {
                    "aspect_ratio": [
                        "COMBO",
                        {
                            "default": "1:1 (Square)",
                            "options": ["1:1 (Square)", "16:9 (Widescreen)"],
                        },
                    ],
                    "megapixels": ["FLOAT", {"default": 1.0}],
                }
            }
        },
        "Legacy": {"input": {"required": {"name": [["a", "b"]]}}},
        "Dict": {"input": {"required": {"name": [{"options": ["x"]}]}}},
    }
    assert combo_options(info, "ResolutionSelector", "aspect_ratio") == [
        "1:1 (Square)",
        "16:9 (Widescreen)",
    ]
    assert combo_options(info, "ResolutionSelector", "megapixels") is None
    assert combo_options(info, "Legacy", "name") == ["a", "b"]
    assert combo_options(info, "Dict", "name") == ["x"]
    assert combo_options(info, "Missing", "name") is None


def test_per_class_ui_only_widgets_are_dropped() -> None:
    api = flatten(_graph("text_to_video")).to_api_prompt()
    [preview] = [e for e in api.values() if e["class_type"] == "ModelPreviewOverrideKJ"]
    assert "preview" not in preview["inputs"]
    assert preview["inputs"]["preview_fps"] == 24
