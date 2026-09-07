"""Reading a newer ComfyUI export, where widget values are positional.

The client sent a fourth workflow on 7 Sep 2026, exported by frontend
1.48.7. Every graph before it carried `widgets_values_named` — a name → value
dict — and the compiler read that directly. The new export has only the
positional `widgets_values` array, so the compiler emitted nodes with no
inputs at all and ComfyUI rejected every one of them with "Required input is
missing".

These pin the resolver that reads the positional form. Each rule here was
found by an actual server rejection, not by reasoning: the extra value a seed
carries, the option inputs a dynamic combo swallows, and the dotted name the
server addresses them by.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_ltx_graphs import _graph
from worker.comfy.ltx_graphs import flatten, load_graph
from worker.comfy.widget_values import is_widget, resolve, subgraph_widget_slots

CLIENT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/client-pack/ltx25/client_original"
    / "LTX2.5_ACTUAL_WORKFLOW_ONLY_FAST_1080_8s_AUDIO.json"
)

#: Enough of the server's catalogue to exercise every rule.
CATALOGUE: dict = {
    "RandomNoise": {
        "input": {"required": {"noise_seed": ["INT", {"control_after_generate": True}]}}
    },
    "LoadImage": {
        "input": {"required": {"image": [["a.png", "b.png"], {"image_upload": True}]}}
    },
    "CFGGuider": {
        "input": {
            "required": {
                "model": ["MODEL", {}],
                "positive": ["CONDITIONING", {}],
                "negative": ["CONDITIONING", {}],
                "cfg": ["FLOAT", {}],
            }
        }
    },
    "LTXVEmptyLatentAudio": {
        "input": {
            "required": {
                "frames_number": ["INT", {}],
                "frame_rate": ["FLOAT,INT", {"widgetType": "FLOAT"}],
                "batch_size": ["INT", {}],
                "audio_vae": ["VAE", {}],
            }
        }
    },
    "ResizeImageMaskNode": {
        "input": {
            "required": {
                "input": ["COMFY_MATCHTYPE_V3", {}],
                "resize_type": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "scale dimensions",
                                "inputs": {
                                    "required": {"width": ["INT", {}], "height": ["INT", {}]}
                                },
                            },
                            {
                                "key": "scale longer dimension",
                                "inputs": {"required": {"longer_size": ["INT", {}]}},
                            },
                        ]
                    },
                ],
                "scale_method": [["lanczos", "nearest"], {}],
            }
        }
    },
    "Enhancer": {
        "input": {
            "required": {
                "clip": ["CLIP", {}],
                "prompt": ["STRING", {}],
                "sampling_mode": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "on",
                                "inputs": {
                                    "required": {
                                        "temperature": ["FLOAT", {}],
                                        "seed": ["INT", {}],
                                    },
                                    "optional": {"presence_penalty": ["FLOAT", {}]},
                                },
                            }
                        ]
                    },
                ],
            },
            "optional": {"thinking": ["BOOLEAN", {}]},
        }
    },
}


def test_link_only_types_never_take_a_widget_slot() -> None:
    for kind in ("MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "IMAGE", "SIGMAS"):
        assert is_widget(kind) is False
    assert is_widget("COMFY_MATCHTYPE_V3") is False
    for kind in ("INT", "FLOAT", "STRING", "BOOLEAN", "FLOAT,INT", "COMFY_DYNAMICCOMBO_V3"):
        assert is_widget(kind) is True
    assert is_widget(["a", "b"]) is True  # a combo, given as its options


def test_linked_inputs_are_skipped_and_widgets_land_in_order() -> None:
    """`CFGGuider` takes three links then one widget; the single value is the cfg."""
    inputs, complaints = resolve("CFGGuider", [1], CATALOGUE)
    assert inputs == {"cfg": 1}
    assert complaints == []

    inputs, complaints = resolve("LTXVEmptyLatentAudio", [193, 25, 1], CATALOGUE)
    assert inputs == {"frames_number": 193, "frame_rate": 25, "batch_size": 1}
    assert complaints == []


def test_a_seed_and_a_file_picker_each_carry_one_value_the_server_will_not_take() -> None:
    inputs, complaints = resolve("RandomNoise", [43, "fixed"], CATALOGUE)
    assert inputs == {"noise_seed": 43}
    assert complaints == []

    inputs, complaints = resolve("LoadImage", ["a.png", "image"], CATALOGUE)
    assert inputs == {"image": "a.png"}
    assert complaints == []


def test_a_dynamic_combo_swallows_its_options_inputs_under_a_dotted_name() -> None:
    """The rule that matters most: without it, 1536 lands in `scale_method`."""
    inputs, complaints = resolve(
        "ResizeImageMaskNode", ["scale longer dimension", 1536, "lanczos"], CATALOGUE
    )
    assert inputs == {
        "resize_type": "scale longer dimension",
        "resize_type.longer_size": 1536,
        "scale_method": "lanczos",
    }
    assert complaints == []
    # A different option key consumes a different number of values.
    inputs, _ = resolve("ResizeImageMaskNode", ["scale dimensions", 512, 384, "nearest"], CATALOGUE)
    assert inputs == {
        "resize_type": "scale dimensions",
        "resize_type.width": 512,
        "resize_type.height": 384,
        "scale_method": "nearest",
    }


def test_an_options_optional_inputs_count_too_and_its_seed_carries_a_mode() -> None:
    values = ["a prompt", "on", 0.01, 7, 0.5, True]
    inputs, complaints = resolve("Enhancer", values, CATALOGUE)
    assert inputs["prompt"] == "a prompt"
    assert inputs["sampling_mode"] == "on"
    assert inputs["sampling_mode.temperature"] == 0.01
    assert inputs["sampling_mode.seed"] == 7
    assert inputs["sampling_mode.presence_penalty"] == 0.5
    assert inputs["thinking"] is True
    assert complaints == []


def test_a_layout_the_resolver_does_not_understand_complains_rather_than_guessing() -> None:
    _, complaints = resolve("CFGGuider", [1, "surprise"], CATALOGUE)
    assert complaints and "left over" in complaints[0]
    _, complaints = resolve("NoSuchClass", [1], CATALOGUE)
    assert complaints and "not offered by the server" in complaints[0]


def test_a_named_export_is_unaffected_by_any_of_this() -> None:
    """The pack's own graphs carry `widgets_values_named` and must compile
    identically whether or not a catalogue is supplied."""
    for name in ("text_to_video", "first_last_frame", "character_replacement"):
        without = flatten(_graph(name))
        with_it = flatten(_graph(name), catalogue=CATALOGUE)
        without.prune_unreachable()
        with_it.prune_unreachable()
        assert without.to_api_prompt() == with_it.to_api_prompt(), name


# ── The client's own file ──────────────────────────────────────────────────

@pytest.mark.skipif(not CLIENT.exists(), reason="client workflow not present")
def test_the_client_export_carries_no_named_widgets_at_all() -> None:
    raw = json.loads(CLIENT.read_text(encoding="utf-8"))
    everywhere = list(raw["nodes"])
    for sub in raw["definitions"]["subgraphs"]:
        everywhere.extend(sub["nodes"])
    assert not any("widgets_values_named" in n for n in everywhere)
    assert raw["extra"]["frontendVersion"].startswith("1.4")


@pytest.mark.skipif(not CLIENT.exists(), reason="client workflow not present")
def test_without_a_catalogue_the_nodes_own_widgets_are_lost() -> None:
    """What the bug looked like. A value promoted across a subgraph boundary
    still arrives, because that is a link; a value the node holds itself does
    not, and the server rejects the node for a missing required input."""
    flat = flatten(load_graph(CLIENT))
    flat.prune_unreachable()
    api = flat.to_api_prompt()
    [sigmas] = [e for e in api.values() if e["class_type"] == "ManualSigmas"]
    assert sigmas["inputs"] == {}
    [save] = [e for e in api.values() if e["class_type"] == "SaveVideo"]
    assert "filename_prefix" not in save["inputs"]


@pytest.mark.skipif(not CLIENT.exists(), reason="client workflow not present")
def test_a_subgraph_instances_values_follow_the_definitions_widget_inputs() -> None:
    """The `Preprocess` instance omits `img_compression` from its own input
    list while its `widgets_values` still carries the value, so the slots
    have to come from the definition."""
    raw = json.loads(CLIENT.read_text(encoding="utf-8"))
    pre = next(s for s in raw["definitions"]["subgraphs"] if s["name"] == "Preprocess")
    slots = subgraph_widget_slots(pre)
    names = [pre["inputs"][i]["name"] for i in slots]
    assert names == ["length", "a", "bypass", "width", "height", "img_compression", "strength"]
    instance = next(n for n in raw["nodes"] if n["id"] == 5514)
    assert len(instance["widgets_values"]) == len(slots)
