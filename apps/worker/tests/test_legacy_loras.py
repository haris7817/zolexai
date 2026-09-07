"""Switching off the LTX 2.3 adapters and the LTX-2 detailer (7 Sep 2026).

The client's ComfyUI operator read our Text to Video graph and found it
mixes two LTX 2.3 LoRAs and an LTX-2 19B detailer with LTX 2.5 core files:
"the active older LoRAs are the biggest problem and could contribute to
darkening, color changes, weak prompt adherence, and inconsistent motion".
Their instruction is precise — turn off `LTX-2.3-OmniNFT-RL-Lora` and
`ltx2.3-transition` in the Power Lora Loader, and bypass the detailer so the
model reaches both `LTXVDualCFGGuider` nodes directly.

Both edits are compile-time and off by default: the frozen files are never
touched, and a deployment that says nothing renders exactly what it rendered
before. What the switches do is pinned here, because "the graph as delivered"
is the contract and any drift from it has to be deliberate and visible.
"""

from __future__ import annotations

import pytest

from tests.test_ltx_graphs import _graph
from worker.adapters.base import AdapterJob
from worker.adapters.ltx_comfy import LtxComfyAdapter
from worker.comfy.ltx_graphs import (
    LAST_MODEL_CHAIN,
    GenerationEdits,
    compile_first_last_frame,
    compile_text_to_video,
    model_files_referenced,
)
from worker.core.config import settings

OMNI = "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"
TRANSITION = "ltx2.3-transition.safetensors"
DETAILER = "ltx-2-19b-ic-lora-detailer.safetensors"

_BASE = dict(
    positive="a boxer in a ring",
    negative="blurry",
    seconds=30,
    aspect_label="9:16 (Portrait)",
    seed_base=7,
    filename_prefix="zolexai/job/output",
)


def _power_loras(api: dict) -> dict[str, dict]:
    [loader] = [e for e in api.values() if e["class_type"] == "Power Lora Loader (rgthree)"]
    return {k: v for k, v in loader["inputs"].items() if k.lower().startswith("lora_")}


def _lora_files(api: dict) -> list[str]:
    return [
        e["inputs"]["lora_name"] for e in api.values() if e["class_type"] == "LoraLoaderModelOnly"
    ]


def _guider_models(api: dict) -> list[str]:
    """The class feeding every dual guider's `model` input."""
    out = []
    for entry in api.values():
        if entry["class_type"] == "LTXVDualCFGGuider":
            src = entry["inputs"].get("model")
            out.append(api[src[0]]["class_type"] if isinstance(src, list) else "?")
    return out


def test_the_pack_runs_as_delivered_when_nothing_is_switched_off() -> None:
    api = compile_text_to_video(_graph("text_to_video"), GenerationEdits(**_BASE))
    assert [v["on"] for v in _power_loras(api).values()] == [True, True]
    assert DETAILER in _lora_files(api)
    # And an explicit "nothing" is the same thing.
    quiet = compile_text_to_video(
        _graph("text_to_video"), GenerationEdits(**_BASE, disabled_loras=(), bypass_detailer=False)
    )
    assert quiet == api


def test_the_two_legacy_loras_go_off_and_nothing_else_moves() -> None:
    graph = _graph("text_to_video")
    before = compile_text_to_video(graph, GenerationEdits(**_BASE))
    after = compile_text_to_video(
        graph, GenerationEdits(**_BASE, disabled_loras=("OmniNFT", "ltx2.3-transition"))
    )
    loras = _power_loras(after)
    assert [v["on"] for v in loras.values()] == [False, False]
    # The entries keep their file and strength: this is the operator's own
    # toggle, not a different graph.
    assert [v["lora"] for v in loras.values()] == [OMNI, TRANSITION]
    assert [v["strength"] for v in loras.values()] == [0.4, 0.8]
    changed = [nid for nid in before if before[nid] != after.get(nid)]
    assert changed == [nid for nid, e in before.items() if e["class_type"].startswith("Power Lora")]
    assert set(before) == set(after)


def test_a_fragment_that_matches_nothing_changes_nothing() -> None:
    graph = _graph("text_to_video")
    assert compile_text_to_video(
        graph, GenerationEdits(**_BASE, disabled_loras=("no-such-lora",))
    ) == compile_text_to_video(graph, GenerationEdits(**_BASE))


def test_the_detailer_is_bypassed_and_both_guiders_take_the_model_directly() -> None:
    """The delivered graph is not quite what the client's operator described.

    They wrote "connect the model directly to BOTH LTXVDualCFGGuider nodes",
    which reads as though both currently run through the detailer. Only one
    does: the second guider already takes the model straight from the preview
    override. So the detailer is active in one of the two sampling stages,
    and bypassing it makes the two stages agree.
    """
    graph = _graph("text_to_video")
    before = compile_text_to_video(graph, GenerationEdits(**_BASE))
    assert sorted(_guider_models(before)) == ["LoraLoaderModelOnly", "ModelPreviewOverrideKJ"]

    after = compile_text_to_video(graph, GenerationEdits(**_BASE, bypass_detailer=True))
    assert DETAILER not in _lora_files(after)
    assert _guider_models(after) == ["ModelPreviewOverrideKJ", "ModelPreviewOverrideKJ"]
    assert len(after) == len(before) - 1


def test_the_first_last_frame_graph_takes_the_same_two_switches() -> None:
    """image-to-video and extend-video run this graph, so a deployment that
    cleans Text to Video and forgets this one would ship two different
    models under two names."""
    edits = dict(_BASE, first_image="first.png")
    api = compile_first_last_frame(
        _graph("first_last_frame"),
        GenerationEdits(
            **edits,
            disabled_loras=("OmniNFT", "ltx2.3-transition"),
            bypass_detailer=True,
        ),
    )
    assert [v["on"] for v in _power_loras(api).values()] == [False, False]
    assert DETAILER not in _lora_files(api)
    assert _guider_models(api) == ["ModelPreviewOverrideKJ", "ModelPreviewOverrideKJ"]


def _job(**execution: object) -> AdapterJob:
    return AdapterJob(
        job_id="j",
        workflow_id="text-to-video",
        workflow_version="1",
        prompt="",
        parameters={},
        execution={"runtime": "ltx_comfy", **execution},
    )


def test_the_switches_are_off_by_default_and_a_deployment_can_flip_them() -> None:
    assert settings.ltx_comfy_disabled_loras == ""
    assert settings.ltx_comfy_bypass_detailer is False
    assert LtxComfyAdapter.disabled_loras(_job()) == ()
    assert LtxComfyAdapter.bypasses_detailer(_job()) is False

    listed = _job(disabled_loras="OmniNFT, ltx2.3-transition")
    assert LtxComfyAdapter.disabled_loras(listed) == ("OmniNFT", "ltx2.3-transition")
    assert LtxComfyAdapter.disabled_loras(_job(disabled_loras=["a", "b"])) == ("a", "b")
    assert LtxComfyAdapter.disabled_loras(_job(disabled_loras="")) == ()
    assert LtxComfyAdapter.bypasses_detailer(_job(bypass_detailer=True)) is True
    assert LtxComfyAdapter.bypasses_detailer(_job(bypass_detailer="false")) is False


def test_the_settings_are_read_when_the_workflow_says_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ltx_comfy_disabled_loras", "OmniNFT,ltx2.3-transition")
    monkeypatch.setattr(settings, "ltx_comfy_bypass_detailer", True)
    assert LtxComfyAdapter.disabled_loras(_job()) == ("OmniNFT", "ltx2.3-transition")
    assert LtxComfyAdapter.bypasses_detailer(_job()) is True
    # The workflow still wins.
    assert LtxComfyAdapter.bypasses_detailer(_job(bypass_detailer="no")) is False


# ── The transformer the generation graphs load ──────────────────────────────

GGUF = "LTX-2.5-Distilled-Q8_0.gguf"
INT8 = "LTXVideo/v2/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"


def _transformer(api: dict) -> tuple[str, dict]:
    [node] = [e for e in api.values() if e["class_type"] in ("UNETLoader", "UnetLoaderGGUF")]
    return node["class_type"], node["inputs"]


def test_the_generation_graphs_load_the_community_gguf_as_delivered() -> None:
    """Worth pinning: the character graph loads Lightricks' own int8 file, so
    the pack ships two different transformers under three product names."""
    api = compile_text_to_video(_graph("text_to_video"), GenerationEdits(**_BASE))
    cls, inputs = _transformer(api)
    assert (cls, inputs["unet_name"]) == ("UnetLoaderGGUF", GGUF)


def test_the_transformer_can_be_swapped_for_the_official_safetensors() -> None:
    graph = _graph("text_to_video")
    before = compile_text_to_video(graph, GenerationEdits(**_BASE))
    after = compile_text_to_video(graph, GenerationEdits(**_BASE, transformer=INT8))

    cls, inputs = _transformer(after)
    assert cls == "UNETLoader"
    assert inputs == {"unet_name": INT8, "weight_dtype": "default"}
    # The swap is the loader and nothing else.
    changed = [nid for nid in before if before[nid] != after.get(nid)]
    assert len(changed) == 1 and before[changed[0]]["class_type"] == "UnetLoaderGGUF"
    assert set(before) == set(after)
    # And the health check follows the file that will actually be loaded.
    assert INT8 in model_files_referenced(after)["UNETLoader.unet_name"]
    assert GGUF in model_files_referenced(before)["UnetLoaderGGUF.unet_name"]


def test_a_gguf_name_puts_the_gguf_loader_back() -> None:
    api = compile_text_to_video(_graph("text_to_video"), GenerationEdits(**_BASE, transformer=GGUF))
    assert _transformer(api) == ("UnetLoaderGGUF", {"unet_name": GGUF})


def test_the_switches_are_reported_so_a_render_can_be_traced_to_its_model_chain() -> None:
    compile_text_to_video(_graph("text_to_video"), GenerationEdits(**_BASE))
    assert LAST_MODEL_CHAIN == {}

    compile_text_to_video(
        _graph("text_to_video"),
        GenerationEdits(
            **_BASE,
            disabled_loras=("OmniNFT", "ltx2.3-transition"),
            bypass_detailer=True,
            transformer=INT8,
        ),
    )
    assert LAST_MODEL_CHAIN["loras_off"] == [OMNI, TRANSITION]
    assert LAST_MODEL_CHAIN["detailer_off"] == DETAILER
    assert LAST_MODEL_CHAIN["transformer"] == INT8
    assert LAST_MODEL_CHAIN["transformer_was"] == GGUF
    assert "loras_not_found" not in LAST_MODEL_CHAIN


def test_a_fragment_that_matches_nothing_is_reported_rather_than_swallowed() -> None:
    """A typo in a deployment's LTX_COMFY_DISABLED_LORAS would otherwise look
    exactly like a successful run."""
    compile_text_to_video(
        _graph("text_to_video"), GenerationEdits(**_BASE, disabled_loras=("OmniNFT", "typo-here"))
    )
    assert LAST_MODEL_CHAIN["loras_off"] == [OMNI]
    assert LAST_MODEL_CHAIN["loras_not_found"] == ["typo-here"]
