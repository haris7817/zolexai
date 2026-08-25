"""Frozen client workflow JSON → executable ComfyUI API prompt.

The three graphs under `benchmarks/client-pack/` are the client's delivered
workflows, transcribed byte-for-byte and structurally validated. They are the
contract; this module does not redesign them. It performs the mechanical
UI→API conversion (linked inputs from the link table, widget inputs from
`widgets_values_named`) plus exactly the edits the pack sanctions:

  * the duration index — the guide's own "Set the duration index first";
  * Prompts 1–5 — the guide's own "review Prompts 1–5" (editing them is the
    intended use; the shipped text is placeholder);
  * disconnecting Picture 3 — the guide's own "disconnect unused loaders";
  * replacing the placeholder image filenames — the guide's own
    "REPLACE THIS FILE" node titles;
  * the width/height primitives — the guide documents them as "authoritative
    and must remain multiples of 32", which is what makes the measured
    544x320 draft / 960x544 delivery tiers legitimate settings rather than
    graph surgery;
  * the Final Decode `filename_prefix` / `output_directory` widgets — exposed
    by the pack for exactly this purpose, and how a job's output lands in the
    job's own workspace instead of a shared folder.

Deliberately NOT supported here: the 4-step Turbo LoRA (rejected on quality,
25 Aug — it loses the reference subject), any sampler/steps/scheduler change,
and any structural rewiring. Those are experiments, not the client build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: widgets_values_named keys that are UI state, not node inputs.
_UI_ONLY_KEYS = {
    "fixed",
    "control_after_generate",
    "h3_extender_timeline",
    "h3_live_preview",
    "upload",
}

#: Widget names whose string values carry Windows path separators in the
#: frozen graphs. The guide's own Linux note: "reselect the same files".
_PATH_WIDGETS = {"unet_name", "clip_name", "vae_name"}

#: Nominal duration presets, index 0–4, exactly as the graphs define them.
#: (frames are what the duration-plan JSON inside each graph resolves to.)
DURATION_PRESETS: dict[int, float] = {0: 5.0, 1: 10.0, 2: 15.0, 3: 30.0, 4: 60.0}

#: How many segment prompts each index consumes (Prompts 1..N).
PROMPTS_PER_INDEX: dict[int, int] = {0: 1, 1: 1, 2: 1, 3: 2, 4: 5}


@dataclass(frozen=True)
class GraphEdits:
    """The sanctioned edit set for one submission."""

    duration_index: int
    prompts: dict[int, str] = field(default_factory=dict)
    """PROMPT N → replacement text. Missing numbers keep the graph's text."""

    images: dict[str, str] = field(default_factory=dict)
    """LoadImage title prefix → filename inside ComfyUI's input directory.

    Keys are matched against node titles ("REFERENCE IMAGE 1", "I2V SOURCE
    IMAGE"), which are stable in the frozen graphs.
    """

    drop_reference_3: bool = False
    width: int | None = None
    height: int | None = None
    filename_prefix: str | None = None
    output_directory: str | None = None

    audio_context_length: int | None = None
    """Frames of AUDIO context carried between segments. The pack ships 0,
    and the measured consequence is a 9.8 dB loudness step at one of four
    60 s seams (25 Aug). None keeps the pack's behaviour; a value is an
    explicit, recorded experiment — never a silent default change."""

    seed_base: int | None = None
    """Per-job seed base. None keeps the pack's fixed seeds — which, with a
    fully deterministic model and ComfyUI's execution cache, means a customer
    who regenerates the same prompt receives the byte-identical video in
    seconds forever (observed in production 26 Aug: a 30 s "render" served
    from cache in 35 s). A value shifts every SEGMENT N SEED primitive and
    RandomNoise node to base + (pack seed % 1000), preserving the pack's
    per-segment distinctness while making each job its own video. Derive it
    stably from the job id so a retry reproduces its own attempt."""

    steps: int | None = None
    """Sampler step count on the Extender. The pack pins 20; None keeps that.

    A value is a recorded, user-decided deviation: measured 25 Aug on the
    RTX PRO 6000 (quality canvas, same seed) — 20 steps 150.3 s, 15 steps
    130.4 s, 12 steps 108.5 s for 5 s of video — and judged acceptable at 12
    by the user on 26 Aug. Applied to the compiled API inputs directly, which
    also closes the T2V gap where the list-form widgets never reached the
    submission and the server's defaults silently applied. Sampler and
    scheduler remain untouchable."""


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def duration_index_for(seconds: float) -> int | None:
    """The preset index whose nominal duration matches, or None.

    Exact-match on the pack's own presets. Client-test offers the durations
    the pack actually implements rather than silently rounding a request to a
    different length — a 20 s ask answered with 15 s of video is a wrong
    answer wearing a valid file.
    """
    for index, nominal in DURATION_PRESETS.items():
        if abs(seconds - nominal) < 0.25:
            return index
    return None


def nearest_duration_index(seconds: float) -> int:
    """The preset closest to a length the customer did not choose.

    Used ONLY for `duration_mode: source` workflows (reference V2V), where the
    API sends no duration and the source clip's own length is the stated
    intent. Unlike `duration_index_for`, approximation is correct here — the
    customer asked for "about my clip's length", not for a number.
    """
    return min(
        DURATION_PRESETS,
        key=lambda index: abs(DURATION_PRESETS[index] - seconds),
    )


def to_api_prompt(graph: dict[str, Any], edits: GraphEdits) -> dict[str, Any]:
    """The proven mechanical conversion, with the sanctioned edits applied."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}

    api: dict[str, dict[str, Any]] = {}
    titles: dict[str, str] = {}
    for nid, n in nodes.items():
        inputs: dict[str, Any] = {}
        for inp in n.get("inputs", []):
            li = inp.get("link")
            if li is not None:
                _, src, s_slot, _, _, _ = links[li]
                inputs[inp["name"]] = [str(src), s_slot]
        for key, value in (n.get("widgets_values_named") or {}).items():
            if key in _UI_ONLY_KEYS or key in inputs:
                continue
            if isinstance(value, str) and "\\" in value and key in _PATH_WIDGETS:
                value = value.replace("\\", "/")
            inputs[key] = value
        api[str(nid)] = {"class_type": n["type"], "inputs": inputs}
        titles[str(nid)] = n.get("title") or ""

    def having(fragment: str, class_type: str | None = None) -> list[str]:
        return [
            nid
            for nid, title in titles.items()
            if fragment in title
            and (class_type is None or api[nid]["class_type"] == class_type)
        ]

    for nid in having("TOTAL DURATION INDEX"):
        api[nid]["inputs"]["value"] = edits.duration_index

    for number, text in edits.prompts.items():
        for nid in having(f"PROMPT {number} ", "PrimitiveStringMultiline"):
            api[nid]["inputs"]["value"] = text

    for title_prefix, filename in edits.images.items():
        for nid in having(title_prefix, "LoadImage"):
            api[nid]["inputs"]["image"] = filename

    if edits.drop_reference_3:
        for entry in api.values():
            if entry["class_type"] == "MiniMaxH3ReferencePackBridge":
                entry["inputs"].pop("ref_3", None)
        for nid in having("REFERENCE IMAGE 3"):
            del api[nid]

    if edits.width is not None:
        for nid in having("OUTPUT WIDTH"):
            api[nid]["inputs"]["value"] = edits.width
    if edits.height is not None:
        for nid in having("OUTPUT HEIGHT"):
            api[nid]["inputs"]["value"] = edits.height

    if edits.seed_base is not None:
        # Literal seeds shift; a linked seed (a [node, slot] reference, as in
        # I2V's RandomNoise fed by the SEGMENT SEED primitives) inherits the
        # shifted value through its link and must not be touched.
        for nid in having("SEED", "PrimitiveInt"):
            old = api[nid]["inputs"].get("value", 0)
            if isinstance(old, int):
                api[nid]["inputs"]["value"] = edits.seed_base + (old % 1000)
        for entry in api.values():
            if entry["class_type"] == "RandomNoise":
                old = entry["inputs"].get("noise_seed", 0)
                if isinstance(old, int):
                    entry["inputs"]["noise_seed"] = edits.seed_base + (old % 1000)

    if edits.steps is not None:
        # On the API inputs, not the UI widgets: the T2V graph's list-form
        # widgets never reach the submission (verified in production history,
        # 25 Aug — the server's defaults applied), so the inputs dict is the
        # only layer where a step count reliably lands on every graph.
        # R2V/T2V carry steps on the Extender itself; the I2V graph holds its
        # schedule in a single BasicScheduler ("Validated schedule — beta /
        # 20 steps"). Both spellings of the same pinned 20.
        for entry in api.values():
            if entry["class_type"] in ("MiniMaxH3Extender", "BasicScheduler"):
                entry["inputs"]["steps"] = edits.steps

    if edits.audio_context_length is not None:
        for entry in api.values():
            if entry["class_type"] in ("MiniMaxH3Extender", "MiniMaxH3MotionContextRAM"):
                entry["inputs"]["audio_context_length"] = edits.audio_context_length

    for entry in api.values():
        if entry["class_type"] == "MiniMaxH3MotionContextDiskFinalDecode":
            if edits.filename_prefix is not None:
                entry["inputs"]["filename_prefix"] = edits.filename_prefix
            if edits.output_directory is not None:
                entry["inputs"]["output_directory"] = edits.output_directory

    return api
