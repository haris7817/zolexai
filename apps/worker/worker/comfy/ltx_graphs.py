"""The client's LTX 2.5 graphs → executable ComfyUI API prompts.

The three graphs under `benchmarks/client-pack/ltx25/` are the production
workflows for Text to Video, First/Last Frame Video and Character Replacement
(client decision, 5 Sep 2026: "use the same as in the zip"). They are the
contract. This module does not redesign them: it performs the mechanical
frontend→API conversion the ComfyUI browser client would perform on "Queue",
then applies exactly the runtime inputs a job has to supply — prompt, seed,
duration, input media, output location, and the canvas selections the graphs
themselves expose as user settings.

## Why this is more than the H3 compiler

`worker/comfy/graph.py` converts the H3 pack: flat graphs, `widgets_values_named`
on every node. The LTX graphs carry the same named widgets (every node, zero
exceptions — verified in the Phase 0 audit) but add three things that
compiler never met:

  * **Subgraphs.** Frontend format 0.4 stores reusable blocks under
    `definitions.subgraphs[]`; an instance node's `type` is the subgraph's
    UUID, and boundary links run to virtual nodes `-10` (inputs) and `-20`
    (outputs). The character graph nests one subgraph inside another. The
    server only accepts a flat prompt, so instances are inlined here with the
    frontend's own id convention, `<instance>:<inner>`.
  * **Set/Get virtual links.** KJNodes `SetNode`/`GetNode` are frontend-only
    wires keyed by a `Constant` name — a `GetNode` inside a subgraph reads a
    root `SetNode`. They are resolved to the real source before anything
    else, then dropped.
  * **A per-graph edit set** that includes structural bypass (the optional
    last frame) and dual-orientation canvas for a graph that pins one.

## What "sanctioned" means here

Every edit is one the pack's author exposed as a user input: the prompt boxes,
the "Clip Length" slider, the aspect-ratio selector, the seed, the two image
loaders, the video loader, the width/height/length constants, the output
prefix. The one structural edit — bypassing the last-frame conditioning node
when no last frame was supplied — is ComfyUI's own bypass semantics applied
to one node, which is what a user would click to run the graph with a first
frame only. Nothing here touches a sampler, a schedule, a LoRA strength or a
model file.

## Verification without a GPU

`flatten()` enforces structural invariants (every link resolves, no virtual or
UUID-typed node survives, node totals match). `verify_against_object_info()`
checks a compiled prompt against a live ComfyUI's `/object_info` — unknown
classes, unknown inputs, and combo values (model files, LoRA files, the
aspect label) that the installed node does not offer. The health check runs
it before any job; `scripts/ltx_comfy_health.py` prints it on GPU day.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Frontend-only node types. They never reach the server; the compiler
#: resolves what they wire and drops them.
VIRTUAL_TYPES = frozenset(
    {"GetNode", "SetNode", "Reroute", "MarkdownNote", "Note", "PrimitiveNode"}
)

#: `widgets_values_named` keys that are UI state rather than node inputs.
UI_ONLY_KEYS = frozenset(
    {
        "control_after_generate",
        "videopreview",
        "choose video to upload",
        "choose file to upload",
        "upload",
        "fixed",
        "PowerLoraLoaderHeaderWidget",
        "➕ Add Lora",
        "divider",
    }
)

#: UI-only widget keys that exist on one node class only (a preview image
#: widget the KJ preview override draws into; the server declares no such
#: input — verified against ComfyUI 0.34.5 + KJNodes e8e88f7 on 5 Sep 2026).
UI_ONLY_KEYS_BY_CLASS: dict[str, frozenset[str]] = {
    "ModelPreviewOverrideKJ": frozenset({"preview"}),
    "LTX2SamplingPreviewOverride": frozenset({"preview"}),
}

#: Node classes whose presence makes a prompt executable (ComfyUI OUTPUT_NODE
#: classes used by the pack). Everything not reachable backwards from one of
#: these is pruned: it would never execute, and an orphaned loader with a
#: placeholder filename must not be able to fail validation.
OUTPUT_TYPES = frozenset(
    {"VHS_VideoCombine", "SaveImage", "SaveVideo", "SaveAudio", "PreviewImage", "SaveAnimatedWEBP"}
)

#: The frame lattice both distilled graphs compute for themselves.
FRAME_RATE = 24


class GraphError(ValueError):
    """The graph is not the one this compiler was written against."""


# ── Flattened representation ────────────────────────────────────────────────

Source = tuple[str, int]
"""A reference to another flat node's output: (flat node id, output slot)."""


@dataclass(frozen=True)
class Literal:
    """A subgraph boundary input carried as a widget value on the instance."""

    value: Any


@dataclass
class FlatNode:
    id: str
    type: str
    title: str
    inputs: dict[str, Source | Literal]
    """Linked inputs by name. A `Literal` is a promoted widget value that
    crossed a subgraph boundary."""
    widgets: dict[str, Any]
    """`widgets_values_named` minus UI-only keys. A key also present in
    `inputs` is shadowed by the link, as on the frontend."""
    output_types: list[str]
    input_types: dict[str, str]


@dataclass
class FlatGraph:
    nodes: dict[str, FlatNode] = field(default_factory=dict)

    # ── Lookups ─────────────────────────────────────────────────────────

    def of_type(self, class_type: str) -> list[FlatNode]:
        return [n for n in self.nodes.values() if n.type == class_type]

    def one_of_type(self, class_type: str) -> FlatNode:
        found = self.of_type(class_type)
        if len(found) != 1:
            raise GraphError(f"expected exactly one {class_type}, found {len(found)}")
        return found[0]

    def titled(self, fragment: str, class_type: str | None = None) -> list[FlatNode]:
        return [
            n
            for n in self.nodes.values()
            if fragment.lower() in n.title.lower() and (class_type is None or n.type == class_type)
        ]

    def one_titled(self, fragment: str, class_type: str | None = None) -> FlatNode:
        found = self.titled(fragment, class_type)
        if len(found) != 1:
            raise GraphError(
                f"expected exactly one node titled ~'{fragment}'"
                f"{' of type ' + class_type if class_type else ''}, found {len(found)}"
            )
        return found[0]

    def consumers_of(self, node_id: str) -> list[tuple[FlatNode, str, int]]:
        """(consumer, input name, slot) for every input reading `node_id`."""
        out: list[tuple[FlatNode, str, int]] = []
        for node in self.nodes.values():
            for name, src in node.inputs.items():
                if isinstance(src, tuple) and src[0] == node_id:
                    out.append((node, name, src[1]))
        return out

    # ── Structural edits ────────────────────────────────────────────────

    def bypass(self, node_id: str) -> None:
        """Removes a node the way ComfyUI's bypass mode does.

        Each consumer of an output is rewired to the node's own input of the
        same type (the frontend's rule), falling back to the input at the
        same slot index. A consumer that cannot be rewired loses the input,
        exactly as it would on the frontend.
        """
        node = self.nodes.pop(node_id)
        for consumer, name, slot in self.consumers_of(node_id):
            out_type = node.output_types[slot] if slot < len(node.output_types) else ""
            replacement: Source | Literal | None = None
            for in_name, src in node.inputs.items():
                if node.input_types.get(in_name) == out_type:
                    replacement = src
                    break
            if replacement is None:
                by_index = list(node.inputs.values())
                if slot < len(by_index):
                    replacement = by_index[slot]
            if replacement is None:
                consumer.inputs.pop(name, None)
            else:
                consumer.inputs[name] = replacement

    def prune_unreachable(self) -> list[str]:
        """Drops every node no output node depends on. Returns what was dropped."""
        keep: set[str] = set()
        stack = [n.id for n in self.nodes.values() if n.type in OUTPUT_TYPES]
        if not stack:
            raise GraphError("graph has no output node; nothing would execute")
        while stack:
            nid = stack.pop()
            if nid in keep:
                continue
            keep.add(nid)
            for src in self.nodes[nid].inputs.values():
                if isinstance(src, tuple):
                    stack.append(src[0])
        dropped = [nid for nid in self.nodes if nid not in keep]
        for nid in dropped:
            del self.nodes[nid]
        return dropped

    # ── Output ──────────────────────────────────────────────────────────

    def to_api_prompt(self) -> dict[str, Any]:
        api: dict[str, Any] = {}
        for node in self.nodes.values():
            inputs: dict[str, Any] = {}
            for name, src in node.inputs.items():
                inputs[name] = list(src) if isinstance(src, tuple) else src.value
            for key, value in node.widgets.items():
                if key in inputs:
                    continue
                inputs[key] = value
            entry: dict[str, Any] = {"class_type": node.type, "inputs": inputs}
            if node.title:
                entry["_meta"] = {"title": node.title}
            api[node.id] = entry
        return api


# ── Flattening ─────────────────────────────────────────────────────────────


def _links_by_id(raw: list[Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in raw or []:
        if isinstance(item, dict):
            link = item
        else:
            link_id, origin, origin_slot, target, target_slot, link_type = item[:6]
            link = {
                "id": link_id,
                "origin_id": origin,
                "origin_slot": origin_slot,
                "target_id": target,
                "target_slot": target_slot,
                "type": link_type,
            }
        out[int(link["id"])] = link
    return out


@dataclass
class _Scope:
    nodes: dict[int, dict[str, Any]]
    links: dict[int, dict[str, Any]]
    prefix: str
    parent: _Scope | None
    boundary_in: dict[int, Any] = field(default_factory=dict)
    """subgraph input index → ("link", outer scope, outer link id) | Literal | None"""

    @property
    def set_nodes(self) -> dict[str, int]:
        found: dict[str, int] = {}
        for nid, node in self.nodes.items():
            if node.get("type") == "SetNode":
                found[_constant_of(node)] = nid
        return found


def _constant_of(node: dict[str, Any]) -> str:
    named = node.get("widgets_values_named") or {}
    if "Constant" in named:
        return str(named["Constant"])
    values = node.get("widgets_values") or [""]
    return str(values[0])


class _Flattener:
    def __init__(self, graph: dict[str, Any]) -> None:
        self.graph = graph
        self.subgraphs: dict[str, dict[str, Any]] = {
            sg["id"]: sg for sg in (graph.get("definitions") or {}).get("subgraphs", [])
        }
        self.flat = FlatGraph()
        self._expanded: dict[str, tuple[_Scope, dict[str, Any]]] = {}
        self._resolving: set[tuple[str, int]] = set()

    def run(self) -> FlatGraph:
        root = _Scope(
            nodes={int(n["id"]): n for n in self.graph["nodes"]},
            links=_links_by_id(self.graph.get("links") or []),
            prefix="",
            parent=None,
        )
        self._emit_scope(root)
        return self.flat

    # ── Emission ────────────────────────────────────────────────────────

    def _emit_scope(self, scope: _Scope) -> None:
        for nid, node in scope.nodes.items():
            node_type = node["type"]
            if node_type in self.subgraphs:
                self._ensure_expanded(scope, node)
                continue
            if node_type in VIRTUAL_TYPES:
                continue
            mode = node.get("mode", 0)
            if mode in (2, 4):
                # Muted nodes vanish; bypassed nodes are folded away at
                # resolution time by whoever reads their outputs.
                continue
            flat_id = scope.prefix + str(nid)
            if flat_id in self.flat.nodes:
                continue
            inputs: dict[str, Source | Literal] = {}
            input_types: dict[str, str] = {}
            for inp in node.get("inputs", []):
                input_types[inp["name"]] = str(inp.get("type") or "")
                link = inp.get("link")
                if link is None:
                    continue
                src = self._resolve(scope, int(link))
                if src is None:
                    continue
                inputs[inp["name"]] = src
            ui_only = UI_ONLY_KEYS | UI_ONLY_KEYS_BY_CLASS.get(node_type, frozenset())
            widgets = {
                k: v
                for k, v in (node.get("widgets_values_named") or {}).items()
                if k not in ui_only
            }
            self.flat.nodes[flat_id] = FlatNode(
                id=flat_id,
                type=node_type,
                title=str(node.get("title") or ""),
                inputs=inputs,
                widgets=widgets,
                output_types=[str(o.get("type") or "") for o in node.get("outputs", [])],
                input_types=input_types,
            )

    def _ensure_expanded(
        self, scope: _Scope, node: dict[str, Any]
    ) -> tuple[_Scope, dict[str, Any]]:
        flat_id = scope.prefix + str(node["id"])
        if flat_id in self._expanded:
            return self._expanded[flat_id]
        sg = self.subgraphs[node["type"]]
        child = _Scope(
            nodes={int(n["id"]): n for n in sg.get("nodes", [])},
            links=_links_by_id(sg.get("links") or []),
            prefix=flat_id + ":",
            parent=scope,
        )
        instance_inputs = node.get("inputs", [])
        by_name = {inp["name"]: inp for inp in instance_inputs}
        named = node.get("widgets_values_named") or {}
        for index, sg_input in enumerate(sg.get("inputs", [])):
            name = sg_input["name"]
            inst = by_name.get(name)
            if inst is None and index < len(instance_inputs):
                candidate = instance_inputs[index]
                if candidate.get("name") == name:
                    inst = candidate
            if inst is not None and inst.get("link") is not None:
                child.boundary_in[index] = ("link", scope, int(inst["link"]))
            elif name in named:
                child.boundary_in[index] = Literal(named[name])
            else:
                child.boundary_in[index] = None
        self._expanded[flat_id] = (child, sg)
        self._emit_scope(child)
        return child, sg

    # ── Resolution ──────────────────────────────────────────────────────

    def _resolve(self, scope: _Scope, link_id: int) -> Source | Literal | None:
        link = scope.links.get(link_id)
        if link is None:
            raise GraphError(f"link {link_id} is not defined in scope '{scope.prefix or 'root'}'")
        origin = int(link["origin_id"])
        slot = int(link["origin_slot"])

        if origin == -10:
            entry = scope.boundary_in.get(slot)
            if entry is None:
                return None
            if isinstance(entry, Literal):
                return entry
            _, outer_scope, outer_link = entry
            return self._resolve(outer_scope, outer_link)

        node = scope.nodes.get(origin)
        if node is None:
            raise GraphError(
                f"link {link_id} originates at node {origin}, which scope "
                f"'{scope.prefix or 'root'}' does not contain"
            )
        key = (scope.prefix + str(origin), slot)
        if key in self._resolving:
            raise GraphError(f"cycle through {key[0]}")
        self._resolving.add(key)
        try:
            return self._resolve_node_output(scope, node, slot)
        finally:
            self._resolving.discard(key)

    def _resolve_node_output(
        self, scope: _Scope, node: dict[str, Any], slot: int
    ) -> Source | Literal | None:
        node_type = node["type"]

        if node_type == "GetNode":
            constant = _constant_of(node)
            set_scope, set_node = self._find_set(scope, constant)
            set_inputs = set_node.get("inputs") or []
            in_link = set_inputs[0].get("link") if set_inputs else None
            if in_link is None:
                return None
            return self._resolve(set_scope, int(in_link))

        if node_type == "Reroute":
            inputs = node.get("inputs") or []
            in_link = inputs[0].get("link") if inputs else None
            return None if in_link is None else self._resolve(scope, int(in_link))

        if node_type in self.subgraphs:
            child, sg = self._ensure_expanded(scope, node)
            for inner in child.links.values():
                if int(inner["target_id"]) == -20 and int(inner["target_slot"]) == slot:
                    return self._resolve(child, int(inner["id"]))
            return None

        mode = node.get("mode", 0)
        if mode == 2:
            return None
        if mode == 4:
            outputs = node.get("outputs") or []
            out_type = str(outputs[slot].get("type") or "") if slot < len(outputs) else ""
            inputs = node.get("inputs") or []
            for inp in inputs:
                if str(inp.get("type") or "") == out_type and inp.get("link") is not None:
                    return self._resolve(scope, int(inp["link"]))
            if slot < len(inputs) and inputs[slot].get("link") is not None:
                return self._resolve(scope, int(inputs[slot]["link"]))
            return None

        return (scope.prefix + str(node["id"]), slot)

    def _find_set(self, scope: _Scope, constant: str) -> tuple[_Scope, dict[str, Any]]:
        current: _Scope | None = scope
        while current is not None:
            sets = current.set_nodes
            if constant in sets:
                return current, current.nodes[sets[constant]]
            current = current.parent
        raise GraphError(f"GetNode '{constant}' has no matching SetNode in any enclosing scope")


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten(graph: dict[str, Any]) -> FlatGraph:
    """The frontend's own "Queue" conversion, minus the server round trip.

    Raises `GraphError` on anything the compiler cannot account for — a
    dangling boundary, an orphan `GetNode`, a cycle — so a graph edit that
    breaks the conversion breaks here, without a GPU.
    """
    flat = _Flattener(graph).run()
    for node in flat.nodes.values():
        if node.type in VIRTUAL_TYPES:
            raise GraphError(f"virtual node {node.id} ({node.type}) survived flattening")
        for name, src in node.inputs.items():
            if isinstance(src, tuple) and src[0] not in flat.nodes:
                raise GraphError(f"{node.id}.{name} references missing node {src[0]}")
    return flat


def graph_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── Frame lattice ───────────────────────────────────────────────────────────


def frames_for_seconds(seconds: float, fps: int = FRAME_RATE) -> int:
    """`fps × seconds + 1` — the count the T2V and FLF graphs compute.

    Every product duration (5/10/15/30 s at 24 fps) lands on the model's
    8k+1 lattice: 121, 241, 361, 721. Anything else is refused rather than
    silently rounded, because the graph would round it and the delivered
    length would differ from the promised one.
    """
    frames = int(round(fps * seconds)) + 1
    if (frames - 1) % 8 != 0:
        raise GraphError(f"{seconds}s at {fps} fps is {frames} frames, not on the 8k+1 lattice")
    return frames


def character_frames_for_seconds(seconds: float, fps: int = FRAME_RATE) -> int:
    """The character graph's own formula: `round((fps·s − 1) / 8) · 8 + 1`."""
    return int(round((fps * seconds - 1) / 8.0)) * 8 + 1


# ── Canvas ──────────────────────────────────────────────────────────────────

#: Core `ResolutionSelector` labels (docs.comfy.org/built-in-nodes/ResolutionSelector,
#: verified 5 Sep 2026). There is no 4:5, so the product does not offer it.
ASPECT_LABELS: dict[str, str] = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}


def aspect_label_for(ratio: str, options: list[str] | None = None) -> str:
    """The selector's label for a product aspect ratio.

    With `options` (the live combo list from `/object_info`) the label is
    whatever the installed node actually offers for that ratio; without it,
    the documented table. Either way an unknown ratio is refused.
    """
    ratio = ratio.strip()
    if options:
        for option in options:
            if option == ratio or option.startswith(ratio + " "):
                return option
        raise GraphError(f"aspect ratio {ratio!r} is not offered by ResolutionSelector: {options}")
    try:
        return ASPECT_LABELS[ratio]
    except KeyError as exc:
        raise GraphError(f"aspect ratio {ratio!r} has no ResolutionSelector label") from exc


def oriented_canvas(
    budget: tuple[int, int], *, source_width: int | None, source_height: int | None
) -> tuple[int, int]:
    """The character graph's pixel budget, oriented like the source.

    The pack pins 736×1280 (portrait) and its own note lists both
    orientations as user settings. Same pixels, source's orientation; a
    square-ish source keeps the pack's portrait default.
    """
    short, long = sorted(budget)
    if source_width and source_height and source_width > source_height:
        return long, short
    return short, long


# ── Edits ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SeedPlan:
    base: int

    def for_index(self, index: int) -> int:
        # Distinct per RandomNoise node, stable per job, inside ComfyUI's range.
        return (self.base + index * 7_919) % (2**48)


def set_prompts(flat: FlatGraph, positive: str, negative: str) -> None:
    """Fills the positive and negative text encoders.

    Graphs 01 and 02 title them; graph 03 does not, so the encoders are found
    through the `LTXVConditioning` node they feed.
    """
    encoders = flat.of_type("CLIPTextEncode")
    if len(encoders) != 2:
        raise GraphError(f"expected two CLIPTextEncode nodes, found {len(encoders)}")
    pos = [n for n in encoders if "positive" in n.title.lower()]
    neg = [n for n in encoders if "negative" in n.title.lower()]
    if len(pos) != 1 or len(neg) != 1:
        conditioning = flat.one_of_type("LTXVConditioning")
        pos_src = conditioning.inputs.get("positive")
        neg_src = conditioning.inputs.get("negative")
        if not isinstance(pos_src, tuple) or not isinstance(neg_src, tuple):
            raise GraphError("cannot tell the positive encoder from the negative one")
        pos = [flat.nodes[pos_src[0]]]
        neg = [flat.nodes[neg_src[0]]]
    pos[0].widgets["text"] = positive
    neg[0].widgets["text"] = negative


def set_clip_length(flat: FlatGraph, seconds: float) -> None:
    """The "Clip Length ( in seconds )" slider on graphs 01 and 02."""
    slider = flat.one_titled("Clip Length", "mxSlider")
    slider.widgets["Xi"] = int(round(seconds))
    slider.widgets["Xf"] = float(seconds)
    slider.widgets["isfloatX"] = 0


def set_aspect(flat: FlatGraph, label: str) -> None:
    flat.one_of_type("ResolutionSelector").widgets["aspect_ratio"] = label


def set_seeds(flat: FlatGraph, plan: SeedPlan) -> list[int]:
    """Every `RandomNoise` gets its own seed; returns them in node order."""
    seeds: list[int] = []
    for index, node in enumerate(flat.of_type("RandomNoise")):
        seed = plan.for_index(index)
        node.widgets["noise_seed"] = seed
        seeds.append(seed)
    if not seeds:
        raise GraphError("graph has no RandomNoise node")
    return seeds


def set_output_prefix(flat: FlatGraph, prefix: str) -> None:
    node = flat.one_of_type("VHS_VideoCombine")
    node.widgets["filename_prefix"] = prefix
    node.widgets["save_output"] = True


def set_load_image(flat: FlatGraph, title_fragment: str, filename: str) -> None:
    flat.one_titled(title_fragment, "LoadImage").widgets["image"] = filename


def set_only_load_image(flat: FlatGraph, filename: str) -> None:
    flat.one_of_type("LoadImage").widgets["image"] = filename


def set_load_video(flat: FlatGraph, filename: str) -> None:
    flat.one_of_type("VHS_LoadVideoFFmpeg").widgets["video"] = filename


def set_int_constant(flat: FlatGraph, title_fragment: str, value: int) -> None:
    flat.one_titled(title_fragment, "INTConstant").widgets["value"] = int(value)


def drop_last_frame(flat: FlatGraph) -> None:
    """Runs graph 02 with the first frame only.

    The last frame is consumed by the subgraph's `LTXVImgToVideoInplaceKJ`
    node (both stills, indices 0 and −1, on the stage-1 latent). Bypassing it
    is the ComfyUI gesture for "no last frame": the empty latent passes
    straight through to the audio/video concat, and the stage-2 first-frame
    conditioning (`LTXVImgToVideoInplace`, strength 0.8) keeps the first
    frame pinned. The second loader and its resize node become unreachable
    and are pruned, so no placeholder filename is ever validated.
    """
    flat.bypass(flat.one_of_type("LTXVImgToVideoInplaceKJ").id)


# ── Per-graph compilers ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationEdits:
    """Runtime inputs for graphs 01 and 02."""

    positive: str
    negative: str
    seconds: float
    aspect_label: str
    seed_base: int
    filename_prefix: str
    first_image: str | None = None
    """Filename inside ComfyUI's input directory; graph 02 only."""
    last_image: str | None = None
    """Graph 02 only. None bypasses the last-frame conditioning."""


def compile_text_to_video(graph: dict[str, Any], edits: GenerationEdits) -> dict[str, Any]:
    flat = flatten(graph)
    set_prompts(flat, edits.positive, edits.negative)
    set_clip_length(flat, edits.seconds)
    set_aspect(flat, edits.aspect_label)
    set_seeds(flat, SeedPlan(edits.seed_base))
    set_output_prefix(flat, edits.filename_prefix)
    flat.prune_unreachable()
    return flat.to_api_prompt()


def compile_first_last_frame(graph: dict[str, Any], edits: GenerationEdits) -> dict[str, Any]:
    if not edits.first_image:
        raise GraphError("first/last frame graph needs a first image")
    flat = flatten(graph)
    set_prompts(flat, edits.positive, edits.negative)
    set_clip_length(flat, edits.seconds)
    set_aspect(flat, edits.aspect_label)
    set_seeds(flat, SeedPlan(edits.seed_base))
    set_output_prefix(flat, edits.filename_prefix)
    set_load_image(flat, "Load Image1", edits.first_image)
    if edits.last_image:
        set_load_image(flat, "Load Image2", edits.last_image)
    else:
        drop_last_frame(flat)
    flat.prune_unreachable()
    return flat.to_api_prompt()


@dataclass(frozen=True)
class ReplacementEdits:
    """Runtime inputs for graph 03."""

    positive: str
    negative: str
    video: str
    """Source clip filename inside ComfyUI's input directory."""
    image: str
    """Reference character image filename inside ComfyUI's input directory."""
    seconds: int
    width: int
    height: int
    seed_base: int
    filename_prefix: str


def compile_character_replacement(graph: dict[str, Any], edits: ReplacementEdits) -> dict[str, Any]:
    flat = flatten(graph)
    set_prompts(flat, edits.positive, edits.negative)
    set_load_video(flat, edits.video)
    set_only_load_image(flat, edits.image)
    set_int_constant(flat, "Set Length", edits.seconds)
    set_int_constant(flat, "Set Width", edits.width)
    set_int_constant(flat, "Set Height", edits.height)
    set_seeds(flat, SeedPlan(edits.seed_base))
    set_output_prefix(flat, edits.filename_prefix)
    flat.prune_unreachable()
    return flat.to_api_prompt()


# ── Verification against a live server ─────────────────────────────────────


def class_types_of(api: dict[str, Any]) -> set[str]:
    return {entry["class_type"] for entry in api.values()}


def combo_options(
    object_info: dict[str, Any], class_type: str, input_name: str
) -> list[str] | None:
    """The option list of a COMBO input, or None when it is not a combo."""
    spec = object_info.get(class_type)
    if not spec:
        return None
    for section in ("required", "optional"):
        entry = (spec.get("input") or {}).get(section, {}).get(input_name)
        if entry is None:
            continue
        head = entry[0] if isinstance(entry, (list, tuple)) and entry else None
        if isinstance(head, list):
            return [str(o) for o in head]
        if isinstance(head, dict) and isinstance(head.get("options"), list):
            return [str(o) for o in head["options"]]
        # ComfyUI v3-schema nodes (core 0.34: ResolutionSelector and friends)
        # report `["COMBO", {"options": [...], ...}]` — verified live on the
        # GPU node, 5 Sep 2026.
        if head == "COMBO" and len(entry) > 1 and isinstance(entry[1], dict):
            options = entry[1].get("options")
            if isinstance(options, list):
                return [str(o) for o in options]
    return None


def verify_against_object_info(api: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    """Problems a live ComfyUI would raise at submit — found before submitting.

    Three classes of finding, in the order the deploy should fix them:
    a node class the server does not have (a missing node pack), an input
    name the node does not declare (a pack revision that renamed it), and a
    combo value the node does not offer — which for loader nodes is a
    missing model or LoRA file, and for `ResolutionSelector` a wrong label.
    """
    problems: list[str] = []
    for node_id, entry in api.items():
        class_type = entry["class_type"]
        spec = object_info.get(class_type)
        if spec is None:
            problems.append(f"{node_id}: node class {class_type!r} is not installed")
            continue
        declared: dict[str, Any] = {}
        for section in ("required", "optional"):
            declared.update((spec.get("input") or {}).get(section, {}))
        for name, value in entry["inputs"].items():
            if name not in declared:
                if "." in name or class_type in _DYNAMIC_INPUT_CLASSES:
                    continue
                problems.append(f"{node_id} ({class_type}): input {name!r} is not declared")
                continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                continue  # a link
            if (class_type, name) in _PER_JOB_FILE_INPUTS:
                continue  # uploaded per job, never in the catalogue at check time
            options = combo_options(object_info, class_type, name)
            if options is not None and str(value) not in options:
                problems.append(
                    f"{node_id} ({class_type}): {name}={value!r} is not among the "
                    f"{len(options)} offered values"
                )
    return problems


#: Inputs whose combo lists are the server's input directory: a job uploads
#: its own file right before submitting, so a health probe cannot expect it.
_PER_JOB_FILE_INPUTS = frozenset({("LoadImage", "image"), ("VHS_LoadVideoFFmpeg", "video")})

#: Node classes whose inputs are generated at runtime and legitimately absent
#: from `/object_info`.
_DYNAMIC_INPUT_CLASSES = frozenset(
    {
        "Power Lora Loader (rgthree)",
        "BatchImagesNode",
        "ComfyMathExpression",
        "SimpleCalculatorKJ",
        "LTXVImgToVideoInplaceKJ",
        "MathExpression|pysssss",
        "easy cleanGpuUsed",
        "easy ifElse",
        "VHS_VideoCombine",
        "VHS_LoadVideoFFmpeg",
    }
)


def model_files_referenced(api: dict[str, Any]) -> dict[str, list[str]]:
    """Every weight file a compiled prompt will ask ComfyUI to load, by loader input.

    The health check turns this into a presence check against the loader
    nodes' own combo lists — which is the only source of truth for what the
    server can see.
    """
    wanted: dict[str, list[str]] = {}
    for entry in api.values():
        for name in _MODEL_INPUTS.get(entry["class_type"], ()):
            value = entry["inputs"].get(name)
            if isinstance(value, str):
                wanted.setdefault(f"{entry['class_type']}.{name}", []).append(value)
        if entry["class_type"] == "Power Lora Loader (rgthree)":
            for key, value in entry["inputs"].items():
                if key.startswith("lora_") and isinstance(value, dict) and value.get("lora"):
                    wanted.setdefault("Power Lora Loader (rgthree).lora", []).append(
                        str(value["lora"])
                    )
    return wanted


_MODEL_INPUTS: dict[str, tuple[str, ...]] = {
    "UnetLoaderGGUF": ("unet_name",),
    "UNETLoader": ("unet_name",),
    "CLIPLoader": ("clip_name",),
    "VAELoader": ("vae_name",),
    "VAELoaderKJ": ("vae_name",),
    "LatentUpscaleModelLoader": ("model_name",),
    "LoraLoaderModelOnly": ("lora_name",),
}


def structural_summary(flat: FlatGraph) -> dict[str, Any]:
    """What a test asserts against the graph's own totals."""
    linked = sum(1 for n in flat.nodes.values() for s in n.inputs.values() if isinstance(s, tuple))
    return {
        "nodes": len(flat.nodes),
        "links": linked,
        "types": sorted({n.type for n in flat.nodes.values()}),
    }


def canvas_pixels(width: int, height: int) -> float:
    return width * height / 1e6


def megapixel_canvas(ratio: str, megapixels: float = 0.9, multiple: int = 32) -> tuple[int, int]:
    """What the selector's arithmetic yields — for documentation and tests only.

    The server computes the real value; this mirrors the documented rule
    (`sqrt(mp·1e6·w/h)` rounded to the nearest multiple) so a benchmark can
    label a run before the file exists.
    """
    num, den = (int(p) for p in ratio.split(":"))
    width = math.sqrt(megapixels * 1e6 * num / den)
    height = width * den / num
    round_to = lambda v: int(round(v / multiple)) * multiple  # noqa: E731
    return round_to(width), round_to(height)
