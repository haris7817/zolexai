"""Positional widget values → named inputs, the way ComfyUI's frontend does it.

The three graphs in the client's pack were exported by a frontend that wrote
`widgets_values_named` on every node — a name → value dict the compiler can
read directly. Newer frontends do not write it: the client's FAST 1080
workflow (frontend 1.48.7, 7 Sep 2026) carries only the positional
`widgets_values` array, and the compiler emitted no inputs at all for it, so
every node failed the server's validation with "Required input is missing".

This module reads the positional array the way the browser does, against the
server's own `/object_info`:

* walk the class's `required` then `optional` inputs in declaration order;
* skip the ones that can only arrive over a link (MODEL, LATENT, VAE …);
* take one value per widget slot, in order;
* a slot marked `control_after_generate` (a seed) or `image_upload` (a file
  picker) is followed by one extra value in the array — the
  "fixed"/"randomize" mode, or the upload button's state — which the server
  does not accept as an input and which is therefore consumed and dropped;
* a `COMFY_DYNAMICCOMBO` slot selects one of its `options` by the value just
  read, and that option's own `inputs` consume the values that follow —
  submitted as `<combo>.<input>`, which is how the server names them —
  before the walk continues with the next top-level slot.

That last rule is not a nicety. `ResizeImageMaskNode` in the client's
workflow reads `["scale longer dimension", 1536, "lanczos"]` for what look
like two widgets; the middle value belongs to the chosen option, and a
resolver that does not expand it would silently write 1536 into
`scale_method`.

Nothing here changes a graph that has `widgets_values_named` — that path is
untouched, so the pack's three graphs still compile byte for byte.
"""

from __future__ import annotations

from typing import Any

#: Types that arrive over a link and never occupy a widget slot.
LINK_ONLY = frozenset(
    {
        "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
        "AUDIO", "VIDEO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET",
        "STYLE_MODEL", "CLIP_VISION", "CLIP_VISION_OUTPUT", "GLIGEN", "UPSCALE_MODEL",
        "PHOTOMAKER", "WEBCAM", "BOOLEAN_LINK", "*",
    }
)

#: Scalar types that do occupy a widget slot.
SCALAR = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})

DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO"
MATCH_TYPE = "COMFY_MATCHTYPE"


def is_widget(spec_type: Any) -> bool:
    """Does this input occupy a positional widget slot?"""
    if isinstance(spec_type, list):
        return True  # a combo, given as its list of options
    text = str(spec_type)
    if text.startswith(DYNAMIC_COMBO):
        return True
    if text.startswith(MATCH_TYPE):
        return False
    parts = {p.strip() for p in text.split(",") if p.strip()}
    if parts and parts <= SCALAR:
        return True
    return not (parts & LINK_ONLY) and parts <= SCALAR


def _trailing(options: dict[str, Any]) -> int:
    """Extra positional values a widget carries that are not server inputs.

    A seed widget is followed by its `control_after_generate` mode, and a
    file picker by its upload-button state. Both sit in `widgets_values` and
    neither is accepted by the server.
    """
    return 1 if (options.get("control_after_generate") or options.get("image_upload")) else 0


def ordered_inputs(class_spec: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """(name, type, options) for every input, required first, in order.

    Accepts a class spec (which keys its sections under `input`) or one
    option of a dynamic combo (which uses `inputs`).
    """
    out: list[tuple[str, Any, dict[str, Any]]] = []
    spec = class_spec.get("input") or class_spec.get("inputs") or {}
    for section in ("required", "optional"):
        for name, definition in (spec.get(section) or {}).items():
            if not isinstance(definition, list) or not definition:
                continue
            second = definition[1] if len(definition) > 1 else None
            options = second if isinstance(second, dict) else {}
            out.append((name, definition[0], options))
    return out


def resolve(
    class_type: str, values: list[Any], catalogue: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Positional `widgets_values` → named inputs for one node.

    Returns the inputs and a list of complaints — an unknown class, or values
    left over after the walk, which means the class's widget layout is not
    what this resolver expects. A complaint is never silent: the caller logs
    or raises, because a mis-aligned widget writes a plausible wrong number
    into a real parameter.
    """
    spec = catalogue.get(class_type)
    if spec is None:
        return {}, [f"{class_type}: not offered by the server"]

    inputs: dict[str, Any] = {}
    index = 0
    for name, spec_type, options in ordered_inputs(spec):
        if index >= len(values):
            break
        if not is_widget(spec_type):
            continue
        value = values[index]
        index += 1
        inputs[name] = value
        index += _trailing(options)
        if str(spec_type).startswith(DYNAMIC_COMBO):
            for option in options.get("options") or []:
                if option.get("key") != value:
                    continue
                # The chosen option's own inputs, required then optional —
                # the same order the browser lays them out.
                for sub_name, sub_type, sub_options in ordered_inputs(option):
                    if index >= len(values):
                        break
                    if not is_widget(sub_type):
                        continue
                    # The server addresses an option's input by the combo it
                    # belongs to: `resize_type.longer_size`, not `longer_size`
                    # (its own validation error names it that way, and the
                    # workflow's linked inputs use the same form).
                    inputs[f"{name}.{sub_name}"] = values[index]
                    index += 1
                    index += _trailing(sub_options)
                break

    complaints = []
    if index < len(values):
        complaints.append(
            f"{class_type}: {len(values) - index} widget value(s) left over "
            f"({values[index:]!r}) — the layout is not what was expected"
        )
    return inputs, complaints


def subgraph_widget_slots(definition: dict[str, Any]) -> list[int]:
    """Which of a subgraph's boundary inputs occupy a widget slot.

    A subgraph instance's `widgets_values` covers the promoted inputs, in the
    definition's own order, and only those that land on an inner node's
    widget. The instance's `inputs` array cannot be used for this: in the
    client's workflow the `Preprocess` instance omits `img_compression`
    entirely while its `widgets_values` still carries the value.
    """
    inner = {int(n["id"]): n for n in definition.get("nodes") or []}
    from_boundary: dict[int, tuple[int, int]] = {}
    for link in definition.get("links") or []:
        if int(link.get("origin_id", 0)) != -10:
            continue
        slot = int(link.get("origin_slot", 0))
        from_boundary.setdefault(slot, (int(link["target_id"]), int(link["target_slot"])))

    slots: list[int] = []
    for index, _ in enumerate(definition.get("inputs") or []):
        target = from_boundary.get(index)
        if target is None:
            continue
        node = inner.get(target[0])
        if node is None:
            continue
        node_inputs = node.get("inputs") or []
        if target[1] < len(node_inputs) and node_inputs[target[1]].get("widget"):
            slots.append(index)
    return slots
