"""ComfyUI as a generation service.

The client H3 pack runs inside a pinned ComfyUI (v0.33.3 + two custom-node
commits) that the worker treats exactly the way it treats ACE-Step: a
long-lived local service that holds the weights and answers HTTP. The worker
never manages its lifecycle, never imports its code, and never reaches around
it to the GPU.

Two modules:

  * `graph`  — turns a frozen client workflow JSON (UI format) into the API
    prompt ComfyUI executes, applying only the edit points the pack itself
    sanctions. The conversion is the one proven on the GPU on 25 Aug 2026.
  * `client` — submit / wait / collect / interrupt / health against the
    ComfyUI HTTP API, with the same cancellation contract the other adapters
    honour.
"""

from worker.comfy.client import ComfyClient, ComfyError, evict_comfy_vram
from worker.comfy.graph import GraphEdits, load_graph, to_api_prompt

__all__ = [
    "ComfyClient",
    "ComfyError",
    "evict_comfy_vram",
    "GraphEdits",
    "load_graph",
    "to_api_prompt",
]
