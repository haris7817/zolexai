"""Adapter selection.

The workflow's private `execution.runtime` decides which adapter runs a job. In
M1 every definition says `mock`. In M2, changing one line of YAML routes a
workflow to a real provider — the frontend, the public API and the job schema
are untouched, which is the entire purpose of the abstraction (directive §12).

An unknown runtime is a hard error, not a silent fallback to mock: a typo that
quietly produced placeholder images instead of real video would be far more
expensive to discover than a failed job.
"""

from __future__ import annotations

from worker.adapters.base import AdapterError, GenerationAdapter
from worker.adapters.h3_comfy import H3ComfyAdapter
from worker.adapters.harness import HarnessAdapter
from worker.adapters.ltx import LtxAdapter
from worker.adapters.mock import MockAdapter
from worker.adapters.music import MusicAdapter

_ADAPTERS: dict[str, GenerationAdapter] = {
    "mock": MockAdapter(),
    # Real media through ffmpeg, no model. Registered so it can be selected for
    # local testing; no shipped workflow points at it. See adapters/harness.py.
    "harness": HarnessAdapter(),
    # LTX-2.5 on a GPU node (NVFP4, RTX 5090). Registered but not yet routed:
    # no shipped workflow says `runtime: ltx`, and only a worker started with
    # RUNTIMES including "ltx" will ever claim such a job. See adapters/ltx.py.
    "ltx": LtxAdapter(),
    # Music generation through whichever provider MUSIC_PROVIDER selects
    # (currently ACE-Step 1.5 XL, reached over HTTP). Registered but not yet
    # routed: no shipped workflow says `runtime: music`. The provider is
    # resolved lazily on first use, so a node that never runs music pays
    # nothing for this entry. See adapters/music.py and music/provider.py.
    "music": MusicAdapter(),
    # MiniMax H3 through the pinned ComfyUI INT8 client pack (proven 25 Aug
    # 2026). Registered but not routed: no shipped workflow says
    # `runtime: h3_comfy` — the client-test environment flips the YAML line
    # for image-to-video and reference video-to-video, exactly the M2
    # mechanism. Connects to a local ComfyUI like music connects to ACE-Step.
    # See adapters/h3_comfy.py and docs/internal/h3-client-runtime-freeze.md.
    "h3_comfy": H3ComfyAdapter(),
}


def get_adapter(runtime: str) -> GenerationAdapter:
    adapter = _ADAPTERS.get(runtime)
    if adapter is None:
        raise AdapterError(
            "This tool is temporarily unavailable.",
            internal_detail=(
                f"No adapter registered for runtime '{runtime}'. "
                f"Available: {sorted(_ADAPTERS)}"
            ),
            # Retrying will not conjure an adapter — fail immediately rather
            # than burning all three attempts on a configuration mistake.
            retriable=False,
        )
    return adapter


def available_runtimes() -> list[str]:
    return sorted(_ADAPTERS)
