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
from worker.adapters.mock import MockAdapter

_ADAPTERS: dict[str, GenerationAdapter] = {
    "mock": MockAdapter(),
    # M2: "ltx": LtxAdapter(), etc. Registered here and nowhere else.
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
