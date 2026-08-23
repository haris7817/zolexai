"""Dual-engine preparation: LTX today, H3 under evaluation.

This package answers "which engine, and what exactly would it be sent?"
without running anything. It sits beside `worker.adapters` rather than inside
it: adapters own execution, providers own comparison, and keeping them apart
is what lets a second engine be evaluated without touching the first.

Today `auto` routes every workflow to LTX. Nothing in here decides otherwise
until the benchmark in `docs/internal/ltx-h3-comparison-framework.md` has been
run on real hardware.
"""

from worker.providers.base import (
    ProviderRefusal,
    ProviderUnavailable,
    VideoProvider,
)
from worker.providers.capabilities import (
    MATRIX,
    Capability,
    Support,
    gpu_test_rows,
    structural_winners,
)
from worker.providers.h3 import H3Provider
from worker.providers.ltx import LtxProvider
from worker.providers.manifest import (
    AudioWindow,
    GenerationManifest,
    ReferenceSpec,
    SectionPlan,
)
from worker.providers.router import (
    AUTO,
    auto_routes,
    compare,
    get_provider,
    requested_provider,
    resolve,
)

__all__ = [
    "AUTO",
    "MATRIX",
    "AudioWindow",
    "Capability",
    "GenerationManifest",
    "H3Provider",
    "LtxProvider",
    "ProviderRefusal",
    "ProviderUnavailable",
    "ReferenceSpec",
    "SectionPlan",
    "Support",
    "VideoProvider",
    "auto_routes",
    "compare",
    "get_provider",
    "gpu_test_rows",
    "requested_provider",
    "resolve",
    "structural_winners",
]
