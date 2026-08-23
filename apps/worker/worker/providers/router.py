"""Which engine serves a job — and, today, deliberately only one answer.

`auto` resolves to LTX for every workflow. That is not a placeholder to be
filled in later by whoever touches this file next; it is the finding of the
audit. No comparison has been run, so there is no evidence to route on, and a
router that guessed would quietly become the decision nobody made.

The override exists so the benchmark can put the same job through both engines
on demand. It is internal: QA sets it, the public API does not expose it, and
a request that asks for an engine this node cannot run is refused with the
reason rather than silently served by the other one — a silent fallback would
make an A/B compare LTX against LTX and call it a tie.
"""

from __future__ import annotations

from worker.adapters.base import AdapterError, AdapterJob
from worker.core.logging import get_logger
from worker.providers.base import VideoProvider
from worker.providers.h3 import H3Provider
from worker.providers.ltx import LtxProvider

logger = get_logger(__name__)

AUTO = "auto"

_PROVIDERS: dict[str, VideoProvider] = {
    "ltx": LtxProvider(),
    "h3": H3Provider(),
}

#: The routing table `auto` reads. Every entry is LTX because every entry is
#: unmeasured; the benchmark's final decision matrix is what may change these,
#: and each change should arrive with the evidence in the commit message.
_AUTO_ROUTES: dict[str, str] = {
    "text-to-video": "ltx",
    "image-to-video": "ltx",
    "video-to-video": "ltx",
    "extend-video": "ltx",
    "music-video": "ltx",
}


def get_provider(name: str) -> VideoProvider:
    provider = _PROVIDERS.get(name)
    if provider is None:
        raise AdapterError(
            "This tool is temporarily unavailable.",
            internal_detail=(
                f"no provider registered as '{name}'. Available: {sorted(_PROVIDERS)}"
            ),
            retriable=False,
        )
    return provider


def requested_provider(job: AdapterJob) -> str:
    """The override, from either side of the job, normalised.

    `execution.provider` is the workflow-level lever (a YAML edit, the way
    every other tier switch works); `parameters.provider` is the per-request
    one the benchmark harness uses. The request wins, because a QA run has to
    be able to override a node's own default without editing its YAML.
    """
    for source in (job.parameters.get("provider"), job.execution.get("provider")):
        value = str(source or "").strip().lower()
        if value:
            return value
    return AUTO


def resolve(job: AdapterJob) -> tuple[str, VideoProvider]:
    """(name, provider) for this job. Never falls back silently."""
    requested = requested_provider(job)
    if requested == AUTO:
        name = _AUTO_ROUTES.get(job.workflow_id, "ltx")
    else:
        name = requested

    provider = get_provider(name)
    if requested != AUTO:
        logger.info(
            "provider_override",
            extra={"workflow_id": job.workflow_id, "provider": name},
        )
    return name, provider


def auto_routes() -> dict[str, str]:
    """The current routing table, for documentation and tests to assert on."""
    return dict(_AUTO_ROUTES)


def compare(job: AdapterJob) -> dict[str, object]:
    """Both engines' plans for one job, for the benchmark and for review.

    A provider that refuses records its reason instead of a manifest: "H3
    cannot do this" is a finding, not an error, and the comparison table needs
    it as much as it needs a section count.
    """
    from worker.providers.base import ProviderRefusal

    out: dict[str, object] = {}
    for name, provider in _PROVIDERS.items():
        try:
            out[name] = provider.compile(job).to_dict()
        except ProviderRefusal as refusal:
            out[name] = {"refused": refusal.reason, "capability": refusal.capability}
    return out
