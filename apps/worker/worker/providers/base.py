"""The engine-facing contract: what a video provider must be able to answer.

ZolexAI already had one provider abstraction — `GenerationAdapter`, selected
by `execution.runtime` — and it is the right seam for *running* a job. This
one sits beside it and answers the questions a second engine forces on us:

  * what can you structurally do (`capabilities`);
  * would you accept this request, and why not (`validate`);
  * what exactly would you send the model (`compile`);
  * are you usable on this node right now (`health`).

`compile` is the important one and it is the reason this layer exists at all.
A benchmark that cannot say what each engine was asked to do is a beauty
contest, not a measurement — and a manifest that can be produced without a GPU
is also the only way to prove, from a laptop, that adding a second engine did
not move the first one by a byte.

Generation itself deliberately stays where it already works: `LtxProvider`
delegates to the existing `LtxAdapter`, and nothing in this package is on the
path of a running job unless a workflow asks for it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.adapters.base import AdapterJob, AdapterResult, ProgressCallback
from worker.providers.capabilities import Capability
from worker.providers.manifest import GenerationManifest


class ProviderUnavailable(Exception):
    """The provider cannot run here — no weights, no service, no licence.

    Distinct from a refusal: an unavailable provider is a deployment fact,
    and the router may fall back past it. A refusal is about the request.
    """


class ProviderRefusal(Exception):
    """This provider will not accept THIS request, with a reason.

    Carries the capability that failed so a refusal reads as "H3 cannot render
    31 seconds in one pass" rather than as a generic validation error.
    """

    def __init__(self, reason: str, *, capability: str = "") -> None:
        self.reason = reason
        self.capability = capability
        super().__init__(reason)


@runtime_checkable
class VideoProvider(Protocol):
    name: str

    def capabilities(self) -> dict[str, Capability]:
        """This engine's column of the matrix, keyed the same for both."""
        ...

    def validate(self, job: AdapterJob) -> list[str]:
        """Reasons this provider would refuse the job. Empty means acceptable.

        A list rather than a raise, because the router wants to compare
        refusals across providers, and the benchmark wants to record them.
        """
        ...

    def compile(self, job: AdapterJob) -> GenerationManifest:
        """What this provider WOULD send, without sending it.

        Pure: no subprocess, no network, no weights, no GPU. Must raise
        `ProviderRefusal` rather than invent a plan it could not execute.
        """
        ...

    async def generate(
        self, job: AdapterJob, on_progress: ProgressCallback
    ) -> AdapterResult:
        """Run it. May raise `ProviderUnavailable` where the engine is not
        installed on this node — which is every node, for H3, today."""
        ...

    def health(self) -> tuple[bool, str]:
        """(usable, why not). Cheap and side-effect free — file existence and
        configuration only, never a model load."""
        ...
