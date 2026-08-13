"""One progress vocabulary for every long-form workflow.

A four-minute music video is fifteen model passes, a stitch, a mux and a
validation pass. Reported naively that is fifteen bars each running 0→100, and
the customer reads it as the job restarting over and over.

`StageReporter` fixes that structurally rather than by convention:

  * **Progress only ever moves forward.** Every value is clamped against the
    highest already sent, so a stage that computes a lower number reports the
    previous one instead of walking the bar backwards.
  * **Status only ever moves forward.** The API ranks lifecycle states strictly
    and rejects a backwards transition, which does not just look wrong — a
    rejected report is treated as a lost lease and the job is abandoned. Any
    adapter that hopped into `post_processing` per segment would kill its own
    job, so the ranking is enforced here where it cannot be forgotten.
  * **Named stages, not numbers.** Callers say `stitching()`, not `(88,
    "post_processing", "…")`. The number and the wording live in one place, so
    every workflow's bar behaves identically and the customer-facing copy can
    be changed once.

Nothing here names a model, a provider, a GPU or a file format. The strings in
this module are read by customers.
"""

from __future__ import annotations

from worker.adapters.base import ProgressCallback

#: Lifecycle statuses in the order the API ranks them. Anything not listed
#: sorts last, which keeps an unknown status from silently going backwards.
_RANK: dict[str, int] = {
    "preparing": 0,
    "generating": 1,
    "post_processing": 2,
    "uploading": 3,
}

#: The generation band. Everything a model does is compressed into it, because
#: leaving `generating` and coming back is the illegal transition above.
GENERATE_FROM = 15
GENERATE_TO = 85


class StageReporter:
    """A monotonic, vocabulary-owning wrapper around the runner's callback."""

    def __init__(self, on_progress: ProgressCallback) -> None:
        self._on_progress = on_progress
        self._progress = 0
        self._status = "preparing"

    @property
    def progress(self) -> int:
        """The highest value reported so far."""
        return self._progress

    async def report(self, status: str, progress: int, message: str) -> None:
        """The one exit. Clamps both axes forward, then forwards the report."""
        if _RANK.get(status, len(_RANK)) < _RANK.get(self._status, len(_RANK)):
            status = self._status
        self._status = status
        self._progress = max(self._progress, min(100, progress))
        await self._on_progress(status, self._progress, message)

    # ── Named stages ─────────────────────────────────────────────────────

    async def preparing(self, message: str = "Setting up your generation…") -> None:
        await self.report("preparing", 8, message)

    async def probing(self, message: str = "Reading your file…") -> None:
        await self.report("preparing", 12, message)

    async def generating(
        self,
        progress: int,
        message: str = "This usually takes a couple of minutes.",
    ) -> None:
        await self.report("generating", progress, message)

    async def section(self, index: int, total: int, progress: int) -> None:
        """Per-segment copy. Silent about sections when there is only one.

        A single-pass job announcing "Section 1 of 1" would expose machinery
        for no benefit; a four-minute job with no counter looks stuck.
        """
        message = (
            f"Generating section {index} of {total}…"
            if total > 1
            else "This usually takes a couple of minutes."
        )
        await self.generating(progress, message)

    async def stitching(self, message: str = "Assembling your video…") -> None:
        await self.report("post_processing", 88, message)

    async def muxing(self, message: str = "Adding your audio…") -> None:
        await self.report("post_processing", 92, message)

    async def finalizing(self, message: str = "Finishing up…") -> None:
        await self.report("post_processing", 94, message)

    async def uploading(self, message: str = "Almost ready…") -> None:
        await self.report("uploading", 96, message)


def band_for(index: int, total: int) -> tuple[int, int]:
    """The slice of the generating band belonging to segment `index` of `total`.

    Integer arithmetic on the ends rather than a running counter, so the last
    segment always finishes exactly at `GENERATE_TO` no matter how the
    divisions round.
    """
    span = GENERATE_TO - GENERATE_FROM
    return (
        GENERATE_FROM + span * index // total,
        GENERATE_FROM + span * (index + 1) // total,
    )
