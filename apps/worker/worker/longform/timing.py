"""Choosing where a music video is allowed to cut.

Visual sections have to end somewhere. Ending them at exact multiples of the
generation ceiling means every cut lands on an arbitrary moment of the song,
and a cut that lands mid-phrase reads as a mistake even to someone who could
not say why.

So the boundary is chosen from the music instead: nominally one pass long, then
pulled back to the most recent moment where the track's energy actually rises.
The pull is bounded, and only ever backwards — a window may end early, never
late — which is what keeps a timing decision from quietly handing the GPU a
pass longer than it survives.

**What this is not.** `detect_onsets` finds energy rises. It does not find
beats, downbeats, bars, phrases, or where the chorus starts, and this module
therefore aligns cuts to *events*, not to musical structure. The honest
description of the result is "cuts tend to land on hits rather than mid-note".
Anything stronger would be a claim the implementation does not support.
"""

from __future__ import annotations

from worker.core.logging import get_logger

logger = get_logger(__name__)

#: How far back a boundary may be pulled to reach an onset, as a fraction of
#: the nominal window. Beyond this the sections become noticeably uneven and
#: the count starts to climb, which costs GPU passes for a cut nobody notices.
_MAX_PULL_FRACTION = 0.2

#: A window shorter than this is not worth generating as its own pass — the
#: model has too little to work with and the seam arrives immediately. Capped
#: at half the pass ceiling by `_minimum_window`, because a fixed floor larger
#: than the ceiling itself would make every window illegal.
_MIN_WINDOW_SECONDS = 2.0


def _minimum_window(per_pass_seconds: float) -> float:
    return min(_MIN_WINDOW_SECONDS, per_pass_seconds / 2)


def plan_musical_boundaries(
    total_seconds: float,
    *,
    per_pass_seconds: float,
    onsets: list[float],
    max_pull_fraction: float = _MAX_PULL_FRACTION,
) -> list[float]:
    """Cut times (seconds, exclusive of 0 and `total_seconds`) for a track.

    Returns an empty list when the whole track fits in one pass, or when no
    onset falls in any window's pull range — in both cases the caller falls
    back to even windows, which is the correct answer rather than a failure.

    Every returned window is at most `per_pass_seconds` long, by construction:
    candidates are only ever taken from *before* the nominal boundary.
    """
    if total_seconds <= per_pass_seconds:
        return []

    pull = max(0.0, min(max_pull_fraction, 0.5)) * per_pass_seconds
    minimum = _minimum_window(per_pass_seconds)
    boundaries: list[float] = []
    position = 0.0

    while total_seconds - position > per_pass_seconds:
        # `latest` is never past the nominal boundary, which is the invariant
        # that matters: every window this produces is at most one pass long,
        # so a timing preference can never hand the GPU an oversized render.
        latest = position + per_pass_seconds
        earliest = max(position + minimum, latest - pull)

        # A cut that leaves an unusable sliver at the end is worse than one
        # slightly early, so pull it back when there is room to.
        if total_seconds - latest < minimum and total_seconds - minimum >= earliest:
            latest = total_seconds - minimum

        # The LAST onset in the window keeps sections as long as possible, so
        # the pass count stays near the minimum the ceiling allows.
        candidates = [time for time in onsets if earliest <= time <= latest]
        boundaries.append(max(candidates) if candidates else latest)
        position = boundaries[-1]

    if boundaries:
        logger.info(
            "musical_boundaries",
            extra={
                "sections": len(boundaries) + 1,
                "onsets_available": len(onsets),
                "cuts": [round(value, 2) for value in boundaries],
            },
        )
    return boundaries
