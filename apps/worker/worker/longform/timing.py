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

import math

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
    candidates are only ever taken from *before* a nominal boundary that is
    itself never more than one pass past where the previous cut landed.

    **The pass count is decided first, and every window is then near the same
    length.** Filling `per_pass_seconds` greedily and letting whatever is left
    be the final section produced slivers on real tracks, measured 2026-08-21:
    a 60.02-second track at a 20s ceiling planned 19.77 / 19.89 / 17.90 /
    **2.46**, and a 300.04-second track at a 60s ceiling planned five full
    windows and then **0.18 seconds**. The count is the same either way — that
    is set by the ceiling — but a 0.18-second section costs the same 22B
    transformer load as a sixty-second one, and delivers four frames for it.

    `plan_segments` already avoids this for even windows and says why. This is
    the same argument for the case where the cuts come from the music: nominal
    cuts at `total · k / count`, each pulled back to an onset within a bounded
    tolerance. A track with no usable onsets degrades to exactly `plan_segments`'
    answer, because it picks the count from the same two numbers.
    """
    if total_seconds <= per_pass_seconds:
        return []

    # The same arithmetic `plan_segments` uses, deliberately: cuts on the music
    # are a preference about WHERE a seam lands, never a reason to pay for a
    # section the even plan would not have needed.
    count = math.ceil(total_seconds / per_pass_seconds - 1e-9)
    nominal = total_seconds / count
    pull = max(0.0, min(max_pull_fraction, 0.5)) * nominal
    minimum = _minimum_window(per_pass_seconds)
    boundaries: list[float] = []
    position = 0.0

    for index in range(1, count):
        # `latest` is never past this cut's nominal position and never more
        # than one pass after the previous cut, which together are the
        # invariant that matters: no window this produces is longer than one
        # pass, so a timing preference can never hand the GPU an oversized
        # render.
        latest = min(total_seconds * index / count, position + per_pass_seconds)
        # …and never so EARLY that what is left cannot fit in the passes that
        # remain. Without this the deficit from each backward pull has nowhere
        # to go but the final window, which then quietly exceeds the ceiling —
        # the exact oversized request the ceiling exists to prevent. A track
        # that fills its passes exactly has no slack and gets no pull, which is
        # the honest answer: moving a cut there would mean buying a pass.
        earliest = min(
            latest,
            max(
                position + minimum,
                latest - pull,
                total_seconds - per_pass_seconds * (count - index),
            ),
        )

        # The LAST onset in the window keeps sections as long as possible, so
        # a cut is pulled back to the music rather than dragged to the front of
        # its own tolerance.
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
