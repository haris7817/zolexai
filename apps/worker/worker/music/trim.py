"""Fitting a longer take to the requested length around its vocals.

## Why this exists

The music model decides its own arrangement. Asked for a two-minute song
with vocals throughout, it opens with a ten-second intro and ends with a
twenty-second instrumental tail whatever the brief says — measured on every
take of 9 Sep 2026 — and no prompt wording moved the vocal entry earlier
than about nine seconds. The client's rule (10 Sep) is explicit and
backend-enforced: intro 2–3 s, outro 3–4 s, no break longer than 2 s, at
least 90% sung.

So the workflow stops asking and starts cutting. It generates a take
LONGER than the song it needs, finds where the singing actually is (from
the vocal stem, or the transcript's word spans without a separator), and
cuts a window of exactly the requested length that begins a couple of
seconds before the first sung note and ends a few seconds after the last
one. The intro and the tail the model insisted on fall outside the window.

What this cannot fix — and reports honestly — is a long instrumental
break in the middle of the singing. That is what the retry loop and the
break gate in `verify.py` are for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.media.ffmpeg import ffmpeg

#: Seconds of music allowed before the first sung note and after the last.
#: The client's rule is 2–3 s and 3–4 s; the window aims at the middle.
INTRO_ALLOWANCE = 2.5
OUTRO_ALLOWANCE = 3.5

#: Fades at the cut points so a mid-phrase cut never clicks. The fade-out is
#: long because it is the song's ending as the listener will hear it.
FADE_IN = 0.4
FADE_OUT = 2.5


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    first_vocal: float | None
    last_vocal: float | None
    source_seconds: float

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def intro_seconds(self) -> float | None:
        return None if self.first_vocal is None else max(0.0, self.first_vocal - self.start)

    @property
    def outro_seconds(self) -> float | None:
        return None if self.last_vocal is None else max(0.0, self.end - self.last_vocal)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "source_seconds": round(self.source_seconds, 2),
            "first_vocal": None if self.first_vocal is None else round(self.first_vocal, 2),
            "last_vocal": None if self.last_vocal is None else round(self.last_vocal, 2),
            "intro_seconds": None if self.intro_seconds is None else round(self.intro_seconds, 2),
            "outro_seconds": None if self.outro_seconds is None else round(self.outro_seconds, 2),
        }


def choose_window(
    spans: list[tuple[float, float]] | None,
    *,
    source_seconds: float,
    target_seconds: float,
    intro_allowance: float = INTRO_ALLOWANCE,
    outro_allowance: float = OUTRO_ALLOWANCE,
) -> Window:
    """The `target_seconds` window of the take that holds the most singing.

    With no spans the window is the head of the take — there is nothing
    better to know. Otherwise it starts `intro_allowance` before the first
    sung note, always: the intro rule is the one the client hears first,
    and a take whose singing is shorter than the window has failed the
    coverage gate whatever the window does. When the singing runs past the
    window the ending is what is cut (a song's beginning matters more than
    its final repeat); when it falls short, the slack lands on the ending.
    """
    target = min(target_seconds, source_seconds)
    if not spans:
        return Window(0.0, target, None, None, source_seconds)
    first = spans[0][0]
    last = spans[-1][1]
    start = max(0.0, first - intro_allowance)
    end = start + target
    if end > source_seconds:
        end = source_seconds
        start = max(0.0, end - target)
    return Window(start, end, first, last, source_seconds)


async def cut_window(source: Path, destination: Path, window: Window, *, timeout: float = 600.0) -> Path:
    """Writes `window` of `source` to `destination` with fades at the cuts."""
    length = window.seconds
    fade_out_start = max(0.0, length - FADE_OUT)
    filters = [f"afade=t=in:st=0:d={FADE_IN:.2f}", f"afade=t=out:st={fade_out_start:.3f}:d={FADE_OUT:.2f}"]
    await ffmpeg(
        [
            "-y",
            "-ss", f"{window.start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(source),
            "-af", ",".join(filters),
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(destination),
        ],
        timeout=timeout,
    )
    return destination


def shift_spans(spans: list[tuple[float, float]], window: Window) -> list[tuple[float, float]]:
    """The sung spans re-based to the cut window's own clock, clipped."""
    out: list[tuple[float, float]] = []
    for start, end in spans:
        s = max(start, window.start) - window.start
        e = min(end, window.end) - window.start
        if e > s:
            out.append((s, e))
    return out


__all__ = ["FADE_IN", "FADE_OUT", "INTRO_ALLOWANCE", "OUTRO_ALLOWANCE", "Window", "choose_window", "cut_window", "shift_spans"]
