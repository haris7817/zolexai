"""Which seconds of a track are actually sung — from the vocal stem itself.

The music-video prompt used to command performance unconditionally, so the
singer mouthed words straight through instrumental passages — the client's
27 Aug frame-audit measured mouth-vs-audio correlation at 0.027, and the
give-away was her singing through a forty-second beats-only intro. Simple
band-energy heuristics cannot separate a vocal from a dense pop mix (measured
on that same track: no feature separated the known regions), and ACE-Step
returns no timing. Stem separation does it properly: split the vocals out of
the mix once per job, and the stem's own loudness envelope IS the answer,
genre-proof, identical for uploaded and generated songs.

The separator lives in its own venv (`vocal_separator_python`), invoked as a
subprocess exactly the way every other heavy tool here is — the worker's own
environment stays light and the pinned pipeline venvs stay untouched. Every
failure path returns None and the caller keeps today's behaviour: this layer
improves prompts, it must never fail a job.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from worker.core.config import settings
from worker.media.audio import audio_envelope
from worker.media.ffmpeg import FfmpegError

logger = logging.getLogger("zolexai.worker.vocals")

#: Envelope hop — inherited from audio_envelope's default resolution.
_HOP_SECONDS = 0.05

#: A hop counts as sung when the stem's RMS clears both an absolute floor
#: (stem silence is truly near-zero — separation leakage sits well below
#: this) and a fraction of the stem's own loud parts, which adapts the
#: threshold to however hot the mix was mastered.
_ABS_FLOOR = 0.015
_REL_FRACTION = 0.15

#: Post-processing: a breath between lines is not an instrumental, and a
#: single leaked drum hit is not a vocal.
_BRIDGE_GAP_SECONDS = 2.0
_MIN_SPAN_SECONDS = 1.5


def spans_from_envelope(
    envelope: list[float],
    *,
    hop_seconds: float = _HOP_SECONDS,
    abs_floor: float = _ABS_FLOOR,
    rel_fraction: float = _REL_FRACTION,
    bridge_gap_seconds: float = _BRIDGE_GAP_SECONDS,
    min_span_seconds: float = _MIN_SPAN_SECONDS,
) -> list[tuple[float, float]]:
    """(start, end) spans where the vocal stem is audibly active."""
    if not envelope:
        return []
    loud = sorted(envelope)[int(len(envelope) * 0.95)]
    threshold = max(abs_floor, rel_fraction * loud)
    active = [value >= threshold for value in envelope]

    spans: list[list[float]] = []
    for index, on in enumerate(active):
        time = index * hop_seconds
        if on:
            if spans and time - spans[-1][1] <= bridge_gap_seconds:
                spans[-1][1] = time + hop_seconds
            else:
                spans.append([time, time + hop_seconds])
    return [
        (start, end) for start, end in spans if end - start >= min_span_seconds
    ]


def vocal_fraction(
    spans: list[tuple[float, float]], start: float, end: float
) -> float:
    """How much of [start, end) is sung, 0..1."""
    if end <= start:
        return 0.0
    covered = 0.0
    for span_start, span_end in spans:
        covered += max(0.0, min(end, span_end) - max(start, span_start))
    return covered / (end - start)


async def vocal_activity(track: Path) -> list[tuple[float, float]] | None:
    """Sung spans of `track`, or None when separation is unavailable.

    None means "no opinion" — the caller must behave exactly as it did
    before this module existed.
    """
    python = settings.vocal_separator_python
    if python is None:
        return None
    with tempfile.TemporaryDirectory(prefix="vocal-sep-") as scratch:
        out = Path(scratch)
        try:
            process = await asyncio.create_subprocess_exec(
                str(python), "-m", "demucs.separate",
                "--two-stems", "vocals",
                "-n", "htdemucs",
                "-o", str(out),
                str(track),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=settings.vocal_separator_timeout
                )
            except TimeoutError:
                process.kill()
                logger.warning("vocal_separation_timed_out")
                return None
            if process.returncode != 0:
                logger.warning(
                    "vocal_separation_failed",
                    extra={"detail": (stderr or b"")[-500:].decode(errors="replace")},
                )
                return None
        except OSError as exc:
            logger.warning("vocal_separation_failed", extra={"detail": str(exc)})
            return None
        stems = list(out.glob("htdemucs/*/vocals.wav"))
        if not stems:
            logger.warning("vocal_separation_no_stem")
            return None
        try:
            envelope = await audio_envelope(stems[0], hop_seconds=_HOP_SECONDS)
        except FfmpegError as exc:
            logger.warning("vocal_stem_unreadable", extra={"detail": str(exc)})
            return None
    return spans_from_envelope(envelope)
