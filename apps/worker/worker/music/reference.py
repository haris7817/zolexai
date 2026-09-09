"""Reference audio: what a customer's example song is *like*, never what it is.

The client's lyrics workflow lets a customer hand over a song — an upload
or a link — and asks the platform to follow its tempo, energy, structure and
vocal cadence while writing something original. Two halves, kept apart on
purpose:

  * **Analysis** produces a `ReferenceProfile` of high-level numbers: BPM
    and how steady it is, an energy curve, where the sections change, how
    much of it is sung, how fast the singing is. That is what the blueprint
    and the production brief consume, and it is what a customer sees.
  * **The transcript** — the reference's own words — is used for exactly
    one thing: checking that the new lyrics do not copy them
    (`worker/music/gates.py`). It is kept on the profile under a private
    name, never written to the customer-facing report and never handed to
    the writer.

Everything is measured with the tools already on the node: ffmpeg for the
envelope, Demucs for the vocal stem, faster-whisper for cadence. Each of the
three degrades to "not measured" rather than failing the job; the fetch and
the basic media validation do fail it, because a reference that cannot be
read is a request that cannot be honoured.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from worker.core.config import settings
from worker.core.logging import get_logger
from worker.media import FfmpegError, audio_envelope, detect_onsets, probe_media
from worker.media.vocals import spans_from_envelope, vocal_activity, vocal_fraction
from worker.music.keys import Key, Tempo, analyse_tempo_and_key, bpm_matches
from worker.music.syllables import syllables
from worker.music.transcribe import Transcript, transcribe

logger = get_logger(__name__)

#: Hosts the reference fetcher accepts. Mirrors the API's list; a link on any
#: other host is refused there before a job exists.
REFERENCE_AUDIO_HOSTS: tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "soundcloud.com",
)

_ENVELOPE_HOP = 0.05
_CURVE_STEP_SECONDS = 5.0


class ReferenceUnavailable(RuntimeError):
    """The link could not be fetched. Code REFERENCE_UNAVAILABLE."""


class ReferenceInvalid(RuntimeError):
    """The media is unusable: no audio, silent, too short, too long.
    Code REFERENCE_AUDIO_INVALID."""


@dataclass(frozen=True)
class ReferenceProfile:
    source: str
    """"upload" or "url"."""
    duration_seconds: float
    bpm: int | None
    tempo_stability: float | None
    """0..1 — how sharply one tempo dominates the onset pattern."""
    time_signature: str
    energy_curve: tuple[float, ...]
    """Normalised loudness per `_CURVE_STEP_SECONDS`, 0..1."""
    section_boundaries: tuple[float, ...]
    vocal_density: float | None
    """Fraction of the song that is sung, from the vocal stem."""
    vocal_cadence: float | None
    """Sung syllables per second, from the transcript over the sung time."""
    words_per_second: float | None
    language: str | None
    language_probability: float | None
    mood_hint: str
    analysis: dict[str, str] = field(default_factory=dict)
    """Per-feature status: "measured" or the reason it was not."""
    key: str | None = None
    """"B major", "F# minor" — Krumhansl–Schmuckler on the chroma."""
    mode: str | None = None
    key_confidence: float | None = None
    syllables_per_bar: float | None = None
    _transcript_lines: tuple[str, ...] = ()
    """PRIVATE. The reference's own words, for the originality check only."""

    @property
    def transcript_lines(self) -> tuple[str, ...]:
        return self._transcript_lines

    def to_dict(self) -> dict[str, Any]:
        """High-level attributes only — the transcript is deliberately absent."""
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 2),
            "bpm": self.bpm,
            "tempo_stability": None if self.tempo_stability is None else round(self.tempo_stability, 3),
            "time_signature": self.time_signature,
            "energy_curve": [round(value, 3) for value in self.energy_curve],
            "section_boundaries": [round(value, 2) for value in self.section_boundaries],
            "vocal_density": None if self.vocal_density is None else round(self.vocal_density, 3),
            "vocal_cadence_syllables_per_second": (
                None if self.vocal_cadence is None else round(self.vocal_cadence, 2)
            ),
            "words_per_second": None if self.words_per_second is None else round(self.words_per_second, 2),
            "language": self.language,
            "language_probability": (
                None if self.language_probability is None else round(self.language_probability, 3)
            ),
            "mood_hint": self.mood_hint,
            "key": self.key,
            "mode": self.mode,
            "key_confidence": None if self.key_confidence is None else round(self.key_confidence, 3),
            "syllables_per_bar": None if self.syllables_per_bar is None else round(self.syllables_per_bar, 2),
            "analysis": dict(self.analysis),
        }

    def describe(self) -> str:
        """One sentence of musical direction for the production brief."""
        parts: list[str] = []
        if self.bpm:
            parts.append(f"{self.bpm} BPM")
        if self.key:
            parts.append(f"in {self.key}")
        if self.mood_hint:
            parts.append(self.mood_hint)
        if self.vocal_density is not None and self.vocal_density >= 0.75:
            parts.append("vocals nearly throughout")
        if self.vocal_cadence:
            if self.vocal_cadence >= 4.0:
                parts.append("fast, dense vocal delivery")
            elif self.vocal_cadence <= 2.2:
                parts.append("slow, spacious vocal phrasing")
        return ", ".join(parts)


def host_allowed(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").casefold().rstrip(".")
    return parsed.scheme.casefold() == "https" and any(
        host == h or host.endswith(f".{h}") for h in REFERENCE_AUDIO_HOSTS
    )


# ── Fetch ────────────────────────────────────────────────────────────────


def _fetch_python() -> str:
    configured = getattr(settings, "music_reference_fetch_python", None)
    if configured:
        return str(configured)
    candidate = settings.ltx_repo_dir / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else "python"


async def fetch_reference_url(url: str, destination: Path) -> Path:
    """Downloads the audio of a public link with yt-dlp into `destination`."""
    if not host_allowed(url):
        raise ReferenceUnavailable("the link is not on a supported host")
    destination.mkdir(parents=True, exist_ok=True)
    template = destination / "reference.%(ext)s"
    argv = [
        _fetch_python(), "-m", "yt_dlp",
        "--no-playlist", "--no-progress", "--no-warnings",
        "--socket-timeout", "30", "--retries", "3",
        "--max-filesize", str(int(getattr(settings, "music_reference_max_bytes", 200 * 1024 * 1024))),
        "-f", "bestaudio/best",
        "-x", "--audio-format", "m4a",
        "-o", str(template),
        url.strip(),
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=float(getattr(settings, "music_reference_fetch_timeout", 300.0))
            )
        except TimeoutError as exc:
            process.kill()
            raise ReferenceUnavailable("fetching the link timed out") from exc
    except OSError as exc:
        raise ReferenceUnavailable(f"the reference fetcher could not start: {exc}") from exc
    if process.returncode != 0:
        detail = (stderr or b"")[-400:].decode(errors="replace").strip()
        raise ReferenceUnavailable(f"yt-dlp failed: {detail or process.returncode}")
    files = [p for p in destination.glob("reference.*") if p.suffix not in {".part", ".ytdl", ".json"}]
    if len(files) != 1:
        raise ReferenceUnavailable("the fetcher did not produce exactly one audio file")
    return files[0]


# ── Analysis ─────────────────────────────────────────────────────────────


def estimate_bpm(onsets: list[float], duration: float) -> tuple[int | None, float | None]:
    """Tempo from onset times by autocorrelating an impulse train.

    Lags from 0.3 s (200 BPM) to 1.2 s (50 BPM) are scored; the best lag is
    folded into 70–180 BPM, the range where a listener would count. The
    stability is the winning peak's share of the correlation mass — near 1
    for a click track, small for rubato.
    """
    if len(onsets) < 8 or duration <= 0:
        return None, None
    resolution = 0.01
    length = int(duration / resolution) + 1
    train = [0.0] * length
    for onset in onsets:
        index = int(onset / resolution)
        if 0 <= index < length:
            train[index] = 1.0
    best_lag, best_score, scores = 0, 0.0, []
    for lag in range(int(0.3 / resolution), int(1.2 / resolution) + 1):
        score = sum(train[i] * train[i + lag] for i in range(length - lag))
        scores.append(score)
        if score > best_score:
            best_score, best_lag = score, lag
    if best_lag == 0 or best_score == 0:
        return None, None
    bpm = 60.0 / (best_lag * resolution)
    while bpm > 180:
        bpm /= 2
    while bpm < 70:
        bpm *= 2
    stability = best_score / sum(scores) if sum(scores) else 0.0
    return int(round(bpm)), min(1.0, stability * len(scores) / 10)


def energy_curve(envelope: list[float], *, hop: float = _ENVELOPE_HOP, step: float = _CURVE_STEP_SECONDS) -> list[float]:
    if not envelope:
        return []
    per_step = max(1, int(step / hop))
    curve = []
    for start in range(0, len(envelope), per_step):
        chunk = envelope[start : start + per_step]
        curve.append(sum(chunk) / len(chunk))
    peak = max(curve) or 1.0
    return [value / peak for value in curve]


def section_boundaries(curve: list[float], *, step: float = _CURVE_STEP_SECONDS, threshold: float = 0.25) -> list[float]:
    """Times where the 5-second energy changes by more than `threshold`."""
    boundaries: list[float] = []
    for index in range(1, len(curve)):
        if abs(curve[index] - curve[index - 1]) >= threshold:
            time = index * step
            if not boundaries or time - boundaries[-1] >= 2 * step:
                boundaries.append(time)
    return boundaries


def _mood(curve: list[float], bpm: int | None) -> str:
    if not curve:
        return ""
    mean = sum(curve) / len(curve)
    if bpm and bpm >= 125 and mean >= 0.55:
        return "high-energy and driving"
    if bpm and bpm <= 85 and mean <= 0.5:
        return "slow and intimate"
    if mean >= 0.65:
        return "loud and full"
    if mean <= 0.35:
        return "quiet and sparse"
    return "steady mid-energy"


def _lines_from_transcript(transcript: Transcript, *, words_per_line: int = 8) -> tuple[str, ...]:
    words = [w.text for w in transcript.words]
    return tuple(" ".join(words[i : i + words_per_line]) for i in range(0, len(words), words_per_line))


async def analyse_reference(
    path: Path,
    *,
    source: str,
    language_hint: str | None,
    max_seconds: float | None = None,
    min_seconds: float = 5.0,
) -> ReferenceProfile:
    """The profile of a reference track. Raises `ReferenceInvalid` for media
    that cannot serve as one."""
    try:
        info = await probe_media(path)
    except FfmpegError as exc:
        raise ReferenceInvalid(f"the reference could not be read: {exc}") from exc
    if not info.has_audio:
        raise ReferenceInvalid("the reference has no audio stream")
    duration = float(info.duration_seconds or 0.0)
    if duration < min_seconds:
        raise ReferenceInvalid(f"the reference is {duration:.1f}s long; at least {min_seconds:.0f}s is needed")
    ceiling = max_seconds or float(getattr(settings, "music_reference_max_seconds", 600))
    if duration > ceiling:
        raise ReferenceInvalid(f"the reference is {duration:.0f}s long; the limit is {ceiling:.0f}s")

    analysis: dict[str, str] = {}
    try:
        envelope = await audio_envelope(path, hop_seconds=_ENVELOPE_HOP)
    except FfmpegError as exc:
        raise ReferenceInvalid(f"the reference could not be decoded: {exc}") from exc
    if not envelope or max(envelope) < 0.002:
        raise ReferenceInvalid("the reference is silent")

    tempo, key = await analyse_tempo_and_key(path)
    bpm = tempo.bpm if tempo else None
    stability = tempo.confidence if tempo else None
    if bpm is None:
        onsets = detect_onsets(envelope, hop_seconds=_ENVELOPE_HOP)
        bpm, stability = estimate_bpm(onsets, duration)
    analysis["tempo"] = "measured" if bpm else "too few onsets"
    analysis["key"] = "measured (chroma)" if key else "unmeasured (no numpy or too short)"
    curve = energy_curve(envelope)
    boundaries = section_boundaries(curve)
    analysis["energy"] = "measured"

    spans = await vocal_activity(path)
    density: float | None = None
    if spans is not None:
        density = vocal_fraction(spans, 0.0, duration)
        analysis["vocals"] = "measured (stem)"
    else:
        analysis["vocals"] = "no stem separator on this node"

    transcript = await transcribe(path, language=language_hint)
    cadence: float | None = None
    wps: float | None = None
    language: str | None = None
    probability: float | None = None
    lines: tuple[str, ...] = ()
    if transcript is not None and transcript.words:
        sung = sum(e - s for s, e in spans) if spans else None
        if not sung:
            sung = sum(max(0.05, w.end - w.start) for w in transcript.words)
        text = transcript.text
        cadence = syllables(text, language_hint or transcript.language or "en") / max(1.0, sung)
        wps = len(transcript.words) / max(1.0, sung)
        language = transcript.language
        probability = transcript.language_probability
        lines = _lines_from_transcript(transcript)
        analysis["cadence"] = "measured (transcript)"
        if density is None:
            # Word spans bridged by breaths as a fallback density measure.
            word_env = [0.0] * (int(duration / _ENVELOPE_HOP) + 1)
            for w in transcript.words:
                for i in range(int(w.start / _ENVELOPE_HOP), min(len(word_env), int(w.end / _ENVELOPE_HOP) + 1)):
                    word_env[i] = 1.0
            density = vocal_fraction(spans_from_envelope(word_env, abs_floor=0.5, rel_fraction=0.5), 0.0, duration)
            analysis["vocals"] = "estimated (transcript)"
    else:
        analysis["cadence"] = "no transcriber on this node" if transcript is None else "no words heard"

    profile = ReferenceProfile(
        source=source,
        duration_seconds=duration,
        bpm=bpm,
        tempo_stability=stability,
        time_signature="4/4",
        energy_curve=tuple(curve),
        section_boundaries=tuple(boundaries),
        vocal_density=density,
        vocal_cadence=cadence,
        words_per_second=wps,
        language=language,
        language_probability=probability,
        mood_hint=_mood(curve, bpm),
        analysis=analysis,
        key=key.name if key else None,
        mode=key.mode if key else None,
        key_confidence=key.confidence if key else None,
        syllables_per_bar=(cadence * 240.0 / bpm) if (cadence and bpm) else None,
        _transcript_lines=lines,
    )
    logger.info(
        "music_reference_analysed",
        extra={
            "source": source,
            "duration": round(duration, 1),
            "bpm": bpm,
            "key": key.name if key else None,
            "vocal_density": None if density is None else round(density, 3),
            "cadence": None if cadence is None else round(cadence, 2),
            "analysis": analysis,
        },
    )
    return profile


# ── Comparing a finished song with its reference ─────────────────────────


@dataclass(frozen=True)
class ReferenceComparison:
    similarity: float
    bpm_ok: bool | None
    key_ok: bool | None
    candidate_bpm: int | None
    candidate_key: str | None
    energy_correlation: float | None
    density_delta: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity": round(self.similarity, 3),
            "bpm_ok": self.bpm_ok,
            "key_ok": self.key_ok,
            "candidate_bpm": self.candidate_bpm,
            "candidate_key": self.candidate_key,
            "energy_correlation": None if self.energy_correlation is None else round(self.energy_correlation, 3),
            "density_delta": None if self.density_delta is None else round(self.density_delta, 3),
            "reasons": list(self.reasons),
        }


def _correlate(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    # Resample the longer curve onto the shorter by nearest index.
    n = min(len(a), len(b))
    ra = [a[int(i * len(a) / n)] for i in range(n)]
    rb = [b[int(i * len(b) / n)] for i in range(n)]
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def compare_to_reference(
    reference: ReferenceProfile,
    candidate: ReferenceProfile,
    *,
    bpm_tolerance: float = 0.03,
) -> ReferenceComparison:
    """How close a finished song is to what the reference asked for.

    Weighted: tempo 0.35, key 0.30, energy curve 0.20, sung density 0.15.
    A feature unmeasured on either side is scored as neutral (its weight's
    half) and named in the reasons — the score says what was measured.
    """
    reasons: list[str] = []
    score = 0.0

    bpm_ok: bool | None = None
    if reference.bpm and candidate.bpm:
        bpm_ok = bpm_matches(reference.bpm, candidate.bpm, tolerance=bpm_tolerance)
        score += 0.35 if bpm_ok else 0.0
        if not bpm_ok:
            reasons.append(f"tempo {candidate.bpm} BPM vs reference {reference.bpm} BPM")
    else:
        score += 0.175
        reasons.append("tempo unmeasured on one side")

    key_ok: bool | None = None
    if reference.key and candidate.key:
        same_tonic = reference.key.split()[0] == candidate.key.split()[0]
        if reference.key == candidate.key:
            key_ok = True
            score += 0.30
        elif same_tonic:
            # Major against minor on the same tonic is the one confusion a
            # chroma profile cannot settle on a dense mix (measured 10 Sep
            # 2026: F major heard as F minor on a generated song). Partial
            # credit and a reason, not a hard mismatch.
            key_ok = None
            score += 0.20
            reasons.append(f"mode uncertain: {candidate.key} heard against reference {reference.key}")
        else:
            key_ok = False
            reasons.append(f"key {candidate.key} vs reference {reference.key}")
    else:
        score += 0.15
        reasons.append("key unmeasured on one side")

    correlation = _correlate(reference.energy_curve, candidate.energy_curve)
    if correlation is None:
        score += 0.10
    else:
        # Saturates at 0.5: two songs with the same shape of build and
        # release correlate about that well; above it is noise.
        score += 0.20 * min(1.0, max(0.0, correlation) / 0.5)
        if correlation < 0.3:
            reasons.append(f"energy progression differs (correlation {correlation:.2f})")

    delta: float | None = None
    if reference.vocal_density is not None and candidate.vocal_density is not None:
        delta = abs(reference.vocal_density - candidate.vocal_density)
        # Within a fifth of the song is the same kind of vocal presence.
        score += 0.15 * min(1.0, max(0.0, (0.5 - delta) / 0.3))
    else:
        score += 0.075

    return ReferenceComparison(
        similarity=min(1.0, score),
        bpm_ok=bpm_ok,
        key_ok=key_ok,
        candidate_bpm=candidate.bpm,
        candidate_key=candidate.key,
        energy_correlation=correlation,
        density_delta=delta,
        reasons=tuple(reasons),
    )


__all__ = [
    "REFERENCE_AUDIO_HOSTS",
    "ReferenceComparison",
    "ReferenceInvalid",
    "ReferenceProfile",
    "ReferenceUnavailable",
    "compare_to_reference",
    "analyse_reference",
    "energy_curve",
    "estimate_bpm",
    "fetch_reference_url",
    "host_allowed",
    "section_boundaries",
]
