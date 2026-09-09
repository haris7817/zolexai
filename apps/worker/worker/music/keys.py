"""Tempo and key of a track, measured from its audio.

The client's reference-matching rule (10 Sep 2026) needs two numbers a
listener would check first — BPM and key — both from the reference and
from the finished song, so the song can be validated against the reference
rather than described as "inspired by" it.

* **Tempo** comes from the onset train (`worker/media/audio.py`) by
  autocorrelation over 50–200 BPM, refined by a second pass over the
  energy envelope itself, with octave folding into 70–180. It is a
  measurement, so both the value and how sharply it won are reported.
* **Key** is Krumhansl–Schmuckler: a chroma vector from a short-time
  spectrum (numpy; the worker has it wherever faster-whisper is) correlated
  with the major and minor key profiles. Correct on clearly tonal music,
  approximate on dense mixes, and the correlation gap between the best two
  candidates is reported as the confidence.

No numpy → key is None and says so. Nothing here fails a job by itself.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from worker.media.audio import audio_envelope, detect_onsets
from worker.media.ffmpeg import ffmpeg_stdout

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: Krumhansl–Kessler key profiles (major, minor), C-rooted.
_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

_RATE = 11025
_FRAME = 4096
_HOP = 2048


@dataclass(frozen=True)
class Tempo:
    bpm: int
    confidence: float

    def to_dict(self) -> dict:
        return {"bpm": self.bpm, "confidence": round(self.confidence, 3)}


@dataclass(frozen=True)
class Key:
    tonic: str
    mode: str
    confidence: float

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> dict:
        return {"tonic": self.tonic, "mode": self.mode, "name": self.name, "confidence": round(self.confidence, 3)}


def fold_bpm(bpm: float) -> float:
    while bpm > 180:
        bpm /= 2
    while bpm < 70:
        bpm *= 2
    return bpm


def _autocorrelate(train: list[float], resolution: float) -> tuple[float | None, float]:
    best_lag, best, scores = 0, 0.0, []
    low, high = int(0.3 / resolution), int(1.2 / resolution)
    for lag in range(low, high + 1):
        score = sum(train[i] * train[i + lag] for i in range(len(train) - lag))
        scores.append(score)
        if score > best:
            best, best_lag = score, lag
    if best_lag == 0 or not best:
        return None, 0.0
    mean = sum(scores) / len(scores)
    confidence = 0.0 if not mean else min(1.0, (best - mean) / (best + 1e-9))
    return 60.0 / (best_lag * resolution), confidence


def tempo_from_envelope(envelope: list[float], *, hop_seconds: float = 0.05) -> Tempo | None:
    """BPM from onsets, cross-checked against the envelope's own periodicity."""
    if len(envelope) < 40:
        return None
    duration = len(envelope) * hop_seconds
    onsets = detect_onsets(envelope, hop_seconds=hop_seconds)
    resolution = 0.01
    length = int(duration / resolution) + 1
    train = [0.0] * length
    for onset in onsets:
        index = int(onset / resolution)
        if 0 <= index < length:
            train[index] = 1.0
    bpm_onsets, conf_onsets = _autocorrelate(train, resolution) if len(onsets) >= 8 else (None, 0.0)

    # Second opinion: the envelope's flux autocorrelation at the envelope hop.
    flux = [max(0.0, b - a) for a, b in zip(envelope, envelope[1:], strict=False)]
    mean = sum(flux) / len(flux) if flux else 0.0
    centred = [value - mean for value in flux]
    bpm_flux, conf_flux = _autocorrelate([max(0.0, v) for v in centred], hop_seconds)

    candidates = [(b, c) for b, c in ((bpm_onsets, conf_onsets), (bpm_flux, conf_flux)) if b]
    if not candidates:
        return None
    bpm, confidence = max(candidates, key=lambda item: item[1])
    folded = fold_bpm(bpm)
    # Agreement between the two measures (allowing an octave) raises confidence.
    if len(candidates) == 2:
        other = fold_bpm(candidates[0][0] if candidates[1][0] == bpm else candidates[1][0])
        if abs(other - folded) <= 0.03 * folded:
            confidence = min(1.0, confidence + 0.25)
    return Tempo(int(round(folded)), confidence)


async def _pcm(path: Path, *, timeout: float) -> list[float]:
    raw = await ffmpeg_stdout(
        ["-i", str(path), "-vn", "-ac", "1", "-ar", str(_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        timeout=timeout,
    )
    count = len(raw) // 2
    return [value / 32768.0 for value in struct.unpack(f"<{count}h", raw[: count * 2])]


def chroma_from_samples(samples: list[float]) -> list[float] | None:
    try:
        import numpy as np
    except ImportError:
        return None
    if len(samples) < _FRAME * 2:
        return None
    signal = np.asarray(samples, dtype=np.float32)
    window = np.hanning(_FRAME).astype(np.float32)
    freqs = np.fft.rfftfreq(_FRAME, 1.0 / _RATE)
    usable = (freqs >= 55.0) & (freqs <= 2000.0)
    midi = 69 + 12 * np.log2(np.where(usable, freqs, 440.0) / 440.0)
    pitch_class = (np.rint(midi).astype(int) % 12)
    chroma = np.zeros(12, dtype=np.float64)
    frames = range(0, len(signal) - _FRAME, _HOP)
    for start in frames:
        spectrum = np.abs(np.fft.rfft(signal[start : start + _FRAME] * window))
        power = (spectrum**2)[usable]
        np.add.at(chroma, pitch_class[usable], power)
    total = chroma.sum()
    if total <= 0:
        return None
    return [float(v / total) for v in chroma]


def key_from_chroma(chroma: list[float]) -> Key | None:
    if not chroma or len(chroma) != 12:
        return None

    def correlation(profile: tuple[float, ...], shift: int) -> float:
        rotated = [profile[(i - shift) % 12] for i in range(12)]
        mean_c = sum(chroma) / 12
        mean_p = sum(rotated) / 12
        num = sum((c - mean_c) * (p - mean_p) for c, p in zip(chroma, rotated, strict=True))
        den = math.sqrt(sum((c - mean_c) ** 2 for c in chroma) * sum((p - mean_p) ** 2 for p in rotated))
        return num / den if den else 0.0

    scores = []
    for shift in range(12):
        scores.append((correlation(_MAJOR, shift), _NOTE_NAMES[shift], "major"))
        scores.append((correlation(_MINOR, shift), _NOTE_NAMES[shift], "minor"))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    confidence = max(0.0, min(1.0, (best[0] - second[0]) * 4 + 0.3))
    return Key(best[1], best[2], confidence)


async def analyse_tempo_and_key(path: Path, *, timeout: float = 600.0) -> tuple[Tempo | None, Key | None]:
    envelope = await audio_envelope(path, hop_seconds=0.05, timeout=timeout)
    tempo = tempo_from_envelope(envelope, hop_seconds=0.05)
    key: Key | None = None
    try:
        samples = await _pcm(path, timeout=timeout)
        chroma = chroma_from_samples(samples)
        key = key_from_chroma(chroma) if chroma else None
    except Exception:  # no numpy, or an undecodable file: the key stays unknown
        key = None
    return tempo, key


def parse_key(text: str | None) -> Key | None:
    """"B major" / "F# minor" / "Bb Major" → Key, or None."""
    if not text:
        return None
    parts = text.strip().replace("♭", "b").replace("♯", "#").split()
    if not parts:
        return None
    tonic = parts[0]
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
    tonic = flats.get(tonic, tonic)
    if tonic not in _NOTE_NAMES:
        tonic = tonic[:1].upper() + tonic[1:]
        tonic = flats.get(tonic, tonic)
        if tonic not in _NOTE_NAMES:
            return None
    mode = "minor" if any(p.lower().startswith("min") or p == "m" for p in parts[1:]) else "major"
    return Key(tonic, mode, 1.0)


def bpm_matches(reference: int, candidate: int, *, tolerance: float = 0.03) -> bool:
    """Within `tolerance`, allowing the half/double-time ambiguity."""
    for factor in (1.0, 2.0, 0.5):
        if abs(candidate * factor - reference) <= tolerance * reference:
            return True
    return False


__all__ = [
    "Key",
    "Tempo",
    "analyse_tempo_and_key",
    "bpm_matches",
    "chroma_from_samples",
    "fold_bpm",
    "key_from_chroma",
    "parse_key",
    "tempo_from_envelope",
]
