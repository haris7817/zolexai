from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .media import load_wav_mono


@dataclass(frozen=True)
class SongSection:
    id: str
    start_frame: int
    end_frame: int
    energy: float
    vocals_present: bool
    vocal_confidence: float


@dataclass(frozen=True)
class SongAnalysis:
    fps: int
    total_frames: int
    tempo_bpm: float | None
    transients: tuple[int, ...]
    sections: tuple[SongSection, ...]
    analyzer: str = "numpy-spectral-flux-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "tempo_bpm": self.tempo_bpm,
            "transients": list(self.transients),
            "sections": [asdict(item) for item in self.sections],
            "analyzer": self.analyzer,
        }


def _spectral_features(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float]:
    window_size = 2048
    hop = 512
    if len(samples) < window_size:
        samples = np.pad(samples, (0, window_size - len(samples)))
    count = 1 + (len(samples) - window_size) // hop
    shape = (count, window_size)
    strides = (samples.strides[0] * hop, samples.strides[0])
    framed = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()
    framed *= np.hanning(window_size)[None, :]
    spectrum = np.abs(np.fft.rfft(framed, axis=1))
    flux = np.maximum(spectrum[1:] - spectrum[:-1], 0.0).sum(axis=1)
    energy = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    frame_seconds = hop / sample_rate
    return flux, energy, frame_seconds


def _find_transient_frames(flux: np.ndarray, frame_seconds: float, fps: int, total_frames: int) -> tuple[int, ...]:
    if len(flux) < 3:
        return (0, total_frames)
    threshold = float(np.percentile(flux, 72))
    candidates: list[tuple[int, float]] = []
    min_gap = max(1, round(0.15 / frame_seconds))
    last_index = -min_gap
    for index in range(1, len(flux) - 1):
        if flux[index] < threshold or flux[index] < flux[index - 1] or flux[index] < flux[index + 1]:
            continue
        if index - last_index < min_gap:
            if candidates and flux[index] > candidates[-1][1]:
                candidates[-1] = (index, float(flux[index]))
                last_index = index
            continue
        candidates.append((index, float(flux[index])))
        last_index = index
    frames = {0, total_frames}
    for index, _ in candidates:
        value = round((index + 1) * frame_seconds * fps)
        if 0 < value < total_frames:
            frames.add(value)
    return tuple(sorted(frames))


def _estimate_tempo(flux: np.ndarray, frame_seconds: float) -> float | None:
    if len(flux) < 32 or not np.any(flux):
        return None
    centered = flux - flux.mean()
    fft_size = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: len(centered)]
    min_lag = max(1, round((60 / 180) / frame_seconds))
    max_lag = min(len(correlation) - 1, round((60 / 60) / frame_seconds))
    if max_lag <= min_lag:
        return None
    lag = min_lag + int(np.argmax(correlation[min_lag : max_lag + 1]))
    bpm = 60.0 / (lag * frame_seconds)
    return round(bpm, 2)


def _build_sections(
    energy: np.ndarray,
    frame_seconds: float,
    *,
    fps: int,
    total_frames: int,
) -> tuple[SongSection, ...]:
    names = ("opening", "introduction", "development", "emotional_middle", "climax", "ending")
    fractions = (0.0, 0.07, 0.25, 0.50, 0.70, 0.93, 1.0)
    global_high = float(np.percentile(energy, 85)) if len(energy) else 1.0
    result: list[SongSection] = []
    for index, name in enumerate(names):
        start = round(total_frames * fractions[index])
        end = total_frames if index == len(names) - 1 else round(total_frames * fractions[index + 1])
        start_sample = round((start / fps) / frame_seconds)
        end_sample = round((end / fps) / frame_seconds)
        span = energy[max(0, start_sample) : max(start_sample + 1, end_sample)]
        value = float(span.mean()) if len(span) else 0.0
        normalized = min(1.0, value / max(global_high, 1e-6))
        vocals = name not in {"opening", "ending"} and normalized > 0.12
        result.append(
            SongSection(
                id=name,
                start_frame=start,
                end_frame=end,
                energy=round(normalized, 4),
                vocals_present=vocals,
                vocal_confidence=0.35 if vocals else 0.25,
            )
        )
    return tuple(result)


def analyze_song(wav_path: Path, *, fps: int, total_frames: int) -> SongAnalysis:
    sample_rate, samples = load_wav_mono(wav_path)
    flux, energy, frame_seconds = _spectral_features(samples, sample_rate)
    return SongAnalysis(
        fps=fps,
        total_frames=total_frames,
        tempo_bpm=_estimate_tempo(flux, frame_seconds),
        transients=_find_transient_frames(flux, frame_seconds, fps, total_frames),
        sections=_build_sections(energy, frame_seconds, fps=fps, total_frames=total_frames),
    )
