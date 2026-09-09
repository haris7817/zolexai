"""Word-level transcription of a song, for verification — never for lyrics.

faster-whisper is already on the music-video node (the `music-video` extra),
so the music path borrows the same model and the same settings rather than
adding a second copy. It runs in a thread: the model call is synchronous and
would otherwise stall the worker's event loop for the length of the song.

Every failure returns None. This module *measures*; it never decides. What a
None means for a job is the caller's policy (`worker/music/verify.py`).
"""

from __future__ import annotations

import asyncio
import functools
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker.core.config import settings
from worker.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class Transcript:
    language: str | None
    language_probability: float
    words: tuple[Word, ...]
    model: str

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "model": self.model,
            "words": [
                {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3), "p": round(w.probability, 3)}
                for w in self.words
            ],
        }


@functools.lru_cache(maxsize=1)
def _model(name: str, device: str, compute_type: str, download_root: str | None):
    from faster_whisper import WhisperModel  # heavy; imported on first use only

    return WhisperModel(name, device=device, compute_type=compute_type, download_root=download_root)


def available() -> bool:
    if getattr(settings, "music_verify_transcription", "auto") == "disabled":
        return False
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _transcribe_sync(path: Path, language: str | None) -> Transcript:
    model_name = settings.music_video_whisper_model
    root = settings.music_video_whisper_download_root
    model = _model(
        model_name,
        settings.music_video_whisper_device,
        settings.music_video_whisper_compute_type,
        str(root) if root else None,
    )
    segments, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=True,
    )
    words: list[Word] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or ():
            text = str(getattr(word, "word", "")).strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(getattr(word, "start", 0.0)),
                    end=float(getattr(word, "end", 0.0)),
                    probability=float(getattr(word, "probability", 0.0)),
                )
            )
    return Transcript(
        language=getattr(info, "language", None) or language,
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        words=tuple(words),
        model=model_name,
    )


async def transcribe(path: Path, *, language: str | None) -> Transcript | None:
    """Words with times, or None when transcription is not available here.

    `language` is the code the song is *supposed* to be in. Passing it stops
    the model guessing — Whisper has called an English pop song Khmer — and
    the report still records the model's own language estimate separately.
    """
    if not available():
        return None
    try:
        return await asyncio.to_thread(_transcribe_sync, path, language)
    except Exception as exc:  # the model raises many types; none should fail a job here
        logger.warning("music_transcription_failed", extra={"detail": str(exc)[:300]})
        return None


# ── Normalisation shared with the verifier ──────────────────────────────

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    """Case-folded, accent-stripped, punctuation-free."""
    folded = unicodedata.normalize("NFD", text.casefold())
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return _PUNCT.sub(" ", folded).strip()


__all__ = ["Transcript", "Word", "available", "normalise", "transcribe"]
