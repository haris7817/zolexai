"""Optional AI speech-to-speech voice replacement for Video-to-Video.

The worker intentionally does not embed a specific cloning model. GPU deployments
point ``VOICE_CLONE_URL`` at a private/local AI voice-conversion service. That
service owns diarization, speaker isolation and conversion; this module owns the
stable ZolexAI contract and the invariants around it:

* up to four ordered target voices (Person 1 .. Person 4), with holes allowed;
* an empty slot means keep that source speaker's original voice;
* dialogue words and timing are preserved (speech-to-speech, not script rewrite);
* music, ambience and sound effects are preserved;
* returned audio is conformed to the exact source-video duration before muxing.

Expected provider endpoint (multipart/form-data):
  source_media: source video file
  voice_1 .. voice_4: only supplied target voice samples
  active_slots: JSON array of 1-based slots, e.g. [1,3]
  mapping_mode: ``visual_slot_order`` by default
  preserve_unmapped_voices: ``true``
  preserve_non_speech_audio: ``true``
  duration_seconds: source duration

The response body must be an audio file (WAV/FLAC/MP3/M4A are all fine). A
provider may additionally return ``X-ZolexAI-Voice-Slots`` for diagnostics, but
it is not required.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import httpx

from worker.core.config import settings
from worker.media.ffmpeg import ffmpeg
from worker.media.probe import probe_media


class VoiceCloneError(RuntimeError):
    """AI voice replacement could not produce a valid exact-length mix."""


async def clone_source_voices(
    source_media: Path,
    voice_slots: Sequence[Path | None],
    dest: Path,
    *,
    duration_seconds: float,
    mapping_mode: str = "visual_slot_order",
    preserve_unmapped_voices: bool = True,
    preserve_non_speech_audio: bool = True,
) -> Path:
    """Clone selected voices and return one full replacement audio mix.

    ``voice_slots`` is positional and may contain holes. This matters: supplying
    only slot 3 must not silently turn it into slot 1. The remote/local AI
    service receives the original media so it can diarize against the real
    soundtrack and preserve every component that is not being replaced.
    """
    slots = list(voice_slots[:4])
    if len(slots) < 4:
        slots.extend([None] * (4 - len(slots)))
    active = [index + 1 for index, path in enumerate(slots) if path is not None]
    if not active:
        raise VoiceCloneError("voice cloning was requested with no voice references")
    if not settings.voice_clone_url.strip():
        raise VoiceCloneError(
            "VOICE_CLONE_URL is not configured on this worker; voice references cannot be used"
        )

    timeout = httpx.Timeout(
        connect=min(30.0, settings.voice_clone_timeout_seconds),
        read=settings.voice_clone_timeout_seconds,
        write=settings.voice_clone_timeout_seconds,
        pool=30.0,
    )
    headers: dict[str, str] = {}
    if settings.voice_clone_api_key:
        headers["Authorization"] = f"Bearer {settings.voice_clone_api_key}"

    data = {
        "active_slots": json.dumps(active),
        "mapping_mode": mapping_mode,
        "preserve_unmapped_voices": str(bool(preserve_unmapped_voices)).lower(),
        "preserve_non_speech_audio": str(bool(preserve_non_speech_audio)).lower(),
        "preserve_words": "true",
        "preserve_timing": "true",
        "duration_seconds": f"{duration_seconds:.6f}",
    }

    raw = dest.with_suffix(".provider-audio")
    with ExitStack() as stack:
        source_handle = stack.enter_context(source_media.open("rb"))
        files: dict[str, tuple[str, object, str]] = {
            "source_media": (source_media.name, source_handle, "application/octet-stream")
        }
        for index, path in enumerate(slots, 1):
            if path is None:
                continue
            handle = stack.enter_context(path.open("rb"))
            files[f"voice_{index}"] = (path.name, handle, "application/octet-stream")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    settings.voice_clone_url,
                    data=data,
                    files=files,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise VoiceCloneError(f"voice-clone provider request failed: {exc}") from exc

    if response.status_code < 200 or response.status_code >= 300:
        detail = response.text[:500].replace("\n", " ")
        raise VoiceCloneError(
            f"voice-clone provider returned HTTP {response.status_code}: {detail}"
        )
    if not response.content:
        raise VoiceCloneError("voice-clone provider returned an empty audio file")
    raw.write_bytes(response.content)

    try:
        info = await probe_media(raw)
    except Exception as exc:
        raw.unlink(missing_ok=True)
        raise VoiceCloneError("voice-clone provider returned unreadable media") from exc
    if not info.has_audio:
        raw.unlink(missing_ok=True)
        raise VoiceCloneError("voice-clone provider response contains no audio stream")

    # Exact source length is a V2V product guarantee. atrim removes provider
    # overrun and apad fills only a short tail without changing speech speed.
    await ffmpeg(
        [
            "-i", str(raw),
            "-vn",
            "-filter:a", f"apad,atrim=0:{max(0.001, duration_seconds):.6f}",
            "-ar", "48000",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(dest),
        ],
        timeout=max(600.0, settings.voice_clone_timeout_seconds),
    )
    raw.unlink(missing_ok=True)

    conformed = await probe_media(dest)
    if not conformed.has_audio:
        raise VoiceCloneError("conformed voice-clone mix contains no audio")
    actual = conformed.duration_seconds or 0.0
    if abs(actual - duration_seconds) > 0.12:
        raise VoiceCloneError(
            f"voice-clone mix duration {actual:.3f}s does not match source {duration_seconds:.3f}s"
        )
    return dest
