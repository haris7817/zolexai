from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import WorkerConfig
from .errors import ValidationError
from .models import AudioInfo, EditorialPlan, LyricSegment, MusicVideoRequest
from .utils import atomic_write_json, ensure_existing_file, format_template_argv, read_json, run_command


LYRIC_REQUEST_TERMS = (
    "according to the lyrics",
    "based on the lyrics",
    "follow the lyrics",
    "match the lyrics",
    "lyrics of the song",
    "according to lyrics",
    "según la letra",
    "de acuerdo con la letra",
    "basado en la letra",
    "sigue la letra",
    "letra de la canción",
)

THEMES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "love and connection",
        ("love", "heart", "baby", "kiss", "amor", "corazón", "beso", "querer"),
        "an intimate symbol of connection, warm reflected light, and restrained hopeful performance",
    ),
    (
        "loss and isolation",
        ("alone", "goodbye", "miss", "pain", "cry", "lost", "solo", "adiós", "dolor", "llorar"),
        "the performer separated from the surrounding space, with receding light and a meaningful empty foreground",
    ),
    (
        "ambition and success",
        ("money", "rich", "boss", "win", "gold", "diamond", "success", "dinero", "rico", "ganar", "éxito"),
        "the performer advancing through polished urban architecture as reflections and height suggest earned progress",
    ),
    (
        "street identity and resilience",
        ("street", "block", "hood", "city", "corner", "barrio", "calle", "ciudad", "esquina"),
        "grounded neighborhood details, direct performance, and forward movement through a recognizable urban environment",
    ),
    (
        "celebration and release",
        ("party", "dance", "club", "night", "drink", "fiesta", "baila", "discoteca", "noche"),
        "rhythmic practical lights, confident movement, and lively environmental motion without introducing unassigned people",
    ),
    (
        "faith and hope",
        ("god", "pray", "faith", "bless", "heaven", "dios", "reza", "fe", "bendición", "cielo"),
        "an opening horizon, upward natural light, and a calm physical gesture that suggests renewed hope",
    ),
    (
        "memory and time",
        ("remember", "memory", "yesterday", "time", "past", "recuerdo", "ayer", "tiempo", "pasado"),
        "a personal keepsake, layered reflections, and a slow change in light that makes the past feel present",
    ),
    (
        "movement and escape",
        ("road", "drive", "run", "fly", "leave", "camino", "conducir", "correr", "volar", "irme"),
        "purposeful travel through the environment with clear screen direction and expanding geography",
    ),
)

SENSITIVE_TERMS = {
    "gun",
    "guns",
    "glock",
    "shoot",
    "kill",
    "murder",
    "blood",
    "cocaine",
    "heroin",
    "arma",
    "pistola",
    "matar",
    "asesinar",
    "sangre",
    "cocaína",
}


@dataclass(frozen=True)
class LyricTranscript:
    requested: bool
    source: str
    language: str | None
    language_probability: float | None
    segments: tuple[LyricSegment, ...]
    transcription_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["segments"] = [item.to_dict() for item in self.segments]
        return result


def lyrics_requested(request: MusicVideoRequest) -> bool:
    if request.lyrics:
        return True
    if request.lyric_mode == "off":
        return False
    if request.lyric_mode == "always":
        return True
    prompt = request.prompt.casefold()
    return any(term in prompt for term in LYRIC_REQUEST_TERMS)


def _seconds_from_stamp(value: str) -> float:
    clean = value.strip().replace(",", ".")
    parts = clean.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Invalid timestamp: {value}")


def _normalize_segments(
    raw: list[tuple[float, float, str, float | None]],
    *,
    fps: int,
    total_frames: int,
    language: str | None,
) -> tuple[LyricSegment, ...]:
    cleaned: list[tuple[int, int, str, float | None]] = []
    for start_seconds, end_seconds, text, confidence in raw:
        line = re.sub(r"\s+", " ", text).strip()
        if not line:
            continue
        start = min(total_frames - 1, max(0, round(start_seconds * fps)))
        end = min(total_frames, max(start + 1, round(end_seconds * fps)))
        if end <= start:
            continue
        cleaned.append((start, end, line, confidence))
    cleaned.sort(key=lambda item: (item[0], item[1]))
    normalized_text = [re.sub(r"[^\w]+", " ", item[2].casefold()).strip() for item in cleaned]
    counts = Counter(item for item in normalized_text if len(item.split()) >= 3)
    result: list[LyricSegment] = []
    for index, ((start, end, text, confidence), normalized) in enumerate(zip(cleaned, normalized_text), start=1):
        result.append(
            LyricSegment(
                id=f"lyric-{index:04d}",
                start_frame=start,
                end_frame=end,
                text=text,
                language=language,
                confidence=confidence,
                is_refrain=bool(normalized and counts[normalized] > 1),
            )
        )
    return tuple(result)


def _parse_lrc(text: str, *, fps: int, total_frames: int, language: str | None) -> tuple[LyricSegment, ...]:
    stamped: list[tuple[float, str]] = []
    pattern = re.compile(r"\[(\d{1,3}:\d{2}(?:[.,]\d{1,3})?)\]")
    for line in text.splitlines():
        matches = list(pattern.finditer(line))
        lyric = pattern.sub("", line).strip()
        for match in matches:
            stamped.append((_seconds_from_stamp(match.group(1)), lyric))
    if not stamped:
        return ()
    stamped.sort(key=lambda item: item[0])
    total_seconds = total_frames / fps
    raw = []
    for index, (start, line) in enumerate(stamped):
        end = stamped[index + 1][0] if index + 1 < len(stamped) else total_seconds
        raw.append((start, max(start + 0.25, end), line, 1.0))
    return _normalize_segments(raw, fps=fps, total_frames=total_frames, language=language)


def _parse_srt(text: str, *, fps: int, total_frames: int, language: str | None) -> tuple[LyricSegment, ...]:
    raw: list[tuple[float, float, str, float | None]] = []
    timing = re.compile(
        r"^(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})$"
    )
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines and lines[0].isdigit():
            lines.pop(0)
        if not lines:
            continue
        match = timing.fullmatch(lines.pop(0))
        if not match:
            continue
        raw.append(
            (
                _seconds_from_stamp(match.group(1)),
                _seconds_from_stamp(match.group(2)),
                " ".join(lines),
                1.0,
            )
        )
    return _normalize_segments(raw, fps=fps, total_frames=total_frames, language=language)


def _parse_plain(text: str, *, fps: int, total_frames: int, language: str | None) -> tuple[LyricSegment, ...]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ()
    start_frame = round(total_frames * 0.03)
    end_frame = max(start_frame + len(lines), round(total_frames * 0.97))
    raw = []
    for index, line in enumerate(lines):
        start = start_frame + round((end_frame - start_frame) * index / len(lines))
        end = start_frame + round((end_frame - start_frame) * (index + 1) / len(lines))
        raw.append((start / fps, max(start + 1, end) / fps, line, 1.0))
    return _normalize_segments(raw, fps=fps, total_frames=total_frames, language=language)


def parse_supplied_lyrics(
    text: str,
    *,
    fps: int,
    total_frames: int,
    language: str | None,
) -> tuple[LyricSegment, ...]:
    for parser in (_parse_lrc, _parse_srt):
        parsed = parser(text, fps=fps, total_frames=total_frames, language=language)
        if parsed:
            return parsed
    return _parse_plain(text, fps=fps, total_frames=total_frames, language=language)


@lru_cache(maxsize=2)
def _whisper_model(model_name: str, device: str, compute_type: str, download_root: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ValidationError(
            "Automatic lyric transcription requires the lyrics extra: pip install 'zolexai-music-video-worker[lyrics]'"
        ) from exc
    return WhisperModel(model_name, device=device, compute_type=compute_type, download_root=download_root)


def _transcribe_faster_whisper(
    audio: AudioInfo,
    request: MusicVideoRequest,
    config: WorkerConfig,
) -> LyricTranscript:
    model = _whisper_model(
        config.transcription_model,
        config.transcription_device,
        config.transcription_compute_type,
        str(config.transcription_download_root) if config.transcription_download_root else None,
    )
    segments, info = model.transcribe(
        audio.decoded_wav,
        language=request.lyrics_language,
        beam_size=config.transcription_beam_size,
        condition_on_previous_text=True,
        vad_filter=False,
        word_timestamps=True,
    )
    language = getattr(info, "language", None) or request.lyrics_language
    raw = []
    for segment in segments:
        average_log_probability = float(getattr(segment, "avg_logprob", -10.0))
        confidence = max(0.0, min(1.0, math.exp(average_log_probability)))
        raw.append((float(segment.start), float(segment.end), str(segment.text), confidence))
    return LyricTranscript(
        requested=True,
        source="faster_whisper",
        language=language,
        language_probability=float(getattr(info, "language_probability", 0.0)),
        segments=_normalize_segments(
            raw,
            fps=audio.fps,
            total_frames=audio.total_frames,
            language=language,
        ),
        transcription_model=config.transcription_model,
    )


def _transcribe_command(
    audio: AudioInfo,
    request: MusicVideoRequest,
    config: WorkerConfig,
    output_dir: Path,
) -> LyricTranscript:
    if not config.transcription_command:
        raise ValidationError("command transcription backend requires ZOLEX_TRANSCRIPTION_COMMAND_JSON")
    request_path = output_dir / "transcription-request.json"
    output_path = output_dir / "transcription-command-output.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": 1,
            "audio": audio.decoded_wav,
            "language": request.lyrics_language,
            "fps": audio.fps,
            "total_frames": audio.total_frames,
            "output": str(output_path),
        },
    )
    values = {
        "request_json": str(request_path),
        "audio": audio.decoded_wav,
        "language": request.lyrics_language or "",
        "output": str(output_path),
    }
    run_command(
        format_template_argv(config.transcription_command, values),
        timeout=config.transcription_timeout_seconds,
        log_path=output_dir / "transcription-command.json",
    )
    payload = read_json(ensure_existing_file(output_path, "Transcription output"))
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValidationError("Transcription command output must contain a segments array")
    language = str(payload.get("language") or request.lyrics_language or "").strip() or None
    raw = []
    for item in payload["segments"]:
        if not isinstance(item, dict):
            raise ValidationError("Every transcription segment must be an object")
        raw.append(
            (
                float(item["start_seconds"]),
                float(item["end_seconds"]),
                str(item["text"]),
                float(item["confidence"]) if item.get("confidence") is not None else None,
            )
        )
    return LyricTranscript(
        requested=True,
        source="command",
        language=language,
        language_probability=float(payload["language_probability"])
        if payload.get("language_probability") is not None
        else None,
        segments=_normalize_segments(raw, fps=audio.fps, total_frames=audio.total_frames, language=language),
        transcription_model=str(payload.get("model") or "command"),
    )


def prepare_lyrics(
    request: MusicVideoRequest,
    audio: AudioInfo,
    config: WorkerConfig,
    output_dir: Path,
) -> LyricTranscript:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = lyrics_requested(request)
    if not requested:
        transcript = LyricTranscript(False, "not_requested", request.lyrics_language, None, ())
    elif request.lyrics:
        transcript = LyricTranscript(
            True,
            "supplied",
            request.lyrics_language,
            1.0 if request.lyrics_language else None,
            parse_supplied_lyrics(
                request.lyrics,
                fps=audio.fps,
                total_frames=audio.total_frames,
                language=request.lyrics_language,
            ),
        )
    elif config.transcription_backend == "faster_whisper":
        transcript = _transcribe_faster_whisper(audio, request, config)
    elif config.transcription_backend == "command":
        transcript = _transcribe_command(audio, request, config, output_dir)
    elif config.transcription_backend == "disabled":
        raise ValidationError(
            "The prompt requests lyric-driven scenes, but automatic transcription is disabled and no lyrics were supplied"
        )
    else:  # Configuration validation normally catches this first.
        raise ValidationError(f"Unsupported transcription backend: {config.transcription_backend}")
    if requested and not transcript.segments:
        raise ValidationError("No usable lyrics were found; supply lyrics or verify the transcription backend")
    atomic_write_json(output_dir / "lyrics.json", transcript.to_dict())
    return transcript


def segments_for_window(
    transcript: LyricTranscript,
    start_frame: int,
    end_frame: int,
) -> tuple[LyricSegment, ...]:
    return tuple(
        item
        for item in transcript.segments
        if item.start_frame < end_frame and item.end_frame > start_frame
    )


def interpret_lyric_text(text: str, *, is_refrain: bool, performer_strategy: str) -> tuple[str, str, bool]:
    words = set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))
    scored: list[tuple[int, int, str, str]] = []
    for index, (theme, keywords, direction) in enumerate(THEMES):
        score = sum(1 for keyword in keywords if keyword in words)
        scored.append((score, -index, theme, direction))
    score, _, theme, direction = max(scored)
    if score == 0:
        theme = "personal reflection and determination"
        direction = "a concrete personal object, expressive performance, and environmental movement that echoes the cadence of the line"
    sensitive = bool(words & SENSITIVE_TERMS)
    if theme == "love and connection" and performer_strategy not in {
        "two_romantic_leads",
        "generate_consistent_adult_couple",
    }:
        direction = "a solitary but hopeful performance using warm light, a personal keepsake, and open space as symbols of connection"
    if sensitive:
        direction = (
            "symbolic tension through shadows, weather, distance, and controlled performance; do not depict weapons, "
            "graphic harm, drug use, or illegal instructions"
        )
    if is_refrain:
        direction += "; repeat the established chorus motif so the refrain is visually recognizable"
    return theme, direction, sensitive


def apply_lyric_directions(
    plan: EditorialPlan,
    transcript: LyricTranscript,
    treatment: dict[str, Any],
    config: WorkerConfig,
    output_dir: Path,
) -> None:
    for shot in plan.shots:
        active = segments_for_window(transcript, shot.start_frame, shot.end_frame)
        if not active:
            continue
        excerpt = re.sub(r"\s+", " ", " ".join(item.text for item in active)).strip()[:300]
        theme, direction, sensitive = interpret_lyric_text(
            excerpt,
            is_refrain=any(item.is_refrain for item in active),
            performer_strategy=str(treatment.get("performer_strategy", "solo")),
        )
        shot.lyric_segment_ids = [item.id for item in active]
        shot.lyric_excerpt = excerpt
        shot.lyric_theme = theme
        shot.lyric_visual_direction = direction
        shot.lyric_sensitive = sensitive
        if shot.family in {"close_performance", "medium_performance", "firelight_performance"}:
            shot.vocals_present = True

    if not config.lyric_director_command or not transcript.segments:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "lyric-director-request.json"
    output_path = output_dir / "lyric-director-output.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": 1,
            "task": "Turn each timestamped lyric window into one filmable, non-graphic visual direction while preserving the treatment and assigned performers.",
            "treatment": treatment,
            "transcript": transcript.to_dict(),
            "shots": [
                {
                    "id": shot.id,
                    "start_seconds": shot.start_frame / plan.fps,
                    "end_seconds": shot.end_frame / plan.fps,
                    "section": shot.section,
                    "family": shot.family,
                    "performer_ids": shot.performer_ids,
                    "lyric_excerpt": shot.lyric_excerpt,
                    "built_in_theme": shot.lyric_theme,
                    "built_in_visual_direction": shot.lyric_visual_direction,
                }
                for shot in plan.shots
            ],
            "requirements": {
                "one_action_per_shot": True,
                "one_camera_behavior_per_shot": True,
                "no_unassigned_people": True,
                "no_graphic_or_instructional_illegal_content": True,
                "recurring_refrain_motif": True,
                "coherent_beginning_middle_climax_ending": True,
            },
            "output": str(output_path),
        },
    )
    run_command(
        format_template_argv(
            config.lyric_director_command,
            {"request_json": str(request_path), "output": str(output_path)},
        ),
        timeout=config.command_timeout_seconds,
        log_path=output_dir / "lyric-director-command.json",
    )
    payload = read_json(ensure_existing_file(output_path, "Lyric director output"))
    if not isinstance(payload, dict) or not isinstance(payload.get("shots"), list):
        raise ValidationError("Lyric director output must contain a shots array")
    by_id = {shot.id: shot for shot in plan.shots}
    seen: set[str] = set()
    for item in payload["shots"]:
        if not isinstance(item, dict) or str(item.get("id", "")) not in by_id:
            raise ValidationError("Lyric director returned an unknown shot id")
        shot_id = str(item["id"])
        if shot_id in seen:
            raise ValidationError(f"Lyric director returned duplicate shot id: {shot_id}")
        seen.add(shot_id)
        direction = re.sub(r"\s+", " ", str(item.get("visual_direction", ""))).strip()
        theme = re.sub(r"\s+", " ", str(item.get("lyric_theme", ""))).strip()
        if not direction or len(direction) > 800 or len(theme) > 120:
            raise ValidationError(f"Lyric director returned an invalid direction for {shot_id}")
        by_id[shot_id].lyric_visual_direction = direction
        if theme:
            by_id[shot_id].lyric_theme = theme
