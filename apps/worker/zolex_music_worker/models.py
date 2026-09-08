from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import ValidationError
from .utils import validate_job_id


FORMAT_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (3840, 2160),
    "9:16": (2160, 3840),
    "1:1": (2160, 2160),
}


# LTX scenes stay at dimensions divisible by 32. Picture assembly happens at
# this working size; a separate GPU finishing stage creates the 4K delivery.
RENDER_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 704),
    "9:16": (704, 1280),
    "1:1": (1024, 1024),
}


@dataclass(frozen=True)
class Performer:
    id: str
    reference_images: tuple[str, ...] = ()
    role: str = "lead"
    description: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Performer":
        performer_id = str(value.get("id", "")).strip()
        if not performer_id:
            raise ValidationError("Every performer requires a nonempty id")
        refs = tuple(str(item) for item in value.get("reference_images", []))
        if len(refs) > 4:
            raise ValidationError(f"Performer {performer_id} has more than four references")
        return cls(
            id=performer_id,
            reference_images=refs,
            role=str(value.get("role", "lead")),
            description=str(value.get("description", "")),
        )


@dataclass(frozen=True)
class MusicVideoRequest:
    job_id: str
    audio: str
    prompt: str
    performers: tuple[Performer, ...] = ()
    formats: tuple[str, ...] = ("16:9",)
    style_preset: str = "cinematic_tropical_performance"
    automatic_creative_direction: bool = True
    lyrics: str | None = None
    lyrics_language: str | None = None
    lyric_mode: str = "automatic"
    allow_background_change: bool = True
    allow_wardrobe_change: bool = False
    lip_sync: bool = True
    max_people_visible_per_shot: int = 1
    anchor_backend: str | None = None
    render_backend: str | None = None
    reference_video: str | None = None
    reference_video_url: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MusicVideoRequest":
        job_id = validate_job_id(str(value.get("job_id", "")))
        audio = str(value.get("audio", "")).strip()
        prompt = str(value.get("prompt", value.get("creative_brief", ""))).strip()
        if not audio:
            raise ValidationError("audio is required")
        if not prompt:
            raise ValidationError("prompt or creative_brief is required")
        performers = tuple(Performer.from_dict(item) for item in value.get("performers", []))
        if len(performers) > 5:
            raise ValidationError("A job supports zero to five performers")
        performer_ids = [item.id for item in performers]
        if len(performer_ids) != len(set(performer_ids)):
            raise ValidationError("Performer ids must be unique")
        formats = tuple(str(item) for item in value.get("formats", ["16:9"]))
        if not formats or len(formats) != len(set(formats)):
            raise ValidationError("formats must be a nonempty unique list")
        unknown_formats = sorted(set(formats) - FORMAT_DIMENSIONS.keys())
        if unknown_formats:
            raise ValidationError(f"Unsupported formats: {', '.join(unknown_formats)}")
        default_visible_people = min(5, max(1, len(performers)))
        max_people = int(value.get("max_people_visible_per_shot", default_visible_people))
        if max_people < 1 or max_people > 5:
            raise ValidationError("max_people_visible_per_shot must be between 1 and 5")
        raw_lyrics = value.get("lyrics")
        lyrics = str(raw_lyrics).strip() if raw_lyrics is not None else None
        if lyrics is not None and len(lyrics) > 200_000:
            raise ValidationError("lyrics must be 200,000 characters or fewer")
        lyric_mode = str(value.get("lyric_mode", "automatic")).strip().casefold()
        if lyric_mode not in {"automatic", "always", "off"}:
            raise ValidationError("lyric_mode must be automatic, always, or off")
        raw_reference_video = value.get("reference_video", value.get("style_reference_video"))
        reference_video = str(raw_reference_video).strip() if raw_reference_video is not None else None
        if reference_video == "":
            reference_video = None
        raw_reference_url = value.get("reference_video_url", value.get("style_reference_video_url"))
        reference_video_url = str(raw_reference_url).strip() if raw_reference_url is not None else None
        if reference_video_url == "":
            reference_video_url = None
        if reference_video and reference_video_url:
            raise ValidationError("Use either reference_video or reference_video_url, not both")
        if reference_video_url:
            if len(reference_video_url) > 2048:
                raise ValidationError("reference_video_url must be 2048 characters or fewer")
            if any(ord(character) < 32 for character in reference_video_url):
                raise ValidationError("reference_video_url cannot contain control characters")
            parsed_reference_url = urlparse(reference_video_url)
            if parsed_reference_url.scheme.casefold() != "https" or not parsed_reference_url.hostname:
                raise ValidationError("reference_video_url must be a valid HTTPS URL")
            if parsed_reference_url.username or parsed_reference_url.password:
                raise ValidationError("reference_video_url cannot contain credentials")
        return cls(
            job_id=job_id,
            audio=audio,
            prompt=prompt,
            performers=performers,
            formats=formats,
            style_preset=str(value.get("style_preset", "cinematic_tropical_performance")),
            automatic_creative_direction=bool(value.get("automatic_creative_direction", True)),
            lyrics=lyrics,
            lyrics_language=(str(value["lyrics_language"]).strip() or None)
            if value.get("lyrics_language") is not None
            else None,
            lyric_mode=lyric_mode,
            allow_background_change=bool(value.get("allow_background_change", True)),
            allow_wardrobe_change=bool(value.get("allow_wardrobe_change", False)),
            lip_sync=bool(value.get("lip_sync", True)),
            max_people_visible_per_shot=max_people,
            anchor_backend=value.get("anchor_backend"),
            render_backend=value.get("render_backend"),
            reference_video=reference_video,
            reference_video_url=reference_video_url,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["formats"] = list(self.formats)
        result["performers"] = []
        for performer in self.performers:
            item = asdict(performer)
            item["reference_images"] = list(performer.reference_images)
            result["performers"].append(item)
        return result


@dataclass(frozen=True)
class AudioInfo:
    source_path: str
    decoded_wav: str
    aligned_wav: str
    conditioning_wav: str
    sample_rate: int
    channels: int
    decoded_samples: int
    total_frames: int
    fps: int
    decoded_duration: float
    aligned_duration: float
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LyricSegment:
    id: str
    start_frame: int
    end_frame: int
    text: str
    language: str | None = None
    confidence: float | None = None
    is_refrain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Shot:
    id: str
    start_frame: int
    frame_count: int
    section: str
    family: str
    energy: float
    vocals_present: bool
    performer_ids: list[str]
    anchor_role: str
    anchor_image: str = ""
    prompt: str = ""
    transition_out: str = "hard_cut"
    seed: int = 0
    lyric_segment_ids: list[str] = field(default_factory=list)
    lyric_excerpt: str = ""
    lyric_theme: str = ""
    lyric_visual_direction: str = ""
    lyric_sensitive: bool = False

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frame_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditorialPlan:
    schema_version: int
    fps: int
    format: str
    width: int
    height: int
    total_frames: int
    shots: list[Shot] = field(default_factory=list)
    render_width: int | None = None
    render_height: int | None = None

    @property
    def scene_width(self) -> int:
        return self.render_width or self.width

    @property
    def scene_height(self) -> int:
        return self.render_height or self.height

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shots"] = [shot.to_dict() for shot in self.shots]
        result["render_width"] = self.scene_width
        result["render_height"] = self.scene_height
        return result
