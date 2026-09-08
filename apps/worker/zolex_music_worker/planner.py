from __future__ import annotations

import math
import os
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .analysis import SongAnalysis, SongSection
from .config import WorkerConfig
from .errors import ValidationError
from .lyrics import LyricTranscript
from .models import EditorialPlan, FORMAT_DIMENSIONS, RENDER_DIMENSIONS, MusicVideoRequest, Shot


SECTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "opening": ("environment", "profile_portrait"),
    "introduction": ("close_performance", "medium_performance", "wide_movement", "detail_jewelry"),
    "development": (
        "medium_performance",
        "detail_hands",
        "close_performance",
        "wide_movement",
        "profile_portrait",
    ),
    "emotional_middle": (
        "seated_reflection",
        "environment",
        "profile_portrait",
        "detail_object",
        "close_performance",
    ),
    "climax": (
        "firelight_performance",
        "medium_performance",
        "wide_movement",
        "close_performance",
        "detail_hands",
    ),
    "ending": ("profile_portrait", "ending_environment"),
}


PERFORMER_FAMILIES = {
    "close_performance",
    "medium_performance",
    "wide_movement",
    "profile_portrait",
    "seated_reflection",
    "firelight_performance",
}


GROUP_FAMILIES = {
    "medium_performance",
    "wide_movement",
    "firelight_performance",
}


def choose_boundaries(
    total_frames: int,
    transients: Iterable[int],
    config: WorkerConfig,
) -> list[int]:
    if total_frames <= 0:
        raise ValidationError("total_frames must be positive")
    minimum = round(config.shot_min_seconds * config.fps)
    target_span = round(config.shot_target_seconds * config.fps)
    maximum = round(config.shot_max_seconds * config.fps)
    candidates = sorted({int(value) for value in transients if 0 < int(value) < total_frames})
    boundaries = [0]
    current = 0
    while total_frames - current > maximum:
        low = current + minimum
        high = min(current + maximum, total_frames - minimum)
        target = min(current + target_span, high)
        eligible = [value for value in candidates if low <= value <= high]
        if eligible:
            selected = min(eligible, key=lambda value: (abs(value - target), value))
        else:
            selected = max(low, min(target, high))
        if selected <= current:
            raise ValidationError("Shot planner failed to advance the timeline")
        boundaries.append(selected)
        current = selected
    boundaries.append(total_frames)
    if len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < minimum:
        boundaries.pop(-2)
    if boundaries[0] != 0 or boundaries[-1] != total_frames:
        raise ValidationError("Shot boundaries do not cover the entire song")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValidationError("Shot boundaries are not strictly increasing")
    return boundaries


def _section_for(frame: int, sections: tuple[SongSection, ...]) -> SongSection:
    for section in sections:
        if section.start_frame <= frame < section.end_frame:
            return section
    return sections[-1]


def _seed(job_id: str, output_format: str, index: int) -> int:
    return zlib.crc32(f"{job_id}:{output_format}:{index}".encode("utf-8")) & 0x7FFFFFFF


def build_plan(
    request: MusicVideoRequest,
    treatment: dict,
    analysis: SongAnalysis,
    output_format: str,
    config: WorkerConfig,
    transcript: LyricTranscript | None = None,
) -> EditorialPlan:
    width, height = FORMAT_DIMENSIONS[output_format]
    render_width, render_height = RENDER_DIMENSIONS[output_format]
    lyric_boundaries: list[int] = []
    if transcript:
        for segment in transcript.segments:
            lyric_boundaries.extend((segment.start_frame, segment.end_frame))
    planning_config = config
    reference_profile = treatment.get("reference_style_profile") or {}
    reference_shot_seconds = reference_profile.get("average_shot_seconds")
    if isinstance(reference_shot_seconds, (int, float)) and math.isfinite(reference_shot_seconds):
        target_seconds = min(
            config.shot_max_seconds,
            max(config.shot_min_seconds, float(reference_shot_seconds)),
        )
        planning_config = replace(config, shot_target_seconds=target_seconds)
    boundaries = choose_boundaries(
        analysis.total_frames,
        (*analysis.transients, *lyric_boundaries),
        planning_config,
    )
    performer_ids = list(treatment.get("performer_ids", []))
    counters: dict[str, int] = {}
    shots: list[Shot] = []
    previous_family = ""
    performer_shot_index = 0
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        midpoint = start + (end - start) // 2
        section = _section_for(midpoint, analysis.sections)
        sequence = SECTION_FAMILIES.get(section.id, SECTION_FAMILIES["development"])
        offset = counters.get(section.id, 0)
        family = sequence[offset % len(sequence)]
        counters[section.id] = offset + 1
        if family == previous_family and len(sequence) > 1:
            family = sequence[(offset + 1) % len(sequence)]
        previous_family = family
        has_performer = family in PERFORMER_FAMILIES and bool(performer_ids)
        assigned: list[str] = []
        if has_performer:
            visible_count = min(request.max_people_visible_per_shot, len(performer_ids))
            group_scene = visible_count > 1 and family in GROUP_FAMILIES and (
                family == "wide_movement"
                or section.id == "climax"
                or performer_shot_index % 4 == 3
            )
            if group_scene:
                assigned = [
                    performer_ids[(performer_shot_index + offset) % len(performer_ids)]
                    for offset in range(visible_count)
                ]
            else:
                assigned = [performer_ids[performer_shot_index % len(performer_ids)]]
            performer_shot_index += 1
        vocals = bool(section.vocals_present and family in {"close_performance", "medium_performance", "firelight_performance"})
        shots.append(
            Shot(
                id=f"shot-{index:03d}",
                start_frame=start,
                frame_count=end - start,
                section=section.id,
                family=family,
                energy=section.energy,
                vocals_present=vocals,
                performer_ids=assigned,
                anchor_role=family,
                seed=_seed(request.job_id, output_format, index),
            )
        )

    validate_plan(shots, analysis.total_frames)
    return EditorialPlan(
        schema_version=3,
        fps=config.fps,
        format=output_format,
        width=width,
        height=height,
        total_frames=analysis.total_frames,
        shots=shots,
        render_width=render_width,
        render_height=render_height,
    )


def validate_plan(shots: list[Shot], total_frames: int) -> None:
    if not shots:
        raise ValidationError("Editorial plan has no shots")
    if shots[0].start_frame != 0:
        raise ValidationError("Editorial plan must begin at frame zero")
    ids = [shot.id for shot in shots]
    if len(ids) != len(set(ids)):
        raise ValidationError("Editorial plan has duplicate shot ids")
    expected = 0
    for shot in shots:
        if shot.start_frame != expected:
            raise ValidationError(f"Gap or overlap before {shot.id}")
        if shot.frame_count <= 0:
            raise ValidationError(f"Shot {shot.id} has invalid frame count")
        expected = shot.end_frame
    if expected != total_frames:
        raise ValidationError(f"Plan ends at frame {expected}, expected {total_frames}")


def compile_legacy_manifest(plan: EditorialPlan, manifest_dir: Path) -> dict:
    base = manifest_dir.resolve()
    shots = []
    for shot in plan.shots:
        if shot.anchor_image is None:
            raise ValidationError(f"Shot {shot.id} does not have an anchor image")
        anchor_path = Path(shot.anchor_image).resolve()
        anchor = os.path.relpath(anchor_path, base).replace(os.sep, "/")
        shots.append(
            {
                "id": shot.id,
                "start_seconds": shot.start_frame / plan.fps,
                "duration_seconds": shot.frame_count / plan.fps,
                "prompt": shot.prompt,
                "image": anchor,
            }
        )
    return {"schema_version": 1, "shots": shots}
