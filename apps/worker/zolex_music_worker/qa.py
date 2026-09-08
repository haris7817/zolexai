from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import WorkerConfig
from .media import brightness_statistics, video_probe
from .models import EditorialPlan, Shot
from .utils import atomic_write_json, format_template_argv, run_command, sha256_file


@dataclass(frozen=True)
class QaResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, Any]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_shot_qa(
    clip: Path,
    shot: Shot,
    plan: EditorialPlan,
    config: WorkerConfig,
    report_path: Path,
) -> QaResult:
    probe = video_probe(clip, config, count_frames=True)
    brightness = brightness_statistics(clip, config)
    checks = {
        "frame_count": probe["frame_count"] == shot.frame_count,
        "fps": abs(probe["fps"] - plan.fps) < 0.01,
        "dimensions": probe["width"] == plan.scene_width and probe["height"] == plan.scene_height,
        "decodes": probe["frame_count"] > 0,
        "no_abrupt_darkening": brightness["min_to_median_ratio"] >= 0.30,
        "no_extreme_luminance_jump": brightness["max_frame_step"] <= 65.0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    metrics: dict[str, Any] = {
        "probe": probe,
        "brightness": brightness,
        "sha256": sha256_file(clip),
    }

    if config.external_qa_command:
        external_report = report_path.with_suffix(".external.json")
        shot_json = report_path.with_suffix(".shot.json")
        atomic_write_json(shot_json, shot.to_dict())
        values = {
            "clip": str(clip),
            "report": str(external_report),
            "shot_json": str(shot_json),
            "format": plan.format,
            "width": plan.scene_width,
            "height": plan.scene_height,
            "delivery_width": plan.width,
            "delivery_height": plan.height,
            "fps": plan.fps,
        }
        try:
            run_command(
                format_template_argv(config.external_qa_command, values),
                timeout=min(config.command_timeout_seconds, 1800),
                log_path=report_path.with_suffix(".external-command.json"),
            )
            checks["external_creative_qa"] = True
        except Exception as exc:  # Preserve evidence and let the retry policy handle it.
            checks["external_creative_qa"] = False
            failures.append("external_creative_qa")
            metrics["external_qa_error"] = str(exc)

    result = QaResult(not failures, checks, metrics, tuple(failures))
    atomic_write_json(report_path, result.to_dict())
    return result


def run_delivery_qa(
    output: Path,
    plan: EditorialPlan,
    config: WorkerConfig,
    report_path: Path,
) -> QaResult:
    probe = video_probe(output, config, count_frames=True)
    checks = {
        "frame_count": probe["frame_count"] == plan.total_frames,
        "fps": abs(probe["fps"] - plan.fps) < 0.01,
        "dimensions": probe["width"] == plan.width and probe["height"] == plan.height,
        "has_audio": bool(probe["has_audio"]),
        "duration": abs(probe["duration"] - plan.total_frames / plan.fps) <= 0.08,
    }
    failures = [name for name, passed in checks.items() if not passed]
    result = QaResult(
        passed=not failures,
        checks=checks,
        metrics={"probe": probe, "sha256": sha256_file(output)},
        failures=tuple(failures),
    )
    atomic_write_json(report_path, result.to_dict())
    return result
