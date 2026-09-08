from __future__ import annotations

import math
from typing import Any

from .config import WorkerConfig


def build_latency_capacity_plan(
    *,
    audio_seconds: float,
    format_count: int,
    render_backend: str,
    config: WorkerConfig,
) -> dict[str, Any]:
    """Translate a measured backend baseline into a transparent capacity target.

    This is deliberately based on the deployer's real end-to-end benchmark. It
    does not pretend that a configuration flag can make generation faster.
    """
    workload_ratio = (audio_seconds / config.baseline_audio_seconds) * max(1, format_count)
    baseline_lane_seconds = (
        config.baseline_job_seconds * config.baseline_render_concurrency * workload_ratio
    )
    required_concurrency = max(1, math.ceil(baseline_lane_seconds / config.target_job_seconds))
    projected_seconds = baseline_lane_seconds / config.render_concurrency
    warm_renderer = render_backend == "command"
    capacity_ready = warm_renderer and config.render_concurrency >= required_concurrency
    return {
        "target_seconds": config.target_job_seconds,
        "target_minutes": round(config.target_job_seconds / 60, 2),
        "baseline_audio_seconds": config.baseline_audio_seconds,
        "baseline_job_seconds": config.baseline_job_seconds,
        "baseline_render_concurrency": config.baseline_render_concurrency,
        "requested_format_count": max(1, format_count),
        "required_render_concurrency": required_concurrency,
        "configured_render_concurrency": config.render_concurrency,
        "required_capacity_multiplier": round(
            config.baseline_job_seconds * workload_ratio / config.target_job_seconds,
            3,
        ),
        "capacity_only_projection_seconds": round(projected_seconds, 1),
        "warm_renderer_required": True,
        "warm_renderer_configured": warm_renderer,
        "finishing_stage": "separate_4k_upscale",
        "upscale_backend": config.upscale_backend,
        "upscale_included_in_target": True,
        "capacity_ready": capacity_ready,
        "guaranteed": False,
        "note": (
            "Capacity is configured for the target; real completion time still depends on renderer throughput, retries, queue load, storage, and measured 4K finishing speed."
            if capacity_ready
            else "Target needs the existing warm command renderer and at least the calculated parallel render capacity."
        ),
    }
