"""Turns a claimed job into something an adapter can run.

The worker holds no copy of the workflow registry. Everything it needs about a
workflow — parameters, input roles, and the private execution block — arrives
with the claim. That means the YAML definitions stay the single source of truth
and a worker can never disagree with the API about what a workflow is
(directive §11, §26).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from worker.adapters.base import AdapterInput, AdapterJob, GenerationAdapter
from worker.adapters.registry import get_adapter


def build_adapter_job(
    claim: dict[str, Any],
    *,
    workspace: Path | None = None,
    cancelled: asyncio.Event | None = None,
    deadline_monotonic: float | None = None,
) -> AdapterJob:
    return AdapterJob(
        job_id=str(claim["job_id"]),
        workflow_id=str(claim["workflow_id"]),
        workflow_version=str(claim.get("workflow_version", "1")),
        prompt=str(claim.get("prompt", "")),
        parameters=dict(claim.get("parameters") or {}),
        inputs=[
            AdapterInput(
                role=str(item["role"]),
                kind=str(item["kind"]),
                content_type=str(item["content_type"]),
                download_url=str(item["download_url"]),
            )
            for item in (claim.get("inputs") or [])
        ],
        execution=dict(claim.get("execution") or {}),
        output_content_type=str(claim.get("output_content_type", "application/octet-stream")),
        workspace=workspace or Path(),
        _cancelled=cancelled,
        _deadline_monotonic=deadline_monotonic,
    )


def resolve_adapter(job: AdapterJob) -> GenerationAdapter:
    """Picks the adapter from the workflow's private execution block.

    Routing lives in version-controlled YAML rather than in worker code, so
    moving a workflow onto a real provider in M2 is a configuration change and a
    deploy — not a code change in three places.

    `runtime_by_quality` is the Fast/Best toggle (client-approved 27 Aug
    2026): a mapping from the public `quality` parameter's value to a
    runtime, e.g. `{fast: ltx, best: h3_comfy}`. A missing or unmapped
    quality falls back to plain `runtime`, so the toggle being absent from a
    request — or a value the YAML never named — routes exactly as before.
    """
    runtime = str(job.execution.get("runtime") or "mock")
    by_quality = job.execution.get("runtime_by_quality")
    if isinstance(by_quality, dict):
        quality = str(job.parameters.get("quality") or "").strip().lower()
        runtime = str(by_quality.get(quality) or runtime)
    return get_adapter(runtime)
