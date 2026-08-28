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
from worker.core.logging import get_logger

logger = get_logger(__name__)


def _execution_for(claim: dict[str, Any]) -> dict[str, Any]:
    """The job's execution block, with any per-quality overlay applied.

    `execution_by_quality` is the companion to `runtime_by_quality`: the same
    Fast/Best control that picks an engine may also need to pick that engine's
    settings — Best is only meaningfully "best" if something about the render
    differs. Keys in the overlay replace keys in the base block; everything
    unmentioned is inherited, and a request whose quality names no overlay is
    byte-identical to the block as written.

    It stays a shallow merge on purpose. A deep merge would let a half-stated
    nested value silently combine with its base and produce a configuration
    nobody wrote down.
    """
    execution = dict(claim.get("execution") or {})
    overlay = execution.get("execution_by_quality")
    if isinstance(overlay, dict):
        quality = str((claim.get("parameters") or {}).get("quality") or "").strip().lower()
        for key, value in (overlay.get(quality) or {}).items():
            execution[str(key)] = value
    return execution


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
        execution=_execution_for(claim),
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
    base = str(job.execution.get("runtime") or "mock")
    runtime = base
    by_quality = job.execution.get("runtime_by_quality")
    if isinstance(by_quality, dict):
        quality = str(job.parameters.get("quality") or "").strip().lower()
        runtime = str(by_quality.get(quality) or base)

    adapter = get_adapter(runtime)
    if runtime != base and not adapter.supports(job.workflow_id):
        # A quality level pointed at an engine that does not run this workflow.
        # The engine's own refusal would be a failed job for a customer who
        # only chose a quality setting, so the base runtime — the one the
        # workflow is defined against — serves it instead.
        #
        # This is a safety net, not a routing strategy: the mapping is still
        # the YAML's business and a deployment that means to withdraw an
        # engine should say so there. It exists because the mapping lives in
        # deployment-local YAML that no test in this repository can see.
        fallback = get_adapter(base)
        if fallback.supports(job.workflow_id):
            logger.warning(
                "runtime_by_quality_unsupported",
                extra={
                    "workflow_id": job.workflow_id,
                    "requested_runtime": runtime,
                    "served_by": base,
                },
            )
            return fallback
    return adapter
