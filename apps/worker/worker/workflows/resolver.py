"""Turns a claimed job into something an adapter can run.

The worker holds no copy of the workflow registry. Everything it needs about a
workflow — parameters, input roles, and the private execution block — arrives
with the claim. That means the YAML definitions stay the single source of truth
and a worker can never disagree with the API about what a workflow is
(directive §11, §26).
"""

from __future__ import annotations

from typing import Any

from worker.adapters.base import AdapterInput, AdapterJob, GenerationAdapter
from worker.adapters.registry import get_adapter


def build_adapter_job(claim: dict[str, Any]) -> AdapterJob:
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
    )


def resolve_adapter(job: AdapterJob) -> GenerationAdapter:
    """Picks the adapter from the workflow's private execution block.

    Routing lives in version-controlled YAML rather than in worker code, so
    moving a workflow onto a real provider in M2 is a configuration change and a
    deploy — not a code change in three places.
    """
    runtime = str(job.execution.get("runtime") or "mock")
    return get_adapter(runtime)
