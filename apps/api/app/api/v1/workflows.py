"""Public workflow catalogue.

Everything the frontend knows about what ZolexAI can do comes from here. The
responses are built by explicit projection in `WorkflowDefinition.to_public()`,
so no provider, model, runtime or hardware detail can reach a browser
(directive §11, §12; guarded by a test).
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.api.deps import Registry
from app.schemas.workflow import WorkflowListResponse, WorkflowPublic

router = APIRouter(prefix="/workflows", tags=["workflows"])

#: Definitions are version-controlled files loaded at startup, so they change
#: only on deploy. A short shared cache spares the API a request per page load
#: without risking a stale catalogue for long.
_CACHE_CONTROL = "public, max-age=60, stale-while-revalidate=300"


@router.get("", response_model=WorkflowListResponse, summary="List available tools")
async def list_workflows(registry: Registry, response: Response) -> WorkflowListResponse:
    response.headers["Cache-Control"] = _CACHE_CONTROL
    return WorkflowListResponse(workflows=registry.list_public())


@router.get("/{workflow_id}", response_model=WorkflowPublic, summary="One tool")
async def get_workflow(
    workflow_id: str, registry: Registry, response: Response
) -> WorkflowPublic:
    response.headers["Cache-Control"] = _CACHE_CONTROL
    return registry.get_public(workflow_id)
