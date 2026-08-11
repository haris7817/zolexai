"""Assembles the v1 API surface.

Versioned from the very first endpoint (directive §15). When a breaking change
is needed, `/api/v2` is added alongside and clients migrate on their own
schedule — nothing is ever silently reshaped underneath a running frontend.
"""

from fastapi import APIRouter

from app.api.v1 import assets, generations, health, internal, workflows

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(workflows.router)
api_router.include_router(generations.router)
api_router.include_router(assets.router)
# Service-token guarded; excluded from the public OpenAPI document.
api_router.include_router(internal.router)
