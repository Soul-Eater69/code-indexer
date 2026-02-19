"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health status.

    Always returns HTTP 200 while the process is running.  Container
    orchestrators (Kubernetes, ECS) use this for liveness probes.
    """
    return HealthResponse(status="ok", version="1.0.0")


@router.get("/ping")
async def ping() -> dict:
    """Minimal ping endpoint for load balancer health checks."""
    return {"pong": True}
