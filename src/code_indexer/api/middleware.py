"""
FastAPI middleware components.

RequestLoggingMiddleware
------------------------
Logs every HTTP request with method, path, status code, and duration.
Uses ``structlog`` for structured JSON output in production and
human-readable coloured output in development.

APIKeyMiddleware
---------------
Simple bearer-token authentication.  When ``settings.api.api_key`` is
non-empty, all requests to non-health endpoints must include:

    Authorization: Bearer <api_key>

Health check endpoints (``/health``, ``/ping``) are always exempted so that
load balancers and container orchestrators can probe liveness without a token.
"""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Paths that are always public (no auth required).
_PUBLIC_PATHS = frozenset({"/health", "/ping", "/docs", "/redoc", "/openapi.json"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with timing information."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=request.client.host if request.client else "unknown",
        )
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce bearer token authentication on non-public endpoints.

    Args:
        api_key: The expected token value.
    """

    def __init__(self, app: object, *, api_key: str) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)  # type: ignore[operator]

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Missing Authorization header"},
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != self._api_key:
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "detail": "Invalid API key"},
            )

        return await call_next(request)  # type: ignore[operator]
