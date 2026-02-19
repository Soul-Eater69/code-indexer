"""Tests for the health endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from code_indexer.api.app import create_app
from code_indexer.core.config import get_settings


@pytest.fixture
def client() -> TestClient:
    """Create a test client with mocked pipeline/embedder/vector_store."""
    app = create_app()

    # Override lifespan by setting state directly.
    app.state.pipeline = MagicMock()
    app.state.embedder = MagicMock()
    app.state.vector_store = MagicMock()
    app.state.settings = get_settings()

    return TestClient(app, raise_server_exceptions=False)


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ping_returns_pong(client: TestClient) -> None:
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["pong"] is True
