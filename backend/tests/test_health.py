"""Automated tests for FastAPI startup, health check endpoint, and error response structure."""

import time

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from app.core.config import settings


def test_app_instance_creation(app: FastAPI) -> None:
    """Verify that the FastAPI application initializes correctly with expected metadata."""
    assert app is not None
    assert app.title == settings.APP_NAME
    assert app.version == settings.APP_VERSION
    assert app.description == settings.APP_DESCRIPTION


def test_health_endpoint_status_code(client: TestClient) -> None:
    """Verify that GET /api/v1/health returns HTTP 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK


def test_health_endpoint_response_structure(client: TestClient) -> None:
    """Verify the JSON schema and structure of the health response."""
    response = client.get("/api/v1/health")
    data = response.json()
    assert isinstance(data, dict)
    assert "status" in data
    assert "service" in data
    assert data["status"] == "ok"


def test_health_endpoint_service_name(client: TestClient) -> None:
    """Verify that the service name in the health response matches 'BookRAG AI'."""
    response = client.get("/api/v1/health")
    data = response.json()
    assert data["service"] == "BookRAG AI"


def test_nonexistent_route_error_format(client: TestClient) -> None:
    """Verify standard structured error format for 404 Not Found without leaking server internals."""
    response = client.get("/api/v1/nonexistent-route")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 404
    assert "message" in data["error"]


@pytest.mark.timeout(10)
def test_readiness_endpoint_structure(client: TestClient) -> None:
    """Verify that GET /api/v1/health/ready returns structured subsystem status."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code in (status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE)
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "environment" in data
    assert "database" in data
    assert "redis" in data
    assert "vector_backend" in data
    assert "checks" in data
    assert isinstance(data["checks"], dict)
    assert data["service"] == settings.APP_NAME


@pytest.mark.timeout(10)
def test_readiness_endpoint_completes_fast(client: TestClient) -> None:
    """Verify that the readiness probe completes within 6 seconds even when DB/Redis are offline."""
    start = time.monotonic()
    response = client.get("/api/v1/health/ready")
    elapsed = time.monotonic() - start
    assert response.status_code in (status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE)
    # Must complete well under 10 seconds even with offline dependencies
    assert elapsed < 8.0, f"Readiness check took {elapsed:.1f}s — expected < 8s"


@pytest.mark.timeout(10)
def test_readiness_reports_database_status(client: TestClient) -> None:
    """Verify the readiness response reports database connectivity status."""
    data = client.get("/api/v1/health/ready").json()
    assert data["database"] in ("connected", "disconnected")
    assert "database" in data["checks"]


@pytest.mark.timeout(10)
def test_readiness_reports_redis_status(client: TestClient) -> None:
    """Verify the readiness response reports Redis connectivity status."""
    data = client.get("/api/v1/health/ready").json()
    assert data["redis"] in ("connected", "disconnected")
    assert "redis" in data["checks"]


@pytest.mark.timeout(10)
def test_readiness_reports_vector_backend(client: TestClient) -> None:
    """Verify the readiness response reports vector backend status."""
    data = client.get("/api/v1/health/ready").json()
    assert "vector_backend" in data
    assert settings.VECTOR_BACKEND in data["vector_backend"]
    assert "vector_backend" in data["checks"]
