"""Automated tests for FastAPI startup, health check endpoint, and error response structure."""

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

