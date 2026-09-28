"""Pytest configuration and shared test fixtures."""

from typing import AsyncGenerator, Generator
import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_application


@pytest.fixture(scope="session")
def app() -> FastAPI:
    """Provide a test application instance."""
    return create_application()


@pytest.fixture(scope="session")
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Provide a synchronous FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an asynchronous HTTPX test client."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client

