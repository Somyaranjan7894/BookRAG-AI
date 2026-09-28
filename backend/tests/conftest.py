"""Pytest configuration and shared test fixtures."""

from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_application
from tests.fixtures.pdf_factory import (
    create_corrupt_pdf,
    create_deterministic_pdf,
    create_zero_page_pdf,
)


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


@pytest.fixture
def deterministic_sample_pdf(tmp_path: Path) -> Path:
    """Provide a 3-page deterministic PDF with verified text content and metadata."""
    pdf_path = tmp_path / "sample_book.pdf"
    pages = [
        "BookRAG AI Phase 1",
        "This is a PDF ingestion test.",
        "Retrieval comes later.",
    ]
    metadata = {
        "title": "BookRAG AI Test Document",
        "author": "Antigravity Engineering",
        "subject": "Phase 1 Ingestion Verification",
    }
    return create_deterministic_pdf(pdf_path, pages_content=pages, metadata=metadata)


@pytest.fixture
def empty_and_low_text_pdf(tmp_path: Path) -> Path:
    """Provide a 3-page PDF containing a normal page, an empty page, and a low-text page."""
    pdf_path = tmp_path / "diagnostic_pages.pdf"
    pages = [
        "This is a normal page with sufficient descriptive text content for ingestion validation.",
        "",  # completely empty page
        "Short note.",  # low text page (< 50 characters)
    ]
    return create_deterministic_pdf(pdf_path, pages_content=pages)


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Provide a non-PDF file with a .pdf extension."""
    return create_corrupt_pdf(tmp_path / "corrupt_document.pdf")


@pytest.fixture
def zero_page_pdf(tmp_path: Path) -> Path:
    """Provide a 0-page PDF document."""
    return create_zero_page_pdf(tmp_path / "zero_pages.pdf")


