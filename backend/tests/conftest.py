"""Pytest configuration and shared test fixtures."""

import os

# Configure test environment BEFORE importing app modules so the cached
# Settings object uses these values.  Mirrors CI environment in .github/workflows/ci.yml.
os.environ.setdefault("DATABASE_URL", "sqlite:///test_bookrag.db")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("VECTOR_BACKEND", "faiss")

from pathlib import Path
import sys
from typing import AsyncGenerator, Generator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_application
from app.core import config as _config_module
from app.core.config import get_settings
from app.db.session import get_engine, reset_engine
from app.db.base import Base
import app.models  # noqa: F401

# Clear any previously cached settings so test env vars take effect
get_settings.cache_clear()
_config_module.settings = get_settings()
reset_engine()

# Initialize tables for SQLite test database if using SQLite
if "sqlite" in _config_module.settings.DATABASE_URL:
    Base.metadata.create_all(bind=get_engine())

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


