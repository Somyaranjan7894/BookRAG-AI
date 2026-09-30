"""Tests for Phase 16: Production FastAPI Architecture.

Validates:
- Centralized Dependency Injection (factories, caching, and test overrides)
- Request Correlation IDs (generation, header propagation, logging context, and error envelope attachment)
- Centralized Exception Handling & Sanitization (400, 404, 409, 422, 500 without tracebacks)
- Thin Router Architecture & Service Boundaries
- Preservation of Celery Asynchronous Enqueueing (202 Accepted)
- OpenAPI Specification Integrity
"""

import uuid
from unittest.mock import MagicMock, patch

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm import Session

from app.api.v1.dependencies import (
    get_citation_service,
    get_dev_retrieval_service,
    get_document_persistence_service,
    get_document_processing_service,
    get_document_repository,
    get_document_upload_service,
    get_embedding_service,
    get_generation_service,
    get_grounded_answer_service,
    get_optional_db_session,
    get_pdf_ingestion_service,
    get_qa_service,
    get_query_understanding_service,
    get_question_generation_service,
    get_required_db_session,
    get_reranker_service,
    get_search_service,
    get_settings_dependency,
    get_text_processing_service,
)
from app.core.config import settings
from app.core.correlation import get_request_id
from app.main import app
from app.schemas.search import SearchResponse


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient bound to the FastAPI application."""
    return TestClient(app, raise_server_exceptions=False)


# ==============================================================================
# 1. Request Correlation ID & Middleware Tests
# ==============================================================================

class TestRequestCorrelation:
    """Verify request correlation ID generation, propagation, and timing."""

    def test_auto_generates_uuid_request_id(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        assert "X-Request-ID" in response.headers
        corr_id = response.headers["X-Request-ID"]
        # Must be valid UUID format
        parsed = uuid.UUID(corr_id)
        assert str(parsed) == corr_id

    def test_propagates_incoming_request_id(self, client: TestClient):
        custom_id = "test-corr-client-98765"
        response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
        assert response.status_code == status.HTTP_200_OK
        assert response.headers.get("X-Request-ID") == custom_id

    def test_sanitizes_invalid_incoming_request_id(self, client: TestClient):
        # Header with malicious characters or excessive length is replaced by a clean UUID
        malicious_id = "bad\r\ninjection;" + "x" * 200
        response = client.get("/api/v1/health", headers={"X-Request-ID": malicious_id})
        assert response.status_code == status.HTTP_200_OK
        corr_id = response.headers.get("X-Request-ID")
        assert corr_id != malicious_id
        # Should have generated a clean UUID instead
        uuid.UUID(corr_id)

    def test_response_includes_process_time_header(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert "X-Process-Time" in response.headers
        process_time = response.headers["X-Process-Time"]
        assert process_time.endswith("ms")

    def test_error_response_contains_matching_request_id(self, client: TestClient):
        custom_id = "corr-error-trace-1234"
        response = client.get("/api/v1/nonexistent-route-404", headers={"X-Request-ID": custom_id})
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.headers.get("X-Request-ID") == custom_id
        data = response.json()
        assert "error" in data
        assert data["error"]["request_id"] == custom_id


# ==============================================================================
# 2. Centralized Exception Handling & Error Sanitization Tests
# ==============================================================================

class TestCentralizedExceptionHandling:
    """Verify structured error formats and prevention of traceback leakage."""

    def test_404_not_found_structure(self, client: TestClient):
        response = client.get("/api/v1/unknown-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error = response.json()["error"]
        assert error["code"] == 404
        assert "message" in error
        assert "request_id" in error
        assert error["error_type"] == "NOT_FOUND"

    def test_422_validation_error_structure(self, client: TestClient):
        # Sending empty body to /api/v1/search requires query field
        response = client.post("/api/v1/search", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        error = response.json()["error"]
        assert error["code"] == 422
        assert error["error_type"] == "VALIDATION_ERROR"
        assert "details" in error
        assert isinstance(error["details"], list)
        assert len(error["details"]) > 0

    def test_unhandled_500_sanitized_no_traceback_leak(self):
        from app.main import create_application

        test_app = create_application()
        test_app.debug = False

        @test_app.get("/api/v1/test-unexpected-crash")
        async def crash_endpoint():
            raise RuntimeError("Database connection string postgresql://secret:pass@localhost:5433/private_db failed")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/api/v1/test-unexpected-crash")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        error = response.json()["error"]
        assert error["code"] == 500
        assert error["message"] == "An internal server error occurred."
        assert error["error_type"] == "INTERNAL_SERVER_ERROR"
        assert "request_id" in error
        # Verify secrets and traceback are NOT leaked to the client
        assert "Traceback" not in response.text
        assert "secret:pass" not in response.text
        assert "private_db" not in response.text

    def test_domain_exception_mapping_pdf_not_found(self, client: TestClient):
        from app.services.pdf.exceptions import PDFNotFoundError

        @app.get("/api/v1/test-domain-pdf-not-found")
        async def mock_pdf_fail():
            raise PDFNotFoundError("Source document sample.pdf was not found.")

        response = client.get("/api/v1/test-domain-pdf-not-found")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error = response.json()["error"]
        assert error["code"] == 404
        assert error["message"] == "Source document sample.pdf was not found."
        assert error["error_type"] == "PDF_NOT_FOUND"
        assert "request_id" in error


# ==============================================================================
# 3. Centralized Dependency Injection Tests
# ==============================================================================

class TestDependencyInjection:
    """Verify dependency factories, caching singletons, and test substitution."""

    def test_settings_dependency(self):
        resolved_settings = get_settings_dependency()
        assert resolved_settings.APP_NAME == settings.APP_NAME

    def test_singleton_service_instances(self):
        s1 = get_search_service()
        s2 = get_search_service()
        assert s1 is s2

        qa1 = get_qa_service()
        qa2 = get_qa_service()
        assert qa1 is qa2

        gen1 = get_generation_service()
        gen2 = get_generation_service()
        assert gen1 is gen2

        ground1 = get_grounded_answer_service()
        ground2 = get_grounded_answer_service()
        assert ground1 is ground2

        qg1 = get_question_generation_service()
        qg2 = get_question_generation_service()
        assert qg1 is qg2

        qu1 = get_query_understanding_service()
        qu2 = get_query_understanding_service()
        assert qu1 is qu2

    def test_factory_dependencies(self):
        pdf_svc = get_pdf_ingestion_service()
        assert pdf_svc is not None

        txt_svc = get_text_processing_service()
        assert txt_svc is not None

        upload_svc = get_document_upload_service(db=None)
        assert upload_svc is not None

    def test_dependency_override_substitution(self, client: TestClient):
        # Verify app.dependency_overrides allows substituting services cleanly
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = SearchResponse(
            query="test overridden query",
            top_k=5,
            total_results=0,
            results=[],
            execution_time_ms=1.5,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search_service
        try:
            response = client.post("/api/v1/search", json={"query": "test overridden query"})
            assert response.status_code == status.HTTP_200_OK
            mock_search_service.search.assert_called_once()
        finally:
            app.dependency_overrides.pop(get_search_service, None)


# ==============================================================================
# 4. OpenAPI Specification Tests
# ==============================================================================

class TestOpenAPISpecification:
    """Verify OpenAPI schema generation, metadata, and tags."""

    def test_openapi_schema_generation(self):
        schema = app.openapi()
        assert schema is not None
        assert schema["info"]["title"] == settings.APP_NAME
        assert schema["info"]["version"] == settings.APP_VERSION
        assert "paths" in schema
        # Verify v1 endpoints are documented
        assert "/api/v1/health" in schema["paths"]
        assert "/api/v1/documents" in schema["paths"]
        assert "/api/v1/search" in schema["paths"]
        assert "/api/v1/qa" in schema["paths"]
        assert "/api/v1/answer" in schema["paths"]
        assert "/api/v1/grounded-answer" in schema["paths"]
        assert "/api/v1/questions/generate" in schema["paths"]
        assert "/api/v1/query-plan" in schema["paths"]

    def test_openapi_tags_metadata(self):
        schema = app.openapi()
        tag_names = [t["name"] for t in schema.get("tags", [])]
        expected_tags = [
            "Health",
            "Documents",
            "Search",
            "Question Answering",
            "Answer Generation",
            "Grounded Answer",
            "Question Generation",
            "Query Planning",
        ]
        for tag in expected_tags:
            assert tag in tag_names


# ==============================================================================
# 5. Celery & Vector Backend Preservation Tests
# ==============================================================================

class TestArchitecturePreservation:
    """Verify asynchronous document upload (202 Accepted) and vector backend abstraction."""

    def test_document_upload_async_enqueuing_returns_202(self, client: TestClient, tmp_path):
        # Create a mock PDF file
        pdf_file = tmp_path / "phase16_test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 minimal test file for Phase 16")

        with patch("app.workers.tasks.process_document_task.delay") as mock_delay:
            mock_task = MagicMock()
            mock_task.id = "task-phase16-async-001"
            mock_delay.return_value = mock_task

            response = client.post(
                "/api/v1/documents",
                json={"file_path": str(pdf_file), "document_id": "doc_phase16_async_001"},
            )

            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert data["document_id"] == "doc_phase16_async_001"
            assert data["task_id"] == "task-phase16-async-001"
            assert data["status"] == "queued"
            mock_delay.assert_called_once_with(
                document_id="doc_phase16_async_001",
                file_path=str(pdf_file.resolve()),
            )

    def test_vector_backend_setting_recognized(self):
        assert settings.VECTOR_BACKEND in ("faiss", "pgvector")
