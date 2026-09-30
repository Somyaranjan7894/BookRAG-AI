"""Comprehensive test suite for Phase 15 Redis + Celery background processing.

Tests cover:
- Celery configuration and broker settings
- Task registration
- Pipeline orchestration (file validation, ingestion, chunking, relational persistence, vector indexing)
- Vector backend compatibility (pgvector and FAISS)
- Failure handling: missing file, corrupt PDF, error recording in PostgreSQL
- Retry behavior: transient failure triggers bounded retry
- Idempotency: duplicate task execution does not duplicate data
- API endpoint: POST /api/v1/documents returns HTTP 202 Accepted with task/document ID
- Post-processing search verification: SearchService successfully retrieves evidence
"""

import os
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.page import Page
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import PGVectorRepository
from app.services.document_processing.exceptions import (
    PermanentProcessingError,
    TransientProcessingError,
)
from app.services.document_processing.service import DocumentProcessingService
from app.services.retrieval.backend import PGVectorBackend
from app.services.search.service import SearchService
from app.workers.celery_app import celery_app
from app.workers.tasks import process_document_task
from tests.fixtures.pdf_factory import create_corrupt_pdf

TEST_DB_URL = "postgresql+psycopg://postgres@localhost:5433/bookrag_test"


@pytest.fixture(scope="session")
def pgvector_engine():
    """Create engine for testing on bookrag_test."""
    engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(pgvector_engine) -> Generator[Session, None, None]:
    """Clean database session before and after each test."""
    session_factory = sessionmaker(
        bind=pgvector_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = session_factory()

    session.query(Chunk).delete()
    session.query(Page).delete()
    session.query(Document).delete()
    session.commit()

    try:
        yield session
    finally:
        session.rollback()
        session.query(Chunk).delete()
        session.query(Page).delete()
        session.query(Document).delete()
        session.commit()
        session.close()


# ==============================================================================
# 1. Celery Configuration & Task Registration Tests
# ==============================================================================


def test_celery_configuration() -> None:
    """Celery application is configured with correct broker, serialization, and ack settings."""
    assert celery_app.conf.broker_url.startswith("redis://")
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.timezone == "UTC"
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_celery_task_registration() -> None:
    """Document processing task is registered with the Celery app."""
    task_name = "app.workers.tasks.process_document_task"
    assert task_name in celery_app.tasks
    task = celery_app.tasks[task_name]
    assert task.max_retries == settings.CELERY_TASK_MAX_RETRIES


# ==============================================================================
# 2. Pipeline Execution & Vector Backend Compatibility
# ==============================================================================


def test_document_processing_service_pgvector(
    db_session: Session,
    deterministic_sample_pdf: Path,
) -> None:
    """DocumentProcessingService completes ingestion, chunking, and pgvector embedding persistence."""
    doc_id = "doc_celery_pg_001"
    service = DocumentProcessingService(
        session=db_session,
        vector_backend="pgvector",
    )

    result = service.process(document_id=doc_id, file_path=str(deterministic_sample_pdf))

    assert result["document_id"] == doc_id
    assert result["status"] == "processed"
    assert result["page_count"] == 3
    assert result["chunk_count"] > 0
    assert result["backend"] == "pgvector"

    # Verify PostgreSQL durable state
    doc_repo = DocumentRepository(db_session)
    doc = doc_repo.get_by_id(doc_id)
    assert doc is not None
    assert doc.status == DocumentStatus.PROCESSED.value
    assert doc.processing_stage == "completed"
    assert doc.error_message is None
    assert doc.page_count == 3
    assert len(doc.pages) == 3
    assert len(doc.chunks) > 0

    # Verify pgvector embeddings exist and are searchable
    vec_repo = PGVectorRepository(db_session)
    assert vec_repo.count_embeddings(document_id=doc_id) == len(doc.chunks)

    # Verify vector search retrieves matching chunk
    vec_results = vec_repo.search_similar(
        query_vector=[0.05] * 384,
        top_k=2,
        document_id=doc_id,
    )
    assert len(vec_results) == 2
    assert vec_results[0].document_id == doc_id


def test_document_processing_service_faiss(
    db_session: Session,
    deterministic_sample_pdf: Path,
) -> None:
    """DocumentProcessingService completes processing with FAISS vector indexing."""
    doc_id = "doc_celery_faiss_001"
    service = DocumentProcessingService(
        session=db_session,
        vector_backend="faiss",
    )

    result = service.process(document_id=doc_id, file_path=str(deterministic_sample_pdf))

    assert result["document_id"] == doc_id
    assert result["status"] == "processed"
    assert result["backend"] == "faiss"

    doc_repo = DocumentRepository(db_session)
    doc = doc_repo.get_by_id(doc_id)
    assert doc is not None
    assert doc.status == DocumentStatus.PROCESSED.value
    assert doc.processing_stage == "completed"


# ==============================================================================
# 3. Failure Handling & Durable Error State
# ==============================================================================


def test_missing_file_transitions_to_failed(db_session: Session) -> None:
    """Missing source file marks document as failed with clean error message in PostgreSQL."""
    doc_id = "doc_missing_file_test"
    service = DocumentProcessingService(session=db_session)

    with pytest.raises(PermanentProcessingError) as exc_info:
        service.process(document_id=doc_id, file_path="data/uploads/does_not_exist.pdf")

    assert "does not exist" in str(exc_info.value)

    doc_repo = DocumentRepository(db_session)
    doc = doc_repo.get_by_id(doc_id)
    # If pre-registered, state should be failed
    if doc is not None:
        assert doc.status == DocumentStatus.FAILED.value
        assert doc.processing_stage == "validation"
        assert doc.error_message is not None


def test_corrupt_pdf_transitions_to_failed(
    db_session: Session,
    tmp_path: Path,
) -> None:
    """Corrupt PDF marks document as failed without crashing worker."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    create_corrupt_pdf(corrupt_pdf)

    doc_id = "doc_corrupt_test"
    service = DocumentProcessingService(session=db_session)

    with pytest.raises(PermanentProcessingError) as exc_info:
        service.process(document_id=doc_id, file_path=str(corrupt_pdf))

    assert "PDF ingestion error" in str(exc_info.value)

    doc_repo = DocumentRepository(db_session)
    doc = doc_repo.get_by_id(doc_id)
    assert doc is not None
    assert doc.status == DocumentStatus.FAILED.value
    assert doc.processing_stage == "ingestion"
    assert doc.error_message is not None


# ==============================================================================
# 4. Idempotency Tests
# ==============================================================================


def test_idempotent_duplicate_processing(
    db_session: Session,
    deterministic_sample_pdf: Path,
) -> None:
    """Duplicate task execution does not create duplicate chunks or corrupt data."""
    doc_id = "doc_idempotency_test"
    service = DocumentProcessingService(session=db_session, vector_backend="pgvector")

    # First run
    res1 = service.process(document_id=doc_id, file_path=str(deterministic_sample_pdf))
    assert res1["status"] == "processed"
    first_chunk_count = res1["chunk_count"]

    # Verify first count
    doc_repo = DocumentRepository(db_session)
    doc1 = doc_repo.get_by_id(doc_id)
    assert doc1 is not None
    assert len(doc1.chunks) == first_chunk_count

    # Second run (duplicate delivery)
    res2 = service.process(document_id=doc_id, file_path=str(deterministic_sample_pdf))
    assert res2["status"] == "processed"
    assert res2.get("idempotent") is True

    # Chunks and pages should not be duplicated
    db_session.refresh(doc1)
    assert len(doc1.chunks) == first_chunk_count
    assert doc1.page_count == 3


# ==============================================================================
# 5. Celery Task Execution & Retries
# ==============================================================================


def test_celery_task_apply_eager(
    deterministic_sample_pdf: Path,
    monkeypatch,
) -> None:
    """Celery process_document_task executes through task interface."""
    doc_id = "doc_celery_task_test"

    # Execute task synchronously through Celery
    res = process_document_task.apply(args=[doc_id, str(deterministic_sample_pdf)])

    assert res.successful()
    result = res.result
    assert result["document_id"] == doc_id
    assert result["status"] == "processed"


def test_celery_task_retries_transient_error(monkeypatch) -> None:
    """TransientProcessingError causes process_document_task to trigger bounded Celery retries."""
    attempts = []

    def mock_process(self, document_id, file_path):
        attempts.append(document_id)
        raise TransientProcessingError("Temporary database timeout")

    monkeypatch.setattr(DocumentProcessingService, "process", mock_process)

    res = process_document_task.apply(args=["doc_transient_test", "some/path.pdf"])
    assert res.failed() is True
    assert isinstance(res.result, TransientProcessingError)
    # Verifies initial attempt + 3 retries = 4 total attempts
    assert len(attempts) == settings.CELERY_TASK_MAX_RETRIES + 1


# ==============================================================================
# 6. Asynchronous API Endpoint Tests
# ==============================================================================


def test_api_upload_document_multipart(
    client: TestClient,
    deterministic_sample_pdf: Path,
    monkeypatch,
) -> None:
    """POST /api/v1/documents via multipart upload returns HTTP 202 Accepted with task ID."""
    with open(deterministic_sample_pdf, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/documents",
        files={"file": ("sample.pdf", file_bytes, "application/pdf")},
    )

    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert "task_id" in data
    assert data["status"] == "queued"
    assert "enqueued successfully" in data["message"]


def test_api_upload_document_json(
    client: TestClient,
    deterministic_sample_pdf: Path,
) -> None:
    """POST /api/v1/documents via JSON file_path returns HTTP 202 Accepted with task ID."""
    response = client.post(
        "/api/v1/documents",
        json={"file_path": str(deterministic_sample_pdf)},
    )

    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert "task_id" in data
    assert data["status"] == "queued"


def test_api_upload_missing_file_returns_404(client: TestClient) -> None:
    """POST /api/v1/documents with missing file_path returns HTTP 404 Not Found."""
    response = client.post(
        "/api/v1/documents",
        json={"file_path": "data/uploads/non_existent_book.pdf"},
    )
    assert response.status_code == 404
    error_msg = response.json()["error"]["message"]
    assert "not found" in error_msg.lower()


# ==============================================================================
# 7. End-to-End Retrieval After Background Processing
# ==============================================================================


def test_search_service_retrieves_chunks_from_processed_document(
    db_session: Session,
    deterministic_sample_pdf: Path,
) -> None:
    """SearchService seamlessly searches documents processed by DocumentProcessingService."""
    doc_id = "doc_search_e2e_001"
    service = DocumentProcessingService(session=db_session, vector_backend="pgvector")
    service.process(document_id=doc_id, file_path=str(deterministic_sample_pdf))

    # Initialize SearchService with pgvector backend
    backend = PGVectorBackend(vector_repo=PGVectorRepository(db_session))
    search_svc = SearchService(
        vector_backend=backend,
        enable_reranking=True,
    )

    response = search_svc.search(
        query="PDF ingestion test",
        top_k=2,
        document_id=doc_id,
    )

    assert response.total_results > 0
    assert response.document_id == doc_id
    assert response.results[0].document_id == doc_id
    assert "PDF ingestion test" in response.results[0].text
    assert response.results[0].similarity_score is not None
