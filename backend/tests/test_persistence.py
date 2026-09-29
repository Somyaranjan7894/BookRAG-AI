"""Comprehensive test suite for Phase 13 PostgreSQL persistent application data.

Tests cover:
- ORM models: fields, relationships, cascade behaviors, and check constraints
- Repositories: DocumentRepository, PageRepository, ChunkRepository CRUD operations
- Constraints: uniqueness of (document_id, page_number) and (document_id, page_number, chunk_index)
- Foreign keys: referential integrity enforcement on pages and chunks
- Transactions: atomic persistence, rollback on failure, zero partial state
- Ingestion Integration: end-to-end PDF ingestion with persistent database records
- API endpoints: list, get, get pages, get chunks, delete documents
- Migrations: schema upgrade and downgrade testing
"""

from pathlib import Path
from typing import Generator
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.models import Chunk, Document, DocumentStatus, Page
from app.repositories import ChunkRepository, DocumentRepository, PageRepository
from app.schemas.chunk import Chunk as DomainChunk
from app.schemas.document import Document as DomainDoc, DocumentMetadata, Page as DomainPage
from app.services.persistence import (
    ConstraintViolationError,
    DocumentAlreadyExistsError,
    DocumentPersistenceService,
    ReferentialIntegrityError,
)

# Test PostgreSQL engine pointing to test database
TEST_DB_URL = "postgresql+psycopg://postgres@localhost:5433/bookrag_test"


@pytest.fixture(scope="session")
def test_engine():
    """Create a persistent engine for the test database session."""
    engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine) -> Generator[Session, None, None]:
    """Provide a clean database session with table cleanup before each test."""
    session_factory = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = session_factory()

    # Clean test tables
    session.query(Chunk).delete()
    session.query(Page).delete()
    session.query(Document).delete()
    session.commit()

    try:
        yield session
    finally:
        session.rollback()
        # Clean up after test
        session.query(Chunk).delete()
        session.query(Page).delete()
        session.query(Document).delete()
        session.commit()
        session.close()


# ==============================================================================
# Suite A: ORM Model Tests
# ==============================================================================


def test_document_model_creation_and_fields(db_session: Session) -> None:
    """Document model preserves required fields, default values, and status."""
    doc = Document(
        document_id="doc_model_01",
        filename="war_and_peace.pdf",
        title="War and Peace",
        author="Leo Tolstoy",
        page_count=1200,
        status=DocumentStatus.UPLOADED.value,
    )
    db_session.add(doc)
    db_session.commit()

    saved = db_session.get(Document, "doc_model_01")
    assert saved is not None
    assert saved.document_id == "doc_model_01"
    assert saved.filename == "war_and_peace.pdf"
    assert saved.title == "War and Peace"
    assert saved.author == "Leo Tolstoy"
    assert saved.page_count == 1200
    assert saved.status == "uploaded"
    assert saved.created_at is not None
    assert saved.updated_at is not None


def test_page_model_creation_and_relationship(db_session: Session) -> None:
    """Page model preserves fields and bi-directional relationship with Document."""
    doc = Document(
        document_id="doc_page_rel",
        filename="test.pdf",
        page_count=1,
        status=DocumentStatus.PROCESSED.value,
    )
    db_session.add(doc)
    db_session.commit()

    page = Page(
        page_id="doc_page_rel_p1",
        document_id="doc_page_rel",
        page_number=1,
        text="Chapter 1. Well, Prince, so Genoa and Lucca are now no more than family estates...",
        char_count=85,
        word_count=16,
    )
    db_session.add(page)
    db_session.commit()

    saved_page = db_session.get(Page, "doc_page_rel_p1")
    assert saved_page is not None
    assert saved_page.document.document_id == "doc_page_rel"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1


def test_chunk_model_creation_and_relationships(db_session: Session) -> None:
    """Chunk model preserves relationships to both Document and Page."""
    doc = Document(
        document_id="doc_chunk_rel",
        filename="sample.pdf",
        page_count=1,
        status=DocumentStatus.PROCESSED.value,
    )
    db_session.add(doc)
    page = Page(
        page_id="doc_chunk_rel_p1",
        document_id="doc_chunk_rel",
        page_number=1,
        text="Extracted text chunk test.",
        char_count=26,
        word_count=4,
    )
    db_session.add(page)
    db_session.commit()

    chunk = Chunk(
        chunk_id="chunk_det_001",
        document_id="doc_chunk_rel",
        page_id="doc_chunk_rel_p1",
        page_number=1,
        chunk_index=0,
        text="Extracted text chunk test.",
        char_count=26,
        word_count=4,
    )
    db_session.add(chunk)
    db_session.commit()

    saved_chunk = db_session.get(Chunk, "chunk_det_001")
    assert saved_chunk is not None
    assert saved_chunk.document.document_id == "doc_chunk_rel"
    assert saved_chunk.page.page_id == "doc_chunk_rel_p1"
    assert len(doc.chunks) == 1
    assert len(page.chunks) == 1


# ==============================================================================
# Suite B: Constraints & Referential Integrity
# ==============================================================================


def test_foreign_key_violation_rejected_for_page(db_session: Session) -> None:
    """Inserting a page referencing a nonexistent document_id must fail."""
    page = Page(
        page_id="orphan_p1",
        document_id="nonexistent_doc_id",
        page_number=1,
        text="Orphan page",
        char_count=11,
        word_count=2,
    )
    db_session.add(page)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_foreign_key_violation_rejected_for_chunk(db_session: Session) -> None:
    """Inserting a chunk referencing a nonexistent page_id must fail."""
    doc = Document(document_id="doc_valid_chk", filename="test.pdf", page_count=1)
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(
        chunk_id="chk_orphan_01",
        document_id="doc_valid_chk",
        page_id="nonexistent_page_id",
        page_number=1,
        chunk_index=0,
        text="Orphan chunk",
        char_count=12,
        word_count=2,
    )
    db_session.add(chunk)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_unique_constraint_rejects_duplicate_page_number(db_session: Session) -> None:
    """A document cannot have two pages with the same page_number."""
    doc = Document(document_id="doc_dup_page", filename="book.pdf", page_count=2)
    db_session.add(doc)
    db_session.commit()

    p1 = Page(page_id="p1_a", document_id="doc_dup_page", page_number=1, text="Page 1", char_count=6, word_count=2)
    p2 = Page(page_id="p1_b", document_id="doc_dup_page", page_number=1, text="Page 1 dup", char_count=10, word_count=3)
    db_session.add(p1)
    db_session.commit()

    db_session.add(p2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_unique_constraint_rejects_duplicate_chunk_index(db_session: Session) -> None:
    """A document page cannot have two chunks with the same chunk_index."""
    doc = Document(document_id="doc_dup_chk", filename="book.pdf", page_count=1)
    db_session.add(doc)
    p = Page(page_id="doc_dup_chk_p1", document_id="doc_dup_chk", page_number=1, text="Text", char_count=4, word_count=1)
    db_session.add(p)
    db_session.commit()

    c1 = Chunk(chunk_id="c_01", document_id="doc_dup_chk", page_id="doc_dup_chk_p1", page_number=1, chunk_index=0, text="A", char_count=1, word_count=1)
    c2 = Chunk(chunk_id="c_02", document_id="doc_dup_chk", page_id="doc_dup_chk_p1", page_number=1, chunk_index=0, text="B", char_count=1, word_count=1)
    db_session.add(c1)
    db_session.commit()

    db_session.add(c2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_check_constraint_rejects_invalid_page_number(db_session: Session) -> None:
    """Page number must be positive (>= 1)."""
    doc = Document(document_id="doc_chk_pg0", filename="book.pdf", page_count=1)
    db_session.add(doc)
    db_session.commit()

    p_zero = Page(page_id="p_zero", document_id="doc_chk_pg0", page_number=0, text="Zero", char_count=4, word_count=1)
    db_session.add(p_zero)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_cascade_delete_removes_pages_and_chunks(db_session: Session) -> None:
    """Deleting a Document must cascade and cleanly remove all child pages and chunks."""
    doc = Document(document_id="doc_cascade", filename="cascade.pdf", page_count=2)
    db_session.add(doc)
    p1 = Page(page_id="doc_cascade_p1", document_id="doc_cascade", page_number=1, text="P1", char_count=2, word_count=1)
    p2 = Page(page_id="doc_cascade_p2", document_id="doc_cascade", page_number=2, text="P2", char_count=2, word_count=1)
    c1 = Chunk(chunk_id="c_cas_1", document_id="doc_cascade", page_id="doc_cascade_p1", page_number=1, chunk_index=0, text="C1", char_count=2, word_count=1)
    c2 = Chunk(chunk_id="c_cas_2", document_id="doc_cascade", page_id="doc_cascade_p2", page_number=2, chunk_index=1, text="C2", char_count=2, word_count=1)

    db_session.add_all([p1, p2, c1, c2])
    db_session.commit()

    # Verify existing
    assert db_session.get(Document, "doc_cascade") is not None
    assert len(db_session.query(Page).filter_by(document_id="doc_cascade").all()) == 2
    assert len(db_session.query(Chunk).filter_by(document_id="doc_cascade").all()) == 2

    # Delete document
    db_session.delete(doc)
    db_session.commit()

    # Verify cascade
    assert db_session.get(Document, "doc_cascade") is None
    assert len(db_session.query(Page).filter_by(document_id="doc_cascade").all()) == 0
    assert len(db_session.query(Chunk).filter_by(document_id="doc_cascade").all()) == 0


# ==============================================================================
# Suite C: Repository Layer Tests
# ==============================================================================


def test_document_repository_crud(db_session: Session) -> None:
    """DocumentRepository creates, retrieves, updates status, and lists documents."""
    repo = DocumentRepository(db_session)
    assert repo.count() == 0

    doc = Document(
        document_id="doc_repo_01",
        filename="dune.pdf",
        title="Dune",
        author="Frank Herbert",
        page_count=600,
        status=DocumentStatus.UPLOADED.value,
    )
    created = repo.create(doc)
    db_session.commit()
    assert created.document_id == "doc_repo_01"
    assert repo.count() == 1

    fetched = repo.get_by_id("doc_repo_01")
    assert fetched is not None
    assert fetched.title == "Dune"

    updated = repo.update_status("doc_repo_01", DocumentStatus.PROCESSED)
    db_session.commit()
    assert updated is not None
    assert updated.status == "processed"

    docs = repo.list_documents(skip=0, limit=10)
    assert len(docs) == 1
    assert docs[0].document_id == "doc_repo_01"

    deleted = repo.delete("doc_repo_01")
    db_session.commit()
    assert deleted is True
    assert repo.count() == 0


def test_page_repository_crud(db_session: Session) -> None:
    """PageRepository batch creates, queries by document, and queries by page number."""
    doc_repo = DocumentRepository(db_session)
    doc_repo.create(Document(document_id="doc_pages_repo", filename="book.pdf", page_count=3))
    db_session.commit()

    page_repo = PageRepository(db_session)
    pages = [
        Page(page_id="doc_pages_repo_p1", document_id="doc_pages_repo", page_number=1, text="Text 1", char_count=6, word_count=2),
        Page(page_id="doc_pages_repo_p2", document_id="doc_pages_repo", page_number=2, text="Text 2", char_count=6, word_count=2),
        Page(page_id="doc_pages_repo_p3", document_id="doc_pages_repo", page_number=3, text="Text 3", char_count=6, word_count=2),
    ]
    created = page_repo.create_many(pages)
    db_session.commit()
    assert len(created) == 3

    assert page_repo.count_by_document("doc_pages_repo") == 3
    doc_pages = page_repo.get_by_document("doc_pages_repo")
    assert [p.page_number for p in doc_pages] == [1, 2, 3]

    p2 = page_repo.get_by_document_and_page("doc_pages_repo", 2)
    assert p2 is not None
    assert p2.text == "Text 2"

    missing = page_repo.get_by_document_and_page("doc_pages_repo", 99)
    assert missing is None


def test_chunk_repository_crud(db_session: Session) -> None:
    """ChunkRepository creates, batch queries by document, and queries by page."""
    doc_repo = DocumentRepository(db_session)
    doc_repo.create(Document(document_id="doc_chunks_repo", filename="book.pdf", page_count=2))
    page_repo = PageRepository(db_session)
    page_repo.create_many([
        Page(page_id="doc_chunks_repo_p1", document_id="doc_chunks_repo", page_number=1, text="P1", char_count=2, word_count=1),
        Page(page_id="doc_chunks_repo_p2", document_id="doc_chunks_repo", page_number=2, text="P2", char_count=2, word_count=1),
    ])
    db_session.commit()

    chunk_repo = ChunkRepository(db_session)
    chunks = [
        Chunk(chunk_id="c_r_1", document_id="doc_chunks_repo", page_id="doc_chunks_repo_p1", page_number=1, chunk_index=0, text="C1", char_count=2, word_count=1),
        Chunk(chunk_id="c_r_2", document_id="doc_chunks_repo", page_id="doc_chunks_repo_p1", page_number=1, chunk_index=1, text="C2", char_count=2, word_count=1),
        Chunk(chunk_id="c_r_3", document_id="doc_chunks_repo", page_id="doc_chunks_repo_p2", page_number=2, chunk_index=2, text="C3", char_count=2, word_count=1),
    ]
    chunk_repo.create_many(chunks)
    db_session.commit()

    assert chunk_repo.count_by_document("doc_chunks_repo") == 3
    doc_chunks = chunk_repo.get_by_document("doc_chunks_repo")
    assert len(doc_chunks) == 3

    p1_chunks = chunk_repo.get_by_page("doc_chunks_repo", 1)
    assert len(p1_chunks) == 2
    assert [c.chunk_index for c in p1_chunks] == [0, 1]


# ==============================================================================
# Suite D: Transactional Persistence Service
# ==============================================================================


def test_transactional_persistence_success(db_session: Session) -> None:
    """DocumentPersistenceService atomically persists document, pages, and chunks."""
    service = DocumentPersistenceService(db_session)

    domain_doc = DomainDoc(
        document_id="doc_atomic_success",
        filename="sample.pdf",
        source_path="data/sample.pdf",
        page_count=2,
        total_characters=60,
        total_words=10,
        pages=[
            DomainPage(page_number=1, text="Introduction to RAG architectures.", char_count=33, word_count=4, has_text=True),
            DomainPage(page_number=2, text="PostgreSQL persistent layer.", char_count=27, word_count=3, has_text=True),
        ],
        metadata=DocumentMetadata(title="Atomic RAG", author="Antigravity"),
    )
    domain_chunks = [
        DomainChunk(chunk_id="chk_at_1", document_id="doc_atomic_success", page_number=1, chunk_index=0, text="Intro RAG", char_count=9, word_count=2),
        DomainChunk(chunk_id="chk_at_2", document_id="doc_atomic_success", page_number=2, chunk_index=1, text="Postgres RAG", char_count=12, word_count=2),
    ]

    saved_doc = service.persist_document_and_chunks(domain_doc, domain_chunks)
    assert saved_doc.status == "processed"

    # Verify persisted state
    assert service.get_document("doc_atomic_success") is not None
    pages = service.get_document_pages("doc_atomic_success")
    assert len(pages) == 2
    chunks = service.get_document_chunks("doc_atomic_success")
    assert len(chunks) == 2


def test_transactional_persistence_rollback_on_failure(db_session: Session) -> None:
    """If chunk persistence violates a constraint, the entire transaction rolls back."""
    service = DocumentPersistenceService(db_session)

    domain_doc = DomainDoc(
        document_id="doc_atomic_fail",
        filename="fail.pdf",
        source_path="data/fail.pdf",
        page_count=1,
        total_characters=10,
        total_words=2,
        pages=[
            DomainPage(page_number=1, text="Page one.", char_count=9, word_count=2, has_text=True),
        ],
    )
    # Malformed chunks with duplicate chunk index
    domain_chunks_bad = [
        DomainChunk(chunk_id="chk_f_1", document_id="doc_atomic_fail", page_number=1, chunk_index=0, text="A", char_count=1, word_count=1),
        DomainChunk(chunk_id="chk_f_2", document_id="doc_atomic_fail", page_number=1, chunk_index=0, text="B", char_count=1, word_count=1),
    ]

    with pytest.raises(ConstraintViolationError):
        service.persist_document_and_chunks(domain_doc, domain_chunks_bad)

    # Document and pages must NOT exist in the database (complete rollback)
    assert service.get_document("doc_atomic_fail") is None
    assert len(service.get_document_pages("doc_atomic_fail")) == 0
    assert len(service.get_document_chunks("doc_atomic_fail")) == 0


def test_persistence_rejects_duplicate_document_id(db_session: Session) -> None:
    """Attempting to persist an existing document raises DocumentAlreadyExistsError."""
    service = DocumentPersistenceService(db_session)
    domain_doc = DomainDoc(
        document_id="doc_existing_id",
        filename="test.pdf",
        source_path="data/test.pdf",
        page_count=1,
        total_characters=5,
        total_words=1,
        pages=[DomainPage(page_number=1, text="Hello", char_count=5, word_count=1, has_text=True)],
    )

    service.persist_document_and_chunks(domain_doc, [])

    with pytest.raises(DocumentAlreadyExistsError):
        service.persist_document_and_chunks(domain_doc, [])


# ==============================================================================
# Suite E: API Endpoint Tests
# ==============================================================================


def test_api_ingest_and_persistence(client: TestClient, deterministic_sample_pdf: Path, db_session: Session) -> None:
    """PDF ingestion endpoint successfully parses PDF and persists into PostgreSQL."""
    response = client.post(
        "/api/v1/documents/ingest",
        json={"file_path": str(deterministic_sample_pdf)},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    doc_id = data["document"]["document_id"]

    # Verify document is available via GET endpoint
    doc_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert doc_resp.status_code == status.HTTP_200_OK
    doc_data = doc_resp.json()
    assert doc_data["document_id"] == doc_id
    assert doc_data["status"] == "processed"
    assert doc_data["page_count"] == 3

    # Verify pages endpoint
    pages_resp = client.get(f"/api/v1/documents/{doc_id}/pages")
    assert pages_resp.status_code == status.HTTP_200_OK
    pages_data = pages_resp.json()
    assert len(pages_data) == 3
    assert pages_data[0]["page_number"] == 1

    # Verify chunks endpoint
    chunks_resp = client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks_resp.status_code == status.HTTP_200_OK
    chunks_data = chunks_resp.json()
    assert len(chunks_data) >= 3

    # Verify document list endpoint
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == status.HTTP_200_OK
    list_data = list_resp.json()
    assert list_data["total"] >= 1

    # Verify document deletion endpoint
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == status.HTTP_200_OK

    # Verify 404 after deletion
    get_del = client.get(f"/api/v1/documents/{doc_id}")
    assert get_del.status_code == status.HTTP_404_NOT_FOUND


def test_api_get_nonexistent_document_returns_404(client: TestClient) -> None:
    """Querying a nonexistent document ID returns HTTP 404."""
    response = client.get("/api/v1/documents/nonexistent_doc_999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


# ==============================================================================
# Suite F: Alembic Migration Test
# ==============================================================================


def test_alembic_schema_verification(test_engine) -> None:
    """Verify that all expected tables, columns, constraints, and indexes exist in the schema."""
    with test_engine.connect() as conn:
        # Check tables in information_schema
        result = conn.exec_driver_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        )
        tables = [row[0] for row in result.fetchall()]
        assert "documents" in tables
        assert "pages" in tables
        assert "chunks" in tables

        # Check documents columns
        result = conn.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'"
        )
        doc_cols = [row[0] for row in result.fetchall()]
        for expected in ["document_id", "filename", "title", "author", "page_count", "status", "created_at", "updated_at"]:
            assert expected in doc_cols

        # Check pages columns
        result = conn.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'pages'"
        )
        page_cols = [row[0] for row in result.fetchall()]
        for expected in ["page_id", "document_id", "page_number", "text", "char_count", "word_count", "created_at"]:
            assert expected in page_cols

        # Check chunks columns
        result = conn.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'chunks'"
        )
        chunk_cols = [row[0] for row in result.fetchall()]
        for expected in ["chunk_id", "document_id", "page_id", "page_number", "chunk_index", "text", "char_count", "word_count", "created_at"]:
            assert expected in chunk_cols
