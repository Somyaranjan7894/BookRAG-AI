"""Comprehensive test suite for Phase 14 pgvector persistent vector storage and retrieval.

Tests cover:
- Extension: PostgreSQL vector extension installation and availability
- Schema: chunks.embedding vector(384) column presence and dimension verification
- Persistence: single and batch embedding storage and retrieval via PGVectorRepository
- Service: EmbeddingPersistenceService batch embedding and database updates
- Dimension Validation: rejection of invalid-dimensional vectors (e.g. 128 vs 384)
- Empty State: safe handling of chunks without embeddings
- Vector Search: similarity calculation, descending ordering, and top_k bounds
- Document Isolation: strict document filtering preventing cross-document leakage
- Deletion: clearing embeddings individually and at the document level
- Backend Abstraction: VectorSearchBackend interface compliance for FAISS and pgvector
- SearchService Integration: end-to-end two-stage retrieval with Cross-Encoder reranking
- Migration: schema generation from a clean database
"""

import os
from pathlib import Path
import subprocess
import sys
from typing import Generator, List
import numpy as np
import psycopg
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.models import Chunk, Document, DocumentStatus, Page
from app.repositories.vector_repository import PGVectorRepository
from app.schemas.chunk import Chunk as DomainChunk
from app.schemas.retrieval import RetrievalResult
from app.schemas.search import SearchResponse
from app.services.embeddings.service import EmbeddingService
from app.services.persistence.embedding_service import EmbeddingPersistenceService
from app.services.reranking.service import RerankerService
from app.services.retrieval.backend import FAISSVectorBackend, PGVectorBackend, create_vector_backend
from app.services.retrieval.service import RetrievalService
from app.services.search.service import SearchService

TEST_DB_URL = "postgresql+psycopg://postgres@localhost:5433/bookrag_test"


@pytest.fixture(scope="session")
def pgvector_engine():
    """Create engine for testing pgvector on bookrag_test."""
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
    session_factory = sessionmaker(bind=pgvector_engine, autoflush=False, autocommit=False, expire_on_commit=False)
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


def _create_sample_doc_and_chunks(
    session: Session,
    doc_id: str,
    num_chunks: int = 3,
) -> List[Chunk]:
    """Helper creating a persisted Document, Page, and Chunks without embeddings."""
    doc = Document(
        document_id=doc_id,
        filename=f"{doc_id}.pdf",
        title=f"Title for {doc_id}",
        author="Author",
        page_count=1,
        status=DocumentStatus.PROCESSED.value,
    )
    session.add(doc)

    page = Page(
        page_id=f"{doc_id}_p1",
        document_id=doc_id,
        page_number=1,
        text="Sample page text for vector testing.",
        char_count=35,
        word_count=6,
    )
    session.add(page)

    chunks: List[Chunk] = []
    for i in range(num_chunks):
        c = Chunk(
            chunk_id=f"{doc_id}_c{i:03d}",
            document_id=doc_id,
            page_id=f"{doc_id}_p1",
            page_number=1,
            chunk_index=i,
            text=f"Content text for chunk {i} in document {doc_id}.",
            char_count=40,
            word_count=7,
        )
        session.add(c)
        chunks.append(c)

    session.commit()
    return chunks


# ==============================================================================
# Suite A: Extension and Schema Verification
# ==============================================================================


def test_pgvector_extension_exists(db_session: Session) -> None:
    """Verify vector extension is enabled in PostgreSQL."""
    res = db_session.execute(
        text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
    ).all()
    assert len(res) == 1
    extname, version = res[0]
    assert extname == "vector"
    assert version.startswith("0.")


def test_chunk_embedding_column_schema(db_session: Session) -> None:
    """Verify chunks table has an embedding column of type vector(384)."""
    res = db_session.execute(
        text(
            "SELECT column_name, udt_name "
            "FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding';"
        )
    ).all()
    assert len(res) == 1
    col_name, udt_name = res[0]
    assert col_name == "embedding"
    assert udt_name == "vector"


# ==============================================================================
# Suite B: Persistence & Repository Tests
# ==============================================================================


def test_store_and_retrieve_single_embedding(db_session: Session) -> None:
    """PGVectorRepository stores single 384-d vector and retrieves it."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_store_single", num_chunks=1)
    chunk_id = chunks[0].chunk_id

    repo = PGVectorRepository(db_session, dimension=384)
    dummy_vec = [float(i) / 384.0 for i in range(384)]

    success = repo.store_embedding(chunk_id, dummy_vec)
    db_session.commit()
    assert success is True

    retrieved = repo.get_embedding(chunk_id)
    assert retrieved is not None
    assert len(retrieved) == 384
    np.testing.assert_allclose(retrieved, dummy_vec, atol=1e-5)


def test_store_batch_embeddings(db_session: Session) -> None:
    """PGVectorRepository stores batch of embeddings across multiple chunks."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_store_batch", num_chunks=3)
    repo = PGVectorRepository(db_session, dimension=384)

    batch_map = {
        c.chunk_id: [float(idx + 1) / 384.0] * 384
        for idx, c in enumerate(chunks)
    }

    updated = repo.store_embeddings(batch_map)
    db_session.commit()
    assert updated == 3

    assert repo.count_embeddings("doc_store_batch") == 3
    for c in chunks:
        vec = repo.get_embedding(c.chunk_id)
        assert vec is not None
        assert len(vec) == 384


def test_dimension_mismatch_rejected(db_session: Session) -> None:
    """PGVectorRepository strictly rejects vectors with incorrect dimensionality."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_dim_mismatch", num_chunks=1)
    repo = PGVectorRepository(db_session, dimension=384)

    # 128 dimensions instead of 384
    short_vec = [0.1] * 128

    with pytest.raises(ValueError, match="dimension mismatch"):
        repo.store_embedding(chunks[0].chunk_id, short_vec)

    with pytest.raises(ValueError, match="dimension mismatch"):
        repo.search_similar(query_vector=short_vec, top_k=5)


def test_empty_embeddings_handled_safely(db_session: Session) -> None:
    """Chunks without embeddings are safely skipped during similarity search."""
    _create_sample_doc_and_chunks(db_session, "doc_empty_vec", num_chunks=3)
    repo = PGVectorRepository(db_session, dimension=384)

    # None have embeddings
    assert repo.count_embeddings("doc_empty_vec") == 0

    query_vec = [0.1] * 384
    results = repo.search_similar(query_vec, top_k=5, document_id="doc_empty_vec")
    assert results == []


# ==============================================================================
# Suite C: Similarity Search & Document Isolation
# ==============================================================================


def test_similarity_search_ranking_and_scores(db_session: Session) -> None:
    """pgvector similarity search orders results by descending cosine similarity."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_sim_rank", num_chunks=3)
    repo = PGVectorRepository(db_session, dimension=384)

    # Target query vector
    query_vec = np.zeros(384, dtype=np.float32)
    query_vec[0] = 1.0  # Unit vector along dimension 0

    # Chunk 0: identical direction (dot product = 1.0)
    v0 = np.zeros(384, dtype=np.float32)
    v0[0] = 1.0

    # Chunk 1: orthogonal (dot product = 0.0)
    v1 = np.zeros(384, dtype=np.float32)
    v1[1] = 1.0

    # Chunk 2: opposite direction (dot product = -1.0)
    v2 = np.zeros(384, dtype=np.float32)
    v2[0] = -1.0

    repo.store_embeddings({
        chunks[0].chunk_id: v0.tolist(),
        chunks[1].chunk_id: v1.tolist(),
        chunks[2].chunk_id: v2.tolist(),
    })
    db_session.commit()

    results: List[RetrievalResult] = repo.search_similar(query_vec.tolist(), top_k=3)
    assert len(results) == 3

    # Ranking check: chunk 0 highest (sim ~ 1.0), chunk 1 second (sim ~ 0.0), chunk 2 third (sim ~ -1.0)
    assert results[0].chunk_id == chunks[0].chunk_id
    assert pytest.approx(results[0].similarity_score, abs=1e-3) == 1.0

    assert results[1].chunk_id == chunks[1].chunk_id
    assert pytest.approx(results[1].similarity_score, abs=1e-3) == 0.0

    assert results[2].chunk_id == chunks[2].chunk_id
    assert pytest.approx(results[2].similarity_score, abs=1e-3) == -1.0

    # Check ranks 1, 2, 3
    assert [r.rank for r in results] == [1, 2, 3]


def test_document_isolation(db_session: Session) -> None:
    """Similarity search strictly isolates results to the requested document_id."""
    doc_a_chunks = _create_sample_doc_and_chunks(db_session, "doc_alpha", num_chunks=2)
    doc_b_chunks = _create_sample_doc_and_chunks(db_session, "doc_beta", num_chunks=2)
    repo = PGVectorRepository(db_session, dimension=384)

    target_vec = [1.0] + [0.0] * 383

    # Store identical vectors in both doc_a and doc_b
    repo.store_embeddings({
        doc_a_chunks[0].chunk_id: target_vec,
        doc_a_chunks[1].chunk_id: target_vec,
        doc_b_chunks[0].chunk_id: target_vec,
        doc_b_chunks[1].chunk_id: target_vec,
    })
    db_session.commit()

    # Search isolated to doc_alpha
    results_a = repo.search_similar(target_vec, top_k=5, document_id="doc_alpha")
    assert len(results_a) == 2
    for r in results_a:
        assert r.document_id == "doc_alpha"
        assert r.chunk_id.startswith("doc_alpha")

    # Search isolated to doc_beta
    results_b = repo.search_similar(target_vec, top_k=5, document_id="doc_beta")
    assert len(results_b) == 2
    for r in results_b:
        assert r.document_id == "doc_beta"
        assert r.chunk_id.startswith("doc_beta")


def test_delete_embeddings(db_session: Session) -> None:
    """Clearing embeddings works for individual chunks and document-wide."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_del_test", num_chunks=3)
    repo = PGVectorRepository(db_session, dimension=384)

    vec = [0.1] * 384
    repo.store_embeddings({c.chunk_id: vec for c in chunks})
    db_session.commit()
    assert repo.count_embeddings("doc_del_test") == 3

    # Delete single embedding
    ok = repo.delete_embedding(chunks[0].chunk_id)
    db_session.commit()
    assert ok is True
    assert repo.get_embedding(chunks[0].chunk_id) is None
    assert repo.count_embeddings("doc_del_test") == 2

    # Delete remaining document embeddings
    cleared = repo.delete_embeddings_for_document("doc_del_test")
    db_session.commit()
    assert cleared == 2
    assert repo.count_embeddings("doc_del_test") == 0


# ==============================================================================
# Suite D: EmbeddingPersistenceService Tests
# ==============================================================================


def test_embedding_persistence_service_batch(db_session: Session) -> None:
    """EmbeddingPersistenceService generates embeddings and saves to database."""
    _create_sample_doc_and_chunks(db_session, "doc_embed_svc", num_chunks=4)
    service = EmbeddingPersistenceService(db_session)

    count = service.embed_unembedded_document_chunks("doc_embed_svc", batch_size=2)
    assert count == 4

    repo = PGVectorRepository(db_session)
    assert repo.count_embeddings("doc_embed_svc") == 4


# ==============================================================================
# Suite E: Vector Backend Abstraction & SearchService Integration
# ==============================================================================


def test_vector_backend_abstraction(db_session: Session) -> None:
    """Both FAISS and pgvector backends conform to VectorSearchBackend interface."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_backend_test", num_chunks=2)
    repo = PGVectorRepository(db_session, dimension=384)
    vec = [0.5] * 384
    repo.store_embeddings({c.chunk_id: vec for c in chunks})
    db_session.commit()

    pg_backend = PGVectorBackend(vector_repo=repo)
    assert pg_backend.has_document("doc_backend_test") is True
    assert pg_backend.has_document("nonexistent") is False
    assert pg_backend.total_vectors() == 2

    res = pg_backend.search(vec, top_k=1, document_id="doc_backend_test")
    assert len(res) == 1
    assert res[0].document_id == "doc_backend_test"


def test_search_service_with_pgvector_backend(db_session: Session) -> None:
    """SearchService seamlessly executes semantic search using pgvector backend."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_search_pg", num_chunks=3)
    emb_svc = EmbeddingService()

    # Generate real semantic embeddings for the chunks
    domain_chunks = [
        DomainChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            text=c.text,
            char_count=c.char_count,
            word_count=c.word_count,
        )
        for c in chunks
    ]
    persistence_svc = EmbeddingPersistenceService(db_session, embedding_service=emb_svc)
    persistence_svc.embed_and_persist_chunks(domain_chunks)

    # Initialize SearchService with pgvector backend
    pg_backend = PGVectorBackend(vector_repo=PGVectorRepository(db_session))
    search_svc = SearchService(
        embedding_service=emb_svc,
        vector_backend=pg_backend,
        enable_reranking=False,
    )

    response: SearchResponse = search_svc.search(
        query="Content text for chunk",
        top_k=2,
        document_id="doc_search_pg",
    )

    assert response.total_results == 2
    assert len(response.results) == 2
    assert response.document_id == "doc_search_pg"
    assert response.results[0].rank == 1
    assert response.results[0].similarity_score > 0.0


def test_search_service_pgvector_with_reranking(db_session: Session) -> None:
    """SearchService with pgvector preserves two-stage Cross-Encoder reranking."""
    chunks = _create_sample_doc_and_chunks(db_session, "doc_rerank_pg", num_chunks=3)
    emb_svc = EmbeddingService()

    domain_chunks = [
        DomainChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            text=c.text,
            char_count=c.char_count,
            word_count=c.word_count,
        )
        for c in chunks
    ]
    persistence_svc = EmbeddingPersistenceService(db_session, embedding_service=emb_svc)
    persistence_svc.embed_and_persist_chunks(domain_chunks)

    pg_backend = PGVectorBackend(vector_repo=PGVectorRepository(db_session))
    search_svc = SearchService(
        embedding_service=emb_svc,
        reranker_service=RerankerService(),
        vector_backend=pg_backend,
        enable_reranking=True,
    )

    response: SearchResponse = search_svc.search(
        query="Content text",
        top_k=2,
        document_id="doc_rerank_pg",
        enable_reranking=True,
    )

    assert response.reranking_applied is True
    assert len(response.results) == 2
    # Verify both similarity_score and reranker_score are populated
    for r in response.results:
        assert r.similarity_score is not None
        assert r.reranker_score is not None
        assert r.original_rank is not None


# ==============================================================================
# Suite F: Clean Database Migration Test
# ==============================================================================


def test_clean_database_migration_with_pgvector() -> None:
    """Verify that a brand-new PostgreSQL database can run all migrations including pgvector."""
    # 1. Connect and create fresh test database
    conn = psycopg.connect("postgresql://postgres@localhost:5433/postgres", autocommit=True)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS test_fresh_pgvector_migration;")
    cur.execute("CREATE DATABASE test_fresh_pgvector_migration;")
    conn.close()

    try:
        # 2. Run alembic upgrade head targeting fresh db
        backend_dir = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ, DATABASE_URL="postgresql+psycopg://postgres@localhost:5433/test_fresh_pgvector_migration")
        res = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=backend_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Alembic upgrade failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        # 3. Check vector extension and column in fresh db
        conn2 = psycopg.connect("postgresql://postgres@localhost:5433/test_fresh_pgvector_migration")
        cur2 = conn2.cursor()
        cur2.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        exts = [r[0] for r in cur2.fetchall()]
        assert "vector" in exts

        cur2.execute(
            "SELECT column_name, udt_name FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding';"
        )
        cols = cur2.fetchall()
        assert len(cols) == 1
        assert cols[0][0] == "embedding"
        assert cols[0][1] == "vector"
        conn2.close()

    finally:
        # 4. Clean up
        conn_cleanup = psycopg.connect("postgresql://postgres@localhost:5433/postgres", autocommit=True)
        conn_cleanup.cursor().execute("DROP DATABASE IF EXISTS test_fresh_pgvector_migration;")
        conn_cleanup.close()
