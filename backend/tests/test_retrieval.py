"""Comprehensive unit and integration tests for Phase 4 FAISS vector retrieval."""

import json
from pathlib import Path
import faiss
import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.schemas.chunk import Chunk
from app.schemas.embedding import EmbeddingRecord
from app.schemas.retrieval import IndexMetadata, RetrievalResult
from app.services.embeddings.model import (
    DEFAULT_MODEL_NAME,
    EXPECTED_DIMENSION,
    EmbeddingModel,
)
from app.services.embeddings.service import EmbeddingService
from app.services.retrieval.exceptions import (
    CorruptedIndexError,
    IndexDimensionMismatchError,
    IndexNotFoundError,
    InvalidQueryError,
    RetrievalError,
)
from app.services.retrieval.index import VectorIndex
from app.services.retrieval.mapping import VectorToChunkMapping
from app.services.retrieval.service import RetrievalService


@pytest.fixture(scope="session")
def shared_embedding_model() -> EmbeddingModel:
    """Session-scoped embedding model instance to avoid repeated loads."""
    return EmbeddingModel.get_instance(model_name=DEFAULT_MODEL_NAME, device="auto")


@pytest.fixture
def embedding_service(shared_embedding_model: EmbeddingModel) -> EmbeddingService:
    """Provide an EmbeddingService reusing the preloaded session model."""
    return EmbeddingService(model=shared_embedding_model)


@pytest.fixture
def retrieval_service(embedding_service: EmbeddingService) -> RetrievalService:
    """Provide a fresh RetrievalService with the shared EmbeddingService."""
    return RetrievalService(embedding_service=embedding_service)


@pytest.fixture
def sample_embedding_records(embedding_service: EmbeddingService) -> list[EmbeddingRecord]:
    """Provide a deterministic batch of real 384-dimensional EmbeddingRecords with distinct semantic topics."""
    chunks = [
        Chunk(
            chunk_id="doc_ai_p001_c0001",
            document_id="doc_ai",
            page_number=1,
            chunk_index=0,
            text="Machine learning algorithms build mathematical models based on sample training data.",
            char_count=86,
            word_count=11,
        ),
        Chunk(
            chunk_id="doc_ai_p002_c0002",
            document_id="doc_ai",
            page_number=2,
            chunk_index=1,
            text="Deep neural networks with multiple hidden layers learn hierarchical feature representations.",
            char_count=92,
            word_count=11,
        ),
        Chunk(
            chunk_id="doc_bio_p010_c0003",
            document_id="doc_bio",
            page_number=10,
            chunk_index=2,
            text="Photosynthesis is the biological process used by plants to convert light energy into chemical energy.",
            char_count=101,
            word_count=14,
        ),
        Chunk(
            chunk_id="doc_bio_p011_c0004",
            document_id="doc_bio",
            page_number=11,
            chunk_index=3,
            text="Cellular respiration breaks down glucose in the mitochondria to produce adenosine triphosphate.",
            char_count=95,
            word_count=11,
        ),
        Chunk(
            chunk_id="doc_hist_p042_c0005",
            document_id="doc_hist",
            page_number=42,
            chunk_index=4,
            text="The Industrial Revolution began in Great Britain in the late eighteenth century.",
            char_count=80,
            word_count=12,
        ),
    ]
    return embedding_service.embed_chunks(chunks)


# ==============================================================================
# A. FAISS Import & Initialization Tests
# ==============================================================================

def test_faiss_import_and_initialization() -> None:
    """Requirement A: FAISS loads successfully and IndexFlatIP initializes with 384 dimensions."""
    index = faiss.IndexFlatIP(EXPECTED_DIMENSION)
    assert index.d == EXPECTED_DIMENSION
    assert index.ntotal == 0
    assert index.is_trained is True


def test_vector_index_wrapper_initialization() -> None:
    """Requirement A: VectorIndex wrapper creates an empty 384-dimensional IndexFlatIP."""
    vec_index = VectorIndex(dimension=EXPECTED_DIMENSION, index_id="test_init")
    assert vec_index.dimension == EXPECTED_DIMENSION
    assert vec_index.total_vectors == 0
    assert vec_index.metadata.total_vectors == 0
    assert vec_index.metadata.index_id == "test_init"
    assert vec_index.metadata.normalized is True


# ==============================================================================
# B & C. Index Creation & Vector Insertion Tests
# ==============================================================================

def test_index_creation_and_vector_counts(
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement B: Valid embeddings populate index and vector count matches input length."""
    vec_index = VectorIndex(dimension=EXPECTED_DIMENSION, index_id="test_counts")
    added = vec_index.add_records(sample_embedding_records)

    assert added == len(sample_embedding_records)
    assert vec_index.total_vectors == len(sample_embedding_records)
    assert len(vec_index._mapping) == len(sample_embedding_records)
    assert vec_index.metadata.total_vectors == len(sample_embedding_records)


def test_incremental_vector_insertion(
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement C: Incremental batch additions maintain strictly contiguous mapping indices."""
    vec_index = VectorIndex(dimension=EXPECTED_DIMENSION, index_id="test_inc")
    batch_1 = sample_embedding_records[:2]
    batch_2 = sample_embedding_records[2:]

    vec_index.add_records(batch_1)
    assert vec_index.total_vectors == 2
    assert vec_index._mapping.get(0).chunk_id == batch_1[0].chunk_id
    assert vec_index._mapping.get(1).chunk_id == batch_1[1].chunk_id

    vec_index.add_records(batch_2)
    assert vec_index.total_vectors == 5
    assert vec_index._mapping.get(2).chunk_id == batch_2[0].chunk_id
    assert vec_index._mapping.get(3).chunk_id == batch_2[1].chunk_id
    assert vec_index._mapping.get(4).chunk_id == batch_2[2].chunk_id


# ==============================================================================
# D & E. Search, Top-K & Semantic Relevance Tests
# ==============================================================================

def test_semantic_search_retrieves_most_relevant_chunk(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement D: Semantic query retrieves most relevant chunk at rank 1 with high similarity score."""
    index = retrieval_service.build_index_from_records(sample_embedding_records, index_id="test_search")

    query = "How do neural networks and machine learning models learn features?"
    results = retrieval_service.search(query=query, top_k=3, index=index)

    assert len(results) == 3
    # First result should be from the AI document
    assert results[0].rank == 1
    assert results[0].document_id == "doc_ai"
    assert results[0].similarity_score > 0.4
    # Rank ordering: rank 1 must have higher similarity than rank 2 and rank 3
    assert results[0].similarity_score >= results[1].similarity_score
    assert results[1].similarity_score >= results[2].similarity_score


def test_top_k_limits(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement E: top_k=1 and top_k=3 return exact number of candidates requested."""
    index = retrieval_service.build_index_from_records(sample_embedding_records)

    results_k1 = retrieval_service.search(query="biology and mitochondria", top_k=1, index=index)
    assert len(results_k1) == 1
    assert results_k1[0].rank == 1

    results_k3 = retrieval_service.search(query="biology and mitochondria", top_k=3, index=index)
    assert len(results_k3) == 3
    assert [r.rank for r in results_k3] == [1, 2, 3]


def test_top_k_larger_than_index_size(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement E: top_k > total_vectors returns all available vectors without error."""
    index = retrieval_service.build_index_from_records(sample_embedding_records)
    total = len(sample_embedding_records)

    results = retrieval_service.search(query="any general query", top_k=50, index=index)
    assert len(results) == total
    assert [r.rank for r in results] == list(range(1, total + 1))


# ==============================================================================
# F. Metadata Mapping & Provenance Survival Tests
# ==============================================================================

def test_provenance_retention_in_retrieval_result(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement F: chunk_id, document_id, page_number, and text strictly survive in RetrievalResult."""
    index = retrieval_service.build_index_from_records(sample_embedding_records)
    target_record = sample_embedding_records[2]  # doc_bio_p010_c0003

    query = "photosynthesis in plants light energy conversion"
    results = retrieval_service.search(query=query, top_k=1, index=index)

    top = results[0]
    assert top.chunk_id == target_record.chunk_id
    assert top.document_id == target_record.document_id
    assert top.page_number == target_record.page_number
    assert top.text == target_record.text
    assert isinstance(top.similarity_score, float)


# ==============================================================================
# G. Dimension Mismatch Validation Tests
# ==============================================================================

def test_reject_record_with_mismatched_dimension() -> None:
    """Requirement G: Adding vector with wrong dimension (e.g. 768) raises IndexDimensionMismatchError."""
    vec_index = VectorIndex(dimension=384)
    bad_record = EmbeddingRecord(
        chunk_id="bad_dim_chunk",
        document_id="doc_bad",
        page_number=1,
        text="Mismatched dimension test text.",
        embedding=[0.1] * 768,  # 768 instead of 384
        dimension=768,
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cpu",
        normalized=True,
    )
    with pytest.raises(IndexDimensionMismatchError):
        vec_index.add_records([bad_record])


def test_reject_query_vector_with_mismatched_dimension() -> None:
    """Requirement G: Searching with wrong dimension vector raises IndexDimensionMismatchError."""
    vec_index = VectorIndex(dimension=384)
    bad_query_vec = np.random.randn(1, 512).astype(np.float32)

    with pytest.raises(IndexDimensionMismatchError):
        vec_index.search(bad_query_vec, top_k=3)


# ==============================================================================
# H. Empty Index Handling Tests
# ==============================================================================

def test_empty_index_search_returns_empty_list(retrieval_service: RetrievalService) -> None:
    """Requirement H: Searching an empty index returns an empty list without raising exceptions."""
    empty_index = retrieval_service.create_index(index_id="empty_idx")
    results = retrieval_service.search(query="any search term", top_k=5, index=empty_index)
    assert results == []


def test_search_unregistered_index_raises_index_not_found(retrieval_service: RetrievalService) -> None:
    """Requirement H: Searching for non-existent index ID raises IndexNotFoundError."""
    with pytest.raises(IndexNotFoundError):
        retrieval_service.search(query="test", index_id="non_existent_index")


# ==============================================================================
# I. Document Isolation / Filtering Tests
# ==============================================================================

def test_document_isolation_excludes_other_books(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement I: Specifying document_id isolates retrieval strictly to that document."""
    # Index containing doc_ai, doc_bio, and doc_hist
    index = retrieval_service.build_index_from_records(sample_embedding_records)

    # Query for biological topic, but filter for doc_ai
    results = retrieval_service.search(
        query="cellular respiration mitochondria glucose",
        document_id="doc_ai",
        top_k=5,
        index=index,
    )

    # Every returned chunk MUST belong to doc_ai, none to doc_bio
    assert len(results) > 0
    for r in results:
        assert r.document_id == "doc_ai"
        assert r.document_id != "doc_bio"


# ==============================================================================
# J. Persistence & Loading Tests
# ==============================================================================

def test_index_save_and_load_preserves_search_fidelity(
    tmp_path: Path,
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement J: Saved index on disk reloads cleanly and yields identical search results."""
    original_index = retrieval_service.build_index_from_records(sample_embedding_records, index_id="persisted_test")

    # Save to temporary directory
    index_file, meta_file = original_index.save(directory=tmp_path, base_name="book_test")
    assert index_file.exists()
    assert meta_file.exists()

    # Load into new VectorIndex instance
    loaded_index = VectorIndex.load(directory=tmp_path, base_name="book_test")
    assert loaded_index.total_vectors == original_index.total_vectors
    assert loaded_index.dimension == original_index.dimension
    assert loaded_index.index_id == "persisted_test"

    # Compare search on original vs loaded
    query = "Industrial revolution in Great Britain"
    orig_results = original_index.search(retrieval_service.embedding_service.embed_query(query), top_k=2)
    loaded_results = loaded_index.search(retrieval_service.embedding_service.embed_query(query), top_k=2)

    assert len(orig_results) == len(loaded_results)
    for orig, loaded in zip(orig_results, loaded_results):
        assert orig.chunk_id == loaded.chunk_id
        assert orig.document_id == loaded.document_id
        assert orig.page_number == loaded.page_number
        assert abs(orig.similarity_score - loaded.similarity_score) < 1e-5


# ==============================================================================
# K. Corrupted & Inconsistent Metadata Validation Tests
# ==============================================================================

def test_load_non_existent_index_raises_error(tmp_path: Path) -> None:
    """Requirement K: Loading non-existent index raises IndexNotFoundError."""
    with pytest.raises(IndexNotFoundError):
        VectorIndex.load(directory=tmp_path, base_name="missing_index")


def test_load_inconsistent_vector_count_raises_corrupted_error(
    tmp_path: Path,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement K: Vector count mismatch between FAISS and metadata raises CorruptedIndexError."""
    index = VectorIndex(dimension=384, index_id="corrupt_test")
    index.add_records(sample_embedding_records)
    index_file, meta_file = index.save(directory=tmp_path, base_name="corrupt_count")

    # Tamper with metadata JSON to report different vector count
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["metadata"]["total_vectors"] = 999
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(CorruptedIndexError):
        VectorIndex.load(directory=tmp_path, base_name="corrupt_count")


def test_load_inconsistent_dimension_raises_corrupted_error(
    tmp_path: Path,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement K: Dimension mismatch between FAISS index and metadata raises CorruptedIndexError."""
    index = VectorIndex(dimension=384, index_id="corrupt_dim_test")
    index.add_records(sample_embedding_records)
    index_file, meta_file = index.save(directory=tmp_path, base_name="corrupt_dim")

    # Tamper with metadata JSON to report different dimension
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["metadata"]["dimension"] = 768
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(CorruptedIndexError):
        VectorIndex.load(directory=tmp_path, base_name="corrupt_dim")


# ==============================================================================
# L. Determinism & Ordering Tests
# ==============================================================================

def test_retrieval_determinism(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement L: Identical queries across identical index builds yield deterministic results."""
    idx1 = retrieval_service.build_index_from_records(sample_embedding_records, index_id="det_1")
    idx2 = retrieval_service.build_index_from_records(sample_embedding_records, index_id="det_2")

    query = "deep neural networks feature representations"
    res1 = retrieval_service.search(query=query, top_k=3, index=idx1)
    res2 = retrieval_service.search(query=query, top_k=3, index=idx2)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.chunk_id == r2.chunk_id
        assert abs(r1.similarity_score - r2.similarity_score) < 1e-5


# ==============================================================================
# M. Input Validation & Error Handling Tests
# ==============================================================================

def test_search_rejects_empty_query_string(retrieval_service: RetrievalService) -> None:
    """Requirement M: Empty query raises InvalidQueryError."""
    with pytest.raises(InvalidQueryError):
        retrieval_service.search(query="")


def test_search_rejects_whitespace_query_string(retrieval_service: RetrievalService) -> None:
    """Requirement M: Whitespace-only query raises InvalidQueryError."""
    with pytest.raises(InvalidQueryError):
        retrieval_service.search(query="    \n\t  ")


def test_search_rejects_non_positive_top_k(
    retrieval_service: RetrievalService,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement M: top_k <= 0 raises RetrievalError."""
    index = retrieval_service.build_index_from_records(sample_embedding_records)
    with pytest.raises(RetrievalError):
        retrieval_service.search(query="valid query", top_k=0, index=index)

    with pytest.raises(RetrievalError):
        retrieval_service.search(query="valid query", top_k=-5, index=index)


# ==============================================================================
# N. API Endpoint Tests
# ==============================================================================

def test_api_build_index_and_search(
    client: TestClient,
    sample_embedding_records: list[EmbeddingRecord],
) -> None:
    """Requirement N: Test /api/v1/retrieval/index and /api/v1/retrieval/search endpoints."""
    # 1. Build index via API
    records_payload = [r.model_dump() for r in sample_embedding_records]
    index_res = client.post(
        "/api/v1/retrieval/index",
        json={
            "index_id": "api_index_test",
            "document_id": "doc_ai",
            "records": records_payload,
        },
    )
    assert index_res.status_code == status.HTTP_200_OK
    index_data = index_res.json()
    assert index_data["index_id"] == "api_index_test"
    assert index_data["total_vectors"] == len(sample_embedding_records)
    assert index_data["dimension"] == 384

    # 2. Search via API
    search_res = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "machine learning mathematical models",
            "top_k": 2,
            "index_id": "api_index_test",
        },
    )
    assert search_res.status_code == status.HTTP_200_OK
    search_data = search_res.json()
    assert search_data["query"] == "machine learning mathematical models"
    assert search_data["total_results"] == 2
    assert len(search_data["results"]) == 2
    assert search_data["results"][0]["rank"] == 1
    assert search_data["results"][0]["chunk_id"] == "doc_ai_p001_c0001"
    assert search_data["results"][0]["document_id"] == "doc_ai"


def test_api_search_empty_query_returns_400(client: TestClient) -> None:
    """Requirement N: API returns 400 Bad Request on empty or whitespace query."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "   ",
            "top_k": 3,
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 400
