"""Comprehensive unit and integration tests for Phase 3 semantic embeddings."""

import math
import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.schemas.chunk import Chunk
from app.schemas.embedding import EmbeddingConfig, EmbeddingRecord
from app.services.embeddings.exceptions import InvalidChunkError
from app.services.embeddings.model import (
    DEFAULT_MODEL_NAME,
    EXPECTED_DIMENSION,
    EmbeddingModel,
    resolve_device,
)
from app.services.embeddings.service import EmbeddingService


@pytest.fixture(scope="session")
def shared_embedding_model() -> EmbeddingModel:
    """Session-scoped embedding model instance to avoid repeated loads across tests."""
    return EmbeddingModel.get_instance(model_name=DEFAULT_MODEL_NAME, device="auto")


@pytest.fixture
def embedding_service(shared_embedding_model: EmbeddingModel) -> EmbeddingService:
    """Provide an EmbeddingService reusing the preloaded session model."""
    return EmbeddingService(model=shared_embedding_model)


@pytest.fixture
def sample_chunk() -> Chunk:
    """Provide a single representative text Chunk."""
    return Chunk(
        chunk_id="doc_demo_p001_c0001",
        document_id="doc_demo",
        page_number=1,
        chunk_index=1,
        text="Dense embeddings capture semantic context for vector retrieval.",
        char_count=64,
        word_count=8,
        metadata={"source_page": 1},
    )


@pytest.fixture
def multi_page_chunks() -> list[Chunk]:
    """Provide a batch of chunks spanning multiple distinct pages."""
    return [
        Chunk(
            chunk_id="doc_multi_p001_c0000",
            document_id="doc_multi",
            page_number=1,
            chunk_index=0,
            text="First chapter introduction on page one.",
            char_count=40,
            word_count=6,
        ),
        Chunk(
            chunk_id="doc_multi_p001_c0001",
            document_id="doc_multi",
            page_number=1,
            chunk_index=1,
            text="Second chunk on page one detailing foundational principles.",
            char_count=59,
            word_count=8,
        ),
        Chunk(
            chunk_id="doc_multi_p002_c0002",
            document_id="doc_multi",
            page_number=2,
            chunk_index=2,
            text="Third chunk transitioning to page two analysis.",
            char_count=48,
            word_count=7,
        ),
        Chunk(
            chunk_id="doc_multi_p003_c0003",
            document_id="doc_multi",
            page_number=3,
            chunk_index=3,
            text="Fourth chunk discussing retrieval architectures on page three.",
            char_count=62,
            word_count=8,
        ),
    ]


# ==============================================================================
# A. Model Initialization Tests
# ==============================================================================

def test_model_initialization_and_metadata(shared_embedding_model: EmbeddingModel) -> None:
    """Requirement A: Model loads successfully, has correct identifier, and 384 dimensions."""
    assert shared_embedding_model is not None
    assert shared_embedding_model.model_name == DEFAULT_MODEL_NAME
    assert shared_embedding_model.dimension == EXPECTED_DIMENSION
    assert shared_embedding_model.device in {"cpu", "cuda"}


# ==============================================================================
# B & C. Basic Embedding and Shape Tests
# ==============================================================================

def test_embed_single_chunk(embedding_service: EmbeddingService, sample_chunk: Chunk) -> None:
    """Requirement B: One valid chunk produces exactly one EmbeddingRecord."""
    record = embedding_service.embed_chunk(sample_chunk)

    assert isinstance(record, EmbeddingRecord)
    assert record.chunk_id == sample_chunk.chunk_id
    assert len(record.embedding) == EXPECTED_DIMENSION
    assert record.dimension == EXPECTED_DIMENSION


def test_embedding_batch_shape_and_counts(
    embedding_service: EmbeddingService,
    multi_page_chunks: list[Chunk],
) -> None:
    """Requirement C: N input chunks produce N output records, all with dimension 384."""
    records = embedding_service.embed_chunks(multi_page_chunks)

    assert len(records) == len(multi_page_chunks)
    for record in records:
        assert len(record.embedding) == EXPECTED_DIMENSION
        assert record.dimension == EXPECTED_DIMENSION


# ==============================================================================
# D. Batch Processing & Order Preservation Tests
# ==============================================================================

def test_batch_processing_order_preserved(
    embedding_service: EmbeddingService,
    multi_page_chunks: list[Chunk],
) -> None:
    """Requirement D: Output order strictly matches input chunk order when batch_size < N."""
    config = EmbeddingConfig(batch_size=2, normalize_embeddings=True)
    records = embedding_service.embed_chunks(multi_page_chunks, config=config)

    assert len(records) == len(multi_page_chunks)
    for idx, (chunk, record) in enumerate(zip(multi_page_chunks, records)):
        assert record.chunk_id == chunk.chunk_id
        assert record.text == chunk.text
        assert record.page_number == chunk.page_number


# ==============================================================================
# E. Normalization Tests
# ==============================================================================

def test_l2_normalization_unit_length(
    embedding_service: EmbeddingService,
    sample_chunk: Chunk,
) -> None:
    """Requirement E: When normalize_embeddings=True, vector L2 norm is approximately 1.0."""
    config = EmbeddingConfig(normalize_embeddings=True)
    record = embedding_service.embed_chunk(sample_chunk, config=config)

    norm = math.sqrt(sum(x * x for x in record.embedding))
    assert abs(norm - 1.0) < 1e-4
    assert record.normalized is True


def test_unnormalized_embeddings_toggle(
    embedding_service: EmbeddingService,
    sample_chunk: Chunk,
) -> None:
    """Requirement E: Normalization flag can be toggled to False and recorded on EmbeddingRecord."""
    config = EmbeddingConfig(normalize_embeddings=False)
    record = embedding_service.embed_chunk(sample_chunk, config=config)

    assert record.normalized is False
    assert len(record.embedding) == EXPECTED_DIMENSION
    assert record.dimension == EXPECTED_DIMENSION



# ==============================================================================
# F & J. Metadata & Provenance Retention Tests
# ==============================================================================

def test_provenance_strictly_preserved(
    embedding_service: EmbeddingService,
    sample_chunk: Chunk,
) -> None:
    """Requirement F: chunk_id, document_id, page_number, and text survive embedding."""
    record = embedding_service.embed_chunk(sample_chunk)

    assert record.chunk_id == sample_chunk.chunk_id
    assert record.document_id == sample_chunk.document_id
    assert record.page_number == sample_chunk.page_number
    assert record.text == sample_chunk.text
    assert record.model_name == DEFAULT_MODEL_NAME
    assert record.device in {"cpu", "cuda"}


def test_multiple_pages_preserve_distinct_page_numbers(
    embedding_service: EmbeddingService,
    multi_page_chunks: list[Chunk],
) -> None:
    """Requirement J: Chunks from different pages retain their respective page numbers."""
    records = embedding_service.embed_chunks(multi_page_chunks)

    page_numbers = [r.page_number for r in records]
    assert page_numbers == [1, 1, 2, 3]


def test_raw_chunk_immutability(
    embedding_service: EmbeddingService,
    sample_chunk: Chunk,
) -> None:
    """Requirement 16: Embedding must NOT mutate original Chunk objects."""
    original_text = sample_chunk.text
    original_id = sample_chunk.chunk_id
    original_page = sample_chunk.page_number
    original_doc = sample_chunk.document_id

    embedding_service.embed_chunk(sample_chunk)

    assert sample_chunk.text == original_text
    assert sample_chunk.chunk_id == original_id
    assert sample_chunk.page_number == original_page
    assert sample_chunk.document_id == original_doc


# ==============================================================================
# G. Input Validation Tests
# ==============================================================================

def test_reject_none_chunks_list(embedding_service: EmbeddingService) -> None:
    """Requirement G: Rejecting None as chunks input."""
    with pytest.raises(InvalidChunkError):
        embedding_service.embed_chunks(None)  # type: ignore


def test_reject_none_chunk_in_batch(embedding_service: EmbeddingService, sample_chunk: Chunk) -> None:
    """Requirement G: Rejecting a list containing None."""
    with pytest.raises(InvalidChunkError):
        embedding_service.embed_chunks([sample_chunk, None])  # type: ignore


def test_reject_empty_text_chunk(embedding_service: EmbeddingService) -> None:
    """Requirement G: Rejecting a chunk with empty string."""
    empty_chunk = Chunk(
        chunk_id="doc_bad_p001_c0001",
        document_id="doc_bad",
        page_number=1,
        chunk_index=0,
        text="",
        char_count=0,
        word_count=0,
    )
    with pytest.raises(InvalidChunkError):
        embedding_service.embed_chunk(empty_chunk)


def test_reject_whitespace_only_chunk(embedding_service: EmbeddingService) -> None:
    """Requirement G: Rejecting a chunk with whitespace-only text."""
    ws_chunk = Chunk(
        chunk_id="doc_bad_p001_c0002",
        document_id="doc_bad",
        page_number=1,
        chunk_index=0,
        text="   \n\t   ",
        char_count=7,
        word_count=0,
    )
    with pytest.raises(InvalidChunkError):
        embedding_service.embed_chunk(ws_chunk)


def test_empty_chunks_list_returns_empty_list(embedding_service: EmbeddingService) -> None:
    """Empty input list returns an empty list without error."""
    assert embedding_service.embed_chunks([]) == []


# ==============================================================================
# H. Determinism Tests
# ==============================================================================

def test_embedding_determinism(
    embedding_service: EmbeddingService,
    sample_chunk: Chunk,
) -> None:
    """Requirement H: Repeated embedding of identical input yields identical vector values."""
    rec1 = embedding_service.embed_chunk(sample_chunk)
    rec2 = embedding_service.embed_chunk(sample_chunk)

    v1 = np.array(rec1.embedding, dtype=np.float32)
    v2 = np.array(rec2.embedding, dtype=np.float32)

    assert np.allclose(v1, v2, atol=1e-5)


# ==============================================================================
# I. Configuration Tests
# ==============================================================================

def test_device_resolution_auto_and_cpu() -> None:
    """Requirement I: Device resolution handles 'auto' and 'cpu'."""
    assert resolve_device("cpu") == "cpu"
    auto_dev = resolve_device("auto")
    assert auto_dev in {"cpu", "cuda"}


def test_config_device_validation() -> None:
    """Requirement I: Invalid device strings are rejected by Pydantic validation."""
    with pytest.raises(ValueError):
        EmbeddingConfig(device="tpu")

    with pytest.raises(ValueError):
        EmbeddingConfig(batch_size=0)


# ==============================================================================
# API Endpoint Tests
# ==============================================================================

def test_api_embed_single_chunk(client: TestClient) -> None:
    """Test POST /api/v1/embeddings/embed-chunk endpoint."""
    response = client.post(
        "/api/v1/embeddings/embed-chunk",
        json={
            "chunk_id": "api_chunk_001",
            "document_id": "api_doc",
            "page_number": 5,
            "text": "Retrieval Augmented Generation with dense representations.",
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["chunk_id"] == "api_chunk_001"
    assert data["page_number"] == 5
    assert data["dimension"] == 384
    assert len(data["embedding"]) == 384
    assert data["normalized"] is True


def test_api_embed_chunks_batch(client: TestClient) -> None:
    """Test POST /api/v1/embeddings/embed-chunks batch endpoint."""
    response = client.post(
        "/api/v1/embeddings/embed-chunks",
        json={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "page_number": 1,
                    "text": "First passage on page one.",
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "page_number": 2,
                    "text": "Second passage on page two.",
                },
            ]
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_embedded"] == 2
    assert data["dimension"] == 384
    assert len(data["records"]) == 2
    assert data["records"][0]["page_number"] == 1
    assert data["records"][1]["page_number"] == 2


def test_api_embed_empty_chunk_returns_400(client: TestClient) -> None:
    """API endpoint returns 400 for empty or whitespace text."""
    response = client.post(
        "/api/v1/embeddings/embed-chunk",
        json={
            "chunk_id": "c_empty",
            "document_id": "d1",
            "page_number": 1,
            "text": "   ",
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 400
