"""Comprehensive unit and integration tests for Phase 5 Semantic Search."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.v1.endpoints.search import get_search_service
from app.main import app
from app.schemas.chunk import Chunk
from app.schemas.embedding import EmbeddingRecord
from app.schemas.retrieval import RetrievalResult
from app.schemas.search import DEFAULT_TOP_K, MAX_TOP_K, SearchRequest, SearchResponse, SearchResult
from app.services.embeddings.exceptions import EmbeddingError
from app.services.embeddings.model import DEFAULT_MODEL_NAME, EmbeddingModel
from app.services.embeddings.service import EmbeddingService
from app.services.retrieval.exceptions import IndexNotFoundError, RetrievalError
from app.services.retrieval.index import VectorIndex
from app.services.retrieval.service import RetrievalService
from app.services.search.exceptions import (
    DocumentNotFoundError,
    IndexNotInitializedError,
    InvalidSearchQueryError,
    InvalidTopKError,
    SearchEmbeddingError,
    SearchError,
    SearchRetrievalError,
)
from app.services.search.service import SearchService


# ==============================================================================
# Fixtures and Test Helpers
# ==============================================================================

@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Fast mock embedding service producing deterministic 384-d normalized unit vectors."""
    mock_svc = MagicMock(spec=EmbeddingService)
    unit_vec = np.ones((384,), dtype=np.float32) / np.sqrt(384)
    mock_svc.embed_query.return_value = unit_vec
    return mock_svc


@pytest.fixture
def mock_sample_results() -> list[RetrievalResult]:
    """Sample retrieval results with complete provenance."""
    return [
        RetrievalResult(
            rank=1,
            chunk_id="doc_ai_p001_c0001",
            document_id="doc_ai",
            page_number=1,
            chunk_index=0,
            text="Machine learning algorithms build mathematical models.",
            similarity_score=0.8842,
            metadata={"device": "cpu"},
        ),
        RetrievalResult(
            rank=2,
            chunk_id="doc_ai_p002_c0002",
            document_id="doc_ai",
            page_number=2,
            chunk_index=1,
            text="Deep neural networks learn hierarchical representations.",
            similarity_score=0.8123,
            metadata={"device": "cpu"},
        ),
        RetrievalResult(
            rank=3,
            chunk_id="doc_bio_p010_c0003",
            document_id="doc_bio",
            page_number=10,
            chunk_index=2,
            text="Photosynthesis converts light into chemical energy.",
            similarity_score=0.6510,
            metadata={"device": "cpu"},
        ),
    ]


@pytest.fixture
def mock_retrieval_service(mock_sample_results: list[RetrievalResult]) -> MagicMock:
    """Mock retrieval service with a registered mock index and search method."""
    mock_svc = MagicMock(spec=RetrievalService)
    mock_index = MagicMock(spec=VectorIndex)
    mock_index.index_id = "default"
    mock_index.total_vectors = 3
    mock_index.metadata.document_ids = ["doc_ai", "doc_bio"]

    mock_svc.get_index.return_value = mock_index
    mock_svc._indices = {"default": mock_index, "doc_ai": mock_index}
    mock_svc.search_by_vector.return_value = mock_sample_results
    return mock_svc


@pytest.fixture
def search_service(
    mock_embedding_service: MagicMock,
    mock_retrieval_service: MagicMock,
) -> SearchService:
    """SearchService wired with fast mock dependencies."""
    return SearchService(
        embedding_service=mock_embedding_service,
        retrieval_service=mock_retrieval_service,
    )


# ==============================================================================
# Suite A: Request Validation
# ==============================================================================

class TestRequestValidation:
    """Tests for SearchRequest schema validation rules."""

    def test_valid_request_with_defaults(self):
        req = SearchRequest(query="What is gradient descent?")
        assert req.query == "What is gradient descent?"
        assert req.top_k == DEFAULT_TOP_K
        assert req.document_id is None

    def test_valid_request_with_all_fields(self):
        req = SearchRequest(query="What is backprop?", top_k=10, document_id="doc_42")
        assert req.query == "What is backprop?"
        assert req.top_k == 10
        assert req.document_id == "doc_42"

    def test_empty_query_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")
        assert "query" in str(exc_info.value)

    def test_whitespace_query_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="    \t \n  ")
        assert "Search query cannot be empty or whitespace-only" in str(exc_info.value)

    def test_query_is_trimmed(self):
        req = SearchRequest(query="   trimmed query text   ")
        assert req.query == "trimmed query text"

    def test_top_k_one_is_valid(self):
        req = SearchRequest(query="valid query", top_k=1)
        assert req.top_k == 1

    def test_top_k_upper_bound_is_valid(self):
        req = SearchRequest(query="valid query", top_k=MAX_TOP_K)
        assert req.top_k == MAX_TOP_K

    def test_top_k_zero_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="valid query", top_k=0)
        assert "top_k" in str(exc_info.value)

    def test_negative_top_k_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="valid query", top_k=-5)
        assert "top_k" in str(exc_info.value)

    def test_excessively_large_top_k_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="valid query", top_k=MAX_TOP_K + 1)
        assert "top_k" in str(exc_info.value)

    def test_whitespace_document_id_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="valid query", document_id="   ")
        assert "document_id cannot be empty or whitespace-only" in str(exc_info.value)

    def test_document_id_is_trimmed(self):
        req = SearchRequest(query="valid query", document_id="  doc_abc  ")
        assert req.document_id == "doc_abc"


# ==============================================================================
# Suite B: SearchService Orchestration
# ==============================================================================

class TestSearchServiceOrchestration:
    """Tests for SearchService orchestration, ranking, and provenance preservation."""

    def test_successful_search_orchestration(
        self,
        search_service: SearchService,
        mock_embedding_service: MagicMock,
        mock_retrieval_service: MagicMock,
    ):
        response = search_service.search(query="machine learning", top_k=3)

        assert isinstance(response, SearchResponse)
        assert response.query == "machine learning"
        assert response.total_results == 3
        assert len(response.results) == 3

        # Assert query embedding was invoked
        mock_embedding_service.embed_query.assert_called_once_with("machine learning")

        # Assert retrieval service was invoked
        mock_retrieval_service.search_by_vector.assert_called_once()
        call_kwargs = mock_retrieval_service.search_by_vector.call_args[1]
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["document_id"] is None

    def test_correct_ranking_order(self, search_service: SearchService):
        response = search_service.search(query="test ranking", top_k=3)
        ranks = [r.rank for r in response.results]
        assert ranks == [1, 2, 3]

        scores = [r.similarity_score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_complete_provenance_preservation(self, search_service: SearchService):
        response = search_service.search(query="provenance test", top_k=1)
        top = response.results[0]

        assert isinstance(top, SearchResult)
        assert top.chunk_id == "doc_ai_p001_c0001"
        assert top.document_id == "doc_ai"
        assert top.page_number == 1
        assert top.chunk_index == 0
        assert "Machine learning" in top.text
        assert isinstance(top.similarity_score, float)
        assert top.similarity_score == pytest.approx(0.8842, abs=1e-4)

    def test_internal_faiss_slots_not_exposed(self, search_service: SearchService):
        response = search_service.search(query="slot test", top_k=1)
        top_dict = response.results[0].model_dump()
        assert "vector_index" not in top_dict


# ==============================================================================
# Suite C: Document Isolation
# ==============================================================================

class TestDocumentIsolation:
    """Tests for document-level filtering and isolation in search."""

    def test_requested_document_only(
        self,
        search_service: SearchService,
        mock_retrieval_service: MagicMock,
    ):
        mock_retrieval_service.search_by_vector.return_value = [
            RetrievalResult(
                rank=1,
                chunk_id="doc_ai_p001_c0001",
                document_id="doc_ai",
                page_number=1,
                chunk_index=0,
                text="AI chunk",
                similarity_score=0.9,
            )
        ]

        response = search_service.search(query="AI test", top_k=5, document_id="doc_ai")
        assert response.document_id == "doc_ai"
        assert all(r.document_id == "doc_ai" for r in response.results)

    def test_unknown_document_raises_document_not_found_error(
        self,
        search_service: SearchService,
    ):
        with pytest.raises(DocumentNotFoundError) as exc_info:
            search_service.search(query="query", top_k=5, document_id="nonexistent_doc")

        assert "nonexistent_doc" in str(exc_info.value)

    def test_results_from_other_documents_excluded(
        self,
        search_service: SearchService,
        mock_retrieval_service: MagicMock,
    ):
        search_service.search(query="query", top_k=5, document_id="doc_ai")
        call_kwargs = mock_retrieval_service.search_by_vector.call_args[1]
        assert call_kwargs["document_id"] == "doc_ai"


# ==============================================================================
# Suite D: Error Handling
# ==============================================================================

class TestErrorHandling:
    """Tests for safe error handling across index, embedding, and retrieval components."""

    def test_empty_index_returns_empty_results_safely(
        self,
        mock_embedding_service: MagicMock,
    ):
        empty_retrieval_svc = MagicMock(spec=RetrievalService)
        empty_index = MagicMock(spec=VectorIndex)
        empty_index.index_id = "empty_idx"
        empty_index.total_vectors = 0
        empty_index.metadata.document_ids = []
        empty_retrieval_svc.get_index.return_value = empty_index

        service = SearchService(
            embedding_service=mock_embedding_service,
            retrieval_service=empty_retrieval_svc,
        )

        response = service.search(query="query against empty index")
        assert response.total_results == 0
        assert response.results == []
        mock_embedding_service.embed_query.assert_not_called()

    def test_empty_index_with_document_id_raises_not_found(
        self,
        mock_embedding_service: MagicMock,
    ):
        empty_retrieval_svc = MagicMock(spec=RetrievalService)
        empty_index = MagicMock(spec=VectorIndex)
        empty_index.index_id = "empty_idx"
        empty_index.total_vectors = 0
        empty_index.metadata.document_ids = []
        empty_retrieval_svc.get_index.return_value = empty_index

        service = SearchService(
            embedding_service=mock_embedding_service,
            retrieval_service=empty_retrieval_svc,
        )

        with pytest.raises(DocumentNotFoundError):
            service.search(query="query", document_id="doc_123")

    def test_uninitialized_index_raises_index_not_initialized_error(
        self,
        mock_embedding_service: MagicMock,
    ):
        uninit_retrieval_svc = MagicMock(spec=RetrievalService)
        uninit_retrieval_svc.get_index.side_effect = IndexNotFoundError("No index")

        service = SearchService(
            embedding_service=mock_embedding_service,
            retrieval_service=uninit_retrieval_svc,
        )

        with pytest.raises(IndexNotInitializedError):
            service.search(query="query on uninit index")

    def test_embedding_failure_raises_search_embedding_error(
        self,
        mock_retrieval_service: MagicMock,
    ):
        failing_embed_svc = MagicMock(spec=EmbeddingService)
        failing_embed_svc.embed_query.side_effect = EmbeddingError("Model failure")

        service = SearchService(
            embedding_service=failing_embed_svc,
            retrieval_service=mock_retrieval_service,
        )

        with pytest.raises(SearchEmbeddingError) as exc_info:
            service.search(query="query failing embedding")
        assert "embedding" in str(exc_info.value).lower()

    def test_retrieval_failure_raises_search_retrieval_error(
        self,
        mock_embedding_service: MagicMock,
        mock_retrieval_service: MagicMock,
    ):
        mock_retrieval_service.search_by_vector.side_effect = RetrievalError("FAISS crash")

        service = SearchService(
            embedding_service=mock_embedding_service,
            retrieval_service=mock_retrieval_service,
        )

        with pytest.raises(SearchRetrievalError) as exc_info:
            service.search(query="query failing retrieval")
        assert "retrieval" in str(exc_info.value).lower()

    def test_malformed_retrieval_item_raises_search_retrieval_error(
        self,
        mock_embedding_service: MagicMock,
        mock_retrieval_service: MagicMock,
    ):
        malformed_item = MagicMock()
        malformed_item.similarity_score = float("nan")  # NaN score
        malformed_item.chunk_id = None
        mock_retrieval_service.search_by_vector.return_value = [malformed_item]

        service = SearchService(
            embedding_service=mock_embedding_service,
            retrieval_service=mock_retrieval_service,
        )

        with pytest.raises(SearchRetrievalError) as exc_info:
            service.search(query="query with malformed item")
        assert "malformed" in str(exc_info.value).lower()

    def test_service_validation_empty_query(self, search_service: SearchService):
        with pytest.raises(InvalidSearchQueryError):
            search_service.search(query="   ")

    def test_service_validation_invalid_top_k(self, search_service: SearchService):
        with pytest.raises(InvalidTopKError):
            search_service.search(query="valid query", top_k=0)

        with pytest.raises(InvalidTopKError):
            search_service.search(query="valid query", top_k=MAX_TOP_K + 1)


# ==============================================================================
# Suite E: API Endpoints (POST /api/v1/search)
# ==============================================================================

class TestSearchAPI:
    """Tests for the FastAPI POST /api/v1/search endpoint."""

    @pytest.fixture(autouse=True)
    def setup_api_override(self, client: TestClient, search_service: SearchService):
        """Override the get_search_service dependency with the mock search service."""
        client.app.dependency_overrides[get_search_service] = lambda: search_service
        yield
        client.app.dependency_overrides.pop(get_search_service, None)

    def test_post_search_success(self, client: TestClient):
        payload = {
            "query": "What is machine learning?",
            "top_k": 2,
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["query"] == "What is machine learning?"
        assert data["total_results"] == 3  # based on mock sample results
        assert len(data["results"]) == 3
        top = data["results"][0]
        assert top["rank"] == 1
        assert top["chunk_id"] == "doc_ai_p001_c0001"
        assert top["similarity_score"] == pytest.approx(0.8842, abs=1e-4)

    def test_post_search_with_document_isolation(self, client: TestClient):
        payload = {
            "query": "Deep learning architectures",
            "top_k": 5,
            "document_id": "doc_ai",
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["document_id"] == "doc_ai"

    def test_post_search_validation_empty_query(self, client: TestClient):
        response = client.post("/api/v1/search", json={"query": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "error" in response.json()

    def test_post_search_validation_whitespace_query(self, client: TestClient):
        response = client.post("/api/v1/search", json={"query": "    "})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_search_validation_invalid_top_k(self, client: TestClient):
        response = client.post("/api/v1/search", json={"query": "valid query", "top_k": -1})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_search_validation_excessive_top_k(self, client: TestClient):
        response = client.post("/api/v1/search", json={"query": "valid query", "top_k": 101})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_search_unknown_document_returns_404(self, client: TestClient):
        payload = {
            "query": "valid query",
            "document_id": "doc_unknown_404",
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_body = response.json()["error"]
        assert error_body["code"] == 404
        assert "doc_unknown_404" in error_body["message"]

    def test_post_search_embedding_error_returns_500(
        self,
        client: TestClient,
        search_service: SearchService,
    ):
        search_service.embedding_service.embed_query.side_effect = EmbeddingError("Model failure")

        response = client.post("/api/v1/search", json={"query": "trigger failure"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        error_body = response.json()["error"]
        assert error_body["code"] == 500
        # Verify stack trace or internal details not leaked
        assert "Traceback" not in response.text
        assert "model_weights" not in response.text

    def test_post_search_retrieval_error_returns_500(
        self,
        client: TestClient,
        search_service: SearchService,
    ):
        search_service.retrieval_service.search_by_vector.side_effect = RetrievalError("FAISS crash")

        response = client.post("/api/v1/search", json={"query": "trigger retrieval failure"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        error_body = response.json()["error"]
        assert error_body["code"] == 500
        assert "Traceback" not in response.text


# ==============================================================================
# Suite F: Full Real Integration (Phase 3 + Phase 4 + Phase 5)
# ==============================================================================

class TestSearchEndToEndIntegration:
    """Integration test with real EmbeddingModel and FAISS VectorIndex."""

    @pytest.fixture(scope="class")
    def real_embedding_model(self) -> EmbeddingModel:
        return EmbeddingModel.get_instance(model_name=DEFAULT_MODEL_NAME, device="auto")

    @pytest.fixture
    def real_search_service(self, real_embedding_model: EmbeddingModel) -> SearchService:
        embed_svc = EmbeddingService(model=real_embedding_model)
        retrieval_svc = RetrievalService(embedding_service=embed_svc)

        # Build real index with distinct semantic domains
        chunks = [
            Chunk(
                chunk_id="doc_ai_p001_c0001",
                document_id="doc_ai",
                page_number=1,
                chunk_index=0,
                text="Convolutional neural networks process image data through convolutional filters.",
                char_count=80,
                word_count=10,
            ),
            Chunk(
                chunk_id="doc_bio_p010_c0002",
                document_id="doc_bio",
                page_number=10,
                chunk_index=1,
                text="Mitochondria are the powerhouse of the eukaryotic cell producing ATP.",
                char_count=71,
                word_count=10,
            ),
        ]
        retrieval_svc.build_index_from_chunks(chunks=chunks, index_id="real_test_index", set_as_default=True)
        return SearchService(embedding_service=embed_svc, retrieval_service=retrieval_svc)

    def test_real_semantic_retrieval(self, real_search_service: SearchService):
        response = real_search_service.search(query="image processing with deep networks", top_k=2)
        assert response.total_results == 2
        # Top result should be the AI chunk
        assert response.results[0].chunk_id == "doc_ai_p001_c0001"
        assert response.results[0].similarity_score > response.results[1].similarity_score

    def test_real_document_isolation(self, real_search_service: SearchService):
        # Even if query is about cells, restrict to doc_ai
        response = real_search_service.search(
            query="cellular biology and energy",
            top_k=2,
            document_id="doc_ai",
        )
        assert len(response.results) == 1
        assert response.results[0].document_id == "doc_ai"

    def test_real_unknown_document_raises_404(self, real_search_service: SearchService):
        with pytest.raises(DocumentNotFoundError):
            real_search_service.search(
                query="any query",
                document_id="doc_unknown",
            )
