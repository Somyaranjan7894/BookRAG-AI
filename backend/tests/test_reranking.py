"""Comprehensive unit and integration tests for Phase 6 Cross-Encoder Reranking."""

from typing import List
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.v1.endpoints.search import get_search_service
from app.core.config import settings
from app.main import app
from app.schemas.retrieval import RetrievalResult
from app.schemas.search import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.embeddings.service import EmbeddingService
from app.services.reranking.exceptions import (
    InvalidRerankingConfigError,
    RerankerModelLoadError,
    RerankingError,
    RerankingExecutionError,
    ScoreAlignmentError,
)
from app.services.reranking.model import (
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL_NAME,
    RerankerModel,
)
from app.services.reranking.service import RerankerService
from app.services.retrieval.exceptions import IndexNotFoundError
from app.services.retrieval.index import VectorIndex
from app.services.retrieval.service import RetrievalService
from app.services.search.exceptions import DocumentNotFoundError
from app.services.search.service import SearchService


# ==============================================================================
# Fixtures and Test Helpers
# ==============================================================================

@pytest.fixture
def sample_retrieval_candidates() -> List[RetrievalResult]:
    """Sample first-stage candidates ranked by FAISS cosine similarity."""
    return [
        RetrievalResult(
            rank=1,
            chunk_id="doc_ai_p001_c0001",
            document_id="doc_ai",
            page_number=1,
            chunk_index=0,
            text="Machine learning algorithms build mathematical models from training data.",
            similarity_score=0.9100,
            metadata={"source": "book_intro"},
        ),
        RetrievalResult(
            rank=2,
            chunk_id="doc_ai_p002_c0002",
            document_id="doc_ai",
            page_number=2,
            chunk_index=1,
            text="Backpropagation calculates loss function gradients with respect to neural network weights.",
            similarity_score=0.8400,
            metadata={"source": "book_ch2"},
        ),
        RetrievalResult(
            rank=3,
            chunk_id="doc_ai_p003_c0003",
            document_id="doc_ai",
            page_number=3,
            chunk_index=2,
            text="The history of artificial intelligence dates back to the mid-twentieth century.",
            similarity_score=0.7200,
            metadata={"source": "book_ch1"},
        ),
    ]


@pytest.fixture
def mock_reranker_model() -> MagicMock:
    """Mock RerankerModel returning deterministic cross-encoder scores."""
    mock_model = MagicMock(spec=RerankerModel)
    mock_model.model_name = DEFAULT_RERANKER_MODEL_NAME
    mock_model.device = "cpu"
    mock_model.max_length = DEFAULT_RERANKER_MAX_LENGTH
    # Default: candidate 2 scores highest (8.5), candidate 1 scores lower (3.2), candidate 3 lowest (1.1)
    mock_model.predict.return_value = np.array([3.2, 8.5, 1.1], dtype=np.float32)
    return mock_model


@pytest.fixture
def reranker_service(mock_reranker_model: MagicMock) -> RerankerService:
    """RerankerService wired with mock RerankerModel."""
    return RerankerService(model=mock_reranker_model, batch_size=16, max_length=512)


# ==============================================================================
# Suite A: Reranker Unit Tests
# ==============================================================================

class TestRerankerUnit:
    """Unit tests for RerankerService scoring, sorting, and provenance mapping."""

    def test_successful_reranking(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        results = reranker_service.rerank(
            query="How does backpropagation compute gradients?",
            candidates=sample_retrieval_candidates,
        )
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_descending_score_order(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        results = reranker_service.rerank(
            query="backpropagation",
            candidates=sample_retrieval_candidates,
        )
        scores = [r.reranker_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(8.5, abs=1e-4)
        assert scores[1] == pytest.approx(3.2, abs=1e-4)
        assert scores[2] == pytest.approx(1.1, abs=1e-4)

    def test_rank_reassignment(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        results = reranker_service.rerank(
            query="backpropagation",
            candidates=sample_retrieval_candidates,
        )
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_original_rank_preservation(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        results = reranker_service.rerank(
            query="backpropagation",
            candidates=sample_retrieval_candidates,
        )
        # Winner is candidate 2, whose original FAISS rank was 2
        assert results[0].chunk_id == "doc_ai_p002_c0002"
        assert results[0].original_rank == 2
        assert results[0].rank == 1

        # Second place is candidate 1, whose original FAISS rank was 1
        assert results[1].chunk_id == "doc_ai_p001_c0001"
        assert results[1].original_rank == 1
        assert results[1].rank == 2

    def test_complete_provenance_preservation(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        results = reranker_service.rerank(
            query="backpropagation",
            candidates=sample_retrieval_candidates,
        )
        top = results[0]
        assert top.chunk_id == "doc_ai_p002_c0002"
        assert top.document_id == "doc_ai"
        assert top.page_number == 2
        assert top.chunk_index == 1
        assert "Backpropagation calculates" in top.text
        assert top.similarity_score == pytest.approx(0.8400, abs=1e-4)
        assert top.reranker_score == pytest.approx(8.5, abs=1e-4)
        assert top.metadata == {"source": "book_ch2"}

    def test_empty_candidates_returns_empty_list(
        self,
        reranker_service: RerankerService,
        mock_reranker_model: MagicMock,
    ):
        results = reranker_service.rerank(query="empty test", candidates=[])
        assert results == []
        mock_reranker_model.predict.assert_not_called()

    def test_single_candidate(
        self,
        mock_reranker_model: MagicMock,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        mock_reranker_model.predict.return_value = np.array([4.2], dtype=np.float32)
        service = RerankerService(model=mock_reranker_model)

        results = service.rerank(
            query="single test",
            candidates=[sample_retrieval_candidates[0]],
        )
        assert len(results) == 1
        assert results[0].rank == 1
        assert results[0].original_rank == 1
        assert results[0].reranker_score == pytest.approx(4.2, abs=1e-4)

    def test_top_n_truncation(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        results = reranker_service.rerank(
            query="test",
            candidates=sample_retrieval_candidates,
            top_n=2,
        )
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_duplicate_candidate_texts(
        self,
        mock_reranker_model: MagicMock,
    ):
        # Two identical text chunks from different pages/chunks
        candidates = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk_a",
                document_id="doc_1",
                page_number=1,
                text="Identical repeated text.",
                similarity_score=0.9,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="chunk_b",
                document_id="doc_1",
                page_number=5,
                text="Identical repeated text.",
                similarity_score=0.85,
            ),
        ]
        mock_reranker_model.predict.return_value = np.array([5.0, 5.0], dtype=np.float32)
        service = RerankerService(model=mock_reranker_model)

        results = service.rerank(query="query", candidates=candidates)
        assert len(results) == 2
        assert {r.chunk_id for r in results} == {"chunk_a", "chunk_b"}

    def test_malformed_candidate_missing_text_raises_error(
        self,
        reranker_service: RerankerService,
    ):
        bad_candidate = MagicMock()
        bad_candidate.text = None
        with pytest.raises(RerankingExecutionError) as exc_info:
            reranker_service.rerank(query="query", candidates=[bad_candidate])
        assert "text" in str(exc_info.value).lower()

    def test_score_alignment_mismatch_raises_error(
        self,
        mock_reranker_model: MagicMock,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        # Return 2 scores for 3 candidates
        mock_reranker_model.predict.return_value = np.array([1.0, 2.0], dtype=np.float32)
        service = RerankerService(model=mock_reranker_model)

        with pytest.raises(ScoreAlignmentError) as exc_info:
            service.rerank(query="query", candidates=sample_retrieval_candidates)
        assert "alignment" in str(exc_info.type.__name__).lower() or "match" in str(exc_info.value).lower()

    def test_non_finite_score_raises_error(
        self,
        mock_reranker_model: MagicMock,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        mock_reranker_model.predict.return_value = np.array([float("nan"), 1.0, 2.0], dtype=np.float32)
        service = RerankerService(model=mock_reranker_model)

        with pytest.raises(RerankingExecutionError):
            service.rerank(query="query", candidates=sample_retrieval_candidates)

    def test_empty_query_raises_error(self, reranker_service: RerankerService):
        with pytest.raises(InvalidRerankingConfigError):
            reranker_service.rerank(query="   ", candidates=[MagicMock()])

    def test_negative_top_n_raises_error(
        self,
        reranker_service: RerankerService,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        with pytest.raises(InvalidRerankingConfigError):
            reranker_service.rerank(query="query", candidates=sample_retrieval_candidates, top_n=-1)


# ==============================================================================
# Suite B: Mocked Model Forward Pass & Batching Tests
# ==============================================================================

class TestMockedModelInference:
    """Tests checking input pair structure, batching, and singleton caching."""

    def test_query_passage_pairs_construction(
        self,
        reranker_service: RerankerService,
        mock_reranker_model: MagicMock,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        query = "What is backprop?"
        reranker_service.rerank(query=query, candidates=sample_retrieval_candidates)

        mock_reranker_model.predict.assert_called_once()
        passed_pairs = mock_reranker_model.predict.call_args[0][0]
        assert len(passed_pairs) == 3
        for pair, cand in zip(passed_pairs, sample_retrieval_candidates):
            assert pair[0] == query
            assert pair[1] == cand.text

    def test_batching_parameter_passed(
        self,
        mock_reranker_model: MagicMock,
        sample_retrieval_candidates: List[RetrievalResult],
    ):
        service = RerankerService(model=mock_reranker_model, batch_size=8)
        service.rerank(query="query", candidates=sample_retrieval_candidates)

        call_kwargs = mock_reranker_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 8

    def test_model_singleton_caching(self):
        """Verify RerankerModel.get_instance reuses cached instances for identical specs."""
        with patch("app.services.reranking.model.CrossEncoder") as mock_ce:
            inst1 = RerankerModel.get_instance(model_name="test/model", device="cpu", max_length=256)
            inst2 = RerankerModel.get_instance(model_name="test/model", device="cpu", max_length=256)
            assert inst1 is inst2
            # CrossEncoder constructor should be called exactly once
            mock_ce.assert_called_once()


# ==============================================================================
# Suite C: SearchService Integration (FAISS -> Reranker)
# ==============================================================================

class TestSearchRerankingIntegration:
    """Tests confirming the two-stage search pipeline reorders FAISS candidates."""

    @pytest.fixture
    def mock_embed_service(self) -> MagicMock:
        svc = MagicMock(spec=EmbeddingService)
        svc.embed_query.return_value = np.ones((384,), dtype=np.float32) / np.sqrt(384)
        return svc

    @pytest.fixture
    def mock_retrieval_svc(self) -> MagicMock:
        svc = MagicMock(spec=RetrievalService)
        index = MagicMock(spec=VectorIndex)
        index.index_id = "default"
        index.total_vectors = 3
        index.metadata.document_ids = ["doc_ai"]
        svc.get_index.return_value = index
        svc._indices = {"default": index, "doc_ai": index}

        # 3 FAISS candidates: candidate 1 has highest similarity, candidate 2 has lower
        svc.search_by_vector.return_value = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk_faiss_rank_1",
                document_id="doc_ai",
                page_number=1,
                text="Superficial keyword match.",
                similarity_score=0.95,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="chunk_faiss_rank_2",
                document_id="doc_ai",
                page_number=2,
                text="Deep semantic explanation that answers the user question.",
                similarity_score=0.80,
            ),
            RetrievalResult(
                rank=3,
                chunk_id="chunk_faiss_rank_3",
                document_id="doc_ai",
                page_number=3,
                text="Tangentially related concept.",
                similarity_score=0.70,
            ),
        ]
        return svc

    def test_reranker_changes_faiss_rank_1(
        self,
        mock_embed_service: MagicMock,
        mock_retrieval_svc: MagicMock,
    ):
        """CRITICAL TEST: FAISS rank 1 is NOT the final rank 1 after CrossEncoder scoring."""
        # CrossEncoder assigns Candidate 2 (chunk_faiss_rank_2) the highest score (9.4)
        mock_model = MagicMock(spec=RerankerModel)
        mock_model.predict.return_value = np.array([2.1, 9.4, 1.0], dtype=np.float32)
        reranker_svc = RerankerService(model=mock_model)

        search_svc = SearchService(
            embedding_service=mock_embed_service,
            retrieval_service=mock_retrieval_svc,
            reranker_service=reranker_svc,
            enable_reranking=True,
        )

        response = search_svc.search(query="Why does gradient descent converge?", top_k=2)
        assert response.reranking_applied is True
        assert len(response.results) == 2

        # Winner must be chunk_faiss_rank_2
        assert response.results[0].chunk_id == "chunk_faiss_rank_2"
        assert response.results[0].rank == 1
        assert response.results[0].original_rank == 2  # Proves original FAISS position preserved!
        assert response.results[0].reranker_score == pytest.approx(9.4, abs=1e-4)

        # Runner up must be chunk_faiss_rank_1
        assert response.results[1].chunk_id == "chunk_faiss_rank_1"
        assert response.results[1].rank == 2
        assert response.results[1].original_rank == 1
        assert response.results[1].reranker_score == pytest.approx(2.1, abs=1e-4)

    def test_candidate_k_larger_than_top_k(
        self,
        mock_embed_service: MagicMock,
        mock_retrieval_svc: MagicMock,
    ):
        """Verify FAISS is requested with candidate_k (20) while top_k (2) is returned."""
        mock_model = MagicMock(spec=RerankerModel)
        mock_model.predict.return_value = np.array([5.0, 4.0, 3.0], dtype=np.float32)
        reranker_svc = RerankerService(model=mock_model)

        search_svc = SearchService(
            embedding_service=mock_embed_service,
            retrieval_service=mock_retrieval_svc,
            reranker_service=reranker_svc,
            default_candidate_k=20,
            enable_reranking=True,
        )

        response = search_svc.search(query="test", top_k=2, candidate_k=15)
        # Check FAISS was queried for candidate_k=15
        mock_retrieval_svc.search_by_vector.assert_called_once()
        call_kwargs = mock_retrieval_svc.search_by_vector.call_args[1]
        assert call_kwargs["top_k"] == 15
        assert len(response.results) == 2


# ==============================================================================
# Suite D: Document Isolation Tests
# ==============================================================================

class TestDocumentIsolationReranking:
    """Tests ensuring document isolation is strictly preserved through reranking."""

    def test_document_isolation_reranks_only_target_doc(self):
        mock_embed = MagicMock(spec=EmbeddingService)
        mock_embed.embed_query.return_value = np.zeros((384,), dtype=np.float32)

        mock_retrieval = MagicMock(spec=RetrievalService)
        index = MagicMock(spec=VectorIndex)
        index.total_vectors = 2
        index.metadata.document_ids = ["doc_book_a"]
        mock_retrieval.get_index.return_value = index
        mock_retrieval._indices = {"doc_book_a": index}

        # FAISS returns only doc_book_a
        mock_retrieval.search_by_vector.return_value = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk_a1",
                document_id="doc_book_a",
                page_number=1,
                text="Text from book A",
                similarity_score=0.9,
            )
        ]

        mock_model = MagicMock(spec=RerankerModel)
        mock_model.predict.return_value = np.array([7.2], dtype=np.float32)
        reranker_svc = RerankerService(model=mock_model)

        search_svc = SearchService(
            embedding_service=mock_embed,
            retrieval_service=mock_retrieval,
            reranker_service=reranker_svc,
            enable_reranking=True,
        )

        response = search_svc.search(query="query", document_id="doc_book_a")
        assert response.document_id == "doc_book_a"
        assert len(response.results) == 1
        assert response.results[0].document_id == "doc_book_a"

    def test_unknown_document_raises_404_without_calling_reranker(self):
        mock_retrieval = MagicMock(spec=RetrievalService)
        index = MagicMock(spec=VectorIndex)
        index.total_vectors = 5
        index.metadata.document_ids = ["doc_known"]
        mock_retrieval.get_index.return_value = index
        mock_retrieval._indices = {"doc_known": index}

        mock_reranker = MagicMock(spec=RerankerService)
        search_svc = SearchService(
            retrieval_service=mock_retrieval,
            reranker_service=mock_reranker,
            enable_reranking=True,
        )

        with pytest.raises(DocumentNotFoundError):
            search_svc.search(query="query", document_id="doc_unknown")

        mock_reranker.rerank.assert_not_called()


# ==============================================================================
# Suite E: API Endpoints (POST /api/v1/search)
# ==============================================================================

class TestRerankingAPI:
    """Tests for the FastAPI endpoint POST /api/v1/search with reranking capabilities."""

    @pytest.fixture
    def configured_search_service(self) -> SearchService:
        mock_embed = MagicMock(spec=EmbeddingService)
        mock_embed.embed_query.return_value = np.zeros((384,), dtype=np.float32)

        mock_retrieval = MagicMock(spec=RetrievalService)
        index = MagicMock(spec=VectorIndex)
        index.total_vectors = 2
        index.metadata.document_ids = ["doc_123"]
        mock_retrieval.get_index.return_value = index
        mock_retrieval._indices = {"default": index, "doc_123": index}

        mock_retrieval.search_by_vector.return_value = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk_1",
                document_id="doc_123",
                page_number=10,
                text="Passage 1 content.",
                similarity_score=0.85,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="chunk_2",
                document_id="doc_123",
                page_number=12,
                text="Passage 2 content.",
                similarity_score=0.75,
            ),
        ]

        mock_model = MagicMock(spec=RerankerModel)
        # Rerank passage 2 higher
        mock_model.predict.return_value = np.array([2.5, 8.42], dtype=np.float32)
        reranker_svc = RerankerService(model=mock_model)

        return SearchService(
            embedding_service=mock_embed,
            retrieval_service=mock_retrieval,
            reranker_service=reranker_svc,
            enable_reranking=True,
        )

    @pytest.fixture(autouse=True)
    def setup_api_override(self, client: TestClient, configured_search_service: SearchService):
        client.app.dependency_overrides[get_search_service] = lambda: configured_search_service
        yield
        client.app.dependency_overrides.pop(get_search_service, None)

    def test_post_search_reranking_output_fields(self, client: TestClient):
        payload = {
            "query": "What is backpropagation?",
            "top_k": 2,
            "document_id": "doc_123",
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["query"] == "What is backpropagation?"
        assert data["reranking_applied"] is True
        assert len(data["results"]) == 2

        top = data["results"][0]
        assert top["rank"] == 1
        assert top["original_rank"] == 2
        assert top["chunk_id"] == "chunk_2"
        assert top["similarity_score"] == pytest.approx(0.75, abs=1e-4)
        assert top["reranker_score"] == pytest.approx(8.42, abs=1e-4)

    def test_post_search_with_explicit_candidate_k(self, client: TestClient):
        payload = {
            "query": "Valid query",
            "top_k": 2,
            "candidate_k": 10,
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_200_OK

    def test_post_search_candidate_k_less_than_top_k_returns_422(self, client: TestClient):
        payload = {
            "query": "Valid query",
            "top_k": 10,
            "candidate_k": 5,  # Invalid: candidate_k must be >= top_k
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_search_disabled_reranking(self, client: TestClient):
        payload = {
            "query": "Valid query",
            "top_k": 2,
            "enable_reranking": False,
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["reranking_applied"] is False
        top = data["results"][0]
        assert top["reranker_score"] is None
        assert top["rank"] == 1

    def test_post_search_reranker_model_load_failure_500(
        self,
        client: TestClient,
        configured_search_service: SearchService,
    ):
        configured_search_service.reranker_service.model.predict.side_effect = (
            RerankerModelLoadError("Weights corrupt")
        )
        response = client.post("/api/v1/search", json={"query": "fail query"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Traceback" not in response.text

    def test_post_search_reranker_execution_failure_500(
        self,
        client: TestClient,
        configured_search_service: SearchService,
    ):
        configured_search_service.reranker_service.model.predict.side_effect = (
            RerankingExecutionError("CUDA OOM")
        )
        response = client.post("/api/v1/search", json={"query": "fail query"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Traceback" not in response.text


# ==============================================================================
# Suite F: Real Cross-Encoder Integration Test
# ==============================================================================

class TestRealCrossEncoderIntegration:
    """Isolated integration test loading the real sentence-transformers CrossEncoder."""

    @pytest.fixture(scope="class")
    def real_cross_encoder_model(self) -> RerankerModel:
        """Class-scoped real cross-encoder model."""
        return RerankerModel.get_instance(
            model_name=DEFAULT_RERANKER_MODEL_NAME,
            device="auto",
            max_length=256,
        )

    def test_real_model_scoring_and_discrimination(self, real_cross_encoder_model: RerankerModel):
        service = RerankerService(model=real_cross_encoder_model, batch_size=2)

        candidates = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk_rome",
                document_id="doc_history",
                page_number=5,
                text="The Roman Empire expanded across Europe and the Mediterranean basin under Augustus.",
                similarity_score=0.85,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="chunk_nn",
                document_id="doc_ai",
                page_number=42,
                text=(
                    "Backpropagation computes the gradient of the loss function via the chain rule, "
                    "allowing gradient descent to update neural network weights efficiently."
                ),
                similarity_score=0.80,
            ),
        ]

        query = "What algorithm is used to compute gradients in neural networks?"
        reranked = service.rerank(query=query, candidates=candidates)

        assert len(reranked) == 2
        # Neural network passage must be ranked #1 by the cross-encoder!
        assert reranked[0].chunk_id == "chunk_nn"
        assert reranked[0].rank == 1
        assert reranked[0].original_rank == 2  # Originally #2 in FAISS
        # Cross-encoder score for relevant chunk must be much higher than irrelevant chunk
        assert reranked[0].reranker_score > reranked[1].reranker_score
