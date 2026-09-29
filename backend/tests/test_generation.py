"""Comprehensive unit, integration, and API tests for Phase 8 Abstractive Question Answering (Generation).

Verifies FLAN-T5 model wrapper caching, EvidenceBuilder context budgeting, prompt grounding,
provenance preservation, deterministic beam search, empty-evidence guardrails,
API endpoint contracts, and regression prevention across the retrieval, reranking, and QA pipelines.
"""

from typing import Any, List
from unittest.mock import MagicMock, patch
import pytest
import torch
from fastapi import status
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from app.api.v1.endpoints.generation import get_generation_service
from app.api.v1.endpoints.search import get_search_service
from app.main import app
from app.schemas.generation import GenerationEvidenceItem, GenerationRequest, GenerationResponse
from app.schemas.qa import QAResponse
from app.schemas.search import SearchResponse, SearchResult
from app.services.generation.evidence import BuiltPrompt, EvidenceBuilder, PROMPT_TEMPLATE
from app.services.generation.exceptions import (
    ContextBudgetExceededError,
    GenerationError,
    GenerationInferenceError,
    GenerationModelLoadError,
    InvalidGenerationConfigError,
    InvalidGenerationQueryError,
)
from app.services.generation.model import GenerationModel
from app.services.generation.service import GenerationService
from app.services.search.exceptions import DocumentNotFoundError
from app.services.search.service import SearchService


# ==============================================================================
# Helper Mock Models
# ==============================================================================

class MockGenerationModel:
    """Fast deterministic mock for GenerationModel avoiding full neural generation in unit tests."""

    def __init__(self, response_text: str = "Backpropagation computes gradient vectors recursively.") -> None:
        self.response_text = response_text
        self.model_name = "google/flan-t5-base"
        self.target_device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.last_prompt: str | None = None
        self.last_kwargs: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self.target_device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        self.last_prompt = prompt
        self.last_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        return self.response_text


# ==============================================================================
# Suite A: Generation Model Wrapper Tests
# ==============================================================================

class TestGenerationModelWrapper:
    """Test GenerationModel wrapper lifecycle, device resolution, and caching."""

    def test_model_wrapper_initialization(self) -> None:
        """Verify GenerationModel initializes with model, tokenizer, and evaluation mode."""
        model_wrapper = GenerationModel.get_instance()
        assert model_wrapper is not None
        assert model_wrapper.tokenizer is not None
        assert model_wrapper.model is not None
        assert not model_wrapper.model.training

    def test_model_wrapper_singleton_caching(self) -> None:
        """Verify get_instance returns identical cached instance."""
        inst1 = GenerationModel.get_instance()
        inst2 = GenerationModel.get_instance()
        assert inst1 is inst2

    def test_model_load_failure_raises_domain_error(self) -> None:
        """Verify model load failures raise GenerationModelLoadError."""
        with patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", side_effect=RuntimeError("Corrupt weights")):
            with pytest.raises(GenerationModelLoadError) as exc_info:
                GenerationModel(model_name="nonexistent/fake-t5")
            assert "Could not load generation model" in str(exc_info.value)


# ==============================================================================
# Suite B: EvidenceBuilder and Context Budgeting Tests
# ==============================================================================

class TestEvidenceBuilder:
    """Test grounded prompt assembly, page boundary formatting, and token budgeting."""

    def test_evidence_formatting_with_page_boundaries(self) -> None:
        """Verify evidence blocks include clear [Page X] markers."""
        builder = EvidenceBuilder()
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="chunk_1",
                document_id="dl_book",
                page_number=12,
                chunk_index=0,
                text="Gradient descent optimizes parameters.",
                similarity_score=0.88,
                reranker_score=7.5,
            )
        ]

        built = builder.build_prompt(question="How are parameters optimized?", evidence=evidence)
        assert "[Page 12] Gradient descent optimizes parameters." in built.prompt
        assert "How are parameters optimized?" in built.prompt
        assert len(built.included_evidence) == 1
        assert built.included_evidence[0].page_number == 12
        assert built.included_evidence[0].chunk_id == "chunk_1"

    def test_evidence_context_ordering(self) -> None:
        """Verify higher-ranked evidence appears first in prompt context."""
        builder = EvidenceBuilder()
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="first",
                document_id="doc1",
                page_number=1,
                chunk_index=0,
                text="First priority evidence.",
                similarity_score=0.9,
            ),
            SearchResult(
                rank=2,
                chunk_id="second",
                document_id="doc1",
                page_number=2,
                chunk_index=1,
                text="Second priority evidence.",
                similarity_score=0.8,
            ),
        ]

        built = builder.build_prompt(question="Order test?", evidence=evidence)
        first_pos = built.prompt.find("[Page 1] First priority evidence.")
        second_pos = built.prompt.find("[Page 2] Second priority evidence.")
        assert first_pos != -1
        assert second_pos != -1
        assert first_pos < second_pos

    def test_context_budget_enforcement(self) -> None:
        """Verify evidence is bounded by max_input_tokens budget."""
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        # Extremely small token budget to allow only the first chunk
        builder = EvidenceBuilder(tokenizer=tokenizer, max_input_tokens=35)


        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c1",
                document_id="doc1",
                page_number=1,
                chunk_index=0,
                text="First chunk fits.",
                similarity_score=0.9,
            ),
            SearchResult(
                rank=2,
                chunk_id="c2",
                document_id="doc1",
                page_number=2,
                chunk_index=1,
                text="Second chunk that would exceed the strict budget limit.",
                similarity_score=0.8,
            ),
        ]

        built = builder.build_prompt(question="What fits?", evidence=evidence)
        assert len(built.included_evidence) == 1
        assert built.included_evidence[0].chunk_id == "c1"
        assert "First chunk fits." in built.prompt
        assert "Second chunk that would exceed" not in built.prompt

    def test_long_single_evidence_truncation_within_budget(self) -> None:
        """Verify a single long evidence item exceeding budget is safely truncated with provenance preserved."""
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        builder = EvidenceBuilder(tokenizer=tokenizer, max_input_tokens=60)

        huge_text = "Neural networks are multi-layered computation graphs with activation functions. " * 10
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="huge_1",
                document_id="doc_huge",
                page_number=42,
                chunk_index=0,
                text=huge_text,
                similarity_score=0.95,
            )
        ]

        built = builder.build_prompt(question="Describe networks?", evidence=evidence)
        assert len(built.included_evidence) == 1
        assert built.included_evidence[0].chunk_id == "huge_1"
        assert built.included_evidence[0].page_number == 42
        assert "..." in built.prompt

    def test_question_exceeding_budget_raises_error(self) -> None:
        """Verify an excessively long question exceeding max_input_tokens raises ContextBudgetExceededError."""
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        builder = EvidenceBuilder(tokenizer=tokenizer, max_input_tokens=30)
        long_query = "What is the detailed mechanism of " + ("convolution " * 25) + "?"

        with pytest.raises(ContextBudgetExceededError):
            builder.build_prompt(question=long_query, evidence=[])


# ==============================================================================
# Suite C: Generation Service Unit Tests
# ==============================================================================

class TestGenerationService:
    """Test GenerationService business logic, empty-evidence handling, and provenance."""

    def test_basic_generation_with_mock_model(self) -> None:
        """Verify normal abstractive answer generation with evidence provenance."""
        mock_model = MockGenerationModel(response_text="Backpropagation was introduced in 1986.")
        svc = GenerationService(model=mock_model)

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="chunk_42",
                document_id="ai_book",
                page_number=42,
                chunk_index=3,
                text="In 1986, Rumelhart, Hinton, and Williams introduced backpropagation.",
                similarity_score=0.92,
                reranker_score=8.4,
            )
        ]

        resp = svc.generate_answer(query="When was backpropagation introduced?", evidence=evidence)
        assert resp.answerable is True
        assert resp.answer == "Backpropagation was introduced in 1986."
        assert resp.model_name == "google/flan-t5-base"
        assert resp.evidence_count == 1
        assert len(resp.evidence) == 1
        assert resp.evidence[0].chunk_id == "chunk_42"
        assert resp.evidence[0].page_number == 42
        assert resp.evidence[0].similarity_score == 0.92
        assert resp.evidence[0].reranker_score == 8.4

    def test_empty_evidence_returns_unanswerable_without_calling_model(self) -> None:
        """Verify empty evidence immediately returns answerable=False without calling FLAN-T5."""
        mock_model = MockGenerationModel()
        svc = GenerationService(model=mock_model)

        resp = svc.generate_answer(query="Who built the pyramids on Neptune?", evidence=[])
        assert resp.answerable is False
        assert resp.answer is None
        assert resp.evidence == []
        assert resp.evidence_count == 0
        # Ensure model.generate was NEVER called
        assert mock_model.last_prompt is None

    def test_invalid_query_raises_error(self) -> None:
        """Verify blank or whitespace queries raise InvalidGenerationQueryError."""
        svc = GenerationService()
        with pytest.raises(InvalidGenerationQueryError):
            svc.generate_answer(query="", evidence=[])
        with pytest.raises(InvalidGenerationQueryError):
            svc.generate_answer(query="   \n\t  ", evidence=[])

    def test_invalid_configuration_raises_error(self) -> None:
        """Verify invalid configuration bounds raise InvalidGenerationConfigError."""
        with pytest.raises(InvalidGenerationConfigError):
            GenerationService(max_input_tokens=0)
        with pytest.raises(InvalidGenerationConfigError):
            GenerationService(max_new_tokens=0)
        with pytest.raises(InvalidGenerationConfigError):
            GenerationService(num_beams=0)
        with pytest.raises(InvalidGenerationConfigError):
            GenerationService(temperature=-1.0)

    def test_deterministic_generation_kwargs_passed_to_model(self) -> None:
        """Verify do_sample=False, num_beams=4, max_new_tokens are passed to model.generate."""
        mock_model = MockGenerationModel()
        svc = GenerationService(
            model=mock_model,
            max_new_tokens=64,
            num_beams=5,
            do_sample=False,
        )

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c1",
                document_id="d1",
                page_number=1,
                chunk_index=0,
                text="Some text.",
                similarity_score=0.8,
            )
        ]

        svc.generate_answer(query="Question?", evidence=evidence)
        assert mock_model.last_kwargs["do_sample"] is False
        assert mock_model.last_kwargs["num_beams"] == 5
        assert mock_model.last_kwargs["max_new_tokens"] == 64

    def test_model_inference_failure_raises_generation_inference_error(self) -> None:
        """Verify forward pass exceptions are converted to GenerationInferenceError."""
        mock_model = MagicMock(spec=GenerationModel)
        mock_model.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        mock_model.generate.side_effect = RuntimeError("GPU out of memory")

        svc = GenerationService(model=mock_model)
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c1",
                document_id="d1",
                page_number=1,
                chunk_index=0,
                text="Passage text.",
                similarity_score=0.8,
            )
        ]

        with pytest.raises(GenerationInferenceError) as exc_info:
            svc.generate_answer(query="What is this?", evidence=evidence)
        assert "GPU out of memory" in str(exc_info.value)


# ==============================================================================
# Suite D: API Endpoint Tests (POST /api/v1/answer)
# ==============================================================================

class TestGenerationAPI:
    """Test FastAPI abstractive answer generation endpoint contracts and validation."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_post_answer_successful_generation(self, client: TestClient) -> None:
        """Verify successful POST /api/v1/answer with full search and generation orchestration."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="Why are convolutional layers used?",
            results=[
                SearchResult(
                    rank=1,
                    original_rank=1,
                    chunk_id="conv_p10_c1",
                    document_id="vision_book",
                    page_number=10,
                    chunk_index=0,
                    text="Convolutional layers preserve spatial hierarchies and share weights across inputs.",
                    similarity_score=0.91,
                    reranker_score=8.2,
                )
            ],
            total_results=1,
            document_id="vision_book",
            reranking_applied=True,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="Why are convolutional layers used?",
            answer="Convolutional layers preserve spatial hierarchies through parameter sharing.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=[
                GenerationEvidenceItem(
                    rank=1,
                    chunk_id="conv_p10_c1",
                    document_id="vision_book",
                    page_number=10,
                    chunk_index=0,
                    source_text="Convolutional layers preserve spatial hierarchies and share weights across inputs.",
                    similarity_score=0.91,
                    reranker_score=8.2,
                )
            ],
            evidence_count=1,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        app.dependency_overrides[get_generation_service] = lambda: mock_gen

        try:
            payload = {
                "query": "Why are convolutional layers used?",
                "document_id": "vision_book",
                "top_k": 5,
            }
            res = client.post("/api/v1/answer", json=payload)
            assert res.status_code == status.HTTP_200_OK
            data = res.json()
            assert data["answerable"] is True
            assert "spatial hierarchies" in data["answer"]
            assert data["model_name"] == "google/flan-t5-base"
            assert data["evidence_count"] == 1
            assert data["evidence"][0]["chunk_id"] == "conv_p10_c1"
            assert data["evidence"][0]["page_number"] == 10
            assert data["evidence"][0]["similarity_score"] == 0.91
            assert data["evidence"][0]["reranker_score"] == 8.2
        finally:
            app.dependency_overrides.clear()

    def test_post_answer_empty_evidence_response(self, client: TestClient) -> None:
        """Verify structured unanswerable response when search returns no candidates."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="Who landed on Mars in 1800?",
            results=[],
            total_results=0,
            reranking_applied=False,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="Who landed on Mars in 1800?",
            answer=None,
            answerable=False,
            model_name="google/flan-t5-base",
            evidence=[],
            evidence_count=0,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        app.dependency_overrides[get_generation_service] = lambda: mock_gen

        try:
            res = client.post("/api/v1/answer", json={"query": "Who landed on Mars in 1800?"})
            assert res.status_code == status.HTTP_200_OK
            data = res.json()
            assert data["answerable"] is False
            assert data["answer"] is None
            assert data["evidence"] == []
        finally:
            app.dependency_overrides.clear()

    def test_post_answer_validation_empty_query(self, client: TestClient) -> None:
        """Verify 422 for empty query."""
        res = client.post("/api/v1/answer", json={"query": ""})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_answer_validation_whitespace_query(self, client: TestClient) -> None:
        """Verify 422 for whitespace query."""
        res = client.post("/api/v1/answer", json={"query": "   \n\t  "})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_answer_validation_invalid_top_k(self, client: TestClient) -> None:
        """Verify 422 for non-positive or excessive top_k."""
        res = client.post("/api/v1/answer", json={"query": "What?", "top_k": 0})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        res = client.post("/api/v1/answer", json={"query": "What?", "top_k": 1000})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_answer_validation_candidate_k_less_than_top_k(self, client: TestClient) -> None:
        """Verify 422 when candidate_k < top_k."""
        res = client.post("/api/v1/answer", json={"query": "What?", "top_k": 10, "candidate_k": 5})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_answer_document_not_found_returns_404(self, client: TestClient) -> None:
        """Verify 404 when requested document is not found."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.side_effect = DocumentNotFoundError("Document 'missing_doc' was not found.")

        app.dependency_overrides[get_search_service] = lambda: mock_search
        try:
            res = client.post("/api/v1/answer", json={"query": "What is this?", "document_id": "missing_doc"})
            assert res.status_code == status.HTTP_404_NOT_FOUND
            assert "missing_doc" in res.json()["error"]["message"]
        finally:
            app.dependency_overrides.clear()

    def test_search_and_qa_endpoints_regression(self, client: TestClient) -> None:
        """Verify existing /search and /qa endpoints continue working without regression."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="regression test",
            results=[],
            total_results=0,
            reranking_applied=False,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        try:
            # Test /search
            res_search = client.post("/api/v1/search", json={"query": "regression test"})
            assert res_search.status_code == status.HTTP_200_OK

            # Test /qa
            with patch("app.api.v1.endpoints.qa.get_qa_service") as mock_get_qa:
                mock_qa_svc = MagicMock()
                mock_qa_svc.answer_question.return_value = QAResponse(
                    query="regression test",
                    answerable=False,
                )
                mock_get_qa.return_value = mock_qa_svc
                res_qa = client.post("/api/v1/qa", json={"query": "regression test"})
                assert res_qa.status_code == status.HTTP_200_OK
        finally:
            app.dependency_overrides.clear()


# ==============================================================================
# Suite E: Real Model Integration Test
# ==============================================================================

class TestRealFLANT5Integration:
    """Deterministic integration test loading real google/flan-t5-base weights."""

    def test_real_flan_t5_forward_pass(self) -> None:
        """Verify real FLAN-T5 generates an abstractive answer strictly from provided evidence."""
        real_model = GenerationModel.get_instance()
        svc = GenerationService(model=real_model)

        context = (
            "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio were awarded the ACM A.M. Turing Award "
            "in 2018 for conceptual and engineering breakthroughs that made deep neural networks a critical "
            "component of computing."
        )

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="turing_chunk_1",
                document_id="ai_history",
                page_number=1,
                chunk_index=0,
                text=context,
                similarity_score=0.92,
                reranker_score=8.4,
            )
        ]

        resp = svc.generate_answer(
            query="In what year was the Turing Award given to Geoffrey Hinton and colleagues?",
            evidence=evidence,
        )
        assert resp.answerable is True
        assert resp.answer is not None
        assert "2018" in resp.answer
        assert resp.evidence_count == 1
        assert resp.evidence[0].chunk_id == "turing_chunk_1"
        assert resp.evidence[0].page_number == 1
