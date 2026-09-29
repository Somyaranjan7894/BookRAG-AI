"""Comprehensive unit, integration, and API tests for Phase 7 Extractive Question Answering.

Verifies RoBERTa SQuAD2 model wrapper caching, sliding-window tokenizer overflow handling,
span validation, SQuAD 2.0 no-answer thresholding, complete provenance preservation,
API validation, and regression prevention across the retrieval/reranking search pipeline.
"""

from typing import Any, List
from unittest.mock import MagicMock, patch
import pytest
import torch
from fastapi import status
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from app.api.v1.endpoints.qa import get_qa_service
from app.api.v1.endpoints.search import get_search_service
from app.main import app
from app.schemas.qa import QARequest, QAResponse
from app.schemas.search import SearchResponse, SearchResult
from app.services.qa.exceptions import (
    InvalidQAConfigError,
    InvalidQAQueryError,
    QAError,
    QAInferenceError,
    QAModelLoadError,
)
from app.services.qa.model import QAModel
from app.services.qa.service import QAService
from app.services.search.exceptions import DocumentNotFoundError, IndexNotInitializedError
from app.services.search.service import SearchService


# ==============================================================================
# Helper Mock Models
# ==============================================================================

class DeterministicMockQAModel:
    """Mock QAModel using the real fast tokenizer but engineered logits for deterministic testing."""

    def __init__(
        self,
        target_answer_text: str | None = None,
        null_logit: float = 0.0,
        span_logit: float = 5.0,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", use_fast=True)
        self.target_answer_text = target_answer_text
        self.null_logit = null_logit
        self.span_logit = span_logit
        self.target_device = "cpu"

    @property
    def device(self) -> str:
        return self.target_device

    def predict_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        start_logits = torch.full((batch_size, seq_len), -10.0)
        end_logits = torch.full((batch_size, seq_len), -10.0)

        # Set null score on token 0 (<s>)
        start_logits[:, 0] = self.null_logit
        end_logits[:, 0] = self.null_logit

        if self.target_answer_text is not None:
            clean_target = self.target_answer_text.strip().lower()
            for b in range(batch_size):
                row = input_ids[b].tolist()
                found = False
                for i in range(len(row)):
                    for j in range(i, min(i + 20, len(row))):
                        decoded = self.tokenizer.decode(row[i : j + 1]).strip().lower()
                        if decoded == clean_target:
                            start_logits[b, i] = self.span_logit
                            end_logits[b, j] = self.span_logit
                            found = True
                            break
                    if found:
                        break

        return start_logits, end_logits



# ==============================================================================
# Suite A: QA Model Wrapper Tests
# ==============================================================================

class TestQAModelWrapper:
    """Test QAModel wrapper lifecycle, device resolution, and caching."""

    def test_qa_model_initialization(self) -> None:
        """Verify QAModel initializes with fast tokenizer and evaluation mode."""
        model_wrapper = QAModel.get_instance()
        assert model_wrapper is not None
        assert model_wrapper.tokenizer is not None
        assert model_wrapper.tokenizer.is_fast
        assert model_wrapper.model is not None
        assert not model_wrapper.model.training

    def test_qa_model_singleton_caching(self) -> None:
        """Verify get_instance returns identical cached instance."""
        inst1 = QAModel.get_instance()
        inst2 = QAModel.get_instance()
        assert inst1 is inst2

    def test_qa_model_predict_logits_shape(self) -> None:
        """Verify predict_logits returns expected tensor shapes."""
        model_wrapper = QAModel.get_instance()
        tokenizer = model_wrapper.tokenizer
        inputs = tokenizer("Question?", "Context passage.", return_tensors="pt")
        starts, ends = model_wrapper.predict_logits(inputs["input_ids"], inputs["attention_mask"])
        assert starts.shape == inputs["input_ids"].shape
        assert ends.shape == inputs["input_ids"].shape

    def test_qa_model_load_failure_raises_domain_error(self) -> None:
        """Verify model load failures raise QAModelLoadError."""
        with patch("transformers.AutoModelForQuestionAnswering.from_pretrained", side_effect=RuntimeError("Disk read failure")):
            with pytest.raises(QAModelLoadError) as exc_info:
                QAModel(model_name="nonexistent/fake-model")
            assert "Could not load extractive QA model" in str(exc_info.value)


# ==============================================================================
# Suite B: Extractive QA Service Unit Tests
# ==============================================================================

class TestQAServiceExtractiveUnit:
    """Test QAService extractive span parsing, length constraints, and provenance."""

    def test_basic_extractive_answer(self) -> None:
        """Verify basic answer span extraction from single evidence chunk."""
        mock_model = DeterministicMockQAModel(target_answer_text="1998", null_logit=0.0, span_logit=6.0)
        svc = QAService(model=mock_model)

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="doc1_p1_c1",
                document_id="doc1",
                page_number=1,
                chunk_index=0,
                text="Gradient-based learning was applied to document recognition in 1998 by LeCun et al.",
                similarity_score=0.85,
                reranker_score=7.2,
            )
        ]

        resp = svc.answer_question("When was gradient learning applied?", evidence)
        assert resp.answerable is True
        assert resp.answer == "1998"
        assert resp.document_id == "doc1"
        assert resp.chunk_id == "doc1_p1_c1"
        assert resp.page_number == 1
        assert resp.chunk_index == 0
        assert resp.evidence_rank == 1
        assert resp.similarity_score == 0.85
        assert resp.reranker_score == 7.2
        assert resp.total_evidence_evaluated == 1

    def test_correct_answer_span_extraction_and_offsets(self) -> None:
        """Verify answer_start and answer_end match exact character slice in source text."""
        mock_model = DeterministicMockQAModel(target_answer_text="backpropagation", null_logit=0.0, span_logit=5.0)
        svc = QAService(model=mock_model)

        text = "Neural networks are trained using backpropagation and stochastic gradient descent."
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c_01",
                document_id="d_01",
                page_number=10,
                chunk_index=1,
                text=text,
                similarity_score=0.9,
                reranker_score=6.0,
            )
        ]

        resp = svc.answer_question("How are networks trained?", evidence)
        assert resp.answerable is True
        assert resp.answer == "backpropagation"
        assert resp.answer_start is not None
        assert resp.answer_end is not None
        assert text[resp.answer_start : resp.answer_end] == "backpropagation"

    def test_answer_text_matches_source_text(self) -> None:
        """Verify extracted answer is an exact substring of the source text (no hallucination)."""
        mock_model = DeterministicMockQAModel(target_answer_text="Adam optimizer", null_logit=0.0, span_logit=5.0)
        svc = QAService(model=mock_model)

        text = "The Adam optimizer computes adaptive learning rates for each parameter."
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c_02",
                document_id="d_02",
                page_number=5,
                chunk_index=0,
                text=text,
                similarity_score=0.75,
            )
        ]

        resp = svc.answer_question("What computes adaptive learning rates?", evidence)
        assert resp.answerable is True
        assert resp.answer in text

    def test_maximum_answer_length_constraint(self) -> None:
        """Verify spans exceeding max_answer_length are rejected."""
        # Long span with 10 tokens
        mock_model = DeterministicMockQAModel(
            target_answer_text="this is a very very long answer span that exceeds token limit",
            null_logit=0.0,
            span_logit=8.0,
        )
        # Configure max_answer_length to 3 tokens
        svc = QAService(model=mock_model, max_answer_length=3)

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c_03",
                document_id="d_03",
                page_number=2,
                chunk_index=0,
                text="We found that this is a very very long answer span that exceeds token limit in practice.",
                similarity_score=0.8,
            )
        ]

        resp = svc.answer_question("What was found?", evidence)
        # The 10-token span is rejected; should fall back to unanswerable
        assert resp.answerable is False

    def test_multiple_evidence_chunks_selects_best(self) -> None:
        """Verify the service selects the highest scoring candidate across multiple chunks."""
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", use_fast=True)

        class MultiChunkMockModel:
            def __init__(self) -> None:
                self.tokenizer = tokenizer
                self.target_device = "cpu"

            @property
            def device(self) -> str:
                return self.target_device

            def predict_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                starts = torch.full(input_ids.shape, -10.0)
                ends = torch.full(input_ids.shape, -10.0)
                starts[:, 0] = 0.0
                ends[:, 0] = 0.0

                for b in range(input_ids.shape[0]):
                    row = input_ids[b].tolist()
                    for i in range(len(row)):
                        for j in range(i, min(i + 10, len(row))):
                            decoded = tokenizer.decode(row[i : j + 1]).strip()
                            if decoded == "WeakAnswer":
                                starts[b, i] = 3.0
                                ends[b, j] = 3.0
                            elif decoded == "StrongAnswer":
                                starts[b, i] = 8.0
                                ends[b, j] = 8.0
                return starts, ends


        svc = QAService(model=MultiChunkMockModel())

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="chunk_1",
                document_id="doc_1",
                page_number=1,
                chunk_index=0,
                text="Here is a WeakAnswer mentioned in the text.",
                similarity_score=0.9,
                reranker_score=5.0,
            ),
            SearchResult(
                rank=2,
                chunk_id="chunk_2",
                document_id="doc_1",
                page_number=2,
                chunk_index=1,
                text="Here is a StrongAnswer with much clearer evidence.",
                similarity_score=0.7,
                reranker_score=8.5,
            ),
        ]

        resp = svc.answer_question("Find the answer", evidence)
        assert resp.answerable is True
        assert resp.answer == "StrongAnswer"
        assert resp.chunk_id == "chunk_2"
        assert resp.page_number == 2
        assert resp.evidence_rank == 2

    def test_no_answer_when_score_below_threshold(self) -> None:
        """Verify unanswerable response when best span score is below no-answer threshold."""
        # span_score = 1.0 + 1.0 = 2.0; null_score = 2.0 + 2.0 = 4.0; diff = -2.0
        mock_model = DeterministicMockQAModel(target_answer_text="maybe", null_logit=2.0, span_logit=1.0)
        svc = QAService(model=mock_model, no_answer_threshold=0.0)

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c_04",
                document_id="d_04",
                page_number=1,
                chunk_index=0,
                text="The result was maybe true.",
                similarity_score=0.5,
            )
        ]

        resp = svc.answer_question("Was it true?", evidence)
        assert resp.answerable is False
        assert resp.answer is None
        assert resp.qa_score is not None
        assert resp.no_answer_score is not None

    def test_empty_evidence_returns_unanswerable(self) -> None:
        """Verify empty evidence list immediately returns unanswerable without error."""
        svc = QAService()
        resp = svc.answer_question("Any question?", [])
        assert resp.answerable is False
        assert resp.answer is None
        assert resp.total_evidence_evaluated == 0

    def test_invalid_query_raises_error(self) -> None:
        """Verify empty or whitespace question raises InvalidQAQueryError."""
        svc = QAService()
        with pytest.raises(InvalidQAQueryError):
            svc.answer_question("", [])
        with pytest.raises(InvalidQAQueryError):
            svc.answer_question("   ", [])

    def test_invalid_configuration_raises_error(self) -> None:
        """Verify invalid doc_stride >= max_length raises InvalidQAConfigError."""
        with pytest.raises(InvalidQAConfigError):
            QAService(max_length=128, doc_stride=128)
        with pytest.raises(InvalidQAConfigError):
            QAService(max_length=0)

    def test_model_inference_failure_raises_qa_inference_error(self) -> None:
        """Verify inference exceptions are converted to QAInferenceError."""
        mock_model = MagicMock(spec=QAModel)
        mock_model.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", use_fast=True)
        mock_model.predict_logits.side_effect = RuntimeError("CUDA out of memory")

        svc = QAService(model=mock_model)
        evidence = [
            SearchResult(
                rank=1,
                chunk_id="c_err",
                document_id="d_err",
                page_number=1,
                chunk_index=0,
                text="Some passage text.",
                similarity_score=0.8,
            )
        ]

        with pytest.raises(QAInferenceError) as exc_info:
            svc.answer_question("What is this?", evidence)
        assert "CUDA out of memory" in str(exc_info.value)


# ==============================================================================
# Suite C: Sliding Window and Long Context Handling
# ==============================================================================

class TestSlidingWindowAndLongContext:
    """Test tokenizer sliding-window overflow and answers positioned in overflowing windows."""

    def test_answer_in_overflowing_sliding_window(self) -> None:
        """Verify an answer located deep in a long passage is extracted via sliding window."""
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", use_fast=True)

        class SlidingWindowMockModel:
            def __init__(self) -> None:
                self.tokenizer = tokenizer
                self.target_device = "cpu"

            @property
            def device(self) -> str:
                return self.target_device

            def predict_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                starts = torch.full(input_ids.shape, -10.0)
                ends = torch.full(input_ids.shape, -10.0)
                starts[:, 0] = 0.0
                ends[:, 0] = 0.0

                for b in range(input_ids.shape[0]):
                    row = input_ids[b].tolist()
                    for i in range(len(row)):
                        for j in range(i, min(i + 10, len(row))):
                            decoded = tokenizer.decode(row[i : j + 1]).strip()
                            if decoded == "SecretTarget":
                                starts[b, i] = 8.0
                                ends[b, j] = 8.0
                return starts, ends


        # Configure small max_length=64 and doc_stride=16 to trigger sliding windows on small text
        svc = QAService(model=SlidingWindowMockModel(), max_length=64, doc_stride=16)

        # Create a repetitive filler context followed by the target answer
        filler = "Deep learning uses neural networks with multiple layers for hierarchical feature extraction. " * 4
        full_text = filler + "The hidden variable was SecretTarget in all experiments."


        evidence = [
            SearchResult(
                rank=1,
                chunk_id="long_c1",
                document_id="doc_long",
                page_number=42,
                chunk_index=3,
                text=full_text,
                similarity_score=0.88,
            )
        ]

        resp = svc.answer_question("What was the hidden variable?", evidence)
        assert resp.answerable is True
        assert resp.answer == "SecretTarget"
        assert resp.answer_start is not None
        assert resp.answer_end is not None
        assert full_text[resp.answer_start : resp.answer_end] == "SecretTarget"


# ==============================================================================
# Suite D: API Endpoint Tests (POST /api/v1/qa)
# ==============================================================================

class TestQAAPI:
    """Test FastAPI question answering endpoint contracts, validation, and error mappings."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_post_qa_successful_answer(self, client: TestClient) -> None:
        """Verify successful POST /api/v1/qa with full search orchestration."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="When was backpropagation popularized?",
            results=[
                SearchResult(
                    rank=1,
                    original_rank=1,
                    chunk_id="chunk_42",
                    document_id="deep_learning_book",
                    page_number=42,
                    chunk_index=3,
                    text="Backpropagation was popularized in 1986 by Rumelhart, Hinton, and Williams.",
                    similarity_score=0.91,
                    reranker_score=8.5,
                )
            ],
            total_results=1,
            document_id="deep_learning_book",
            reranking_applied=True,
        )

        mock_qa = MagicMock(spec=QAService)
        mock_qa.answer_question.return_value = QAResponse(
            query="When was backpropagation popularized?",
            answer="1986",
            answerable=True,
            qa_score=8.42,
            no_answer_score=-2.15,
            document_id="deep_learning_book",
            chunk_id="chunk_42",
            page_number=42,
            chunk_index=3,
            answer_start=34,
            answer_end=38,
            source_text="Backpropagation was popularized in 1986 by Rumelhart, Hinton, and Williams.",
            evidence_rank=1,
            similarity_score=0.91,
            reranker_score=8.5,
            total_evidence_evaluated=1,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        app.dependency_overrides[get_qa_service] = lambda: mock_qa

        try:
            payload = {
                "query": "When was backpropagation popularized?",
                "document_id": "deep_learning_book",
                "top_k": 5,
            }
            res = client.post("/api/v1/qa", json=payload)
            assert res.status_code == status.HTTP_200_OK
            data = res.json()
            assert data["answerable"] is True
            assert data["answer"] == "1986"
            assert data["document_id"] == "deep_learning_book"
            assert data["page_number"] == 42
            assert data["similarity_score"] == 0.91
            assert data["reranker_score"] == 8.5
            assert data["qa_score"] == 8.42
        finally:
            app.dependency_overrides.clear()

    def test_post_qa_no_answer_response(self, client: TestClient) -> None:
        """Verify structured no-answer response when question is unanswerable."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="Who landed on Mars?",
            results=[
                SearchResult(
                    rank=1,
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    page_number=1,
                    chunk_index=0,
                    text="Nobody has landed on Mars yet.",
                    similarity_score=0.5,
                )
            ],
            total_results=1,
            reranking_applied=False,
        )

        mock_qa = MagicMock(spec=QAService)
        mock_qa.answer_question.return_value = QAResponse(
            query="Who landed on Mars?",
            answer=None,
            answerable=False,
            qa_score=None,
            no_answer_score=3.2,
            total_evidence_evaluated=1,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        app.dependency_overrides[get_qa_service] = lambda: mock_qa

        try:
            payload = {"query": "Who landed on Mars?"}
            res = client.post("/api/v1/qa", json=payload)
            assert res.status_code == status.HTTP_200_OK
            data = res.json()
            assert data["answerable"] is False
            assert data["answer"] is None
        finally:
            app.dependency_overrides.clear()

    def test_post_qa_validation_empty_query(self, client: TestClient) -> None:
        """Verify 422 for empty query."""
        res = client.post("/api/v1/qa", json={"query": ""})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_qa_validation_whitespace_query(self, client: TestClient) -> None:
        """Verify 422 for whitespace-only query."""
        res = client.post("/api/v1/qa", json={"query": "   \n\t  "})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_qa_validation_invalid_top_k(self, client: TestClient) -> None:
        """Verify 422 for non-positive or excessive top_k."""
        res = client.post("/api/v1/qa", json={"query": "What?", "top_k": 0})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        res = client.post("/api/v1/qa", json={"query": "What?", "top_k": 1000})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_qa_validation_candidate_k_less_than_top_k(self, client: TestClient) -> None:
        """Verify 422 when candidate_k < top_k."""
        res = client.post("/api/v1/qa", json={"query": "What?", "top_k": 10, "candidate_k": 5})
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_post_qa_document_not_found_returns_404(self, client: TestClient) -> None:
        """Verify 404 when requested document is not found in the search catalog."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.side_effect = DocumentNotFoundError("Document 'unknown_doc' was not found.")

        app.dependency_overrides[get_search_service] = lambda: mock_search
        try:
            res = client.post("/api/v1/qa", json={"query": "What is this?", "document_id": "unknown_doc"})
            assert res.status_code == status.HTTP_404_NOT_FOUND
            assert "unknown_doc" in res.json()["error"]["message"]
        finally:
            app.dependency_overrides.clear()

    def test_search_endpoint_regression(self, client: TestClient) -> None:
        """Verify existing /api/v1/search endpoint remains functional and unchanged."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test query",
            results=[],
            total_results=0,
            reranking_applied=False,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        try:
            res = client.post("/api/v1/search", json={"query": "test query"})
            assert res.status_code == status.HTTP_200_OK
            assert res.json()["query"] == "test query"
        finally:
            app.dependency_overrides.clear()


# ==============================================================================
# Suite E: Real Model Integration Test
# ==============================================================================

class TestRealQAModelIntegration:
    """Deterministic integration tests loading real deepset/roberta-base-squad2 weights."""

    def test_real_roberta_squad2_forward_pass(self) -> None:
        """Verify real RoBERTa SQuAD2 extracts answer span and identifies unanswerable question."""
        real_model = QAModel.get_instance()
        svc = QAService(model=real_model)

        context = (
            "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio were awarded the ACM A.M. Turing Award "
            "in 2018 for conceptual and engineering breakthroughs that made deep neural networks a critical "
            "component of computing."
        )

        evidence = [
            SearchResult(
                rank=1,
                chunk_id="chunk_turing",
                document_id="ai_history",
                page_number=1,
                chunk_index=0,
                text=context,
                similarity_score=0.92,
                reranker_score=8.1,
            )
        ]

        # 1. Answerable question
        resp1 = svc.answer_question("In what year was the Turing Award given to Geoffrey Hinton?", evidence)
        assert resp1.answerable is True
        assert "2018" in (resp1.answer or "")
        assert resp1.page_number == 1
        assert resp1.document_id == "ai_history"

        # 2. Unanswerable question (SQuAD 2.0 no-answer)
        resp2 = svc.answer_question("What is the speed of sound on Jupiter?", evidence)
        assert resp2.answerable is False
        assert resp2.answer is None
