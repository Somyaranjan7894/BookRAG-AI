"""Phase 22 Comprehensive Test Suite for Controlled Question Generation and QA Expansion in BookRAG AI.

Validates:
1. Generation:
   - Valid evidence produces question
   - Multiple analytical question types (direct_fact, definition, explanation, comparison, numerical_fact, reasoning, multi_page_synthesis)
   - Configurable count respected
   - Diversity balancing across candidate types
2. Validation:
   - Answerable question accepted
   - Unanswerable question rejected with diagnostic reason
   - Incomplete / dangling questions rejected
   - Vague questions rejected
   - Circular / tautological questions rejected
   - Unsupported assumptions rejected
3. Duplicates:
   - Exact duplicates rejected and recorded in diagnostics
   - Normalized duplicates rejected
   - Near-duplicate questions (Jaccard similarity >= 0.75) rejected
4. Provenance:
   - Document ID preserved
   - Page number preserved
   - Chunk ID preserved
   - Source evidence preserved
   - Multi-chunk & multi-page lists preserved (chunk_ids, page_numbers)
5. Safety:
   - External knowledge entities rejected
   - Plausible but ungrounded claims rejected
6. API Endpoint:
   - 200 OK on valid document with validated questions and rejection summary
   - 404 on nonexistent document
   - 422 on invalid parameters (count <= 0)
   - Diagnostic reporting with include_rejected=True
"""

from typing import Any, List, Optional
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.v1.dependencies import get_question_generation_service
from app.main import app
from app.schemas.chunk import Chunk
from app.schemas.qa import QAResponse
from app.schemas.question_generation import (
    AnswerCandidate,
    GeneratedQuestion,
    QuestionCandidate,
    QuestionDifficulty,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionType,
    RejectedCandidateInfo,
)
from app.services.qa.service import QAService
from app.services.question_generation.answers import AnswerCandidateExtractor
from app.services.question_generation.deduplication import QuestionDeduplicator
from app.services.question_generation.service import QuestionGenerationService
from app.services.question_generation.validator import (
    QuestionValidator,
    classify_question_difficulty,
    classify_question_type,
    matches_expected_answer,
    normalize_answer_text,
)
from app.services.search.exceptions import DocumentNotFoundError


# ---------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------

class MockPhase22Model:
    """Deterministic question generation model mock."""

    def __init__(self, mapping: Optional[dict[str, str]] = None) -> None:
        self.mapping = mapping or {}
        self.generated_pairs: List[tuple[str, str]] = []

    def generate_questions_batch(self, pairs: List[tuple[str, str]], **kwargs: Any) -> List[str]:
        results = []
        for ans, ctx in pairs:
            self.generated_pairs.append((ans, ctx))
            if ans in self.mapping:
                results.append(self.mapping[ans])
            else:
                results.append(f"What is {ans}?")
        return results


class MockPhase22QA:
    """Mock QA service returning controllable answerability."""

    def __init__(self, answer_map: Optional[dict[str, Optional[str]]] = None, default_score: float = 0.95) -> None:
        self.answer_map = answer_map or {}
        self.default_score = default_score
        self.recorded_queries: List[str] = []

    def answer_question(self, query: str, evidence: Any, **kwargs: Any) -> QAResponse:
        self.recorded_queries.append(query)
        ans = self.answer_map.get(query, None)
        if ans is None and query in self.answer_map:
            # Explicitly unanswerable
            return QAResponse(
                query=query,
                answer=None,
                answerable=False,
                qa_score=-3.0,
                no_answer_score=2.5,
                document_id="doc_p22",
                chunk_id="chunk_p22",
                page_number=1,
            )
        # Default behavior: extract query key or first word
        chosen = ans if ans is not None else "Mock Answer"
        return QAResponse(
            query=query,
            answer=chosen,
            answerable=True,
            qa_score=self.default_score,
            no_answer_score=0.1,
            document_id="doc_p22",
            chunk_id="chunk_p22",
            page_number=1,
        )


def make_chunk(chunk_id: str, doc_id: str, page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        page_number=page,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        char_count=len(text),
        word_count=len(text.split()),
    )


# =====================================================================
# 1. Generation Suite
# =====================================================================

class TestPhase22Generation:
    """Tests for generation, analytical question types, and diversity."""

    def test_valid_evidence_produces_question(self):
        text = "Supervised learning algorithms map input features to labeled targets using ground truth examples."
        chunk = make_chunk("c1", "doc1", 1, text)
        extractor = AnswerCandidateExtractor()
        cands = extractor.extract_from_chunk(chunk)
        assert len(cands) > 0

    def test_multi_page_synthesis_candidate_extraction(self):
        c1 = make_chunk("c1", "doc1", 1, "The transformer architecture relies on multi-head self-attention.")
        c2 = make_chunk("c2", "doc1", 2, "In contrast, convolutional neural networks process local spatial features.")
        extractor = AnswerCandidateExtractor()
        mp_cands = extractor.extract_multipage_candidates([c1, c2])
        assert len(mp_cands) > 0
        cand = mp_cands[0]
        assert cand.candidate_type == QuestionType.MULTI_PAGE_SYNTHESIS
        assert cand.page_numbers == [1, 2]
        assert cand.chunk_ids == ["c1", "c2"]

    def test_question_types_classification_analytical(self):
        assert classify_question_type("What is the difference between supervised and unsupervised learning?") == QuestionType.COMPARISON
        assert classify_question_type("What is meant by gradient descent?") == QuestionType.DEFINITION
        assert classify_question_type("Explain the mechanism of backpropagation.") == QuestionType.EXPLANATION
        assert classify_question_type("What is the reason why ReLU prevents vanishing gradients?") == QuestionType.REASONING
        assert classify_question_type("What percentage of training data was reserved for validation?") == QuestionType.NUMERICAL_FACT
        assert classify_question_type("How many hidden layers are configured?", candidate_type=QuestionType.NUMERICAL_FACT) == QuestionType.NUMERICAL_FACT
        assert classify_question_type("How does attention aggregate cross-page representations?", is_multipage=True) == QuestionType.MULTI_PAGE_SYNTHESIS

    def test_diversity_balancing_selects_different_types(self):
        chunks = [
            make_chunk("c1", "doc1", 1, "A neural network is defined as a computing system composed of interconnected nodes. The network achieved 98.4% accuracy."),
            make_chunk("c2", "doc1", 2, "In contrast to batch gradient descent, stochastic gradient descent updates parameters per sample."),
        ]
        mock_model = MockPhase22Model(mapping={
            "a computing system composed of interconnected nodes": "What is defined as a neural network?",
            "98.4%": "What percentage accuracy was achieved by the network?",
            "stochastic gradient descent": "What is the difference between batch and stochastic gradient descent?",
        })
        qa_service = MockPhase22QA(answer_map={
            "What is defined as a neural network?": "a computing system composed of interconnected nodes",
            "What percentage accuracy was achieved by the network?": "98.4%",
            "What is the difference between batch and stochastic gradient descent?": "stochastic gradient descent",
        })
        service = QuestionGenerationService(
            model=mock_model,
            qa_service=qa_service,
        )
        resp = service.generate_from_chunks(
            chunks=chunks,
            document_id="doc1",
            count=3,
            ensure_diversity=True,
        )
        assert resp.returned_count >= 2
        types = {q.question_type for q in resp.questions}
        # Verify diversity: at least 2 distinct question types
        assert len(types) >= 2

    def test_duplicate_question_keeps_alternative_answer_until_qa_selects_correct_span(self):
        text = "The ALVINN system drove at 70 miles per hour for 90 miles."
        question = "How long did the ALVINN system drive?"
        extractor = AnswerCandidateExtractor(max_candidates_per_chunk=2)
        model = MockPhase22Model({"70 miles": question, "90 miles": question})
        qa = MockPhase22QA({question: "90 miles"})
        service = QuestionGenerationService(
            model=model,
            extractor=extractor,
            validator=QuestionValidator(qa_service=qa),
        )

        response = service.generate_from_chunks(
            [make_chunk("alvinn", "doc1", 15, text)],
            document_id="doc1",
            count=1,
            include_rejected=True,
        )

        assert response.generated_candidates == 2
        assert response.returned_count == 1
        assert response.questions[0].answer == "90 miles"
        assert response.questions[0].qa_predicted_answer == "90 miles"

    def test_requested_type_is_not_bypassed_when_no_source_candidates_match(self):
        chunk = make_chunk("year", "doc1", 1, "Python was first released in 1991.")
        model = MockPhase22Model({"1991": "When was Python first released?"})
        qa = MockPhase22QA({"When was Python first released?": "1991"})
        service = QuestionGenerationService(model=model, qa_service=qa)

        response = service.generate_from_chunks(
            [chunk],
            document_id="doc1",
            count=1,
            target_type=QuestionType.DEFINITION,
        )

        assert response.validated_count == 0
        assert response.returned_count == 0
        assert response.questions == []


# =====================================================================
# 2. Validation Suite
# =====================================================================

class TestPhase22Validation:
    """Tests for question validation quality checks and unanswerability."""

    def test_answerable_question_accepted(self):
        validator = QuestionValidator(qa_service=MockPhase22QA({"What is the learning rate?": "0.001"}))
        cand = QuestionCandidate(
            question_text="What is the learning rate?",
            answer_candidate=AnswerCandidate(
                answer_text="0.001",
                source_text="The optimizer was initialized with a learning rate of 0.001.",
                document_id="d1",
                chunk_id="c1",
                page_number=1,
            ),
        )
        is_valid, validated_q, reason = validator.validate_candidate(cand)
        assert is_valid is True
        assert validated_q is not None
        assert validated_q.answer == "0.001"
        assert reason is None

    def test_unanswerable_question_rejected(self):
        validator = QuestionValidator(qa_service=MockPhase22QA({"What is quantum computing?": None}))
        cand = QuestionCandidate(
            question_text="What is quantum computing?",
            answer_candidate=AnswerCandidate(
                answer_text="0.001",
                source_text="The optimizer was initialized with a learning rate of 0.001.",
                document_id="d1",
                chunk_id="c1",
                page_number=1,
            ),
        )
        is_valid, validated_q, reason = validator.validate_candidate(cand)
        assert is_valid is False
        assert validated_q is None
        assert "unanswerable" in reason.lower()

    def test_malformed_questions_rejected(self):
        validator = QuestionValidator()
        # Invalid punctuation prefix
        assert validator._check_quality("?What is learning rate?", "0.001", "learning rate 0.001") is not None
        assert validator._check_quality("what is the learning rate?", "0.001", "learning rate 0.001") is not None
        # Missing question mark
        assert validator._check_quality("What is learning rate", "0.001", "learning rate 0.001") is not None
        # Too short
        assert validator._check_quality("Who?", "A", "A") is not None

    def test_dangling_incomplete_questions_rejected(self):
        validator = QuestionValidator()
        assert validator._check_quality("What is the role of?", "weights", "weights update") is not None
        assert validator._check_quality("Why does gradient descent fail because?", "vanishing", "vanishing gradients") is not None
        assert validator._check_quality("Which layer does it connect with?", "layer 2", "connects with layer 2") is not None

    def test_vague_questions_rejected(self):
        validator = QuestionValidator()
        assert validator._check_quality("What happened?", "loss decreased", "loss decreased") is not None
        assert validator._check_quality("What is this?", "matrix multiplication", "matrix multiplication") is not None
        assert validator._check_quality("What did they do?", "trained model", "trained model") is not None

    def test_circular_and_tautological_questions_rejected(self):
        validator = QuestionValidator()
        assert validator._check_quality("What is transformer?", "transformer", "transformer architecture") is not None
        assert validator._check_quality("Who is Alan Turing?", "Alan Turing", "Alan Turing designed ACE") is not None


# =====================================================================
# 3. Duplicate Detection Suite
# =====================================================================

class TestPhase22Duplicates:
    """Tests for exact, normalized, and near-duplicate candidate suppression."""

    def test_exact_and_normalized_duplicates_rejected_with_diagnostics(self):
        dedup = QuestionDeduplicator()
        ac = AnswerCandidate(
            answer_text="ReLU",
            source_text="ReLU has derivative 1 for positive inputs.",
            document_id="d1",
            chunk_id="c1",
            page_number=1,
        )
        cands = [
            QuestionCandidate(candidate_id="c1", question_text="What is the derivative of ReLU?", answer_candidate=ac),
            QuestionCandidate(candidate_id="c2", question_text="What is the derivative of ReLU?", answer_candidate=ac),
            QuestionCandidate(candidate_id="c3", question_text="  what is the derivative of relu?  ", answer_candidate=ac),
        ]
        unique, rejected = dedup.filter_candidates_with_rejected(cands)
        assert len(unique) == 1
        assert len(rejected) == 2
        assert "duplicate" in rejected[0][1].lower()

    def test_near_duplicate_jaccard_suppression(self):
        dedup = QuestionDeduplicator(similarity_threshold=0.75)
        ac = AnswerCandidate(
            answer_text="ReLU",
            source_text="ReLU has derivative 1 for positive inputs.",
            document_id="d1",
            chunk_id="c1",
            page_number=1,
        )
        cands = [
            QuestionCandidate(candidate_id="c1", question_text="What is the derivative of ReLU for positive inputs?", answer_candidate=ac),
            QuestionCandidate(candidate_id="c2", question_text="What is derivative of ReLU for positive inputs?", answer_candidate=ac),
        ]
        unique, rejected = dedup.filter_candidates_with_rejected(cands)
        assert len(unique) == 1
        assert len(rejected) == 1
        assert "near" in rejected[0][1].lower() and "duplicate" in rejected[0][1].lower()


# =====================================================================
# 4. Provenance Preservation Suite
# =====================================================================

class TestPhase22Provenance:
    """Tests ensuring complete multi-level provenance is retained."""

    def test_single_chunk_provenance_preserved(self):
        qa = MockPhase22QA({"What is AdaGrad?": "adaptive gradient algorithm"})
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="What is AdaGrad?",
            answer_candidate=AnswerCandidate(
                answer_text="adaptive gradient algorithm",
                source_text="AdaGrad is an adaptive gradient algorithm designed for sparse data.",
                document_id="opt_book",
                chunk_id="opt_chunk_42",
                page_number=88,
            ),
        )
        is_valid, q, _ = validator.validate_candidate(cand)
        assert is_valid is True
        assert q.document_id == "opt_book"
        assert q.chunk_id == "opt_chunk_42"
        assert q.chunk_ids == ["opt_chunk_42"]
        assert q.page_number == 88
        assert q.page_numbers == [88]
        assert "AdaGrad" in q.source_text

    def test_multipage_provenance_preserved(self):
        qa = MockPhase22QA({"What are the components?": "encoder and decoder"})
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="What are the components?",
            answer_candidate=AnswerCandidate(
                answer_text="encoder and decoder",
                source_text="The model has an encoder and decoder.",
                document_id="nlp_book",
                chunk_id="nlp_p1_c0",
                chunk_ids=["nlp_p1_c0", "nlp_p2_c0"],
                page_number=1,
                page_numbers=[1, 2],
                candidate_type=QuestionType.MULTI_PAGE_SYNTHESIS,
                metadata={"is_multipage": True},
            ),
        )
        is_valid, q, _ = validator.validate_candidate(cand)
        assert is_valid is True
        assert q.question_type == QuestionType.MULTI_PAGE_SYNTHESIS
        assert q.chunk_ids == ["nlp_p1_c0", "nlp_p2_c0"]
        assert q.page_numbers == [1, 2]


# =====================================================================
# 5. Safety & External Knowledge Rejection Suite
# =====================================================================

class TestPhase22Safety:
    """Tests ensuring no model general knowledge or outside entities are allowed."""

    def test_outside_entity_in_question_rejected(self):
        validator = QuestionValidator()
        source = "Support vector machines find the maximum margin hyperplane."
        err = validator._check_outside_knowledge(
            question="Why did Albert Einstein prefer support vector machines?",
            source_text=source,
        )
        assert err is not None
        assert "Albert" in err or "Einstein" in err

    def test_grounded_entity_in_question_accepted(self):
        validator = QuestionValidator()
        source = "Geoffrey Hinton and Terry Sejnowski introduced Boltzmann machines in 1985."
        err = validator._check_outside_knowledge(
            question="Who introduced Boltzmann machines in 1985?",
            source_text=source,
        )
        assert err is None


# =====================================================================
# 6. API Layer Suite
# =====================================================================

class TestPhase22API:
    """Tests for the question generation HTTP endpoints."""

    def test_api_generate_questions_success_with_diagnostics(self):
        chunk = make_chunk(
            "c10",
            "doc_test",
            5,
            "Dropout is defined as a regularization technique for neural networks. The dropout rate was set to 50% during training.",
        )
        mock_model = MockPhase22Model({"50%": "What was the dropout rate set to during training?"})
        mock_qa = MockPhase22QA({"What was the dropout rate set to during training?": "50%"})

        mock_service = QuestionGenerationService(
            model=mock_model,
            qa_service=mock_qa,
        )
        mock_service._get_document_chunks = lambda *args, **kwargs: [chunk]

        app.dependency_overrides[get_question_generation_service] = lambda: mock_service
        try:
            client = TestClient(app)
            req_body = {
                "document_id": "doc_test",
                "count": 2,
                "include_rejected": True,
                "ensure_diversity": True,
            }
            res = client.post("/api/v1/questions/generate", json=req_body)
            assert res.status_code == status.HTTP_200_OK
            data = res.json()
            assert data["document_id"] == "doc_test"
            assert data["requested_count"] == 2
            assert data["returned_count"] >= 1
            assert "questions" in data
            assert "rejection_summary" in data
            assert data["latency_ms"] is not None

            q0 = data["questions"][0]
            assert q0["document_id"] == "doc_test"
            assert q0["page_number"] == 5
            assert "50%" in q0["answer"]
            assert q0["chunk_ids"] == ["c10"]
        finally:
            app.dependency_overrides.pop(get_question_generation_service, None)

    def test_api_generate_questions_404_when_document_missing(self):
        client = TestClient(app)
        req_body = {
            "document_id": "nonexistent_book_9999",
            "count": 5,
        }
        res = client.post("/api/v1/questions/generate", json=req_body)
        assert res.status_code == status.HTTP_404_NOT_FOUND

    def test_api_generate_questions_422_on_invalid_count(self):
        client = TestClient(app)
        req_body = {
            "document_id": "doc_test",
            "count": 0,  # Invalid: ge=1
        }
        res = client.post("/api/v1/questions/generate", json=req_body)
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
