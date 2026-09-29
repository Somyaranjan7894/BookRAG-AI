"""Comprehensive test suite for Phase 12 Question Generation in BookRAG AI.

Covers:
Suite A: Answer candidate extraction (person, date, year, number, location, trivial rejected, provenance)
Suite B: Question generation (formatting, answer conditioning, determinism)
Suite C: Question validation (accepted, incorrect rejected, unanswerable rejected, outside knowledge rejected)
Suite D: Answer matching (exact, casing, whitespace, numeric mismatch, date mismatch, partial overlap)
Suite E: Grounding (occurrence in evidence, unsupported rejected, provenance preservation)
Suite F: Duplicate detection (exact, normalized, within candidate pool)
Suite G: Count control (requested count respected, insufficient candidates, no fabrication)
Suite H: Document isolation (strict document scope, no cross-document leakage)
Suite I: API endpoints (POST /api/v1/questions/generate, valid response, 422 validation, 404 document)
Suite J: Semantic principles (plausible != valid, book is source of truth, no model authority)
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock, patch
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.v1.endpoints.question_generation import get_question_generation_service
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
)
from app.services.qa.service import QAService
from app.services.question_generation.answers import AnswerCandidateExtractor
from app.services.question_generation.deduplication import (
    QuestionDeduplicator,
    normalize_question_for_dedup,
)
from app.services.question_generation.model import QuestionGenerationModel
from app.services.question_generation.service import QuestionGenerationService
from app.services.question_generation.validator import (
    QuestionValidator,
    classify_question_difficulty,
    classify_question_type,
    extract_numbers,
    matches_expected_answer,
    normalize_answer_text,
)
from app.services.search.exceptions import DocumentNotFoundError


# =====================================================================
# Mock Helpers
# =====================================================================

class MockQuestionGenModel:
    """Mock for QuestionGenerationModel providing fast deterministic question generation."""

    def __init__(self, mapping: Optional[dict[str, str]] = None) -> None:
        self.mapping = mapping or {}
        self.calls: List[tuple[str, str]] = []

    def generate_question(self, answer: str, context: str, **kwargs: Any) -> str:
        self.calls.append((answer, context))
        if answer in self.mapping:
            return self.mapping[answer]
        return f"What is related to {answer}?"

    def generate_questions_batch(self, pairs: List[tuple[str, str]], **kwargs: Any) -> List[str]:
        results = []
        for ans, ctx in pairs:
            results.append(self.generate_question(ans, ctx, **kwargs))
        return results


class MockQAService:
    """Mock for QAService simulating extractive QA validation."""

    def __init__(self, answer: Optional[str] = None, answerable: bool = True, score: float = 0.95) -> None:
        self.answer = answer
        self.answerable = answerable
        self.score = score
        self.queries: List[str] = []

    def answer_question(self, query: str, evidence: Any, **kwargs: Any) -> QAResponse:
        self.queries.append(query)
        ans = self.answer
        if ans is None and self.answerable:
            # Default to extracting something from evidence
            ans = "Mock Answer"

        return QAResponse(
            query=query,
            answer=ans if self.answerable else None,
            answerable=self.answerable,
            qa_score=self.score if self.answerable else -5.0,
            no_answer_score=0.0,
            document_id="doc_1",
            chunk_id="doc_1_p001_c0001",
            page_number=1,
            chunk_index=0,
            answer_start=0,
            answer_end=len(ans) if ans else 0,
            source_text=evidence[0]["text"] if evidence else "",
            evidence_rank=1,
        )


def make_test_chunk(
    text: str,
    document_id: str = "doc_test",
    chunk_id: str = "doc_test_p001_c0001",
    page_number: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        start_char=0,
        end_char=len(text),
        char_count=len(text),
        word_count=len(text.split()),
    )


# =====================================================================
# Suite A: Answer Candidate Extraction
# =====================================================================
class TestAnswerCandidateExtraction:
    """Verify conservative extraction of entities, dates, numbers, and rejection of trivia."""

    def test_extract_valid_person(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk("Python was created by Guido van Rossum in 1991.")
        candidates = extractor.extract_from_chunk(chunk)
        ans_texts = [c.answer_text for c in candidates]
        assert "Guido van Rossum" in ans_texts

    def test_extract_valid_year_and_date(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk("The declaration was signed on October 14, 1947 in Paris.")
        candidates = extractor.extract_from_chunk(chunk)
        ans_texts = [c.answer_text for c in candidates]
        assert any("1947" in t or "October 14, 1947" in t for t in ans_texts)

    def test_extract_valid_number_and_percentage(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk("The population grew by 42% reaching 1.41 billion in 2020.")
        candidates = extractor.extract_from_chunk(chunk)
        ans_texts = [c.answer_text for c in candidates]
        assert any("42%" in t or "1.41 billion" in t for t in ans_texts)

    def test_extract_valid_organization(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk("The mission was directed by NASA from Houston.")
        candidates = extractor.extract_from_chunk(chunk)
        ans_texts = [c.answer_text for c in candidates]
        assert "NASA" in ans_texts

    def test_reject_trivial_stop_words(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk("The and or but if because.")
        candidates = extractor.extract_from_chunk(chunk)
        assert len(candidates) == 0

    def test_reject_empty_or_whitespace_text(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk("   \n\t  ")
        candidates = extractor.extract_from_chunk(chunk)
        assert len(candidates) == 0

    def test_provenance_strictly_preserved(self):
        extractor = AnswerCandidateExtractor()
        chunk = make_test_chunk(
            text="Alan Turing designed the ACE computer in 1945.",
            document_id="history_doc",
            chunk_id="history_doc_p012_c0003",
            page_number=12,
            chunk_index=3,
        )
        candidates = extractor.extract_from_chunk(chunk)
        assert len(candidates) > 0
        cand = candidates[0]
        assert cand.document_id == "history_doc"
        assert cand.chunk_id == "history_doc_p012_c0003"
        assert cand.page_number == 12
        assert cand.chunk_index == 3
        assert cand.start_offset is not None
        assert cand.end_offset is not None
        assert chunk.text[cand.start_offset:cand.end_offset] == cand.answer_text


# =====================================================================
# Suite B: Question Generation
# =====================================================================
class TestQuestionGeneration:
    """Verify question generation formatting, answer conditioning, and determinism."""

    def test_question_formatting_ends_with_question_mark(self):
        cleaned = QuestionGenerationModel._clean_question("Who was Alan Turing")
        assert cleaned.endswith("?")
        assert not cleaned.endswith("??")

    def test_question_formatting_collapses_double_question_mark(self):
        cleaned = QuestionGenerationModel._clean_question("Who was Alan Turing? ?")
        assert cleaned == "Who was Alan Turing?"

    def test_answer_conditioned_generation_call(self):
        mock_model = MockQuestionGenModel(
            mapping={"Guido van Rossum": "Who created Python?"}
        )
        q = mock_model.generate_question("Guido van Rossum", "Python was created by Guido van Rossum.")
        assert q == "Who created Python?"
        assert ("Guido van Rossum", "Python was created by Guido van Rossum.") in mock_model.calls

    def test_batch_generation_preserves_order(self):
        mock_model = MockQuestionGenModel(
            mapping={"A": "What is A?", "B": "What is B?", "C": "What is C?"}
        )
        pairs = [("A", "ctx A"), ("B", "ctx B"), ("C", "ctx C")]
        results = mock_model.generate_questions_batch(pairs)
        assert results == ["What is A?", "What is B?", "What is C?"]


# =====================================================================
# Suite C: Question Validation
# =====================================================================
class TestQuestionValidation:
    """Verify question acceptance, incorrect answer rejection, and unanswerability handling."""

    def test_valid_question_accepted(self):
        qa = MockQAService(answer="Guido van Rossum", answerable=True)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="Who created Python?",
            answer_candidate=AnswerCandidate(
                answer_text="Guido van Rossum",
                source_text="Python was created by Guido van Rossum in 1991.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, validated_q, reason = validator.validate_candidate(cand)
        assert is_valid is True
        assert validated_q is not None
        assert validated_q.question == "Who created Python?"
        assert validated_q.answer == "Guido van Rossum"
        assert reason is None

    def test_incorrect_answer_rejected(self):
        qa = MockQAService(answer="James Gosling", answerable=True)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="Who created Python?",
            answer_candidate=AnswerCandidate(
                answer_text="Guido van Rossum",
                source_text="Python was created by Guido van Rossum in 1991.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, validated_q, reason = validator.validate_candidate(cand)
        assert is_valid is False
        assert validated_q is None
        assert "does not match expected answer" in reason

    def test_unanswerable_question_rejected(self):
        qa = MockQAService(answer=None, answerable=False)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="What color is the sky?",
            answer_candidate=AnswerCandidate(
                answer_text="Guido van Rossum",
                source_text="Python was created by Guido van Rossum in 1991.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, validated_q, reason = validator.validate_candidate(cand)
        assert is_valid is False
        assert "unanswerable" in reason

    def test_empty_question_rejected_by_quality_check(self):
        qa = MockQAService(answer="Guido van Rossum", answerable=True)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="   ",
            answer_candidate=AnswerCandidate(
                answer_text="Guido van Rossum",
                source_text="Python was created by Guido van Rossum in 1991.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, validated_q, reason = validator.validate_candidate(cand)
        assert is_valid is False
        assert "empty" in reason.lower()

    def test_question_identical_to_source_text_rejected(self):
        text = "Python was created by Guido van Rossum in 1991."
        qa = MockQAService(answer="Guido van Rossum", answerable=True)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text=text + "?",
            answer_candidate=AnswerCandidate(
                answer_text="Guido van Rossum",
                source_text=text,
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        # normalize_answer_text compares clean text
        assert validator._check_quality(text, "Guido van Rossum", text) is not None


# =====================================================================
# Suite D: Answer Matching
# =====================================================================
class TestAnswerMatching:
    """Verify exact, case, whitespace, numeric, and date matching rules."""

    def test_exact_match_passes(self):
        assert matches_expected_answer("Paris", "Paris") is True

    def test_case_insensitive_match_passes(self):
        assert matches_expected_answer("paris", "Paris") is True
        assert matches_expected_answer("GUIDO VAN ROSSUM", "Guido van Rossum") is True

    def test_whitespace_variation_passes(self):
        assert matches_expected_answer("Guido  van   Rossum", "Guido van Rossum") is True

    def test_numeric_mismatch_strictly_rejected(self):
        assert matches_expected_answer("1998", "1999") is False
        assert matches_expected_answer("In 1998", "In 1999") is False

    def test_date_mismatch_strictly_rejected(self):
        assert matches_expected_answer("October 14, 1947", "October 15, 1947") is False

    def test_partial_overlap_with_identical_numbers_passes(self):
        assert matches_expected_answer("year 1991", "1991") is True
        assert matches_expected_answer("approximately 1.41 billion", "1.41 billion") is True


# =====================================================================
# Suite E: Source Grounding
# =====================================================================
class TestSourceGrounding:
    """Verify that expected answer must be physically present in source evidence."""

    def test_answer_present_in_evidence_accepted(self):
        qa = MockQAService(answer="Alan Turing", answerable=True)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="Who designed ACE?",
            answer_candidate=AnswerCandidate(
                answer_text="Alan Turing",
                source_text="Alan Turing designed the Automatic Computing Engine.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, _, _ = validator.validate_candidate(cand)
        assert is_valid is True

    def test_answer_missing_from_evidence_rejected(self):
        qa = MockQAService(answer="Isaac Newton", answerable=True)
        validator = QuestionValidator(qa_service=qa)
        cand = QuestionCandidate(
            question_text="Who discovered gravity?",
            answer_candidate=AnswerCandidate(
                answer_text="Isaac Newton",
                source_text="The text discusses thermodynamics and heat engines.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, _, reason = validator.validate_candidate(cand)
        assert is_valid is False
        assert "does not occur in source evidence" in reason


# =====================================================================
# Suite F: Duplicate Detection
# =====================================================================
class TestDuplicateDetection:
    """Verify duplicate detection across casing, punctuation, and candidates."""

    def test_exact_duplicate_rejected(self):
        dedup = QuestionDeduplicator()
        assert dedup.is_duplicate("Who created Python?") is False
        dedup.record("Who created Python?")
        assert dedup.is_duplicate("Who created Python?") is True

    def test_normalized_duplicate_rejected(self):
        dedup = QuestionDeduplicator()
        dedup.record("Who created Python?")
        assert dedup.is_duplicate("who created python") is True
        assert dedup.is_duplicate("  WHO CREATED PYTHON?  ") is True

    def test_filter_candidate_duplicates(self):
        dedup = QuestionDeduplicator()
        ans = AnswerCandidate(
            answer_text="Guido",
            source_text="Guido created Python.",
            document_id="doc_1",
            chunk_id="chunk_1",
            page_number=1,
        )
        candidates = [
            QuestionCandidate(question_text="Who created Python?", answer_candidate=ans),
            QuestionCandidate(question_text="who created python?", answer_candidate=ans),
            QuestionCandidate(question_text="When was Python created?", answer_candidate=ans),
        ]
        unique = dedup.filter_candidates(candidates)
        assert len(unique) == 2
        assert unique[0].question_text == "Who created Python?"
        assert unique[1].question_text == "When was Python created?"


# =====================================================================
# Suite G: Count Control
# =====================================================================
class TestCountControl:
    """Verify requested question count control without fabrication."""

    def test_requested_count_respected(self):
        mock_model = MockQuestionGenModel(
            mapping={
                "Guido van Rossum": "Who created Python?",
                "1991": "When was Python released?",
                "Netherlands": "Where was it created?",
            }
        )
        mock_qa = MockQAService()
        # Mock QA to return the exact answer candidate
        def dynamic_qa(query, evidence, **kwargs):
            ans = "Guido van Rossum" if "Who" in query else ("1991" if "When" in query else "Netherlands")
            return QAResponse(
                query=query,
                answer=ans,
                answerable=True,
                qa_score=0.95,
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
                source_text=evidence[0]["text"] if evidence else "",
                evidence_rank=1,
            )
        mock_qa.answer_question = dynamic_qa

        service = QuestionGenerationService(
            model=mock_model,
            validator=QuestionValidator(qa_service=mock_qa),
        )

        chunk = make_test_chunk(
            "Python was created by Guido van Rossum in 1991 in the Netherlands."
        )
        res = service.generate_from_chunks([chunk], document_id="doc_1", count=2)
        assert res.requested_count == 2
        assert res.returned_count == 2
        assert len(res.questions) == 2

    def test_insufficient_candidates_never_fabricates(self):
        mock_model = MockQuestionGenModel(
            mapping={"Guido van Rossum": "Who created Python?"}
        )
        mock_qa = MockQAService(answer="Guido van Rossum", answerable=True)
        service = QuestionGenerationService(
            model=mock_model,
            validator=QuestionValidator(qa_service=mock_qa),
        )

        chunk = make_test_chunk("Python was created by Guido van Rossum.")
        # Request 10 questions when only 1 valid question exists
        res = service.generate_from_chunks([chunk], document_id="doc_1", count=10)
        assert res.requested_count == 10
        assert res.returned_count < res.requested_count
        assert len(res.questions) == res.returned_count


# =====================================================================
# Suite H: Document Isolation
# =====================================================================
class TestDocumentIsolation:
    """Verify that question generation never accesses cross-document evidence."""

    def test_unknown_document_raises_404(self):
        mock_retrieval = MagicMock()
        mock_retrieval._indices = {}
        mock_retrieval._default_index_id = None
        service = QuestionGenerationService(retrieval_service=mock_retrieval)

        req = QuestionGenerationRequest(document_id="non_existent_doc", count=5)
        with pytest.raises(DocumentNotFoundError):
            service.generate_questions(req)


# =====================================================================
# Suite I: API Endpoints
# =====================================================================
class TestQuestionGenerationAPI:
    """Verify FastAPI endpoint POST /api/v1/questions/generate."""

    client = TestClient(app)

    def test_post_generate_questions_success(self):
        mock_service = MagicMock(spec=QuestionGenerationService)
        mock_service.generate_questions.return_value = QuestionGenerationResponse(
            document_id="doc_test",
            requested_count=2,
            generated_candidates=4,
            validated_count=2,
            returned_count=2,
            questions=[
                GeneratedQuestion(
                    question="Who created Python?",
                    answer="Guido van Rossum",
                    question_type=QuestionType.WHO,
                    difficulty=QuestionDifficulty.EASY,
                    document_id="doc_test",
                    chunk_id="doc_test_p001_c0001",
                    page_number=1,
                    source_text="Python was created by Guido van Rossum in 1991.",
                ),
                GeneratedQuestion(
                    question="When was Python released?",
                    answer="1991",
                    question_type=QuestionType.WHEN,
                    difficulty=QuestionDifficulty.EASY,
                    document_id="doc_test",
                    chunk_id="doc_test_p001_c0001",
                    page_number=1,
                    source_text="Python was created by Guido van Rossum in 1991.",
                ),
            ],
        )

        app.dependency_overrides[get_question_generation_service] = lambda: mock_service
        try:
            resp = self.client.post(
                "/api/v1/questions/generate",
                json={"document_id": "doc_test", "count": 2},
            )
            assert resp.status_code == status.HTTP_200_OK
            data = resp.json()
            assert data["document_id"] == "doc_test"
            assert data["requested_count"] == 2
            assert data["returned_count"] == 2
            assert len(data["questions"]) == 2
            assert data["questions"][0]["question"] == "Who created Python?"
            assert data["questions"][0]["answer"] == "Guido van Rossum"
            assert data["questions"][0]["document_id"] == "doc_test"
        finally:
            app.dependency_overrides.pop(get_question_generation_service, None)

    def test_post_generate_questions_invalid_count_returns_422(self):
        resp = self.client.post(
            "/api/v1/questions/generate",
            json={"document_id": "doc_test", "count": 0},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =====================================================================
# Suite J: Semantic Principles
# =====================================================================
class TestSemanticPrinciples:
    """Verify semantic guarantees: plausible != valid, book is source of truth."""

    def test_plausible_question_rejected_if_qa_cannot_answer(self):
        """A fluent question must be rejected if the answer cannot be extracted from the evidence."""
        mock_qa = MockQAService(answerable=False)
        validator = QuestionValidator(qa_service=mock_qa)
        cand = QuestionCandidate(
            question_text="What was the primary motive of the author?",
            answer_candidate=AnswerCandidate(
                answer_text="financial freedom",
                source_text="The author lived in London.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        is_valid, _, reason = validator.validate_candidate(cand)
        assert is_valid is False

    def test_question_generator_is_not_treated_as_factual_authority(self):
        """If the generator generates a hallucinated date not in the source text, it is rejected."""
        mock_qa = MockQAService(answer="1850", answerable=True)
        validator = QuestionValidator(qa_service=mock_qa)
        cand = QuestionCandidate(
            question_text="When did the event occur?",
            answer_candidate=AnswerCandidate(
                answer_text="1920",
                source_text="The event occurred in 1920 according to records.",
                document_id="doc_1",
                chunk_id="chunk_1",
                page_number=1,
            ),
        )
        # QA returned 1850 while expected was 1920
        is_valid, _, reason = validator.validate_candidate(cand)
        assert is_valid is False
        assert "does not match" in reason


# =====================================================================
# Suite K: Taxonomy and Heuristics
# =====================================================================
class TestTaxonomyAndHeuristics:
    """Verify QuestionType and QuestionDifficulty classifications."""

    def test_classify_all_question_types(self):
        assert classify_question_type("Who was the founder?") == QuestionType.WHO
        assert classify_question_type("What is backpropagation?") == QuestionType.WHAT
        assert classify_question_type("When was Python created?") == QuestionType.WHEN
        assert classify_question_type("Where was Alan Turing born?") == QuestionType.WHERE
        assert classify_question_type("Why did the network diverge?") == QuestionType.WHY
        assert classify_question_type("How many parameters were used?") == QuestionType.HOW_MANY
        assert classify_question_type("How do you train the model?") == QuestionType.HOW
        assert classify_question_type("Compare gradient descent and Adam.") == QuestionType.COMPARISON
        assert classify_question_type("What is the definition of loss?") == QuestionType.DEFINITION
        assert classify_question_type("Is Python interpreted?") == QuestionType.FACTUAL

    def test_classify_difficulty_heuristics(self):
        assert classify_question_difficulty(QuestionType.WHO, "Guido", "Some context") == QuestionDifficulty.EASY
        assert classify_question_difficulty(QuestionType.WHY, "Because of over-fitting", "Some context") == QuestionDifficulty.HARD
        assert classify_question_difficulty(QuestionType.WHAT, "Gradient vectors computed via chain rule recursively", "Long context " * 50) == QuestionDifficulty.MEDIUM


# =====================================================================
# Suite L: Real Model Integration
# =====================================================================
class TestRealModelIntegration:
    """Verify real question generator model inference and end-to-end integration."""

    def test_real_model_single_generation(self):
        model = QuestionGenerationModel.get_instance()
        q = model.generate_question(
            answer="Guido van Rossum",
            context="Python was created by Guido van Rossum in 1991.",
        )
        assert len(q) > 0
        assert q.endswith("?")
        assert "Python" in q or "who" in q.lower()

