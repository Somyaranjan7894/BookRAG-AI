"""Unit and Integration Tests for Phase 21 Grounding Robustness and Query-Type-Aware Generation.

Tests all 12 required architectural categories:
1. Short factual fragment grounding
2. Short list answer grounding
3. Numerical answer grounding
4. Direct factual sentence grounding
5. Comparison completeness checking
6. Multi-part completeness checking
7. Multi-page evidence synthesis
8. Unsupported answer safe refusal
9. Unanswerable query safe refusal
10. Bounded regeneration (max 2 attempts)
11. Citation preservation
12. Provenance preservation
"""

from typing import List, Optional, Sequence, Tuple
from unittest.mock import MagicMock
import pytest

from app.schemas.chunk import Chunk
from app.schemas.generation import GenerationEvidenceItem, GenerationResponse
from app.schemas.grounding import (
    ExtractedClaim,
    GroundedAnswerRequest,
    GroundingReport,
)
from app.schemas.query_plan import QueryPlan, QueryType
from app.schemas.search import SearchResponse, SearchResult
from app.services.citation.service import CitationService
from app.services.generation.service import GenerationService
from app.services.grounding.claims import (
    ClaimDecomposer,
    classify_claim_form,
    contextualize_fragment,
)
from app.services.grounding.completeness import (
    AnswerCompletenessChecker,
    CompletenessCheckResult,
)
from app.services.grounding.model import NLIModel, NLIScores
from app.services.grounding.orchestrator import GroundedAnswerService
from app.services.grounding.service import GroundingService
from app.services.query_understanding.planner import QueryPlanner
from app.services.search.query_search import QuerySearchService
from app.services.search.service import SearchService


class MockNLIModel(NLIModel):
    """Deterministic mock NLI model for unit testing."""

    def __init__(self, score_map: Optional[dict] = None, default_score: Optional[NLIScores] = None) -> None:
        self.score_map = score_map or {}
        self.default_score = default_score or NLIScores(entailment=0.95, contradiction=0.01, neutral=0.04)

    def predict_pair(self, premise: str, hypothesis: str) -> NLIScores:
        key = (premise.strip(), hypothesis.strip())
        if key in self.score_map:
            return self.score_map[key]
        for (p, h), sc in self.score_map.items():
            if p in premise and h in hypothesis:
                return sc
        return self.default_score

    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[NLIScores]:
        return [self.predict_pair(p, h) for p, h in pairs]

    def predict_pairs(self, pairs: List) -> List[NLIScores]:
        return self.predict(pairs)


@pytest.fixture
def generic_evidence() -> List[GenerationEvidenceItem]:
    return [
        GenerationEvidenceItem(
            rank=1,
            chunk_id="chunk_page1_0",
            document_id="doc_test_01",
            page_number=1,
            chunk_index=0,
            source_text="The speed of sound in dry air at 20 degrees Celsius is approximately 343 meters per second.",
            similarity_score=0.89,
            reranker_score=3.5,
        ),
        GenerationEvidenceItem(
            rank=2,
            chunk_id="chunk_page5_1",
            document_id="doc_test_01",
            page_number=5,
            chunk_index=1,
            source_text="Photosynthesis converts solar light energy into chemical energy stored in glucose molecules.",
            similarity_score=0.82,
            reranker_score=2.8,
        ),
    ]


class TestPhase21FragmentGrounding:
    """Category 1, 2, 3, 4: Fragment classification, contextualization, and proposition-level evaluation."""

    def test_short_factual_fragment_classification_and_contextualization(self) -> None:
        """Category 1: Short factual fragment (<= 5 words, no finite verb)."""
        frag = "gradient descent"
        assert classify_claim_form(frag) == "short_span"

        q = "What optimization algorithm minimizes the empirical risk?"
        ctx = contextualize_fragment(q, frag)
        assert "optimization algorithm" in ctx.lower()
        assert "gradient descent" in ctx.lower()
        assert ctx.endswith(".")

    def test_short_list_answer_classification_and_contextualization(self) -> None:
        """Category 2: Short list answer without finite verbs."""
        lst = "entailment, contradiction, and neutral"
        assert classify_claim_form(lst) == "list"

        q = "What are the three semantic relationship categories in inference?"
        ctx = contextualize_fragment(q, lst)
        assert "semantic relationship categories" in ctx.lower()
        assert "entailment, contradiction, and neutral" in ctx.lower()
        assert "are" in ctx.lower()

    def test_numerical_answer_classification_and_contextualization(self) -> None:
        """Category 3: Numerical answers (percentage, constant, count)."""
        val1 = "42%"
        assert classify_claim_form(val1) == "numerical"

        val2 = "a constant 1"
        assert classify_claim_form(val2) == "numerical"

        val3 = "343 meters per second"
        assert classify_claim_form(val3) == "short_span"

        q = "What is the exact derivative of the activation function?"
        ctx = contextualize_fragment(q, "a constant 1")
        assert "derivative of the activation function" in ctx.lower()
        assert "is a constant 1" in ctx.lower()

    def test_direct_factual_sentence_declarative(self) -> None:
        """Category 4: Direct factual sentence with subject and predicate."""
        sent = "Backpropagation calculates loss gradients using the chain rule."
        assert classify_claim_form(sent) == "declarative"
        ctx = contextualize_fragment("How does it work?", sent)
        # Declarative sentences are kept as-is
        assert ctx == sent


class TestPhase21CompletenessChecking:
    """Category 5 & 6: Generic completeness checking for comparison and multi-part queries."""

    def test_comparison_completeness_both_aspects_present(self) -> None:
        """Category 5a: Comparison query with both aspects covered."""
        checker = AnswerCompletenessChecker()
        plan = QueryPlan(
            original_query="What is the difference between supervised learning and unsupervised learning?",
            query_type=QueryType.COMPARISON,
            comparison_aspects=["supervised learning", "unsupervised learning"],
        )
        complete_answer = (
            "Supervised learning trains models using labeled input-output pairs, "
            "whereas unsupervised learning discovers hidden patterns in unlabeled data."
        )
        res = checker.check_completeness(plan, complete_answer)
        assert res.is_complete is True
        assert res.missing_aspects == []

    def test_comparison_completeness_one_aspect_missing(self) -> None:
        """Category 5b: Comparison query where one aspect is omitted."""
        checker = AnswerCompletenessChecker()
        plan = QueryPlan(
            original_query="What is the difference between supervised learning and unsupervised learning?",
            query_type=QueryType.COMPARISON,
            comparison_aspects=["supervised learning", "unsupervised learning"],
        )
        incomplete_answer = "Supervised learning trains models using labeled input-output pairs."
        res = checker.check_completeness(plan, incomplete_answer)
        assert res.is_complete is False
        assert "unsupervised learning" in res.missing_aspects
        assert "missing coverage of 'unsupervised learning'" in res.reason

    def test_multipart_completeness_subquestions_covered(self) -> None:
        """Category 6a: Multi-part query with sub-questions fully covered."""
        checker = AnswerCompletenessChecker()
        plan = QueryPlan(
            original_query="Explain the architecture of transformers and how self-attention works.",
            query_type=QueryType.MULTI_PART,
            sub_questions=["architecture of transformers", "how self-attention works"],
        )
        complete_answer = (
            "The architecture of transformers relies on stacked encoder and decoder layers, "
            "where self-attention dynamically weights relationships between token representations."
        )
        res = checker.check_completeness(plan, complete_answer)
        assert res.is_complete is True

    def test_multipart_completeness_subquestion_missing(self) -> None:
        """Category 6b: Multi-part query where one part is omitted."""
        checker = AnswerCompletenessChecker()
        plan = QueryPlan(
            original_query="Explain the architecture of transformers and how self-attention works.",
            query_type=QueryType.MULTI_PART,
            sub_questions=["architecture of transformers", "how self-attention works"],
        )
        incomplete_answer = "The architecture of transformers uses encoder-decoder blocks."
        res = checker.check_completeness(plan, incomplete_answer)
        assert res.is_complete is False
        assert "how self-attention works" in res.missing_aspects


class TestPhase21MultiPageSynthesisAndProvenance:
    """Category 7, 11, 12: Multi-page synthesis, citation preservation, and provenance tracking."""

    def test_multipage_evidence_and_citation_preservation(self, generic_evidence: List[GenerationEvidenceItem]) -> None:
        """Category 7, 11, 12: Validates synthesis across multiple pages with intact provenance."""
        mock_nli = MockNLIModel(default_score=NLIScores(entailment=0.96, contradiction=0.01, neutral=0.03))
        grounding_service = GroundingService(model=mock_nli)

        claims = [
            ExtractedClaim(claim_id="c1", claim_text="The speed of sound in dry air is 343 meters per second.", claim_index=0),
            ExtractedClaim(claim_id="c2", claim_text="Photosynthesis stores light energy in glucose.", claim_index=1),
        ]

        report = grounding_service.validate_claims(claims=claims, evidence=generic_evidence)
        assert report.overall_status == "grounded"
        assert report.supported_claims == 2

        citation_service = CitationService()
        cite_res = citation_service.build_citations(report, expected_document_id="doc_test_01")
        assert len(cite_res.citations) == 2

        # Provenance verification: Page 1 and Page 5 are preserved
        pages = {c.page_number for c in cite_res.citations}
        chunks = {c.chunk_id for c in cite_res.citations}
        assert pages == {1, 5}
        assert chunks == {"chunk_page1_0", "chunk_page5_1"}


class TestPhase21SafetyAndControlledRegeneration:
    """Category 8, 9, 10: Unsupported refusal, unanswerable refusal, and bounded regeneration."""

    def test_unsupported_answer_safely_refused(self, generic_evidence: List[GenerationEvidenceItem]) -> None:
        """Category 8: Answers asserting unsupported facts must be refused under safe policy."""
        mock_nli = MockNLIModel(
            score_map={
                (generic_evidence[0].source_text, "Sound travels faster in cold air than in warm air."): NLIScores(
                    entailment=0.01, contradiction=0.92, neutral=0.07
                )
            },
            default_score=NLIScores(entailment=0.05, contradiction=0.85, neutral=0.10),
        )
        grounding_service = GroundingService(model=mock_nli)
        claims = [
            ExtractedClaim(claim_id="c1", claim_text="Sound travels faster in cold air than in warm air.", claim_index=0)
        ]
        report = grounding_service.validate_claims(claims=claims, evidence=generic_evidence)
        assert report.overall_status in ("contradicted", "unsupported")

    def test_unanswerable_query_refused_due_to_low_evidence_similarity(self) -> None:
        """Category 9: Queries where retrieved evidence has negligible similarity are safely refused."""
        mock_search = MagicMock(spec=SearchService)
        # Returned candidates have low similarity (e.g. 0.05 < 0.20)
        low_sim_results = [
            SearchResult(
                rank=1,
                chunk_id="chunk_unrelated",
                document_id="doc_test_01",
                page_number=3,
                text="Some irrelevant topic.",
                similarity_score=0.05,
            )
        ]
        mock_search.search.return_value = SearchResponse(
            query="What is quantum teleportation?",
            results=low_sim_results,
            total_results=len(low_sim_results),
            candidate_count=1,
            reranking_applied=False,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_grounding = MagicMock(spec=GroundingService)

        service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        resp = service.answer_with_grounding(
            GroundedAnswerRequest(query="What is quantum teleportation?", document_id="doc_test_01")
        )

        assert resp.answerable is False
        assert resp.grounded is False
        assert resp.answer is None
        assert resp.completeness_status == "insufficient_evidence"
        # Generation was not even invoked
        mock_gen.generate_answer.assert_not_called()

    def test_bounded_regeneration_on_incomplete_answer(self, generic_evidence: List[GenerationEvidenceItem]) -> None:
        """Category 10: Incomplete answers trigger at most 1 targeted retry (max 2 attempts total)."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="Compare A and B",
            results=[
                SearchResult(
                    rank=1,
                    chunk_id="chunk_page1_0",
                    document_id="doc_test_01",
                    page_number=1,
                    text="Concept A is fast. Concept B is accurate.",
                    similarity_score=0.85,
                )
            ],
            total_results=1,
            candidate_count=1,
            reranking_applied=False,
        )

        mock_gen = MagicMock(spec=GenerationService)
        # Attempt 1: only mentions A
        # Attempt 2: mentions both A and B
        mock_gen.generate_answer.side_effect = [
            GenerationResponse(
                query="Compare A and B",
                answer="Concept A is fast.",
                answerable=True,
                model_name="test_model",
                evidence=generic_evidence,
            ),
            GenerationResponse(
                query="Compare A and B",
                answer="Concept A is fast, whereas Concept B is accurate.",
                answerable=True,
                model_name="test_model",
                evidence=generic_evidence,
            ),
        ]

        mock_nli = MockNLIModel(default_score=NLIScores(entailment=0.98, contradiction=0.01, neutral=0.01))
        grounding_service = GroundingService(model=mock_nli)

        planner = QueryPlanner()
        mock_qu = MagicMock()
        mock_qu.analyze_query.return_value = QueryPlan(
            original_query="Compare A and B",
            query_type=QueryType.COMPARISON,
            comparison_aspects=["Concept A", "Concept B"],
        )

        service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=grounding_service,
            query_understanding_service=mock_qu,
        )

        resp = service.answer_with_grounding(GroundedAnswerRequest(query="Compare A and B", document_id="doc_test_01"))

        assert resp.answerable is True
        assert resp.grounded is True
        assert resp.regeneration_attempted is True
        assert resp.completeness_status == "complete"
        assert "Concept B is accurate" in resp.answer
        # Exactly 2 generation calls were made (Attempt 1 + Attempt 2)
        assert mock_gen.generate_answer.call_count == 2
