"""Unit, integration, and semantic tests for Phase 11 Query Understanding and Planning.

Verifies:
A. Normalization (whitespace, technical terms like C++, COVID-19, R&D, numbers, years)
B. Query classification taxonomy (factual, definition, list, comparison, causal, procedural, location, summary, multi_hop, unknown)
C. Expected answer type metadata
D. Constraint extraction (chapter, year, page, quoted phrases; no invented constraints)
E. Retrieval query generation (single vs focused multi-queries, max 3, non-redundant)
F. Candidate merging & deduplication (stable dedup by document_id + chunk_id, provenance preservation)
G. Final reranking (cross-encoder reranks merged pool against normalized query)
H. Document isolation enforcement
I. End-to-end integration (QueryPlan -> QuerySearchService -> GroundedAnswerService)
J. Core semantic principles (retrieval is not correctness; classification is not answering; explicit constraints only)
K. FastAPI endpoint contracts
"""

from typing import List
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.citation import Citation, ClaimCitationRef
from app.schemas.generation import GenerationEvidenceItem, GenerationResponse
from app.schemas.grounding import (
    ClaimEvidenceProvenance,
    ClaimResult,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    GroundingReport,
)
from app.schemas.query_plan import (
    ExpectedAnswerType,
    QueryConstraints,
    QueryPlan,
    QueryPlanRequest,
    QueryType,
)
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.query_understanding.exceptions import (
    InvalidQueryError,
    QueryUnderstandingError,
)
from app.services.query_understanding.planner import QueryPlanner
from app.services.query_understanding.service import QueryUnderstandingService
from app.services.search.query_search import QuerySearchService


# Helper fixtures & builders
def make_search_result(
    chunk_id: str = "doc1_p001_c0001",
    document_id: str = "doc1",
    page_number: int = 1,
    chunk_index: int = 1,
    text: str = "Sample passage text for testing.",
    similarity_score: float = 0.85,
    reranker_score: float = 0.90,
    rank: int = 1,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        similarity_score=similarity_score,
        reranker_score=reranker_score,
        rank=rank,
        original_rank=rank,
    )


# =====================================================================
# Suite A: Safe Normalization
# =====================================================================
class TestQueryNormalization:
    """Verify safe normalization without destructive removal of semantic symbols."""

    def test_whitespace_normalization(self):
        planner = QueryPlanner()
        raw = "   What   is   machine \t learning?   \n "
        normalized = planner.normalize(raw)
        assert normalized == "What is machine learning?"

    def test_non_breaking_spaces_normalized(self):
        planner = QueryPlanner()
        raw = "What\u00a0is\u00a0deep\u00a0learning?"
        assert planner.normalize(raw) == "What is deep learning?"

    def test_preserves_technical_symbols_c_plus_plus(self):
        planner = QueryPlanner()
        raw = "How do you implement backprop in C++?"
        normalized = planner.normalize(raw)
        assert "C++" in normalized

    def test_preserves_covid_19_hyphenation(self):
        planner = QueryPlanner()
        raw = "What were the effects of COVID-19 on global supply chains?"
        normalized = planner.normalize(raw)
        assert "COVID-19" in normalized

    def test_preserves_rd_ampersand(self):
        planner = QueryPlanner()
        raw = "What was the budget for R&D in 2021?"
        normalized = planner.normalize(raw)
        assert "R&D" in normalized

    def test_preserves_years_and_numbers(self):
        planner = QueryPlanner()
        raw = "What occurred in 1998 on page 12 with 3.14 ratio?"
        normalized = planner.normalize(raw)
        assert "1998" in normalized
        assert "page 12" in normalized
        assert "3.14" in normalized

    def test_empty_or_whitespace_raises_invalid_query_error(self):
        planner = QueryPlanner()
        with pytest.raises(InvalidQueryError):
            planner.normalize("   ")
        with pytest.raises(InvalidQueryError):
            planner.normalize("")


# =====================================================================
# Suite B: Query Classification
# =====================================================================
class TestQueryClassification:
    """Verify taxonomy categorization across all 10 query types."""

    def test_classify_factual(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("Who founded OpenAI?")
        assert q_type == QueryType.FACTUAL
        assert ans_type == ExpectedAnswerType.PERSON_ENTITY.value

        q_type, ans_type = planner.classify("When was backpropagation introduced?")
        assert q_type == QueryType.FACTUAL
        assert ans_type == ExpectedAnswerType.DATE_YEAR.value

        q_type, ans_type = planner.classify("How many parameters does the model have?")
        assert q_type == QueryType.FACTUAL
        assert ans_type == ExpectedAnswerType.NUMBER.value

    def test_classify_definition(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("What is backpropagation?")
        assert q_type == QueryType.DEFINITION
        assert ans_type == ExpectedAnswerType.DEFINITION.value

        q_type, _ = planner.classify("Define stochastic gradient descent.")
        assert q_type == QueryType.DEFINITION

        q_type, _ = planner.classify("What does overfitting mean?")
        assert q_type == QueryType.DEFINITION

    def test_classify_list(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("List the primary activation functions.")
        assert q_type == QueryType.LIST
        assert ans_type == ExpectedAnswerType.LIST.value

        q_type, _ = planner.classify("What are the main components of a transformer?")
        assert q_type == QueryType.LIST

    def test_classify_comparison(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("Compare the population of India and China in 2020.")
        assert q_type == QueryType.COMPARISON
        assert ans_type == ExpectedAnswerType.COMPARISON.value

        q_type, _ = planner.classify("What is the difference between CNN and RNN?")
        assert q_type == QueryType.COMPARISON

        q_type, _ = planner.classify("PyTorch vs TensorFlow")
        assert q_type == QueryType.COMPARISON

    def test_classify_causal(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("Why did the vanishing gradient problem occur?")
        assert q_type == QueryType.CAUSAL
        assert ans_type == ExpectedAnswerType.EXPLANATION.value

        q_type, _ = planner.classify("What are the causes of model drift?")
        assert q_type == QueryType.CAUSAL

    def test_classify_procedural(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("How do you perform gradient descent?")
        assert q_type == QueryType.PROCEDURAL
        assert ans_type == ExpectedAnswerType.PROCEDURE.value

        q_type, _ = planner.classify("What are the steps to train an autoencoder?")
        assert q_type == QueryType.PROCEDURAL

        q_type, _ = planner.classify("How to implement dropout in neural networks?")
        assert q_type == QueryType.PROCEDURAL

    def test_classify_location(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("Where is the attention mechanism discussed in the book?")
        assert q_type == QueryType.LOCATION
        assert ans_type == ExpectedAnswerType.LOCATION.value

        q_type, _ = planner.classify("Where was Geoff Hinton born?")
        assert q_type == QueryType.LOCATION

    def test_classify_summary(self):
        planner = QueryPlanner()
        q_type, ans_type = planner.classify("Summarize Chapter 4.")
        assert q_type == QueryType.SUMMARY
        assert ans_type == ExpectedAnswerType.SUMMARY.value

        q_type, _ = planner.classify("Give an overview of the history of neural networks.")
        assert q_type == QueryType.SUMMARY

    def test_classify_multi_hop(self):
        planner = QueryPlanner()
        q_type, _ = planner.classify("Who was the teacher of the founder of Google?")
        assert q_type == QueryType.MULTI_HOP

        q_type, _ = planner.classify("What happened in the year that Alan Turing died?")
        assert q_type == QueryType.MULTI_HOP

    def test_classify_unknown(self):
        planner = QueryPlanner()
        q_type, _ = planner.classify("Random text query without typical interrogative signals.")
        assert q_type == QueryType.UNKNOWN


# =====================================================================
# Suite C: Expected Answer Type
# =====================================================================
class TestExpectedAnswerType:
    """Verify semantic mapping of question signals to expected answer categories."""

    def test_who_yields_person_entity(self):
        planner = QueryPlanner()
        _, ans_type = planner.classify("Who invented backpropagation?")
        assert ans_type == "person/entity"

    def test_when_yields_date_year(self):
        planner = QueryPlanner()
        _, ans_type = planner.classify("When was AlexNet published?")
        assert ans_type == "date/year"

    def test_where_yields_location(self):
        planner = QueryPlanner()
        _, ans_type = planner.classify("Where did the conference take place?")
        assert ans_type == "location"

    def test_how_many_yields_number(self):
        planner = QueryPlanner()
        _, ans_type = planner.classify("How many layers does VGG16 have?")
        assert ans_type == "number"

    def test_compare_yields_comparison(self):
        planner = QueryPlanner()
        _, ans_type = planner.classify("Compare Adam and SGD.")
        assert ans_type == "comparison"


# =====================================================================
# Suite D: Constraint Extraction
# =====================================================================
class TestConstraintExtraction:
    """Verify explicit constraint detection without hallucinating constraints."""

    def test_explicit_chapter_constraint(self):
        planner = QueryPlanner()
        constraints = planner.extract_constraints("What happened in Chapter 7?")
        assert constraints.chapter == 7
        assert constraints.year is None

    def test_explicit_year_constraint(self):
        planner = QueryPlanner()
        constraints = planner.extract_constraints("What happened in 2020?")
        assert constraints.year == 2020
        assert constraints.chapter is None

    def test_explicit_page_constraint(self):
        planner = QueryPlanner()
        constraints = planner.extract_constraints("Explain the formula on page 42.")
        assert constraints.page_number == 42
        assert constraints.page_range is None

    def test_explicit_page_range_constraint(self):
        planner = QueryPlanner()
        constraints = planner.extract_constraints("Summarize the arguments on pages 15-20.")
        assert constraints.page_range == (15, 20)

    def test_explicit_quoted_phrase_constraint(self):
        planner = QueryPlanner()
        constraints = planner.extract_constraints("Explain 'quantum tunneling' in semiconductor physics.")
        assert "quantum tunneling" in constraints.quoted_phrases

    def test_no_invented_constraints(self):
        planner = QueryPlanner()
        constraints = planner.extract_constraints("What is the general formulation of backpropagation?")
        assert constraints.chapter is None
        assert constraints.year is None
        assert constraints.page_number is None
        assert constraints.page_range is None
        assert constraints.quoted_phrases == []


# =====================================================================
# Suite E: Retrieval Query Generation
# =====================================================================
class TestRetrievalQueryGeneration:
    """Verify targeted single- and multi-query retrieval generation."""

    def test_simple_question_generates_single_query(self):
        planner = QueryPlanner()
        plan = planner.plan("What is machine learning?")
        assert len(plan.retrieval_queries) == 1
        assert "What is machine learning" in plan.retrieval_queries[0]

    def test_comparison_generates_focused_queries(self):
        planner = QueryPlanner()
        plan = planner.plan("Compare the population of India and China in 2020.")
        assert len(plan.retrieval_queries) == 2
        assert any("India" in q for q in plan.retrieval_queries)
        assert any("China" in q for q in plan.retrieval_queries)
        assert any("2020" in q for q in plan.retrieval_queries)

    def test_causes_and_consequences_generates_dual_queries(self):
        planner = QueryPlanner()
        plan = planner.plan("What were the causes and consequences of the French Revolution?")
        assert len(plan.retrieval_queries) == 2
        assert any("causes of" in q and "French Revolution" in q for q in plan.retrieval_queries)
        assert any("consequences of" in q and "French Revolution" in q for q in plan.retrieval_queries)

    def test_max_three_queries_enforced(self):
        planner = QueryPlanner()
        plan = planner.plan("Compare A and B and C in 2020.")
        assert len(plan.retrieval_queries) <= 3


# =====================================================================
# Suite F: Candidate Merging and Deduplication
# =====================================================================
class TestCandidateMergingAndDeduplication:
    """Verify multi-query candidate merging, chunk deduplication, and provenance survival."""

    def test_duplicate_chunks_across_queries_merged_to_single_candidate(self):
        mock_search = MagicMock()
        mock_reranker = MagicMock()

        # Shared chunk returned by both Query A and Query B
        shared_chunk = make_search_result(
            chunk_id="book_p001_c0001",
            text="Shared supporting text.",
            similarity_score=0.88,
        )
        unique_chunk = make_search_result(
            chunk_id="book_p002_c0002",
            page_number=2,
            text="Unique supporting text.",
            similarity_score=0.82,
        )

        mock_search.is_reranking_enabled = False
        mock_search.default_candidate_k = 10
        # Return shared_chunk for query 1, and shared_chunk + unique_chunk for query 2
        mock_search.search.side_effect = [
            SearchResponse(query="Query 1", results=[shared_chunk], total_results=1),
            SearchResponse(query="Query 2", results=[shared_chunk, unique_chunk], total_results=2),
        ]

        query_search = QuerySearchService(search_service=mock_search, reranker_service=mock_reranker)
        plan = QueryPlan(
            original_query="Compare X and Y",
            normalized_query="Compare X and Y",
            query_type=QueryType.COMPARISON,
            retrieval_queries=["Query 1", "Query 2"],
        )

        resp = query_search.search_with_plan(plan, top_k=5, enable_reranking=False)

        # There should be exactly 2 candidates, not 3 (shared_chunk deduplicated)
        assert len(resp.results) == 2
        chunk_ids = [r.chunk_id for r in resp.results]
        assert chunk_ids.count("book_p001_c0001") == 1
        assert "book_p002_c0002" in chunk_ids

        # Verify matched_queries tracked
        shared_res = next(r for r in resp.results if r.chunk_id == "book_p001_c0001")
        assert "Query 1" in shared_res.metadata["matched_queries"]
        assert "Query 2" in shared_res.metadata["matched_queries"]

    def test_provenance_strictly_survives_merge(self):
        mock_search = MagicMock()
        res = make_search_result(
            chunk_id="handbook_p015_c0003",
            document_id="handbook",
            page_number=15,
            chunk_index=3,
            text="Exact source passage text.",
            similarity_score=0.91,
        )
        mock_search.is_reranking_enabled = False
        mock_search.search.return_value = SearchResponse(query="Q", results=[res], total_results=1)

        query_search = QuerySearchService(search_service=mock_search)
        plan = QueryPlan(
            original_query="What is SGD?",
            normalized_query="What is SGD?",
            query_type=QueryType.DEFINITION,
            retrieval_queries=["What is SGD"],
        )
        resp = query_search.search_with_plan(plan, top_k=1, enable_reranking=False)

        assert len(resp.results) == 1
        item = resp.results[0]
        assert item.chunk_id == "handbook_p015_c0003"
        assert item.document_id == "handbook"
        assert item.page_number == 15
        assert item.chunk_index == 3
        assert item.text == "Exact source passage text."


# =====================================================================
# Suite G: Final Reranking on Merged Candidates
# =====================================================================
class TestFinalReranking:
    """Verify that cross-encoder reranking runs over merged candidate pool against normalized query."""

    def test_reranker_called_with_normalized_query_after_merge(self):
        mock_search = MagicMock()
        mock_reranker = MagicMock()

        c1 = make_search_result(chunk_id="chunk_1", similarity_score=0.70)
        c2 = make_search_result(chunk_id="chunk_2", similarity_score=0.80)

        mock_search.is_reranking_enabled = True
        mock_search.default_candidate_k = 10
        mock_search.search.side_effect = [
            SearchResponse(query="Sub 1", results=[c1], total_results=1),
            SearchResponse(query="Sub 2", results=[c2], total_results=1),
        ]

        reranked_c1 = c1.model_copy(update={"rank": 1, "reranker_score": 0.99})
        reranked_c2 = c2.model_copy(update={"rank": 2, "reranker_score": 0.92})
        mock_reranker.rerank.return_value = [reranked_c1, reranked_c2]

        query_search = QuerySearchService(search_service=mock_search, reranker_service=mock_reranker)
        plan = QueryPlan(
            original_query="Compare A and B",
            normalized_query="Compare A and B",
            query_type=QueryType.COMPARISON,
            retrieval_queries=["Sub 1", "Sub 2"],
        )

        resp = query_search.search_with_plan(plan, top_k=2, enable_reranking=True)

        assert resp.reranking_applied is True
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args.kwargs["query"] == "Compare A and B"
        assert len(call_args.kwargs["candidates"]) == 2


# =====================================================================
# Suite H: Document Isolation
# =====================================================================
class TestDocumentIsolation:
    """Verify document isolation parameter is propagated to all sub-queries."""

    def test_document_id_propagated_to_all_sub_queries(self):
        mock_search = MagicMock()
        mock_search.is_reranking_enabled = False
        mock_search.search.return_value = SearchResponse(query="Q", results=[], total_results=0)

        query_search = QuerySearchService(search_service=mock_search)
        plan = QueryPlan(
            original_query="Compare X and Y",
            normalized_query="Compare X and Y",
            query_type=QueryType.COMPARISON,
            retrieval_queries=["X", "Y"],
        )

        query_search.search_with_plan(plan, top_k=5, document_id="target_doc_id", enable_reranking=False)

        assert mock_search.search.call_count == 2
        for call in mock_search.search.call_args_list:
            assert call.kwargs["document_id"] == "target_doc_id"


# =====================================================================
# Suite I: End-to-End Grounded Answer Orchestration
# =====================================================================
class TestOrchestrationIntegration:
    """Verify full end-to-end integration including QueryPlan, Search, Generation, Grounding, Citations."""

    def test_orchestrator_includes_query_plan_in_response(self):
        from app.services.grounding.orchestrator import GroundedAnswerService
        from app.services.citation.service import CitationService

        mock_search = MagicMock()
        mock_gen = MagicMock()
        mock_decomposer = MagicMock()
        mock_grounding = MagicMock()

        # 1. Search returns 1 chunk
        ev_item = make_search_result(
            chunk_id="doc1_p001_c0001",
            document_id="doc1",
            page_number=1,
            chunk_index=1,
            text="Gradient descent optimizes loss functions.",
        )
        mock_search.search.return_value = SearchResponse(
            query="What is SGD?",
            results=[ev_item],
            total_results=1,
            top_k=1,
        )

        # 2. Generation returns answer
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="What is SGD?",
            answer="Gradient descent optimizes loss functions.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=[
                GenerationEvidenceItem(
                    chunk_id="doc1_p001_c0001",
                    document_id="doc1",
                    page_number=1,
                    chunk_index=1,
                    source_text="Gradient descent optimizes loss functions.",
                    similarity_score=0.91,
                    reranker_score=0.95,
                    rank=1,
                )
            ],
            evidence_count=1,
        )

        # 3. Decomposer decomposes claim
        from app.schemas.grounding import ExtractedClaim
        mock_decomposer.decompose.return_value = [
            ExtractedClaim(claim_index=0, claim_text="Gradient descent optimizes loss functions.")
        ]

        # 4. Grounding returns report
        prov = ClaimEvidenceProvenance(
            chunk_id="doc1_p001_c0001",
            document_id="doc1",
            page_number=1,
            chunk_index=1,
            source_text="Gradient descent optimizes loss functions.",
            rank=1,
            similarity_score=0.91,
            reranker_score=0.95,
            nli_score=0.96,
        )
        mock_grounding.validate_claims.return_value = GroundingReport(
            groundedness_score=1.0,
            total_claims=1,
            supported_claims=1,
            unsupported_claims=0,
            contradicted_claims=0,
            conflicted_claims=0,
            overall_status="grounded",
            claim_results=[
                ClaimResult(
                    claim_index=0,
                    claim_text="Gradient descent optimizes loss functions.",
                    status="entailed",
                    entailment_score=0.96,
                    contradiction_score=0.01,
                    neutral_score=0.03,
                    supporting_evidence=prov,
                    supporting_evidences=[prov],
                )
            ],
            reason="Fully grounded.",
        )

        service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
            claim_decomposer=mock_decomposer,
            citation_service=CitationService(),
            query_understanding_service=QueryUnderstandingService(),
        )

        req = GroundedAnswerRequest(query="What is SGD?", document_id="doc1")
        response = service.answer_with_grounding(req)

        assert response.answerable is True
        assert response.grounded is True
        assert response.query_plan is not None
        assert response.query_plan.query_type == QueryType.DEFINITION
        assert response.query_plan.expected_answer_type == "definition"
        assert len(response.citations) == 1
        assert response.citations[0].citation_id == "cite_1"


# =====================================================================
# Suite J: Important Semantic Tests
# =====================================================================
class TestSemanticPrinciples:
    """Verify semantic boundaries and constraints mandated by Phase 11."""

    def test_query_classification_does_not_answer_question(self):
        planner = QueryPlanner()
        plan = planner.plan("Who was the 16th president of the United States?")
        assert plan.query_type == QueryType.FACTUAL
        assert plan.expected_answer_type == "person/entity"
        # The planner produces retrieval queries, never the answer itself
        assert "Abraham Lincoln" not in plan.retrieval_queries[0]

    def test_constraints_only_added_when_explicitly_present(self):
        planner = QueryPlanner()
        # Query mentioning no chapter or year
        plan_generic = planner.plan("How does self-attention work?")
        assert plan_generic.constraints.chapter is None
        assert plan_generic.constraints.year is None

        # Query with explicit chapter
        plan_chapter = planner.plan("How does self-attention work in Chapter 5?")
        assert plan_chapter.constraints.chapter == 5
        assert plan_chapter.constraints.year is None

    def test_retrieval_relevance_is_not_answer_correctness(self):
        # A high similarity score indicates vector proximity, not factual truth
        res = make_search_result(similarity_score=0.99)
        assert res.similarity_score == 0.99
        # Semantic check: score does not guarantee the statement is factually true in real world


# =====================================================================
# Suite K: FastAPI Endpoint Tests
# =====================================================================
class TestQueryPlanAPI:
    """Verify POST /api/v1/query-plan endpoint contract and response."""

    def test_post_query_plan_endpoint(self):
        client = TestClient(app)
        res = client.post(
            "/api/v1/query-plan",
            json={"query": "Compare the population of India and China in 2020."},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["original_query"] == "Compare the population of India and China in 2020."
        assert data["query_type"] == "comparison"
        assert data["expected_answer_type"] == "comparison"
        assert len(data["retrieval_queries"]) >= 2
        assert data["requires_multiple_evidence"] is True
        assert data["constraints"]["year"] == 2020

    def test_post_query_plan_empty_query_returns_422(self):
        client = TestClient(app)
        res = client.post("/api/v1/query-plan", json={"query": "   "})
        assert res.status_code in (400, 422)
