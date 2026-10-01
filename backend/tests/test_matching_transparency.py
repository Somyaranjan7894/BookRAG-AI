"""Unit and integration tests for Phase 18 Matching Board & Retrieval Transparency metadata contracts."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from app.schemas.generation import GenerationEvidenceItem
from app.schemas.grounding import GroundedAnswerRequest, GroundedAnswerResponse
from app.schemas.query_plan import QueryPlan, QueryType
from app.schemas.retrieval import RetrievalResult
from app.schemas.search import SearchResponse, SearchResult
from app.services.generation.evidence import EvidenceBuilder
from app.services.grounding.orchestrator import GroundedAnswerService
from app.services.search.query_search import QuerySearchService
from app.services.search.service import SearchService


class TestRetrievalTransparencyMetadata:
    """Verifies that retrieval provenance, ranking movement, and candidate metadata are accurately preserved."""

    def test_search_response_preserves_candidate_count_and_ranks(self) -> None:
        """Verify SearchResponse contains candidate_count and SearchResult contains original_rank and final rank."""
        results = [
            SearchResult(
                rank=1,
                original_rank=4,
                chunk_id="chunk_004",
                document_id="book_1",
                page_number=14,
                text="Gradient descent updates weights in the direction of steepest descent.",
                similarity_score=0.78,
                reranker_score=5.42,
                chunk_index=3,
            ),
            SearchResult(
                rank=2,
                original_rank=1,
                chunk_id="chunk_001",
                document_id="book_1",
                page_number=3,
                text="Neural networks are composed of layers of interconnected nodes.",
                similarity_score=0.91,
                reranker_score=3.15,
                chunk_index=0,
            ),
        ]

        response = SearchResponse(
            query="how does gradient descent work?",
            results=results,
            total_results=2,
            document_id="book_1",
            reranking_applied=True,
            candidate_count=20,
        )

        assert response.query == "how does gradient descent work?"
        assert response.total_results == 2
        assert response.candidate_count == 20
        assert response.reranking_applied is True
        assert len(response.results) == 2

        # Verify ranking movement fields
        first = response.results[0]
        assert first.rank == 1
        assert first.original_rank == 4
        # Deterministic ranking movement: original_rank - rank
        rank_delta = first.original_rank - first.rank
        assert rank_delta == 3  # Moved upward by 3 positions

        second = response.results[1]
        assert second.rank == 2
        assert second.original_rank == 1
        rank_delta_2 = second.original_rank - second.rank
        assert rank_delta_2 == -1  # Moved downward by 1 position

    def test_evidence_builder_preserves_original_rank(self) -> None:
        """Verify EvidenceBuilder._normalize_evidence_item copies original_rank from SearchResult or dict."""
        search_result = SearchResult(
            rank=2,
            original_rank=7,
            chunk_id="chunk_abc",
            document_id="doc_x",
            page_number=42,
            text="Attention is all you need.",
            similarity_score=0.85,
            reranker_score=4.90,
            chunk_index=15,
        )

        item = EvidenceBuilder._normalize_evidence_item(search_result, rank=1)
        assert item.rank == 1
        assert item.original_rank == 7
        assert item.chunk_id == "chunk_abc"
        assert item.page_number == 42
        assert item.similarity_score == 0.85
        assert item.reranker_score == 4.90

        # Dict input test
        dict_chunk = {
            "chunk_id": "chunk_dict",
            "document_id": "doc_x",
            "page_number": 9,
            "chunk_index": 2,
            "text": "Self-attention mechanism details.",
            "similarity_score": 0.77,
            "reranker_score": 2.11,
            "original_rank": 5,
        }
        item_from_dict = EvidenceBuilder._normalize_evidence_item(dict_chunk, rank=3)
        assert item_from_dict.rank == 3
        assert item_from_dict.original_rank == 5

    def test_grounded_answer_response_carries_retrieval_transparency_metadata(self) -> None:
        """Verify GroundedAnswerService attaches candidate_count and reranking_applied to GroundedAnswerResponse."""
        mock_query_plan = QueryPlan(
            original_query="What is backpropagation?",
            normalized_query="What is backpropagation?",
            query_type=QueryType.FACTUAL,
            retrieval_queries=["What is backpropagation?"],
        )

        mock_search_results = [
            SearchResult(
                rank=1,
                original_rank=3,
                chunk_id="chunk_bp",
                document_id="book_dl",
                page_number=5,
                text="Backpropagation calculates gradients efficiently using chain rule.",
                similarity_score=0.89,
                reranker_score=6.12,
                chunk_index=0,
            )
        ]

        mock_search_response = SearchResponse(
            query="What is backpropagation?",
            results=mock_search_results,
            total_results=1,
            document_id="book_dl",
            reranking_applied=True,
            candidate_count=20,
        )

        mock_search_svc = MagicMock(spec=SearchService)
        mock_query_search_svc = MagicMock(spec=QuerySearchService)
        mock_query_search_svc.search_with_plan.return_value = mock_search_response

        service = GroundedAnswerService(
            search_service=mock_search_svc,
            query_search_service=mock_query_search_svc,
        )

        # Mock the query understanding service
        mock_qu = MagicMock()
        mock_qu.analyze_query.return_value = mock_query_plan
        service._query_understanding_service = mock_qu

        # Mock generation service
        mock_gen_evidence = [
            GenerationEvidenceItem(
                rank=1,
                original_rank=3,
                chunk_id="chunk_bp",
                document_id="book_dl",
                page_number=5,
                chunk_index=0,
                source_text="Backpropagation calculates gradients efficiently using chain rule.",
                similarity_score=0.89,
                reranker_score=6.12,
            )
        ]
        mock_gen = MagicMock()
        mock_gen.generate_answer.return_value = MagicMock(
            answer="Backpropagation calculates gradients.",
            answerable=True,
            evidence=mock_gen_evidence,
        )
        service._generation_service = mock_gen

        # Mock claim decomposer & grounding
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = [
            MagicMock(claim_index=0, claim_text="Backpropagation calculates gradients.")
        ]
        service._claim_decomposer = mock_decomposer

        mock_grounding = MagicMock()
        mock_grounding.validate_claims.return_value = MagicMock(
            groundedness_score=1.0,
            total_claims=1,
            supported_claims=1,
            unsupported_claims=0,
            contradicted_claims=0,
            conflicted_claims=0,
            overall_status="grounded",
            claim_results=[],
            reason="All claims entailed.",
        )
        service._grounding_service = mock_grounding

        # Mock citation service
        mock_cit = MagicMock()
        mock_cit.build_citations.return_value = MagicMock(citations=[])
        service._citation_service = mock_cit

        request = GroundedAnswerRequest(
            query="What is backpropagation?",
            document_id="book_dl",
        )

        resp = service.answer_with_grounding(request)

        assert isinstance(resp, GroundedAnswerResponse)
        assert resp.candidate_count == 20
        assert resp.reranking_applied is True
        assert len(resp.evidence) == 1
        assert resp.evidence[0].original_rank == 3
        assert resp.evidence[0].rank == 1
        assert resp.evidence[0].similarity_score == 0.89
        assert resp.evidence[0].reranker_score == 6.12
