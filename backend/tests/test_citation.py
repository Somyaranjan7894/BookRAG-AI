"""Unit and integration tests for Phase 10 Citation and Provenance Mapping.

Verifies the core architectural principle:
"The generator generates the answer. The system assigns citations from verified evidence."

Covers suites:
A. Citation creation (page, chunk, exact source text)
B. Deduplication (same chunk across claims reuses citation_id)
C. Multiple evidence (one claim supported by multiple chunks)
D. Unsupported claims (no authoritative citation assigned)
E. Contradictions (contradiction evidence preserved, not converted to support)
F. Conflicts (both supporting and contradicting evidence preserved)
G. Determinism (same input produces identical citation IDs)
H. Provenance retention (exact fields preserved)
I. Document isolation (cross-document evidence rejected)
J. Response schema validation (every claim citation ID exists in top-level citations)
K. Integration (generation + grounding + citation mapping orchestration)
"""

from typing import List
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.citation import Citation, CitationMappingResult, ClaimCitationRef
from app.schemas.generation import GenerationEvidenceItem, GenerationResponse
from app.schemas.grounding import (
    ClaimEvidenceProvenance,
    ClaimResult,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    GroundingReport,
)
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.citation.exceptions import (
    CitationError,
    DocumentIsolationError,
    InvalidCitationInputError,
)
from app.services.citation.service import CitationService
from app.services.grounding.orchestrator import GroundedAnswerService


# Helper fixtures & builders
def make_evidence_prov(
    chunk_id: str = "doc1_p001_c0001",
    document_id: str = "doc1",
    page_number: int = 1,
    chunk_index: int = 1,
    source_text: str = "Neural networks optimize objective functions using gradient descent.",
    rank: int = 1,
    similarity_score: float = 0.88,
    reranker_score: float = 0.95,
    nli_score: float = 0.92,
) -> ClaimEvidenceProvenance:
    return ClaimEvidenceProvenance(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        chunk_index=chunk_index,
        source_text=source_text,
        rank=rank,
        similarity_score=similarity_score,
        reranker_score=reranker_score,
        nli_score=nli_score,
    )


# =====================================================================
# Suite A: Citation Creation
# =====================================================================
class TestCitationCreation:
    """Verify basic citation object creation from verified grounded claims."""

    def test_valid_grounded_claim_creates_citation(self):
        service = CitationService()
        ev = make_evidence_prov(
            chunk_id="book_p012_c0003",
            document_id="book_ml",
            page_number=12,
            chunk_index=3,
            source_text="Backpropagation calculates gradients via the chain rule.",
        )
        claim = ClaimResult(
            claim_index=0,
            claim_text="Backpropagation computes gradients.",
            status="entailed",
            entailment_score=0.94,
            contradiction_score=0.01,
            neutral_score=0.05,
            supporting_evidence=ev,
            supporting_evidences=[ev],
        )

        result = service.build_citations([claim], expected_document_id="book_ml")

        assert len(result.citations) == 1
        cit = result.citations[0]
        assert cit.citation_id == "cite_1"
        assert cit.document_id == "book_ml"
        assert cit.chunk_id == "book_p012_c0003"
        assert cit.page_number == 12
        assert cit.chunk_index == 3
        assert cit.source_text == "Backpropagation calculates gradients via the chain rule."

        # Verify claim reference
        assert len(claim.citations) == 1
        ref = claim.citations[0]
        assert ref.citation_id == "cite_1"
        assert ref.relation == "supports"
        assert ref.page_number == 12

    def test_empty_claims_returns_empty_result(self):
        service = CitationService()
        result = service.build_citations([])
        assert result.citations == []
        assert result.claim_citations == {}

    def test_none_input_raises_invalid_citation_input_error(self):
        service = CitationService()
        with pytest.raises(InvalidCitationInputError):
            service.build_citations(None)  # type: ignore


# =====================================================================
# Suite B: Deduplication
# =====================================================================
class TestCitationDeduplication:
    """Verify deduplication when multiple claims are supported by the same chunk."""

    def test_same_chunk_supports_multiple_claims_reuses_citation_id(self):
        service = CitationService()
        ev1 = make_evidence_prov(
            chunk_id="doc1_p005_c0002",
            document_id="doc1",
            page_number=5,
            chunk_index=2,
            source_text="Adam optimizer combines momentum and RMSProp.",
        )

        claim1 = ClaimResult(
            claim_index=0,
            claim_text="Adam uses momentum.",
            status="entailed",
            entailment_score=0.91,
            contradiction_score=0.02,
            neutral_score=0.07,
            supporting_evidence=ev1,
            supporting_evidences=[ev1],
        )
        claim2 = ClaimResult(
            claim_index=1,
            claim_text="Adam incorporates RMSProp.",
            status="entailed",
            entailment_score=0.89,
            contradiction_score=0.03,
            neutral_score=0.08,
            supporting_evidence=ev1,
            supporting_evidences=[ev1],
        )

        result = service.build_citations([claim1, claim2], expected_document_id="doc1")

        # Top-level collection has only ONE citation
        assert len(result.citations) == 1
        assert result.citations[0].citation_id == "cite_1"
        assert result.citations[0].chunk_id == "doc1_p005_c0002"

        # Both claims reference cite_1
        assert claim1.citations[0].citation_id == "cite_1"
        assert claim2.citations[0].citation_id == "cite_1"

    def test_different_chunks_produce_different_citation_ids(self):
        service = CitationService()
        ev1 = make_evidence_prov(chunk_id="doc1_p001_c0001", page_number=1)
        ev2 = make_evidence_prov(chunk_id="doc1_p002_c0001", page_number=2)

        claim1 = ClaimResult(
            claim_index=0,
            claim_text="Claim one.",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev1,
            supporting_evidences=[ev1],
        )
        claim2 = ClaimResult(
            claim_index=1,
            claim_text="Claim two.",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev2,
            supporting_evidences=[ev2],
        )

        result = service.build_citations([claim1, claim2], expected_document_id="doc1")

        assert len(result.citations) == 2
        assert result.citations[0].citation_id == "cite_1"
        assert result.citations[0].chunk_id == "doc1_p001_c0001"
        assert result.citations[1].citation_id == "cite_2"
        assert result.citations[1].chunk_id == "doc1_p002_c0001"


# =====================================================================
# Suite C: Multiple Evidence per Claim
# =====================================================================
class TestMultipleEvidence:
    """Verify one claim supported by multiple chunks."""

    def test_one_claim_supported_by_multiple_chunks(self):
        service = CitationService()
        ev1 = make_evidence_prov(chunk_id="doc1_p003_c0001", page_number=3)
        ev2 = make_evidence_prov(chunk_id="doc1_p004_c0002", page_number=4)

        claim = ClaimResult(
            claim_index=0,
            claim_text="Transformers use multi-head attention and positional encodings.",
            status="entailed",
            entailment_score=0.95,
            contradiction_score=0.01,
            neutral_score=0.04,
            supporting_evidence=ev1,
            supporting_evidences=[ev1, ev2],
        )

        result = service.build_citations([claim], expected_document_id="doc1")

        assert len(result.citations) == 2
        assert len(claim.citations) == 2
        assert claim.citations[0].citation_id == "cite_1"
        assert claim.citations[1].citation_id == "cite_2"
        assert claim.citations[0].page_number == 3
        assert claim.citations[1].page_number == 4


# =====================================================================
# Suite D: Unsupported Claims
# =====================================================================
class TestUnsupportedClaims:
    """Verify unsupported claims never receive authoritative citations."""

    def test_unsupported_claim_has_no_citations(self):
        service = CitationService()
        ev = make_evidence_prov(chunk_id="doc1_p001_c0001", page_number=1)
        claim = ClaimResult(
            claim_index=0,
            claim_text="The earth is flat.",
            status="unsupported",
            entailment_score=0.15,
            contradiction_score=0.10,
            neutral_score=0.75,
            supporting_evidence=ev,
            supporting_evidences=[ev],
        )

        result = service.build_citations([claim], expected_document_id="doc1")

        assert len(result.citations) == 0
        assert claim.citations == []
        assert claim.contradicting_citations == []
        assert result.claim_citations[0] == []


# =====================================================================
# Suite E & F: Contradictions and Conflicts
# =====================================================================
class TestContradictionsAndConflicts:
    """Verify contradicted and conflicted claims retain diagnostic citations."""

    def test_contradicted_claim_preserves_contradicting_citation_only(self):
        service = CitationService()
        ev_contra = make_evidence_prov(
            chunk_id="doc1_p020_c0001",
            page_number=20,
            source_text="Transformers strictly do not use recurrence.",
        )
        claim = ClaimResult(
            claim_index=0,
            claim_text="Transformers use recurrent cells.",
            status="contradicted",
            entailment_score=0.05,
            contradiction_score=0.92,
            neutral_score=0.03,
            supporting_evidence=None,
            supporting_evidences=[],
            contradicting_evidence=[ev_contra],
        )

        result = service.build_citations([claim], expected_document_id="doc1")

        # Top-level contains the contradicting evidence
        assert len(result.citations) == 1
        assert result.citations[0].citation_id == "cite_1"
        assert result.citations[0].page_number == 20

        # Claim has NO supporting citations (never convert to support)
        assert claim.citations == []
        # Claim preserves contradicting citation reference
        assert len(claim.contradicting_citations) == 1
        assert claim.contradicting_citations[0].citation_id == "cite_1"
        assert claim.contradicting_citations[0].relation == "contradicts"

    def test_conflicted_claim_preserves_both_supporting_and_contradicting(self):
        service = CitationService()
        ev_sup = make_evidence_prov(
            chunk_id="doc1_p012_c0001",
            page_number=12,
            source_text="The company expanded to Europe in 2005.",
        )
        ev_contra = make_evidence_prov(
            chunk_id="doc1_p030_c0002",
            page_number=30,
            source_text="The company never operated in Europe before 2010.",
        )

        claim = ClaimResult(
            claim_index=0,
            claim_text="The company was in Europe in 2005.",
            status="conflicted",
            entailment_score=0.88,
            contradiction_score=0.86,
            neutral_score=0.02,
            supporting_evidence=ev_sup,
            supporting_evidences=[ev_sup],
            contradicting_evidence=[ev_contra],
        )

        result = service.build_citations([claim], expected_document_id="doc1")

        # Top level has both unique chunks
        assert len(result.citations) == 2
        assert result.citations[0].citation_id == "cite_1"
        assert result.citations[0].page_number == 12
        assert result.citations[1].citation_id == "cite_2"
        assert result.citations[1].page_number == 30

        # Claim has supporting citation
        assert len(claim.citations) == 1
        assert claim.citations[0].citation_id == "cite_1"
        assert claim.citations[0].relation == "supports"

        # Claim has contradicting citation
        assert len(claim.contradicting_citations) == 1
        assert claim.contradicting_citations[0].citation_id == "cite_2"
        assert claim.contradicting_citations[0].relation == "contradicts"


# =====================================================================
# Suite G: Determinism
# =====================================================================
class TestDeterminism:
    """Verify that multiple runs with identical input produce identical output."""

    def test_deterministic_citation_generation(self):
        service = CitationService()
        ev1 = make_evidence_prov(chunk_id="doc1_p001_c0001", page_number=1)
        ev2 = make_evidence_prov(chunk_id="doc1_p002_c0002", page_number=2)

        def make_claims():
            c1 = ClaimResult(
                claim_index=0,
                claim_text="A",
                status="entailed",
                entailment_score=0.9,
                contradiction_score=0.0,
                neutral_score=0.1,
                supporting_evidence=ev1,
                supporting_evidences=[ev1],
            )
            c2 = ClaimResult(
                claim_index=1,
                claim_text="B",
                status="entailed",
                entailment_score=0.9,
                contradiction_score=0.0,
                neutral_score=0.1,
                supporting_evidence=ev2,
                supporting_evidences=[ev2],
            )
            return [c1, c2]

        res1 = service.build_citations(make_claims(), expected_document_id="doc1")
        res2 = service.build_citations(make_claims(), expected_document_id="doc1")

        assert [c.citation_id for c in res1.citations] == [c.citation_id for c in res2.citations]
        assert [c.chunk_id for c in res1.citations] == [c.chunk_id for c in res2.citations]


# =====================================================================
# Suite H: Provenance Retention
# =====================================================================
class TestProvenanceRetention:
    """Verify exact provenance fields are retained without alteration."""

    def test_exact_provenance_retention(self):
        service = CitationService()
        exact_text = "   Unedited passage with leading and trailing spaces   "
        ev = make_evidence_prov(
            chunk_id="doc_deep_p042_c0007",
            document_id="doc_deep",
            page_number=42,
            chunk_index=7,
            source_text=exact_text,
            rank=3,
            similarity_score=0.7712,
            reranker_score=0.8845,
        )
        claim = ClaimResult(
            claim_index=0,
            claim_text="Test claim",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev,
            supporting_evidences=[ev],
        )

        result = service.build_citations([claim], expected_document_id="doc_deep")
        cit = result.citations[0]

        assert cit.document_id == "doc_deep"
        assert cit.chunk_id == "doc_deep_p042_c0007"
        assert cit.page_number == 42
        assert cit.chunk_index == 7
        assert cit.source_text == exact_text  # exact text preserved!
        assert cit.similarity_score == 0.7712
        assert cit.reranker_score == 0.8845
        assert cit.evidence_rank == 3


# =====================================================================
# Suite I: Document Isolation
# =====================================================================
class TestDocumentIsolation:
    """Verify cross-document evidence is strictly rejected."""

    def test_mismatched_expected_document_raises_error(self):
        service = CitationService()
        ev = make_evidence_prov(document_id="wrong_doc")
        claim = ClaimResult(
            claim_index=0,
            claim_text="Claim",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev,
            supporting_evidences=[ev],
        )

        with pytest.raises(DocumentIsolationError):
            service.build_citations([claim], expected_document_id="target_doc")

    def test_inconsistent_documents_without_expected_id_raises_error(self):
        service = CitationService()
        ev1 = make_evidence_prov(chunk_id="docA_c1", document_id="docA")
        ev2 = make_evidence_prov(chunk_id="docB_c1", document_id="docB")

        claim1 = ClaimResult(
            claim_index=0,
            claim_text="Claim A",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev1,
            supporting_evidences=[ev1],
        )
        claim2 = ClaimResult(
            claim_index=1,
            claim_text="Claim B",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev2,
            supporting_evidences=[ev2],
        )

        with pytest.raises(DocumentIsolationError):
            service.build_citations([claim1, claim2], expected_document_id=None)


# =====================================================================
# Suite J: Response Schema Validation
# =====================================================================
class TestResponseSchema:
    """Verify that every claim-level citation reference points to an existing top-level citation."""

    def test_claim_citations_exist_in_top_level_citations(self):
        service = CitationService()
        ev1 = make_evidence_prov(chunk_id="doc1_p001_c0001", page_number=1)
        ev2 = make_evidence_prov(chunk_id="doc1_p002_c0002", page_number=2)

        c1 = ClaimResult(
            claim_index=0,
            claim_text="C1",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev1,
            supporting_evidences=[ev1],
        )
        c2 = ClaimResult(
            claim_index=1,
            claim_text="C2",
            status="entailed",
            entailment_score=0.9,
            contradiction_score=0.0,
            neutral_score=0.1,
            supporting_evidence=ev2,
            supporting_evidences=[ev2],
        )

        result = service.build_citations([c1, c2], expected_document_id="doc1")
        top_ids = {c.citation_id for c in result.citations}

        for claim in [c1, c2]:
            for ref in claim.citations:
                assert ref.citation_id in top_ids


# =====================================================================
# Suite K: Orchestration Integration
# =====================================================================
class TestOrchestrationIntegration:
    """Test full pipeline integration with GroundedAnswerService."""

    def test_grounded_answer_includes_citations(self):
        mock_search = MagicMock()
        mock_gen = MagicMock()
        mock_decomposer = MagicMock()
        mock_grounding = MagicMock()

        # 1. Search returns 1 result
        mock_search.search.return_value = SearchResponse(
            query="What is SGD?",
            results=[
                SearchResult(
                    document_id="ml_book",
                    chunk_id="ml_book_p001_c0001",
                    page_number=1,
                    chunk_index=1,
                    text="Stochastic gradient descent updates weights per batch.",
                    similarity_score=0.91,
                    reranker_score=0.95,
                    rank=1,
                )
            ],
            total_results=1,
            top_k=1,
        )

        # 2. Generation returns answer
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="What is SGD?",
            answer="Stochastic gradient descent updates weights per batch.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=[
                GenerationEvidenceItem(
                    chunk_id="ml_book_p001_c0001",
                    document_id="ml_book",
                    page_number=1,
                    chunk_index=1,
                    source_text="Stochastic gradient descent updates weights per batch.",
                    similarity_score=0.91,
                    reranker_score=0.95,
                    rank=1,
                )
            ],
            evidence_count=1,
        )

        # 3. Decomposer decomposes 1 claim
        from app.schemas.grounding import ExtractedClaim
        mock_decomposer.decompose.return_value = [
            ExtractedClaim(claim_index=0, claim_text="Stochastic gradient descent updates weights per batch.")
        ]

        # 4. Grounding returns report with 1 entailed claim
        ev = make_evidence_prov(
            chunk_id="ml_book_p001_c0001",
            document_id="ml_book",
            page_number=1,
            chunk_index=1,
            source_text="Stochastic gradient descent updates weights per batch.",
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
                    claim_text="Stochastic gradient descent updates weights per batch.",
                    status="entailed",
                    entailment_score=0.95,
                    contradiction_score=0.01,
                    neutral_score=0.04,
                    supporting_evidence=ev,
                    supporting_evidences=[ev],
                )
            ],
            reason="Fully grounded.",
        )

        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
            claim_decomposer=mock_decomposer,
            citation_service=CitationService(),
        )

        req = GroundedAnswerRequest(query="What is SGD?", document_id="ml_book")
        response = orchestrator.answer_with_grounding(req)

        assert response.answerable is True
        assert response.grounded is True
        assert len(response.citations) == 1
        assert response.citations[0].citation_id == "cite_1"
        assert response.citations[0].page_number == 1
        assert response.citations[0].chunk_id == "ml_book_p001_c0001"
        assert response.claims[0].citations[0].citation_id == "cite_1"


# =====================================================================
# FastAPI Endpoint Integration
# =====================================================================
class TestFastAPIEndpoint:
    """Verify HTTP contract and JSON serialization for /api/v1/grounded-answer."""

    def test_endpoint_returns_citations(self):
        ev = make_evidence_prov(
            chunk_id="doc1_p010_c0001",
            document_id="doc1",
            page_number=10,
            chunk_index=1,
            source_text="Passage text.",
        )
        mock_response = GroundedAnswerResponse(
            query="Test query",
            answer="Grounded answer text.",
            answerable=True,
            grounded=True,
            groundedness_score=1.0,
            grounding_status="grounded",
            claims=[
                ClaimResult(
                    claim_index=0,
                    claim_text="Grounded answer text.",
                    status="entailed",
                    entailment_score=0.95,
                    contradiction_score=0.01,
                    neutral_score=0.04,
                    supporting_evidence=ev,
                    supporting_evidences=[ev],
                    citations=[
                        ClaimCitationRef(
                            citation_id="cite_1",
                            document_id="doc1",
                            chunk_id="doc1_p010_c0001",
                            page_number=10,
                            chunk_index=1,
                            relation="supports",
                        )
                    ],
                )
            ],
            citations=[
                Citation(
                    citation_id="cite_1",
                    document_id="doc1",
                    chunk_id="doc1_p010_c0001",
                    page_number=10,
                    chunk_index=1,
                    source_text="Passage text.",
                )
            ],
        )

        from app.api.v1.endpoints.grounded_answer import get_grounded_answer_service
        mock_service = MagicMock()
        mock_service.answer_with_grounding.return_value = mock_response
        app.dependency_overrides[get_grounded_answer_service] = lambda: mock_service

        try:
            client = TestClient(app)
            res = client.post("/api/v1/grounded-answer", json={"query": "Test query", "document_id": "doc1"})

            assert res.status_code == 200
            data = res.json()
            assert data["grounded"] is True
            assert len(data["citations"]) == 1
            assert data["citations"][0]["citation_id"] == "cite_1"
            assert data["citations"][0]["page_number"] == 10
            assert data["claims"][0]["citations"][0]["citation_id"] == "cite_1"
        finally:
            app.dependency_overrides.clear()
