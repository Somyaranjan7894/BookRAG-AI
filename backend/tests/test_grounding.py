"""Comprehensive unit, integration, and API tests for Phase 9 Groundedness & Hallucination Control.

Verifies:
A. Claim decomposition (deterministic sentence-splitting, normalization, filtering, indexing)
B. NLI label mapping (dynamic id2label discovery, order independence, case-insensitivity, error handling)
C. Claim/evidence pair construction (premise/hypothesis orientation, provenance retention)
D. Claim classification (entailed, contradicted, unsupported, conflicted)
E. Groundedness aggregation (scores, claim counts, overall status)
F. Threshold behavior (exact equality, above, below, custom configurations)
G. Safe final decision policy (safe acceptance, safe refusal, provenance preservation)
H. Service orchestration (Search -> Generation -> Grounding -> Safe Decision)
I. API tests (POST /api/v1/grounded-answer contracts, HTTP status codes, safe refusal responses)
J. Regression tests (Phase 5 search, Phase 7 extractive QA, Phase 8 abstractive generation endpoints)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.v1.endpoints.generation import get_generation_service
from app.api.v1.endpoints.grounded_answer import get_grounded_answer_service
from app.api.v1.endpoints.search import get_search_service
from app.core.config import settings
from app.main import app
from app.schemas.generation import GenerationEvidenceItem, GenerationResponse
from app.schemas.grounding import (
    ClaimEvidenceProvenance,
    ClaimResult,
    ExtractedClaim,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    GroundingReport,
)
from app.schemas.search import SearchResponse, SearchResult
from app.services.generation.service import GenerationService
from app.services.grounding.claims import ClaimDecomposer
from app.services.grounding.exceptions import (
    InvalidGroundingConfigError,
    InvalidGroundingInputError,
    NLIInferenceError,
    NLILabelMappingError,
    NLIModelLoadError,
)
from app.services.grounding.model import NLIModel, NLIScores
from app.services.grounding.orchestrator import GroundedAnswerService
from app.services.grounding.service import GroundingService
from app.services.search.service import SearchService


# ==============================================================================
# Fast Deterministic Mock for NLIModel
# ==============================================================================

class MockNLIModel:
    """Fast deterministic mock for NLIModel avoiding neural weights download in unit tests."""

    def __init__(
        self,
        default_scores: Optional[NLIScores] = None,
        pair_scores_fn: Optional[Any] = None,
    ) -> None:
        self.model_name = "cross-encoder/nli-deberta-v3-base"
        self.target_device = "cpu"
        self.entailment_idx = 1
        self.contradiction_idx = 0
        self.neutral_idx = 2
        self.default_scores = default_scores or NLIScores(entailment=0.95, contradiction=0.02, neutral=0.03)
        self.pair_scores_fn = pair_scores_fn
        self.predict_calls: List[List[Tuple[str, str]]] = []

    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[NLIScores]:
        self.predict_calls.append(list(pairs))
        if self.pair_scores_fn:
            return [self.pair_scores_fn(premise, hypothesis) for premise, hypothesis in pairs]
        return [self.default_scores for _ in pairs]


# Sample test fixtures
@pytest.fixture
def sample_evidence_items() -> List[GenerationEvidenceItem]:
    return [
        GenerationEvidenceItem(
            rank=1,
            chunk_id="chunk_page_10_0",
            document_id="deep_learning_handbook",
            page_number=10,
            chunk_index=0,
            source_text="Backpropagation calculates gradients of the loss function with respect to weights using the chain rule.",
            similarity_score=0.88,
            reranker_score=4.25,
        ),
        GenerationEvidenceItem(
            rank=2,
            chunk_id="chunk_page_12_1",
            document_id="deep_learning_handbook",
            page_number=12,
            chunk_index=1,
            source_text="Rumelhart, Hinton, and Williams popularized backpropagation in 1986 for neural network training.",
            similarity_score=0.82,
            reranker_score=3.10,
        ),
    ]


# ==============================================================================
# Suite A: Claim Decomposition Tests
# ==============================================================================

class TestClaimDecomposition:
    """Verify deterministic sentence-level claim splitting, whitespace normalization, and filtering."""

    def test_single_sentence(self) -> None:
        decomposer = ClaimDecomposer(min_claim_length=3)
        text = "Backpropagation applies the chain rule recursively."
        claims = decomposer.decompose(text)
        assert len(claims) == 1
        assert claims[0].claim_index == 0
        assert claims[0].claim_text == "Backpropagation applies the chain rule recursively."

    def test_multiple_sentences_with_indexing(self) -> None:
        decomposer = ClaimDecomposer(min_claim_length=3)
        text = "Python was created by Guido in 1991. It was first released in 1991! Is it popular? Yes."
        claims = decomposer.decompose(text)
        assert len(claims) == 4
        assert [c.claim_index for c in claims] == [0, 1, 2, 3]
        assert claims[0].claim_text == "Python was created by Guido in 1991."
        assert claims[1].claim_text == "It was first released in 1991!"
        assert claims[2].claim_text == "Is it popular?"
        assert claims[3].claim_text == "Yes."

    def test_whitespace_normalization(self) -> None:
        decomposer = ClaimDecomposer(min_claim_length=3)
        text = "  Backpropagation   calculates \n  gradients.  \t  It uses\n\nthe chain rule. "
        claims = decomposer.decompose(text)
        assert len(claims) == 2
        assert claims[0].claim_text == "Backpropagation calculates gradients."
        assert claims[1].claim_text == "It uses the chain rule."

    def test_empty_and_short_fragments_discarded(self) -> None:
        decomposer = ClaimDecomposer(min_claim_length=5)
        # "Hi." has 3 chars -> discarded
        text = "Hi. Valid claim text here.   . OK. Another valid claim here."
        claims = decomposer.decompose(text)
        assert len(claims) == 2
        assert claims[0].claim_text == "Valid claim text here."
        assert claims[1].claim_text == "Another valid claim here."

    def test_none_or_empty_text_returns_empty_list(self) -> None:
        decomposer = ClaimDecomposer()
        assert decomposer.decompose(None) == []
        assert decomposer.decompose("") == []
        assert decomposer.decompose("   \n\t  ") == []


# ==============================================================================
# Suite B: NLI Label Mapping Tests
# ==============================================================================

class TestNLILabelMapping:
    """Verify dynamic id2label discovery and strict validation."""

    def test_standard_hf_mapping_deberta(self) -> None:
        """cross-encoder/nli-deberta-v3-base has 0: contradiction, 1: entailment, 2: neutral."""
        model = NLIModel.__new__(NLIModel)
        model.model_name = "test-nli"
        model._model = MagicMock()
        model._model.model.config.id2label = {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
        }
        model._discover_label_indices()
        assert model.contradiction_idx == 0
        assert model.entailment_idx == 1
        assert model.neutral_idx == 2

    def test_reordered_and_case_insensitive_mapping(self) -> None:
        """Ensure arbitrary order and uppercase labels are resolved safely."""
        model = NLIModel.__new__(NLIModel)
        model.model_name = "test-nli"
        model._model = MagicMock()
        model._model.model.config.id2label = {
            0: "ENTAILMENT",
            1: "NEUTRAL",
            2: "CONTRADICTION",
        }
        model._discover_label_indices()
        assert model.entailment_idx == 0
        assert model.neutral_idx == 1
        assert model.contradiction_idx == 2

    def test_missing_label_raises_mapping_error(self) -> None:
        """If one label is missing, raise clear NLILabelMappingError."""
        model = NLIModel.__new__(NLIModel)
        model.model_name = "test-nli"
        model._model = MagicMock()
        model._model.model.config.id2label = {
            0: "positive",
            1: "negative",
            2: "neutral",
        }
        with pytest.raises(NLILabelMappingError) as exc_info:
            model._discover_label_indices()
        assert "Could not safely resolve all 3 NLI labels" in str(exc_info.value)

    def test_missing_id2label_raises_mapping_error(self) -> None:
        model = NLIModel.__new__(NLIModel)
        model.model_name = "test-nli"
        model._model = MagicMock()
        del model._model.model.config.id2label
        del model._model.config
        with pytest.raises(NLILabelMappingError):
            model._discover_label_indices()

    def test_duplicate_or_ambiguous_labels_raises_mapping_error(self) -> None:
        """Ambiguous duplicate labels for the same category raise NLILabelMappingError."""
        model = NLIModel.__new__(NLIModel)
        model.model_name = "test-nli"
        model._model = MagicMock()
        model._model.model.config.id2label = {
            0: "entailment_primary",
            1: "entailment_secondary",
            2: "neutral",
        }
        with pytest.raises(NLILabelMappingError):
            model._discover_label_indices()



# ==============================================================================
# Suite C: Pair Construction & Provenance Preservation
# ==============================================================================

class TestPairConstructionAndProvenance:
    """Verify premise/hypothesis orientation and provenance preservation."""

    def test_pair_orientation_premise_is_evidence(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_nli = MockNLIModel()
        service = GroundingService(model=mock_nli, top_k_evidence=2)
        claims = [
            ExtractedClaim(claim_index=0, claim_text="Gradients are computed via chain rule.")
        ]
        service.validate_claims(claims=claims, evidence=sample_evidence_items)

        assert len(mock_nli.predict_calls) == 1
        pairs = mock_nli.predict_calls[0]
        assert len(pairs) == 2  # 1 claim * 2 evidence chunks
        # First pair: (premise=evidence_0, hypothesis=claim_0)
        assert pairs[0][0] == sample_evidence_items[0].source_text
        assert pairs[0][1] == "Gradients are computed via chain rule."
        # Second pair: (premise=evidence_1, hypothesis=claim_0)
        assert pairs[1][0] == sample_evidence_items[1].source_text
        assert pairs[1][1] == "Gradients are computed via chain rule."

    def test_provenance_preservation(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_nli = MockNLIModel(default_scores=NLIScores(entailment=0.92, contradiction=0.03, neutral=0.05))
        service = GroundingService(model=mock_nli)
        claims = [ExtractedClaim(claim_index=0, claim_text="Chain rule computes gradients.")]
        report = service.validate_claims(claims=claims, evidence=sample_evidence_items)

        assert len(report.claim_results) == 1
        claim_res = report.claim_results[0]
        assert claim_res.supporting_evidence is not None
        assert claim_res.supporting_evidence.chunk_id == "chunk_page_10_0"
        assert claim_res.supporting_evidence.document_id == "deep_learning_handbook"
        assert claim_res.supporting_evidence.page_number == 10
        assert claim_res.supporting_evidence.chunk_index == 0
        assert claim_res.supporting_evidence.rank == 1
        assert claim_res.supporting_evidence.similarity_score == 0.88
        assert claim_res.supporting_evidence.reranker_score == 4.25
        assert claim_res.supporting_evidence.nli_score == 0.92


# ==============================================================================
# Suite D: Claim Classification Logic
# ==============================================================================

class TestClaimClassification:
    """Verify entailment, contradiction, unsupported, and conflicted classification."""

    def test_clearly_entailed_claim(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_nli = MockNLIModel(default_scores=NLIScores(entailment=0.94, contradiction=0.01, neutral=0.05))
        service = GroundingService(model=mock_nli, entailment_threshold=0.80, contradiction_threshold=0.80)
        claims = [ExtractedClaim(claim_index=0, claim_text="Gradients use chain rule.")]
        report = service.validate_claims(claims, sample_evidence_items)

        assert report.claim_results[0].status == "entailed"
        assert report.claim_results[0].entailment_score == 0.94
        assert report.claim_results[0].contradiction_score == 0.01

    def test_clearly_contradicted_claim(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_nli = MockNLIModel(default_scores=NLIScores(entailment=0.02, contradiction=0.91, neutral=0.07))
        service = GroundingService(model=mock_nli, entailment_threshold=0.80, contradiction_threshold=0.80)
        claims = [ExtractedClaim(claim_index=0, claim_text="Backpropagation never uses chain rule.")]
        report = service.validate_claims(claims, sample_evidence_items)

        assert report.claim_results[0].status == "contradicted"
        assert report.claim_results[0].contradiction_score == 0.91
        assert len(report.claim_results[0].contradicting_evidence) > 0

    def test_unsupported_neutral_claim(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_nli = MockNLIModel(default_scores=NLIScores(entailment=0.45, contradiction=0.10, neutral=0.45))
        service = GroundingService(model=mock_nli, entailment_threshold=0.80, contradiction_threshold=0.80)
        claims = [ExtractedClaim(claim_index=0, claim_text="Quantum computers run backpropagation faster.")]
        report = service.validate_claims(claims, sample_evidence_items)

        assert report.claim_results[0].status == "unsupported"
        assert report.supported_claims == 0
        assert report.unsupported_claims == 1

    def test_conflicted_evidence_across_chunks(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        """If chunk 1 strongly entails (0.92) but chunk 2 strongly contradicts (0.89), status is 'conflicted'."""
        def score_fn(premise: str, hypothesis: str) -> NLIScores:
            if "page_10" in premise or "calculates gradients" in premise:
                return NLIScores(entailment=0.92, contradiction=0.02, neutral=0.06)
            return NLIScores(entailment=0.03, contradiction=0.89, neutral=0.08)

        mock_nli = MockNLIModel(pair_scores_fn=score_fn)
        service = GroundingService(model=mock_nli, entailment_threshold=0.80, contradiction_threshold=0.80)
        claims = [ExtractedClaim(claim_index=0, claim_text="The concept was introduced in 1986.")]
        report = service.validate_claims(claims, sample_evidence_items)

        assert report.claim_results[0].status == "conflicted"
        assert report.conflicted_claims == 1
        assert report.overall_status == "conflicted"


# ==============================================================================
# Suite E & F: Aggregation & Threshold Behavior
# ==============================================================================

class TestAggregationAndThresholds:
    """Verify aggregation counts, groundedness score, and threshold boundaries."""

    def test_exact_threshold_boundary(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        # Score exactly at 0.80 -> entailed
        mock_nli = MockNLIModel(default_scores=NLIScores(entailment=0.80, contradiction=0.05, neutral=0.15))
        service = GroundingService(model=mock_nli, entailment_threshold=0.80)
        claims = [ExtractedClaim(claim_index=0, claim_text="Boundary test.")]
        report = service.validate_claims(claims, sample_evidence_items)
        assert report.claim_results[0].status == "entailed"

        # Score just below threshold 0.799 -> unsupported
        mock_nli_below = MockNLIModel(default_scores=NLIScores(entailment=0.799, contradiction=0.05, neutral=0.151))
        service_below = GroundingService(model=mock_nli_below, entailment_threshold=0.80)
        report_below = service_below.validate_claims(claims, sample_evidence_items)
        assert report_below.claim_results[0].status == "unsupported"

    def test_partial_support_score_calculation(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        """2 claims: claim 0 is supported (0.95), claim 1 is unsupported (0.50) -> groundedness_score = 0.5."""
        def score_fn(premise: str, hypothesis: str) -> NLIScores:
            if "First claim" in hypothesis:
                return NLIScores(entailment=0.95, contradiction=0.01, neutral=0.04)
            return NLIScores(entailment=0.50, contradiction=0.05, neutral=0.45)

        mock_nli = MockNLIModel(pair_scores_fn=score_fn)
        service = GroundingService(model=mock_nli, entailment_threshold=0.80)
        claims = [
            ExtractedClaim(claim_index=0, claim_text="First claim."),
            ExtractedClaim(claim_index=1, claim_text="Second claim."),
        ]
        report = service.validate_claims(claims, sample_evidence_items)

        assert report.total_claims == 2
        assert report.supported_claims == 1
        assert report.unsupported_claims == 1
        assert report.groundedness_score == 0.5
        assert report.overall_status == "unsupported"

    def test_zero_claims_explicit_handling(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        service = GroundingService(model=MockNLIModel())
        report = service.validate_claims([], sample_evidence_items)
        assert report.total_claims == 0
        assert report.groundedness_score == 0.0
        assert report.overall_status == "empty"

    def test_invalid_thresholds_raise_config_error(self) -> None:
        with pytest.raises(InvalidGroundingConfigError):
            GroundingService(entailment_threshold=-0.1)
        with pytest.raises(InvalidGroundingConfigError):
            GroundingService(contradiction_threshold=1.5)
        with pytest.raises(InvalidGroundingConfigError):
            GroundingService(top_k_evidence=0)


# ==============================================================================
# Suite G: Safe Final Decision Policy
# ==============================================================================

class TestSafeDecisionPolicy:
    """Verify safe answer acceptance or safe refusal under require_all_claims_supported."""

    def test_fully_grounded_answer_accepted(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=ev.similarity_score or 0.8,
                    reranker_score=ev.reranker_score,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Backpropagation calculates gradients of the loss function.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = GroundingService(
            model=MockNLIModel(default_scores=NLIScores(entailment=0.95, contradiction=0.01, neutral=0.04))
        )

        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        resp = orchestrator.answer_with_grounding(
            GroundedAnswerRequest(query="test", require_all_claims_supported=True)
        )

        assert resp.answer == "Backpropagation calculates gradients of the loss function."
        assert resp.answerable is True
        assert resp.grounded is True
        assert resp.groundedness_score == 1.0
        assert resp.grounding_status == "grounded"
        assert len(resp.claims) == 1
        assert len(resp.evidence) == 2

    def test_partially_unsupported_answer_safely_refused(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.8,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Backpropagation calculates gradients. It was invented on Mars.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        def score_fn(premise: str, hypothesis: str) -> NLIScores:
            if "gradients" in hypothesis:
                return NLIScores(entailment=0.95, contradiction=0.01, neutral=0.04)
            return NLIScores(entailment=0.10, contradiction=0.10, neutral=0.80)

        mock_grounding = GroundingService(
            model=MockNLIModel(pair_scores_fn=score_fn)
        )

        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        resp = orchestrator.answer_with_grounding(
            GroundedAnswerRequest(query="test", require_all_claims_supported=True)
        )

        # Under safe policy: answer text MUST be suppressed (None), answerable=False, grounded=False
        assert resp.answer is None
        assert resp.answerable is False
        assert resp.grounded is False
        assert resp.groundedness_score == 0.5
        assert resp.grounding_status == "unsupported"
        assert "not sufficiently supported" in (resp.reason or "")
        # Provenance and claim results are preserved
        assert len(resp.claims) == 2
        assert len(resp.evidence) == 2

    def test_contradicted_answer_safely_refused(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.8,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Backpropagation does not compute gradients.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = GroundingService(
            model=MockNLIModel(default_scores=NLIScores(entailment=0.01, contradiction=0.96, neutral=0.03))
        )

        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        resp = orchestrator.answer_with_grounding(
            GroundedAnswerRequest(query="test", require_all_claims_supported=True)
        )

        assert resp.answer is None
        assert resp.answerable is False
        assert resp.grounded is False
        assert resp.grounding_status == "contradicted"

    def test_conflicted_answer_safely_refused(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.8,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Backpropagation has conflicting evidence.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        def score_fn(premise: str, hypothesis: str) -> NLIScores:
            if "gradients" in premise:
                return NLIScores(entailment=0.91, contradiction=0.03, neutral=0.06)
            return NLIScores(entailment=0.02, contradiction=0.93, neutral=0.05)

        mock_grounding = GroundingService(model=MockNLIModel(pair_scores_fn=score_fn))
        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        resp = orchestrator.answer_with_grounding(
            GroundedAnswerRequest(query="test", require_all_claims_supported=True)
        )

        assert resp.answer is None
        assert resp.answerable is False
        assert resp.grounded is False
        assert resp.grounding_status == "conflicted"

    def test_permissive_mode_returns_answer(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.8,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Backpropagation computes gradients.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = GroundingService(
            model=MockNLIModel(default_scores=NLIScores(entailment=0.50, contradiction=0.05, neutral=0.45))
        )
        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        # In permissive mode: answer is returned even if grounded=False
        resp = orchestrator.answer_with_grounding(
            GroundedAnswerRequest(query="test", require_all_claims_supported=False)
        )
        assert resp.answer == "Backpropagation computes gradients."
        assert resp.grounded is False
        assert resp.grounding_status == "unsupported"



# ==============================================================================
# Suite H: Service Orchestration & Edge Cases
# ==============================================================================

class TestServiceOrchestration:
    """Verify pipeline orchestration and edge case handling."""

    def test_empty_evidence_from_search(self) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[],
            total_results=0,
        )
        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")

        mock_grounding = MagicMock(spec=GroundingService)
        mock_grounding.model = MagicMock(model_name="cross-encoder/nli-deberta-v3-base")

        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        resp = orchestrator.answer_with_grounding(GroundedAnswerRequest(query="test"))
        assert resp.answer is None
        assert resp.answerable is False
        assert resp.grounded is False
        assert resp.grounding_status == "empty"
        # GenerationService and GroundingService must NOT be called when evidence is empty
        mock_gen.generate_answer.assert_not_called()
        mock_grounding.validate_claims.assert_not_called()

    def test_disabled_grounding_flag(self, sample_evidence_items: List[GenerationEvidenceItem]) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.8,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Unvalidated raw answer.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = MagicMock(spec=GroundingService)
        mock_grounding.model = MagicMock(model_name="cross-encoder/nli-deberta-v3-base")

        orchestrator = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        with patch.object(settings, "GROUNDING_ENABLED", False):
            resp = orchestrator.answer_with_grounding(GroundedAnswerRequest(query="test"))
            assert resp.answer == "Unvalidated raw answer."
            assert resp.grounding_status == "disabled"
            mock_grounding.validate_claims.assert_not_called()


# ==============================================================================
# Suite I: API Endpoint Tests
# ==============================================================================

class TestGroundedAnswerAPI:
    """Verify POST /api/v1/grounded-answer HTTP contracts and responses."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_api_valid_grounded_answer(
        self, client: TestClient, sample_evidence_items: List[GenerationEvidenceItem]
    ) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="What is backpropagation?",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.85,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="What is backpropagation?",
            answer="Backpropagation calculates gradients of the loss function.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = GroundingService(
            model=MockNLIModel(default_scores=NLIScores(entailment=0.96, contradiction=0.01, neutral=0.03))
        )

        test_service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        app.dependency_overrides[get_grounded_answer_service] = lambda: test_service
        try:
            response = client.post(
                "/api/v1/grounded-answer",
                json={"query": "What is backpropagation?"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["answer"] == "Backpropagation calculates gradients of the loss function."
            assert data["answerable"] is True
            assert data["grounded"] is True
            assert data["groundedness_score"] == 1.0
            assert data["grounding_status"] == "grounded"
            assert len(data["claims"]) == 1
            assert len(data["evidence"]) == 2
        finally:
            app.dependency_overrides.clear()

    def test_api_safe_refusal_unsupported_claims(
        self, client: TestClient, sample_evidence_items: List[GenerationEvidenceItem]
    ) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="What is backpropagation?",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.85,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="What is backpropagation?",
            answer="Backpropagation calculates gradients. Mars invented it.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        # First claim entailed, second unsupported
        def score_fn(premise: str, hypothesis: str) -> NLIScores:
            if "gradients" in hypothesis:
                return NLIScores(entailment=0.95, contradiction=0.01, neutral=0.04)
            return NLIScores(entailment=0.10, contradiction=0.10, neutral=0.80)

        mock_grounding = GroundingService(model=MockNLIModel(pair_scores_fn=score_fn))

        test_service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        app.dependency_overrides[get_grounded_answer_service] = lambda: test_service
        try:
            response = client.post(
                "/api/v1/grounded-answer",
                json={"query": "What is backpropagation?"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Under safe policy: answer MUST be null
            assert data["answer"] is None
            assert data["answerable"] is False
            assert data["grounded"] is False
            assert data["groundedness_score"] == 0.5
            assert data["grounding_status"] == "unsupported"
            assert len(data["claims"]) == 2
        finally:
            app.dependency_overrides.clear()

    def test_api_malformed_empty_query_raises_validation_error(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/grounded-answer",
            json={"query": "   "},
        )
        assert response.status_code == 422

    def test_api_safe_refusal_contradicted_claims(
        self, client: TestClient, sample_evidence_items: List[GenerationEvidenceItem]
    ) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.85,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Backpropagation never uses chain rule.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = GroundingService(
            model=MockNLIModel(default_scores=NLIScores(entailment=0.01, contradiction=0.96, neutral=0.03))
        )
        test_service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        app.dependency_overrides[get_grounded_answer_service] = lambda: test_service
        try:
            response = client.post(
                "/api/v1/grounded-answer",
                json={"query": "test"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["answer"] is None
            assert data["answerable"] is False
            assert data["grounded"] is False
            assert data["grounding_status"] == "contradicted"
        finally:
            app.dependency_overrides.clear()

    def test_api_empty_evidence_safe_response(self, client: TestClient) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[],
            total_results=0,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_grounding = MagicMock(spec=GroundingService)
        mock_grounding.model = MagicMock(model_name="cross-encoder/nli-deberta-v3-base")

        test_service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        app.dependency_overrides[get_grounded_answer_service] = lambda: test_service
        try:
            response = client.post(
                "/api/v1/grounded-answer",
                json={"query": "test"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["answer"] is None
            assert data["answerable"] is False
            assert data["grounded"] is False
            assert data["grounding_status"] == "empty"
        finally:
            app.dependency_overrides.clear()

    def test_api_disabled_grounding(
        self, client: TestClient, sample_evidence_items: List[GenerationEvidenceItem]
    ) -> None:
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    page_number=ev.page_number,
                    chunk_index=ev.chunk_index,
                    text=ev.source_text,
                    similarity_score=0.85,
                    rank=ev.rank,
                )
                for ev in sample_evidence_items
            ],
            total_results=2,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.model = MagicMock(model_name="google/flan-t5-base")
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="test",
            answer="Unchecked generated answer.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=sample_evidence_items,
            evidence_count=2,
        )

        mock_grounding = MagicMock(spec=GroundingService)
        mock_grounding.model = MagicMock(model_name="cross-encoder/nli-deberta-v3-base")

        test_service = GroundedAnswerService(
            search_service=mock_search,
            generation_service=mock_gen,
            grounding_service=mock_grounding,
        )

        app.dependency_overrides[get_grounded_answer_service] = lambda: test_service
        try:
            with patch.object(settings, "GROUNDING_ENABLED", False):
                response = client.post(
                    "/api/v1/grounded-answer",
                    json={"query": "test"},
                )
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["answer"] == "Unchecked generated answer."
                assert data["grounding_status"] == "disabled"
        finally:
            app.dependency_overrides.clear()



# ==============================================================================
# Suite J: Regression Tests for Earlier Phases
# ==============================================================================

class TestPhase0To8Regressions:
    """Verify existing search, extractive QA, and abstractive generation endpoints remain intact."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def test_generation_endpoint_still_works(self, client: TestClient) -> None:
        """POST /api/v1/answer must remain operational without modification."""
        mock_search = MagicMock(spec=SearchService)
        mock_search.search.return_value = SearchResponse(
            query="What is deep learning?",
            results=[
                SearchResult(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    page_number=1,
                    chunk_index=0,
                    text="Deep learning uses neural networks with multiple layers.",
                    similarity_score=0.9,
                    rank=1,
                )
            ],
            total_results=1,
        )

        mock_gen = MagicMock(spec=GenerationService)
        mock_gen.generate_answer.return_value = GenerationResponse(
            query="What is deep learning?",
            answer="Deep learning uses neural networks with multiple layers.",
            answerable=True,
            model_name="google/flan-t5-base",
            evidence=[
                GenerationEvidenceItem(
                    rank=1,
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    page_number=1,
                    chunk_index=0,
                    source_text="Deep learning uses neural networks with multiple layers.",
                    similarity_score=0.9,
                )
            ],
            evidence_count=1,
        )

        app.dependency_overrides[get_search_service] = lambda: mock_search
        app.dependency_overrides[get_generation_service] = lambda: mock_gen
        try:
            response = client.post("/api/v1/answer", json={"query": "What is deep learning?"})
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["answer"] == "Deep learning uses neural networks with multiple layers."
            assert data["answerable"] is True
            assert len(data["evidence"]) == 1
        finally:
            app.dependency_overrides.clear()
