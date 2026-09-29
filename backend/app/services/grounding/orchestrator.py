"""Grounded Answer Orchestration Service for BookRAG AI Phase 9.

Orchestrates the complete grounded question answering pipeline:
1. SearchService: Dense retrieval + Cross-Encoder reranking
2. GenerationService: FLAN-T5 abstractive answer synthesis
3. ClaimDecomposer: Deterministic sentence-level claim splitting
4. GroundingService: NLI groundedness verification of each claim
5. Safe Decision Policy: Suppresses ungrounded/contradicted/conflicted answers
"""

from typing import Optional
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.grounding import GroundedAnswerRequest, GroundedAnswerResponse, GroundingReport
from app.services.citation.exceptions import DocumentIsolationError
from app.services.citation.service import CitationService
from app.services.generation.service import GenerationService
from app.services.grounding.claims import ClaimDecomposer
from app.services.grounding.exceptions import InvalidGroundingInputError
from app.services.grounding.service import GroundingService
from app.services.search.service import SearchService

logger = get_logger(__name__)


class GroundedAnswerService:
    """Orchestrates retrieval, generation, claim decomposition, NLI validation, safe decision policy, and citation mapping."""

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        generation_service: Optional[GenerationService] = None,
        grounding_service: Optional[GroundingService] = None,
        claim_decomposer: Optional[ClaimDecomposer] = None,
        citation_service: Optional[CitationService] = None,
    ) -> None:
        """Initialize GroundedAnswerService with injected or lazily resolved components."""
        self._search_service = search_service
        self._generation_service = generation_service
        self._grounding_service = grounding_service
        self._claim_decomposer = claim_decomposer
        self._citation_service = citation_service

    @property
    def citation_service(self) -> CitationService:
        """Return injected or default CitationService."""
        if self._citation_service is None:
            self._citation_service = CitationService()
        return self._citation_service

    @property
    def search_service(self) -> SearchService:
        """Return injected or default SearchService."""
        if self._search_service is None:
            self._search_service = SearchService()
        return self._search_service

    @property
    def generation_service(self) -> GenerationService:
        """Return injected or default GenerationService."""
        if self._generation_service is None:
            self._generation_service = GenerationService()
        return self._generation_service

    @property
    def grounding_service(self) -> GroundingService:
        """Return injected or default GroundingService."""
        if self._grounding_service is None:
            self._grounding_service = GroundingService()
        return self._grounding_service

    @property
    def claim_decomposer(self) -> ClaimDecomposer:
        """Return injected or default ClaimDecomposer."""
        if self._claim_decomposer is None:
            self._claim_decomposer = ClaimDecomposer()
        return self._claim_decomposer

    def answer_with_grounding(self, request: GroundedAnswerRequest) -> GroundedAnswerResponse:
        """Execute two-stage retrieval, abstractive generation, claim-level NLI verification, and safe decision.

        Args:
            request: Validated GroundedAnswerRequest.

        Returns:
            GroundedAnswerResponse with safe answer, validation metadata, and evidence provenance.
        """
        query = request.query.strip()
        if not query:
            raise InvalidGroundingInputError("Question query cannot be empty or whitespace-only.")

        gen_model = getattr(self.generation_service, "model", None)
        raw_gen_name = getattr(gen_model, "model_name", None)
        model_name = raw_gen_name if isinstance(raw_gen_name, str) else settings.GENERATION_MODEL_NAME

        grounding_model = getattr(self.grounding_service, "model", None)
        raw_grounding_name = getattr(grounding_model, "model_name", None)
        grounding_model_name = (
            raw_grounding_name if isinstance(raw_grounding_name, str) else settings.GROUNDING_MODEL_NAME
        )

        # 1. First & Second Stage Retrieval via SearchService
        search_response = self.search_service.search(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
            candidate_k=request.candidate_k,
            enable_reranking=request.enable_reranking,
        )

        # Edge case: No evidence retrieved from index
        if not search_response.results:
            logger.info("No evidence retrieved for query '%s'. Returning safe unanswerable response.", query)
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="empty",
                claims=[],
                evidence=[],
                reason="No relevant evidence was retrieved from the book index.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
            )

        # 2. Abstractive Generation via GenerationService
        gen_response = self.generation_service.generate_answer(
            query=query,
            evidence=search_response.results,
            max_new_tokens=request.max_new_tokens,
            num_beams=request.num_beams,
            do_sample=request.do_sample,
            temperature=request.temperature,
        )

        # Edge case: Generation model produced no answer or unanswerable
        if not gen_response.answerable or not gen_response.answer or not gen_response.answer.strip():
            logger.info("Generation service returned unanswerable for query '%s'.", query)
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="empty",
                claims=[],
                evidence=gen_response.evidence,
                reason="Generation model could not synthesize an answer from the retrieved evidence.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
            )

        raw_answer = gen_response.answer.strip()

        # Check if grounding is globally disabled
        if not settings.GROUNDING_ENABLED:
            logger.info("Grounding validation is disabled via configuration.")
            return GroundedAnswerResponse(
                query=query,
                answer=raw_answer,
                answerable=True,
                grounded=True,
                groundedness_score=1.0,
                grounding_status="disabled",
                claims=[],
                evidence=gen_response.evidence,
                reason="Grounding validation is disabled in settings.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
            )

        # 3. Decompose Answer into Claims
        claims = self.claim_decomposer.decompose(raw_answer)

        # Edge case: Answer has no substantive claims
        if not claims:
            logger.info("Generated answer for query '%s' contained no substantive claims.", query)
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="empty",
                claims=[],
                evidence=gen_response.evidence,
                reason="Generated answer contains no substantive claims to validate.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
            )

        # 4. NLI Claim-Level Validation via GroundingService
        grounding_report: GroundingReport = self.grounding_service.validate_claims(
            claims=claims,
            evidence=gen_response.evidence,
            entailment_threshold=request.entailment_threshold,
            contradiction_threshold=request.contradiction_threshold,
            top_k_evidence=request.top_k_evidence,
        )

        # 5. Citation Mapping via CitationService (Phase 10)
        try:
            citation_result = self.citation_service.build_citations(
                grounding_result=grounding_report,
                expected_document_id=request.document_id,
            )
            citations = citation_result.citations
        except DocumentIsolationError as exc:
            logger.error("Document isolation violation during citation mapping: %s", exc)
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="unsupported",
                claims=grounding_report.claim_results,
                evidence=gen_response.evidence,
                citations=[],
                reason=f"Document isolation error: {exc}",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
            )

        # 6. Safe Decision Policy
        require_all = (
            request.require_all_claims_supported
            if request.require_all_claims_supported is not None
            else settings.GROUNDING_REQUIRE_ALL_CLAIMS_SUPPORTED
        )

        if require_all:
            # Under strict safe policy: all substantive claims must be entailed with zero contradiction/conflict
            is_acceptable = (
                grounding_report.overall_status == "grounded"
                and grounding_report.supported_claims == grounding_report.total_claims
                and grounding_report.contradicted_claims == 0
                and grounding_report.conflicted_claims == 0
            )

            if is_acceptable:
                return GroundedAnswerResponse(
                    query=query,
                    answer=raw_answer,
                    answerable=True,
                    grounded=True,
                    groundedness_score=grounding_report.groundedness_score,
                    grounding_status="grounded",
                    claims=grounding_report.claim_results,
                    evidence=gen_response.evidence,
                    citations=citations,
                    reason=grounding_report.reason,
                    model_name=model_name,
                    grounding_model_name=grounding_model_name,
                )
            else:
                # Safe refusal: suppress ungrounded answer to prevent misleading the user
                logger.warning(
                    "Generated answer rejected by safe policy for query '%s' (status=%s, score=%.2f).",
                    query,
                    grounding_report.overall_status,
                    grounding_report.groundedness_score,
                )
                safe_reason = (
                    grounding_report.reason
                    or "The generated answer contains claims that are not sufficiently supported by the retrieved book evidence."
                )
                return GroundedAnswerResponse(
                    query=query,
                    answer=None,
                    answerable=False,
                    grounded=False,
                    groundedness_score=grounding_report.groundedness_score,
                    grounding_status=grounding_report.overall_status,
                    claims=grounding_report.claim_results,
                    evidence=gen_response.evidence,
                    citations=citations,
                    reason=safe_reason,
                    model_name=model_name,
                    grounding_model_name=grounding_model_name,
                )
        else:
            # Permissive mode: return generated text alongside grounding metadata
            is_grounded = grounding_report.overall_status == "grounded"
            return GroundedAnswerResponse(
                query=query,
                answer=raw_answer,
                answerable=gen_response.answerable,
                grounded=is_grounded,
                groundedness_score=grounding_report.groundedness_score,
                grounding_status=grounding_report.overall_status,
                claims=grounding_report.claim_results,
                evidence=gen_response.evidence,
                citations=citations,
                reason=grounding_report.reason,
                model_name=model_name,
                grounding_model_name=grounding_model_name,
            )
