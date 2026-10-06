"""Grounded Answer Orchestration Service for BookRAG AI Phase 9.

Orchestrates the complete grounded question answering pipeline:
1. SearchService: Dense retrieval + Cross-Encoder reranking
2. GenerationService: FLAN-T5 abstractive answer synthesis
3. ClaimDecomposer: Deterministic sentence-level claim splitting
4. GroundingService: NLI groundedness verification of each claim
5. Safe Decision Policy: Suppresses ungrounded/contradicted/conflicted answers
"""

import time
from typing import Optional
from app.core import config
from app.core.config import settings
from app.core.device import get_device_manager
from app.core.logging import get_logger
from app.schemas.grounding import GroundedAnswerRequest, GroundedAnswerResponse, GroundingReport
from app.services.citation.exceptions import DocumentIsolationError
from app.services.citation.service import CitationService
from app.services.generation.service import GenerationService
from app.services.grounding.claims import ClaimDecomposer
from app.services.grounding.exceptions import InvalidGroundingInputError
from app.schemas.query_plan import QueryType
from app.services.grounding.completeness import AnswerCompletenessChecker
from app.services.grounding.service import GroundingService
from app.services.query_understanding.service import QueryUnderstandingService
from app.services.search.query_search import QuerySearchService
from app.services.search.service import SearchService

logger = get_logger(__name__)


class GroundedAnswerService:
    """Orchestrates query planning, retrieval, generation, claim decomposition, NLI validation, safe decision policy, and citation mapping."""

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        generation_service: Optional[GenerationService] = None,
        grounding_service: Optional[GroundingService] = None,
        claim_decomposer: Optional[ClaimDecomposer] = None,
        citation_service: Optional[CitationService] = None,
        query_understanding_service: Optional[QueryUnderstandingService] = None,
        query_search_service: Optional[QuerySearchService] = None,
        completeness_checker: Optional[AnswerCompletenessChecker] = None,
    ) -> None:
        """Initialize GroundedAnswerService with injected or lazily resolved components."""
        self._search_service = search_service
        self._generation_service = generation_service
        self._grounding_service = grounding_service
        self._claim_decomposer = claim_decomposer
        self._citation_service = citation_service
        self._query_understanding_service = query_understanding_service
        self._query_search_service = query_search_service
        self._completeness_checker = completeness_checker

    @property
    def completeness_checker(self) -> AnswerCompletenessChecker:
        """Return injected or default AnswerCompletenessChecker."""
        if self._completeness_checker is None:
            self._completeness_checker = AnswerCompletenessChecker()
        return self._completeness_checker

    @property
    def query_understanding_service(self) -> QueryUnderstandingService:
        """Return injected or default QueryUnderstandingService."""
        if self._query_understanding_service is None:
            self._query_understanding_service = QueryUnderstandingService()
        return self._query_understanding_service

    @property
    def query_search_service(self) -> QuerySearchService:
        """Return injected or default QuerySearchService."""
        if self._query_search_service is None:
            self._query_search_service = QuerySearchService(search_service=self.search_service)
        return self._query_search_service

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

        t_start = time.perf_counter()
        device_diag = get_device_manager().get_diagnostics()

        # 0. Query Understanding & Query Planning (Phase 11 & Phase 21)
        t0 = time.perf_counter()
        query_plan = self.query_understanding_service.analyze_query(query)
        query_plan_ms = (time.perf_counter() - t0) * 1000

        if query_plan.query_type == QueryType.AMBIGUOUS:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="unsupported",
                claims=[],
                evidence=[],
                citations=[],
                query_plan=query_plan,
                reason="The question is ambiguous or underspecified. Please clarify what you want to know.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=0,
                reranking_applied=False,
                completeness_status="ambiguous",
                latency_breakdown_ms={
                    "query_understanding_ms": round(query_plan_ms, 2),
                    "total_latency_ms": round(elapsed_ms, 2),
                },
                device_info=device_diag,
            )

        # 1. Plan-Driven Retrieval via QuerySearchService (Phase 11)
        t1 = time.perf_counter()
        search_response = self.query_search_service.search_with_plan(
            plan=query_plan,
            top_k=request.top_k,
            document_id=request.document_id,
            candidate_k=request.candidate_k,
            enable_reranking=request.enable_reranking,
        )
        retrieval_ms = (time.perf_counter() - t1) * 1000

        cand_count = getattr(search_response, "candidate_count", None)
        if cand_count is None:
            cand_count = len(search_response.results) if search_response else 0
        rerank_applied = getattr(search_response, "reranking_applied", True)

        # Edge case: No evidence retrieved from index
        if not search_response.results:
            logger.info("No evidence retrieved for query '%s'. Returning safe unanswerable response.", query)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="empty",
                claims=[],
                evidence=[],
                citations=[],
                query_plan=query_plan,
                reason="No relevant evidence was retrieved from the book index.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=cand_count,
                reranking_applied=rerank_applied,
                latency_breakdown_ms={
                    "query_understanding_ms": round(query_plan_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "total_latency_ms": round(elapsed_ms, 2),
                },
                device_info=device_diag,
            )

        # Edge case: Insufficient evidence relevance (evidence sufficiency invariant)
        min_evidence_similarity = getattr(settings, "GROUNDING_MIN_EVIDENCE_SIMILARITY", 0.20)
        scores = [getattr(r, "similarity_score", None) for r in search_response.results]
        has_scores = any(s is not None and s > 0.0 for s in scores)
        if has_scores:
            top_similarity = max((s for s in scores if s is not None), default=0.0)
            if top_similarity < min_evidence_similarity:
                logger.info(
                    "Retrieved evidence for query '%s' has insufficient similarity (max=%.4f < %.4f). Safely refusing.",
                    query,
                    top_similarity,
                    min_evidence_similarity,
                )
                return GroundedAnswerResponse(
                    query=query,
                    answer=None,
                    answerable=False,
                    grounded=False,
                    groundedness_score=0.0,
                    grounding_status="unsupported",
                    claims=[],
                    evidence=[],
                    citations=[],
                    query_plan=query_plan,
                    reason="The retrieved book passages do not contain sufficiently relevant evidence to answer this question.",
                    model_name=model_name,
                    grounding_model_name=grounding_model_name,
                    candidate_count=cand_count,
                    reranking_applied=rerank_applied,
                    regeneration_attempted=False,
                    regeneration_reason=None,
                    completeness_status="insufficient_evidence",
                )

        # 2. Abstractive Generation via GenerationService (Attempt 1: Query-Type-Aware)
        t2 = time.perf_counter()
        gen_response = self.generation_service.generate_answer(
            query=query,
            evidence=search_response.results,
            max_new_tokens=request.max_new_tokens,
            num_beams=request.num_beams,
            do_sample=request.do_sample,
            temperature=request.temperature,
            query_type=query_plan.query_type,
        )

        regeneration_attempted = False
        regeneration_reason: Optional[str] = None
        completeness_status: str = "complete"

        # Completeness Check and Bounded Controlled Regeneration (Attempt 2, max 2 attempts total)
        if gen_response.answerable and gen_response.answer and gen_response.answer.strip():
            raw_answer = gen_response.answer.strip()
            comp_result = self.completeness_checker.check_completeness(query_plan, raw_answer)

            if not comp_result.is_complete:
                logger.info(
                    "Attempt 1 answer incomplete for query '%s': %s. Triggering controlled regeneration...",
                    query,
                    comp_result.reason,
                )
                regeneration_attempted = True
                regeneration_reason = comp_result.reason
                completeness_status = "incomplete_attempt_1"

                # Construct deterministic targeted retry instruction
                retry_instruction = None
                if query_plan.query_type == QueryType.COMPARISON and len(query_plan.comparison_aspects) >= 2:
                    retry_instruction = (
                        f"Compare both concepts using only the provided context. "
                        f"Ensure you explicitly explain both {query_plan.comparison_aspects[0]} "
                        f"and {query_plan.comparison_aspects[1]}:"
                    )
                elif query_plan.query_type in (QueryType.MULTI_PART, QueryType.MULTI_PAGE) and query_plan.sub_questions:
                    sub_qs_str = ", and ".join(query_plan.sub_questions)
                    retry_instruction = (
                        f"Answer all parts of the question thoroughly using only the provided context. "
                        f"Ensure you address: {sub_qs_str}:"
                    )

                if retry_instruction:
                    retry_gen_response = self.generation_service.generate_answer(
                        query=query,
                        evidence=search_response.results,
                        max_new_tokens=request.max_new_tokens,
                        num_beams=request.num_beams,
                        do_sample=request.do_sample,
                        temperature=request.temperature,
                        prompt_instruction=retry_instruction,
                    )
                    if retry_gen_response.answerable and retry_gen_response.answer and retry_gen_response.answer.strip():
                        retry_answer = retry_gen_response.answer.strip()
                        comp_result_2 = self.completeness_checker.check_completeness(query_plan, retry_answer)
                        if comp_result_2.is_complete:
                            logger.info("Controlled regeneration succeeded with complete answer.")
                            gen_response = retry_gen_response
                            completeness_status = "complete"
                        else:
                            logger.warning("Controlled regeneration attempt 2 still incomplete: %s", comp_result_2.reason)
                            completeness_status = "incomplete_after_retry"
                            gen_response = retry_gen_response
        generation_ms = (time.perf_counter() - t2) * 1000

        # Edge case: Generation model produced no answer or unanswerable
        if not gen_response.answerable or not gen_response.answer or not gen_response.answer.strip():
            logger.info("Generation service returned unanswerable for query '%s'.", query)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="empty",
                claims=[],
                evidence=gen_response.evidence,
                citations=[],
                query_plan=query_plan,
                reason="Generation model could not synthesize an answer from the retrieved evidence.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=cand_count,
                reranking_applied=rerank_applied,
                latency_breakdown_ms={
                    "query_understanding_ms": round(query_plan_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_latency_ms": round(elapsed_ms, 2),
                },
                device_info=device_diag,
            )

        raw_answer = gen_response.answer.strip()

        # Check if grounding is globally disabled
        if not settings.GROUNDING_ENABLED:
            logger.info("Grounding validation is disabled via configuration.")
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return GroundedAnswerResponse(
                query=query,
                answer=raw_answer,
                answerable=True,
                grounded=True,
                groundedness_score=1.0,
                grounding_status="disabled",
                claims=[],
                evidence=gen_response.evidence,
                citations=[],
                query_plan=query_plan,
                reason="Grounding validation is disabled in settings.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=cand_count,
                reranking_applied=rerank_applied,
                latency_breakdown_ms={
                    "query_understanding_ms": round(query_plan_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_latency_ms": round(elapsed_ms, 2),
                },
                device_info=device_diag,
            )

        # 3. Decompose Answer into Claims
        claims = self.claim_decomposer.decompose(raw_answer)

        # Edge case: Answer has no substantive claims
        if not claims:
            logger.info("Generated answer for query '%s' contained no substantive claims.", query)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return GroundedAnswerResponse(
                query=query,
                answer=None,
                answerable=False,
                grounded=False,
                groundedness_score=0.0,
                grounding_status="empty",
                claims=[],
                evidence=gen_response.evidence,
                citations=[],
                query_plan=query_plan,
                reason="Generated answer contains no substantive claims to validate.",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=cand_count,
                reranking_applied=rerank_applied,
                latency_breakdown_ms={
                    "query_understanding_ms": round(query_plan_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_latency_ms": round(elapsed_ms, 2),
                },
                device_info=device_diag,
            )

        # 4. NLI Claim-Level Validation via GroundingService
        t3 = time.perf_counter()
        grounding_report: GroundingReport = self.grounding_service.validate_claims(
            claims=claims,
            evidence=gen_response.evidence,
            entailment_threshold=request.entailment_threshold,
            contradiction_threshold=request.contradiction_threshold,
            top_k_evidence=request.top_k_evidence,
            query=query,
        )
        grounding_ms = (time.perf_counter() - t3) * 1000

        # 5. Citation Mapping via CitationService (Phase 10)
        t4 = time.perf_counter()
        try:
            citation_result = self.citation_service.build_citations(
                grounding_result=grounding_report,
                expected_document_id=request.document_id,
            )
            citations = citation_result.citations
        except DocumentIsolationError as exc:
            logger.error("Document isolation violation during citation mapping: %s", exc)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
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
                query_plan=query_plan,
                reason=f"Document isolation error: {exc}",
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=cand_count,
                reranking_applied=rerank_applied,
                regeneration_attempted=regeneration_attempted,
                regeneration_reason=regeneration_reason,
                completeness_status="isolation_error",
                latency_breakdown_ms={
                    "query_understanding_ms": round(query_plan_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "grounding_ms": round(grounding_ms, 2),
                    "total_latency_ms": round(elapsed_ms, 2),
                },
                device_info=device_diag,
            )
        citation_ms = (time.perf_counter() - t4) * 1000

        total_latency_ms = (time.perf_counter() - t_start) * 1000
        latency_breakdown = {
            "query_understanding_ms": round(query_plan_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "grounding_ms": round(grounding_ms, 2),
            "citation_ms": round(citation_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
        }
        logger.info(
            "GroundedAnswer completed in %.1f ms (query_plan=%.1f, retrieval=%.1f, generation=%.1f, grounding=%.1f, citation=%.1f) on %s",
            total_latency_ms,
            query_plan_ms,
            retrieval_ms,
            generation_ms,
            grounding_ms,
            citation_ms,
            device_diag.get("device", "CPU"),
        )

        # 6. Safe Decision Policy
        require_all = (
            request.require_all_claims_supported
            if request.require_all_claims_supported is not None
            else settings.GROUNDING_REQUIRE_ALL_CLAIMS_SUPPORTED
        )

        if require_all:
            # Under strict safe policy: all substantive claims must be entailed with zero contradiction/conflict
            # and answer must be complete across requested components
            is_acceptable = (
                grounding_report.overall_status == "grounded"
                and grounding_report.supported_claims == grounding_report.total_claims
                and grounding_report.contradicted_claims == 0
                and grounding_report.conflicted_claims == 0
                and completeness_status != "incomplete_after_retry"
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
                    query_plan=query_plan,
                    reason=grounding_report.reason,
                    model_name=model_name,
                    grounding_model_name=grounding_model_name,
                    candidate_count=cand_count,
                    reranking_applied=rerank_applied,
                    regeneration_attempted=regeneration_attempted,
                    regeneration_reason=regeneration_reason,
                    completeness_status=completeness_status,
                    latency_breakdown_ms=latency_breakdown,
                    device_info=device_diag,
                )
            else:
                # Safe refusal: suppress ungrounded answer to prevent misleading the user
                logger.warning(
                    "Generated answer rejected by safe policy for query '%s' (status=%s, score=%.2f, completeness=%s).",
                    query,
                    grounding_report.overall_status,
                    grounding_report.groundedness_score,
                    completeness_status,
                )
                safe_reason = (
                    f"Answer is incomplete: {regeneration_reason}"
                    if completeness_status == "incomplete_after_retry"
                    else (
                        grounding_report.reason
                        or "The generated answer contains claims that are not sufficiently supported by the retrieved book evidence."
                    )
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
                    citations=[],
                    query_plan=query_plan,
                    reason=safe_reason,
                    model_name=model_name,
                    grounding_model_name=grounding_model_name,
                    candidate_count=cand_count,
                    reranking_applied=rerank_applied,
                    regeneration_attempted=regeneration_attempted,
                    regeneration_reason=regeneration_reason,
                    completeness_status=completeness_status,
                    latency_breakdown_ms=latency_breakdown,
                    device_info=device_diag,
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
                query_plan=query_plan,
                reason=grounding_report.reason,
                model_name=model_name,
                grounding_model_name=grounding_model_name,
                candidate_count=cand_count,
                reranking_applied=rerank_applied,
                regeneration_attempted=regeneration_attempted,
                regeneration_reason=regeneration_reason,
                completeness_status=completeness_status,
                latency_breakdown_ms=latency_breakdown,
                device_info=device_diag,
            )
